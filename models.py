import numpy as np
import torch
import math
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init


# ---------------------------------------------------------------------------
# Positional Encoding (standard sinusoidal)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1, maxlen: int = 750):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)   # (maxlen, 1, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[:token_embedding.size(0), :]
        )


# ---------------------------------------------------------------------------
# Ring Memory Buffer  (simplified MATR-style, no flag token overhead)
# ---------------------------------------------------------------------------
class RingMemory(nn.Module):
    """
    Maintains a rolling queue of past segment encodings across the video.
    Compatible with simple batch iteration (no DataParallel state issues).
    """
    def __init__(self, memory_len: int, seg_len: int, emb_dim: int, gap: int = 2):
        super(RingMemory, self).__init__()
        self.memory_len = memory_len   # how many past segments to keep
        self.seg_len    = seg_len      # frames per segment
        self.emb_dim    = emb_dim
        self.gap        = gap          # gap-sample every `gap` frames from memory
        self.queue      = None         # (memory_len * seg_len, B, D)
        self.cur_video  = None

    def reset(self, batch_size: int, device: torch.device):
        self.queue = torch.zeros(
            self.memory_len * self.seg_len, batch_size, self.emb_dim, device=device
        )
        self.cur_video = None

    def update(self, current_segment: torch.Tensor, video_names=None):
        """
        Args:
            current_segment: (seg_len, B, D) — encoded current window
            video_names:     list of B strings  (optional, for cross-video reset)
        Returns:
            memory_feats: (memory_len * seg_len // gap + seg_len // gap, B, D)
        """
        B = current_segment.shape[1]
        device = current_segment.device

        if self.queue is None:
            self.reset(B, device)

        # Reset memory for any video that changed
        if video_names is not None and self.cur_video is not None:
            for b, name in enumerate(video_names):
                if b < B and self.cur_video[b] != name:
                    self.queue[:, b, :] = 0.0

        # Shift queue and push current segment
        seg = current_segment.detach()
        self.queue = torch.cat(
            [self.queue[self.seg_len:], seg], dim=0
        )                                               # (memory_len*seg_len, B, D)

        # Gap-sample for efficiency
        memory_feats = torch.cat(
            [self.queue[::self.gap], seg], dim=0
        )                                               # ((mem*seg//gap + seg), B, D)

        self.cur_video = list(video_names) if video_names is not None else None
        return memory_feats


# ---------------------------------------------------------------------------
# History Unit  (adapted from HAT)
#   Two-stage TransformerDecoder:
#     Stage 1:  learnable hist_tokens × long_x  → hist_encoded_1
#     Stage 2:  hist_encoded_1       × encoded_x → hist_encoded
#   Auxiliary snippet classification head.
# ---------------------------------------------------------------------------
class HistoryUnit(nn.Module):
    def __init__(self, n_embedding_dim: int, n_class: int, history_tokens: int = 16,
                 n_dec_head: int = 4, n_dec_layer_1: int = 5, n_dec_layer_2: int = 2,
                 dropout: float = 0.3):
        super(HistoryUnit, self).__init__()
        self.history_tokens = history_tokens

        self.pos_enc = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)

        # Stage-1: compress long history into tokens
        self.enc_block1 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, nhead=n_dec_head,
                                       dropout=dropout, activation='gelu'),
            n_dec_layer_1, nn.LayerNorm(n_embedding_dim)
        )
        # Stage-2: refine with current short-window encodings
        self.enc_block2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, nhead=n_dec_head,
                                       dropout=dropout, activation='gelu'),
            n_dec_layer_2, nn.LayerNorm(n_embedding_dim)
        )

        # Auxiliary snippet head (future-supervised)
        self.snip_head = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim // 4), nn.ReLU()
        )
        snip_in_dim = history_tokens * (n_embedding_dim // 4)
        self.snip_classifier = nn.Sequential(
            nn.Linear(snip_in_dim, snip_in_dim // 4), nn.ReLU(),
            nn.Linear(snip_in_dim // 4, n_class)
        )

        self.hist_token = nn.Parameter(
            torch.zeros(history_tokens, 1, n_embedding_dim)
        )
        self.norm   = nn.LayerNorm(n_embedding_dim)
        self.drop   = nn.Dropout(0.1)

    def forward(self, long_x, encoded_x):
        """
        long_x:    (T_long, B, D)
        encoded_x: (T_short, B, D)
        Returns:
            hist_out: (history_tokens, B, D)   — context for anchor refinement
            snip_cls: (B, n_class)             — auxiliary snippet prediction
        """
        pe_long = self.pos_enc(long_x)
        tokens  = self.hist_token.expand(-1, pe_long.shape[1], -1)

        h1 = self.enc_block1(tokens, pe_long)        # stage-1
        h2 = self.enc_block2(h1, encoded_x)          # stage-2
        h2 = self.norm(h2 + self.drop(h1))           # residual

        # Snippet classification (on stage-1 features)
        snip_feat = self.snip_head(h1)               # (hist_tokens, B, D//4)
        snip_feat = torch.flatten(
            snip_feat.permute(1, 0, 2), start_dim=1  # (B, hist_tokens*D//4)
        )
        snip_cls = self.snip_classifier(snip_feat)   # (B, n_class)

        return h2, snip_cls


# ---------------------------------------------------------------------------
# MYNET  —  STAR (Structured Temporal Action Recognition)
# ---------------------------------------------------------------------------
class MYNET(nn.Module):
    """
    Architecture combines:
      1. OAT's anchor-based Encoder-Decoder
      2. HAT's HistoryUnit with future-supervised snippet head
      3. Simplified ring memory attention (MATR-inspired)
      4. Dual-branch anchor tokens (separate cls / reg decoders)
    """
    def __init__(self, opt):
        super(MYNET, self).__init__()
        self.n_feature      = opt['feat_dim']
        n_class             = opt['num_of_class']
        n_embedding_dim     = opt['hidden_dim']
        n_enc_layer         = opt['enc_layer']
        n_enc_head          = opt['enc_head']
        n_dec_layer         = opt['dec_layer']
        n_dec_head          = opt['dec_head']
        self.anchors        = opt['anchors']
        n_anchors           = len(self.anchors)
        dropout             = 0.3
        self.best_loss      = 1e6
        self.best_map       = 0

        # Short / long window split (same as HAT)
        self.short_window   = opt.get('short_window', 16)
        self.segment_size   = opt['segment_size']
        self.long_window    = self.segment_size - self.short_window

        # ---- Feature projection (two-stream) ----
        self.proj_rgb  = nn.Linear(self.n_feature // 2, n_embedding_dim // 2)
        self.proj_flow = nn.Linear(self.n_feature // 2, n_embedding_dim // 2)

        # ---- Short-window Encoder ----
        self.pos_enc = PositionalEncoding(n_embedding_dim, dropout, maxlen=400)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=n_embedding_dim, nhead=n_enc_head,
                                       dropout=dropout, activation='gelu'),
            n_enc_layer, nn.LayerNorm(n_embedding_dim)
        )

        # ---- Anchor Decoder (short context only, baseline) ----
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, nhead=n_dec_head,
                                       dropout=dropout, activation='gelu'),
            n_dec_layer, nn.LayerNorm(n_embedding_dim)
        )

        # ---- HAT History Unit ----
        self.history_unit = HistoryUnit(
            n_embedding_dim, n_class,
            history_tokens=opt.get('history_tokens', 16),
            n_dec_head=4, n_dec_layer_1=5, n_dec_layer_2=2, dropout=dropout
        )

        # ---- History Anchor Refinement (anchor × history context) ----
        self.hist_refine = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, nhead=4,
                                       dropout=dropout, activation='gelu'),
            5, nn.LayerNorm(n_embedding_dim)
        )

        # ---- Ring Memory ----
        memory_len = opt.get('memory_len', 4)
        mem_gap    = opt.get('memory_gap', 2)
        self.ring_memory = RingMemory(
            memory_len=memory_len, seg_len=self.segment_size,
            emb_dim=n_embedding_dim, gap=mem_gap
        )

        # ---- Memory Anchor Refinement (anchor × ring memory) ----
        self.memory_refine = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=n_embedding_dim, nhead=4,
                                       dropout=dropout, activation='gelu'),
            2, nn.LayerNorm(n_embedding_dim)
        )

        # ---- Dual-branch anchor tokens ----
        #   First n_anchors  → classification branch
        #   Last  n_anchors  → regression branch
        self.anchor_tokens = nn.Parameter(
            torch.zeros(n_anchors * 2, 1, n_embedding_dim)
        )

        # ---- Residual gates ----
        self.norm_hist   = nn.LayerNorm(n_embedding_dim)
        self.drop_hist   = nn.Dropout(0.1)
        self.norm_mem    = nn.LayerNorm(n_embedding_dim)
        self.drop_mem    = nn.Dropout(0.1)

        # ---- Prediction heads ----
        self.classifier = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(),
            nn.Linear(n_embedding_dim, n_class)
        )
        self.regressor = nn.Sequential(
            nn.Linear(n_embedding_dim, n_embedding_dim), nn.ReLU(),
            nn.Linear(n_embedding_dim, 2)
        )

        self.relu       = nn.ReLU(True)
        self.softmaxd1  = nn.Softmax(dim=-1)

        # ---- Weight init ----
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        nn.init.normal_(self.anchor_tokens, std=0.02)
        nn.init.normal_(self.history_unit.hist_token, std=0.02)

    # ------------------------------------------------------------------
    def _project_features(self, inputs):
        """
        inputs: (B, T, feat_dim)
        returns: (T, B, emb_dim)  [seq-first for PyTorch Transformer]
        """
        x_rgb  = self.proj_rgb(inputs[:, :, :self.n_feature // 2])
        x_flow = self.proj_flow(inputs[:, :, self.n_feature // 2:])
        x = torch.cat([x_rgb, x_flow], dim=-1)   # (B, T, D)
        return x.permute(1, 0, 2)                 # (T, B, D)

    # ------------------------------------------------------------------
    def reset_memory(self):
        """Call at the start of each test video (for online inference)."""
        self.ring_memory.queue     = None
        self.ring_memory.cur_video = None

    # ------------------------------------------------------------------
    def forward(self, inputs, video_names=None):
        """
        inputs:      (B, segment_size, feat_dim)
        video_names: optional list[str] for memory reset at video boundaries

        Returns:
            anc_cls:  (B, n_anchors, n_class)
            anc_reg:  (B, n_anchors, 2)
            snip_cls: (B, n_class)           — auxiliary, used during training
        """
        base_x = self._project_features(inputs)   # (T, B, D)

        # ── Split into short (current) and long (history) ──────────────
        short_x = base_x[-self.short_window:]     # (16, B, D)
        long_x  = base_x[:-self.short_window]     # (48, B, D)

        # ── Short-window Encoder ────────────────────────────────────────
        pe_short   = self.pos_enc(short_x)
        encoded_x  = self.encoder(pe_short)       # (16, B, D)

        # ── Anchor token init (decode against short context) ───────────
        n_anchors = len(self.anchors)
        atokens = self.anchor_tokens.expand(-1, encoded_x.shape[1], -1)  # (2A, B, D)
        decoded_x = self.decoder(atokens, encoded_x)                      # (2A, B, D)

        # ── HAT History Unit ────────────────────────────────────────────
        hist_ctx, snip_cls = self.history_unit(long_x, encoded_x)        # (16, B, D), (B, C)

        # ── History-guided anchor refinement ───────────────────────────
        ref1      = self.hist_refine(decoded_x, hist_ctx)
        decoded_x = self.norm_hist(ref1 + self.drop_hist(decoded_x))

        # ── Ring Memory update & attend ────────────────────────────────
        memory_feats = self.ring_memory.update(
            encoded_x, video_names=video_names
        )                                         # ((mem+1)*16//gap, B, D)

        ref2      = self.memory_refine(decoded_x, memory_feats)
        decoded_x = self.norm_mem(ref2 + self.drop_mem(decoded_x))

        # ── Dual-branch prediction ─────────────────────────────────────
        decoded_x = decoded_x.permute(1, 0, 2)   # (B, 2A, D)
        cls_feats  = decoded_x[:, :n_anchors, :]  # (B, A, D)
        reg_feats  = decoded_x[:, n_anchors:, :]  # (B, A, D)

        anc_cls = self.classifier(cls_feats)      # (B, A, n_class)
        anc_reg = self.regressor(reg_feats)        # (B, A, 2)

        return anc_cls, anc_reg, snip_cls


# ---------------------------------------------------------------------------
# SuppressNet  (unchanged from OAT — used for online post-processing)
# ---------------------------------------------------------------------------
class SuppressNet(nn.Module):
    def __init__(self, opt):
        super(SuppressNet, self).__init__()
        n_class       = opt['num_of_class'] - 1
        n_seglen      = opt['segment_size']
        n_embedding_dim = 2 * n_seglen
        self.best_loss = 1e6
        self.best_map  = 0

        self.mlp1    = nn.Linear(n_seglen, n_embedding_dim)
        self.mlp2    = nn.Linear(n_embedding_dim, 1)
        self.norm    = nn.InstanceNorm1d(n_class)
        self.relu    = nn.ReLU(True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # inputs: (B, seg_len, n_class)
        x = inputs.permute(0, 2, 1)   # (B, n_class, seg_len)
        x = self.norm(x)
        x = self.relu(self.mlp1(x))   # (B, n_class, 2*seg_len)
        x = self.sigmoid(self.mlp2(x))# (B, n_class, 1)
        return x.squeeze(-1)           # (B, n_class)
