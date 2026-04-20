import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Adaptive Gradient-Reweighted Focal Loss  (from HAT, generalised)
# ---------------------------------------------------------------------------
class AdaptiveFocalLoss(nn.Module):
    """
    Focal loss with per-class, automatically tracked gamma.

    During .backward() the hook collect_grad() must be called:
        output.register_hook(partial(loss_fn.collect_grad, targets))

    gamma_base : starting focal exponent for every class
    gamma_f    : maximum extra exponent added for hard neg classes
    num_classes: total number of classes (including background)
    """
    def __init__(self, num_classes: int, gamma_base: float = 0.025,
                 gamma_f: float = 0.05, reduce: bool = True):
        super(AdaptiveFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma_base  = gamma_base
        self.gamma_f     = gamma_f
        self.reduce      = reduce

        self.register_buffer('gamma_', torch.zeros(num_classes) + gamma_base)
        self.register_buffer('pos_grad', torch.zeros(num_classes - 1))
        self.register_buffer('neg_grad', torch.zeros(num_classes - 1))
        self.register_buffer('pos_neg',  torch.ones(num_classes - 1))

    # ------------------------------------------------------------------
    def _map_func(self, x: torch.Tensor, s: float = 1.0) -> torch.Tensor:
        """Sigmoid-like monotone normaliser onto [0, 1]."""
        xmin, xmax = x.min(), x.max()
        mu = x.mean()
        x_norm = (x - xmin) / (xmax - xmin + 1e-8)
        return 1.0 / (1.0 + torch.exp(-s * (x_norm - mu)))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def collect_grad(self, targets: torch.Tensor, grad: torch.Tensor):
        """
        Hook: accumulates positive/negative gradient magnitudes per class.
        targets: (B, ..., C) or (N, C) — same shape as model output
        grad:    same shape as model output
        """
        grad    = torch.abs(grad.reshape(-1, grad.shape[-1]))
        targets = targets.reshape(-1, targets.shape[-1]).to(grad.device).float()
        pos_g   = (grad * targets).sum(0)[:-1]
        neg_g   = (grad * (1.0 - targets)).sum(0)[:-1]
        self.pos_grad += pos_g
        self.neg_grad += neg_g
        ratio = self.pos_grad / (self.neg_grad + 1e-10)
        ratio = ratio.clamp(0, 1)
        self.pos_neg = self._map_func(ratio, s=1.0)

    # ------------------------------------------------------------------
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, C) — unnormalised scores
        targets: (N, C) — multi-hot or soft labels (normalised by caller)
        """
        # normalise multi-hot → probabilities-over-classes
        t_sum = targets.sum(dim=1, keepdim=True)
        t_sum = torch.where(t_sum == 0, torch.ones_like(t_sum), t_sum)
        targets_norm = targets / t_sum

        # Build per-class gamma
        gamma = self.gamma_.clone()
        gamma[:-1] = gamma[:-1] + self.gamma_f * (1.0 - self.pos_neg)

        logsoftmax = nn.LogSoftmax(dim=1)
        softmax    = nn.Softmax(dim=1)

        p   = softmax(logits)
        p_clamped = p.clamp(min=1e-8, max=1.0 - 1e-8)
        loss = torch.sum(
            -targets_norm * (1.0 - p_clamped) ** gamma * logsoftmax(logits),
            dim=1
        )
        return loss.mean() if self.reduce else loss


# ---------------------------------------------------------------------------
# 1-D Distance-IoU Loss  (adapted from MATR / fvcore)
# ---------------------------------------------------------------------------
def diou_loss_1d(
    pred: torch.Tensor,   # (N, 2)  — [end_offset, log_length_ratio]
    target: torch.Tensor, # (N, 2)  — same format
    reduction: str = 'mean',
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Converts (end_offset, log_len_ratio) → absolute [start, end] intervals,
    then computes 1-D DIoU.

    OAT regression format:
        pred[:, 0]  = (gt_ed − anchor_ed) / anchor
        pred[:, 1]  = log(gt_len / anchor)
    We work in an arbitrary-scale space: just need relative differences.
    """
    # Recover interval endpoints from OAT-style regression
    # (anchor cancels in IoU computation → keep relative)
    pred_ed  = pred[:, 0]
    pred_len = torch.exp(pred[:, 1].clamp(max=16.0)).clamp(min=eps)
    pred_st  = pred_ed - pred_len

    tgt_ed  = target[:, 0]
    tgt_len = torch.exp(target[:, 1].clamp(max=16.0)).clamp(min=eps)
    tgt_st  = tgt_ed - tgt_len

    # Intersection
    i_st = torch.max(pred_st, tgt_st)
    i_ed = torch.min(pred_ed, tgt_ed)
    inter = (i_ed - i_st).clamp(min=0)

    # Union
    union = (pred_ed - pred_st) + (tgt_ed - tgt_st) - inter
    iou   = inter / (union + eps)

    # Smallest enclosing interval
    enc_st = torch.min(pred_st, tgt_st)
    enc_ed = torch.max(pred_ed, tgt_ed)
    enc_len = (enc_ed - enc_st).clamp(min=eps)

    # Centre distance penalty
    pred_c  = 0.5 * (pred_st + pred_ed)
    tgt_c   = 0.5 * (tgt_st  + tgt_ed)
    rho     = (pred_c - tgt_c).abs()

    loss = 1.0 - iou + (rho / enc_len) ** 2

    if reduction == 'mean':
        return loss.mean() if loss.numel() > 0 else loss.sum() * 0.0
    elif reduction == 'sum':
        return loss.sum()
    return loss


# ---------------------------------------------------------------------------
# Convenience wrappers (match existing call-sites in main.py)
# ---------------------------------------------------------------------------
def cls_loss_func(y, output, loss_fn=None, reduce=True):
    """
    y      : (B, A, C) or (B, C)  multi-hot labels
    output : (B, A, C) or (B, C)  logits
    loss_fn: AdaptiveFocalLoss instance (optional; created fresh if None)
    """
    input_size = y.size()
    y      = y.float().cuda()
    output = output.cuda()

    if loss_fn is None:
        loss_fn = AdaptiveFocalLoss(num_classes=y.shape[-1], reduce=reduce).cuda()

    y_flat  = y.reshape(-1, y.size(-1))
    out_flat = output.reshape(-1, output.size(-1))
    loss = loss_fn(out_flat, y_flat)

    if not reduce:
        loss = loss.reshape(input_size[:-1])
    return loss


def regress_loss_func(y, output, use_diou: bool = True, diou_weight: float = 1.0):
    """
    y      : (B, A, 2) — regression labels (background flagged by y[:,:,1] < -1e2)
    output : (B, A, 2) — regression predictions
    """
    y      = y.float().cuda()
    output = output.cuda()

    y_flat   = y.reshape(-1, y.size(-1))
    out_flat = output.reshape(-1, output.size(-1))

    bgmask   = y_flat[:, 1] < -1e2
    fg_pred  = out_flat[~bgmask]
    fg_tgt   = y_flat[~bgmask]

    if fg_pred.numel() == 0:
        return torch.tensor(0.0, requires_grad=True, device=output.device)

    # L1 loss
    loss_l1 = F.l1_loss(fg_pred, fg_tgt)
    if loss_l1.isnan():
        loss_l1 = torch.tensor(0.0, requires_grad=True, device=output.device)

    if not use_diou:
        return loss_l1

    # DIoU loss
    loss_diou = diou_loss_1d(fg_pred, fg_tgt, reduction='mean')
    if loss_diou.isnan():
        loss_diou = torch.tensor(0.0, requires_grad=True, device=output.device)

    return loss_l1 + diou_weight * loss_diou


def suppress_loss_func(y, output):
    y      = y.float().cuda()
    output = output.cuda()
    y_flat  = y.reshape(-1, y.size(-1))
    out_flat = output.reshape(-1, output.size(-1))
    return F.binary_cross_entropy(out_flat, y_flat)
