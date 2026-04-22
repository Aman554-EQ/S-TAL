import argparse


def parse_opt():
    parser = argparse.ArgumentParser(
        description='STAR: Structured Temporal Action Recognition — THUMOS14'
    )

    # ── Mode / Paths ──────────────────────────────────────────────────
    parser.add_argument('--mode',            type=str,   default='train')
    parser.add_argument('--checkpoint_path', type=str,   default='./checkpoint')
    parser.add_argument('--video_anno',      type=str,
                        default='./data/thumos14_v2.json')
    parser.add_argument('--video_feature_all_train', type=str,
                        default='./data/thumos_all_feature_val_V3.pickle')
    parser.add_argument('--video_feature_all_test',  type=str,
                        default='./data/thumos_all_feature_test_V3.pickle')
    parser.add_argument('--result_file',    type=str,
                        default='./output/result_proposal.json')
    parser.add_argument('--frame_result_file', type=str,
                        default='./output/frame_result.h5')
    parser.add_argument('--video_len_file', type=str,
                        default='./output/video_len_{}.json')
    parser.add_argument('--proposal_label_file', type=str,
                        default='./output/proposal_label_{}.h5')
    parser.add_argument('--suppress_label_file',  type=str,
                        default='./output/suppress_label_{}.h5')
    parser.add_argument('--suppress_result_file', type=str,
                        default='./output/suppress_result.h5')

    # ── Dataset ───────────────────────────────────────────────────────
    parser.add_argument('--num_of_class',   type=int,   default=21)
    parser.add_argument('--data_format',    type=str,   default='pickle')
    parser.add_argument('--data_rescale',   default=False, action='store_true')
    parser.add_argument('--predefined_fps', type=float, default=None)
    parser.add_argument('--rgb_only',       default=False, action='store_true')
    parser.add_argument('--inference_subset', type=str, default='test')
    parser.add_argument('--seed',           type=int,   default=52)

    # ── Segment / Anchor ──────────────────────────────────────────────
    parser.add_argument('--segment_size',   type=int,   default=64)
    parser.add_argument('--anchors',        type=str,   default='4,8,16,32,48,64')
    parser.add_argument('--short_window',   type=int,   default=16,
                        help='Frames in the short (current) window; '
                             'rest is the long (history) window.')

    # ── Network ───────────────────────────────────────────────────────
    parser.add_argument('--feat_dim',       type=int,   default=4096)
    parser.add_argument('--hidden_dim',     type=int,   default=1024)
    parser.add_argument('--enc_layer',      type=int,   default=3)
    parser.add_argument('--enc_head',       type=int,   default=8)
    parser.add_argument('--dec_layer',      type=int,   default=5)
    parser.add_argument('--dec_head',       type=int,   default=4)
    parser.add_argument('--history_tokens', type=int,   default=16,
                        help='Number of learnable history compression tokens.')

    # ── Ring Memory (MATR-inspired) ───────────────────────────────────
    parser.add_argument('--memory_len',     type=int,   default=4,
                        help='Number of past segments stored in the ring buffer.')
    parser.add_argument('--memory_gap',     type=int,   default=2,
                        help='Gap-sampling stride when reading from ring memory.')

    # ── Training ──────────────────────────────────────────────────────
    parser.add_argument('--batch_size',     type=int,   default=128)
    parser.add_argument('--epoch',          type=int,   default=5)
    parser.add_argument('--lr',             type=float, default=1e-4,
                        help='Base (peak) learning rate.')
    parser.add_argument('--lr_hist',        type=float, default=1e-6,
                        help='Learning rate for HistoryUnit parameters.')
    parser.add_argument('--weight_decay',   type=float, default=1e-4)
    parser.add_argument('--lr_step',        type=int,   default=3,
                        help='StepLR step size (used only when --use_cosine_lr is False).')

    # ── Cosine Warm-Restart LR ────────────────────────────────────────
    # Fix #5: default is now False (StepLR) to match OAT-OSN-main exactly.
    # Once NaN is resolved, set --use_cosine_lr True to re-enable cosine.
    parser.add_argument('--use_cosine_lr',  type=bool,  default=False)
    parser.add_argument('--lr_T0',          type=int,   default=5,
                        help='Full cycle length (epochs) for cosine LR.')
    parser.add_argument('--lr_Tup',         type=int,   default=1,
                        help='Warm-up epochs within each cosine cycle.')
    parser.add_argument('--lr_gamma',       type=float, default=0.5,
                        help='Decay factor for peak lr after each restart.')

    # ── Loss weights ──────────────────────────────────────────────────
    parser.add_argument('--alpha',          type=float, default=1.0,
                        help='Classification loss weight.')
    parser.add_argument('--beta',           type=float, default=1.0,
                        help='Regression loss weight.')
    parser.add_argument('--gamma',          type=float, default=0.3,
                        help='Snippet (auxiliary) classification loss weight.')

    # ── DIoU regression ───────────────────────────────────────────────
    parser.add_argument('--use_diou',       type=bool,  default=True,
                        help='Add DIoU loss on top of L1 for regression.')
    parser.add_argument('--diou_weight',    type=float, default=1.0,
                        help='Relative weight of DIoU vs L1.')

    # ── Post-processing ───────────────────────────────────────────────
    parser.add_argument('--pptype',         type=str,   default='net',
                        choices=['nms', 'net'])
    parser.add_argument('--threshold',      type=float, default=0.1)
    parser.add_argument('--sup_threshold',  type=float, default=0.1)
    parser.add_argument('--pos_threshold',  type=float, default=0.5)
    parser.add_argument('--soft_nms',       type=float, default=0.3)
    parser.add_argument('--wterm',          type=bool,  default=False)

    return parser.parse_args()
