import numpy as np
import h5py
import json
import torch
import torch.utils.data as data
import os
import pickle
from multiprocessing import Pool


def load_json(file):
    with open(file) as f:
        return json.load(f)


def calc_iou(a, b):
    """IoU between anchor a=[ed, length] and gt b=[ed, length, cls]."""
    st  = a[0] - a[1];  ed  = a[0]
    tst = b[0] - b[1];  ted = b[0]
    sst = min(st, tst); led = max(ed, ted)
    lst = max(st, tst); sed = min(ed, ted)
    return (sed - lst) / max(led - sst, 1)


def box_include(y, target):
    """True if the target is a larger box that contains the anchor y."""
    st  = y[0] - y[1];       ed  = y[0]
    tst = target[0] - target[1]; ted = target[0]
    detect = tst
    return ed > detect and tst < st and ted > ed


class VideoDataSet(data.Dataset):
    """
    THUMOS14 dataset — 4-tuple output:
        feature, cls_label, reg_label, snip_label
    snip_label is a per-window snippet classification label (HAT-style).
    """
    def __init__(self, opt, subset="train"):
        self.subset         = subset
        self.mode           = opt['mode']
        self.predefined_fps = opt['predefined_fps']
        self.video_anno_path = opt['video_anno']
        self.video_len_path  = opt['video_len_file'].format(self.subset)
        self.num_of_class   = opt['num_of_class']
        self.segment_size   = opt['segment_size']
        self.anchors        = opt['anchors']
        self.pos_threshold  = opt['pos_threshold']
        self.data_rescale   = opt['data_rescale']

        self.label_name     = []
        self.match_score    = {}
        self.gt_action      = {}
        self.cls_label      = {}
        self.reg_label      = {}
        self.snip_label     = {}   # ← NEW: snippet-level auxiliary labels
        self.inputs         = []
        self.inputs_all     = []
        self.video_dict     = {}

        self._getDatasetDict()
        self._loadFeaturelen(opt)
        self._getMatchScore()
        self._makeInputSeq()
        self._loadPropLabel(opt['proposal_label_file'].format(self.subset))

        # ── Load feature files ────────────────────────────────────────
        if subset == 'train':
            feat_path = opt['video_feature_all_train']
        else:
            feat_path = opt['video_feature_all_test']

        if opt['data_format'] == 'pickle':
            feature_all = pickle.load(open(feat_path, 'rb'))
            self.feature_rgb_file  = {k: feature_all[k]['rgb']  for k in self.video_list}
            self.feature_flow_file = {k: feature_all[k]['flow'] for k in self.video_list}
        elif opt['data_format'] == 'h5':
            rgb_key  = 'video_feature_rgb_train'  if subset == 'train' else 'video_feature_rgb_test'
            flow_key = 'video_feature_flow_train' if subset == 'train' else 'video_feature_flow_test'
            rgb_file  = h5py.File(opt[rgb_key], 'r')
            self.feature_rgb_file = {k: np.array(rgb_file[k][:]) for k in self.video_list}
            if opt.get('rgb_only', False):
                self.feature_flow_file = None
            else:
                flow_file = h5py.File(opt[flow_key], 'r')
                self.feature_flow_file = {k: np.array(flow_file[k][:]) for k in self.video_list}
        else:
            raise ValueError("Unsupported data_format: %s" % opt['data_format'])

    # ------------------------------------------------------------------
    def _getDatasetDict(self):
        anno = load_json(self.video_anno_path)['database']
        for vname, vinfo in anno.items():
            if self.subset == 'full' or self.subset in vinfo['subset']:
                self.video_dict[vname] = vinfo
            for seg in vinfo['annotations']:
                if seg['label'] not in self.label_name:
                    self.label_name.append(seg['label'])
        self.label_name.sort()
        self.video_list = list(self.video_dict.keys())
        print("%s subset video numbers: %d" % (self.subset, len(self.video_list)))

    # ------------------------------------------------------------------
    def _loadFeaturelen(self, opt):
        if os.path.exists(self.video_len_path):
            self.video_len = load_json(self.video_len_path)
            return
        self.video_len = {}
        if opt['data_format'] == 'pickle':
            feat_path = (opt['video_feature_all_train']
                         if self.subset == 'train'
                         else opt['video_feature_all_test'])
            ff = pickle.load(open(feat_path, 'rb'))
            for k in self.video_list:
                self.video_len[k] = len(ff[k]['rgb'])
        os.makedirs(os.path.dirname(self.video_len_path), exist_ok=True)
        with open(self.video_len_path, 'w') as f:
            json.dump(self.video_len, f, indent=2)

    # ------------------------------------------------------------------
    def _getMatchScore(self):
        for vname in self.video_list:
            vinfo  = self.video_dict[vname]
            labels = vinfo['annotations']
            s2f    = self.video_len[vname] / float(vinfo['duration'])
            gt_edlen = []
            gt_bbox  = []
            for ann in labels:
                st  = ann['segment'][0] * s2f
                ed  = ann['segment'][1] * s2f
                cls = self.label_name.index(ann['label'])
                gt_bbox.append([st, ed, cls])
                gt_edlen.append([ed, ed - st, cls])
            gt_bbox  = np.array(gt_bbox)  if gt_bbox  else np.zeros((0, 3))
            gt_edlen = np.array(gt_edlen) if gt_edlen else np.zeros((0, 3))
            self.gt_action[vname] = gt_edlen

            mlen = self.video_len[vname]
            ms   = np.zeros((mlen, self.num_of_class - 1), dtype=np.float32)
            for i in range(gt_bbox.shape[0]):
                st_f = int(gt_bbox[i, 0])
                ed_f = int(gt_bbox[i, 1]) + 1
                cls  = int(gt_bbox[i, 2])
                ms[st_f:ed_f, cls] = i + 1
            self.match_score[vname] = ms

    # ------------------------------------------------------------------
    def _makeInputSeq(self):
        data_idx = 0
        for vname in self.video_list:
            duration = self.match_score[vname].shape[0]
            for i in range(1, duration + 1):
                self.inputs_all.append([vname, i - self.segment_size, i, data_idx])
                data_idx += 1
        self.inputs = self.inputs_all.copy()
        print("%s subset seg numbers: %d" % (self.subset, len(self.inputs)))

    # ------------------------------------------------------------------
    def _makePropLabelUnit(self, i):
        vname = self.inputs_all[i][0]
        ed    = self.inputs_all[i][2]

        cls_anc = []
        reg_anc = []

        # ── Anchor-level labels ────────────────────────────────────────
        for j, anc in enumerate(self.anchors):
            v1 = np.zeros(self.num_of_class);  v1[-1] = 1
            v2 = np.zeros(2);                  v2[-1] = -1e3
            y_box = [ed - 1, anc]

            subset_label = self._get_train_label_with_class(vname, ed - anc, ed)
            idx_list = []
            for ii in range(subset_label.shape[0]):
                for jj in range(subset_label.shape[1]):
                    idx = int(subset_label[ii, jj])
                    if idx > 0 and (idx - 1) not in idx_list:
                        idx_list.append(idx - 1)

            for idx in idx_list:
                tb  = self.gt_action[vname][idx]
                cls = int(tb[2])
                iou = calc_iou(y_box, tb)
                if (iou >= self.pos_threshold
                        or (j == len(self.anchors) - 1 and box_include(y_box, tb))
                        or (j == 0                     and box_include(tb, y_box))):
                    v1[cls] = 1;  v1[-1] = 0
                    v2[0] = (tb[0] - y_box[0]) / anc
                    v2[1] = np.log(max(1, tb[1]) / y_box[1])

            cls_anc.append(v1)
            reg_anc.append(v2)

        cls_anc = np.stack(cls_anc, axis=0)   # (A, C)
        reg_anc = np.stack(reg_anc, axis=0)   # (A, 2)

        # ── Snippet-level label (HAT-style future-supervised) ──────────
        v0 = np.zeros(self.num_of_class);  v0[-1] = 1
        y_box_snip = [ed - 1, self.anchors[-1]]
        subset_label_snip = self._get_train_label_with_class(
            vname, ed - self.anchors[-1], ed
        )
        snip_idx_list = []
        for ii in range(subset_label_snip.shape[0]):
            for jj in range(subset_label_snip.shape[1]):
                idx = int(subset_label_snip[ii, jj])
                if idx > 0 and (idx - 1) not in snip_idx_list:
                    snip_idx_list.append(idx - 1)
        for idx in snip_idx_list:
            tb  = self.gt_action[vname][idx]
            cls = int(tb[2])
            iou = calc_iou(y_box_snip, tb)
            if iou >= 0:          # any overlap qualifies
                v0[cls] = 1;  v0[-1] = 0

        return cls_anc, reg_anc, v0   # v0: (C,)

    # ------------------------------------------------------------------
    def _loadPropLabel(self, filename):
        """Load pre-computed labels or compute & cache them."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if os.path.exists(filename):
            pf = h5py.File(filename, 'r')
            self.cls_label  = np.array(pf['cls_label'][:])
            self.reg_label  = np.array(pf['reg_label'][:])
            self.snip_label = np.array(pf['snip_label'][:])
            pf.close()
            cnt = np.sum(self.cls_label.reshape(-1, self.cls_label.shape[-1]), axis=0)
            self.action_frame_count = torch.Tensor(cnt)
            return

        pool   = Pool(os.cpu_count() // 2)
        labels = pool.map(self._makePropLabelUnit, range(len(self.inputs_all)))
        pool.close(); pool.join()

        cls_list  = [l[0] for l in labels]
        reg_list  = [l[1] for l in labels]
        snip_list = [l[2] for l in labels]
        self.cls_label  = np.stack(cls_list,  axis=0)
        self.reg_label  = np.stack(reg_list,  axis=0)
        self.snip_label = np.stack(snip_list, axis=0)

        with h5py.File(filename, 'w') as f:
            f.create_dataset('/cls_label',  data=self.cls_label.astype(np.float32))
            f.create_dataset('/reg_label',  data=self.reg_label.astype(np.float32))
            f.create_dataset('/snip_label', data=self.snip_label.astype(np.float32))

        cnt = np.sum(self.cls_label.reshape(-1, self.cls_label.shape[-1]), axis=0)
        self.action_frame_count = torch.Tensor(cnt)

    # ------------------------------------------------------------------
    def __getitem__(self, index):
        vname, st, ed, data_idx = self.inputs[index]
        if st >= 0:
            feature = self._get_base_data(vname, st, ed)
        else:
            feature = self._get_base_data(vname, 0, ed)
            pad = torch.nn.ConstantPad2d((0, 0, -st, 0), 0)
            feature = pad(feature)

        cls_label  = torch.Tensor(self.cls_label[data_idx])    # (A, C)
        reg_label  = torch.Tensor(self.reg_label[data_idx])    # (A, 2)
        snip_label = torch.Tensor(self.snip_label[data_idx])   # (C,)
        return feature, cls_label, reg_label, snip_label

    # ------------------------------------------------------------------
    def _get_base_data(self, vname, st, ed):
        rgb  = self.feature_rgb_file[vname][st:ed, :]
        if self.feature_flow_file is not None:
            flow = self.feature_flow_file[vname][st:ed, :]
            feat = np.append(rgb, flow, axis=1)
        else:
            feat = rgb
        return torch.from_numpy(np.array(feat, dtype=np.float32))

    # ------------------------------------------------------------------
    def _get_train_label_with_class(self, vname, st, ed):
        dur      = len(self.match_score[vname])
        st_pad   = max(0, -st);        st  = max(0, st)
        ed_pad   = max(0, ed - dur);   ed  = min(ed, dur)
        ms       = torch.Tensor(self.match_score[vname][st:ed])
        if st_pad > 0:
            ms = torch.nn.ConstantPad2d((0, 0,  st_pad, 0),      0)(ms)
        if ed_pad > 0:
            ms = torch.nn.ConstantPad2d((0, 0,  0, ed_pad),      0)(ms)
        return ms

    def __len__(self):
        return len(self.inputs)

    def reset_sample(self):
        self.inputs = self.inputs_all.copy()

    def select_sample(self, idx):
        self.inputs = [self.inputs_all[i] for i in idx]


# ---------------------------------------------------------------------------
# SuppressNet dataset (unchanged)
# ---------------------------------------------------------------------------
class SuppressDataSet(data.Dataset):
    def __init__(self, opt, subset="train"):
        self.data_file = h5py.File(opt['suppress_label_file'].format(subset), 'r')
        self.video_list = list(self.data_file.keys())
        self.inputs = []
        for vname in self.video_list:
            dur = self.data_file[vname + '/input'].shape[0]
            for i in range(dur):
                self.inputs.append([vname, i])
        print("%s subset seg numbers: %d" % (subset, len(self.inputs)))

    def __getitem__(self, index):
        vname, idx = self.inputs[index]
        seq   = torch.from_numpy(self.data_file[vname + '/input'][idx])
        label = torch.from_numpy(self.data_file[vname + '/label'][idx])
        return seq, label

    def __len__(self):
        return len(self.inputs)
