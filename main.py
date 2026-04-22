import os
import json
import math
import torch
import torchvision
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import opts_thumos as opts
import time
import h5py
from functools import partial
from tqdm import tqdm
from iou_utils import *
from eval import evaluation_detection
from tensorboardX import SummaryWriter
from dataset import VideoDataSet
from models import MYNET, SuppressNet
from loss_func import (AdaptiveFocalLoss, cls_loss_func,
                       regress_loss_func, suppress_loss_func)


# ---------------------------------------------------------------------------
# JSON-safe cast: numpy scalars / arrays are not serialisable by default
# ---------------------------------------------------------------------------
def _to_py(x):
    """Recursively cast numpy scalars/arrays to native Python types."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, list):
        return [_to_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_py(v) for k, v in x.items()}
    return x


class _NumpyEncoder(json.JSONEncoder):
    """Drop-in JSONEncoder that handles all numpy scalar/array types."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
def is_main_process():
    """True if this is rank 0 (or DDP is not being used)."""
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def setup_ddp():
    """Initialise the default process group for DDP."""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Cosine Annealing Warm-Up Restart (MATR-style scheduler)
# ---------------------------------------------------------------------------
class CosineAnnealingWarmUpRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warm-up at the start of each restart cycle.
    Args:
        T_0    : cycle length in steps
        T_up   : warm-up steps within each cycle
        eta_max: peak learning rate
        gamma  : decay factor for eta_max after each restart
    """
    def __init__(self, optimizer, T_0, T_up=0, eta_max=1e-4, gamma=1.0,
                 last_epoch=-1):
        self.T_0     = T_0
        self.T_up    = T_up
        self.eta_max = eta_max
        self.gamma   = gamma
        self.cycle   = 0
        self.T_cur   = last_epoch
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            # linear warm-up
            return [
                self.eta_max * self.T_cur / self.T_up
                for _ in self.base_lrs
            ]
        else:
            # cosine decay
            t = self.T_cur - self.T_up
            T = self.T_0 - self.T_up
            return [
                self.eta_max * (1 + math.cos(math.pi * t / T)) / 2
                for _ in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.T_cur = epoch
        if self.T_cur >= self.T_0:
            self.cycle  += 1
            self.T_cur   = self.T_cur - self.T_0
            self.eta_max = self.eta_max * self.gamma
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


# ---------------------------------------------------------------------------
# Training — one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(opt, model, train_dataset, optimizer,
                    cls_loss_fn, snip_loss_fn,
                    warmup=False, sampler=None):

    # Shuffle sampler for DDP; standard DataLoader shuffle otherwise
    use_ddp = dist.is_available() and dist.is_initialized()
    num_workers = opt.get('num_workers', 8)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt['batch_size'],
        shuffle=(sampler is None),       # shuffle only when no sampler
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ── Reset ring memory at epoch start ─────────────────────────────────
    raw = model.module if hasattr(model, 'module') else model
    raw.ring_memory.queue     = None
    raw.ring_memory.cur_video = None

    epoch_cost      = 0.0
    epoch_cost_cls  = 0.0
    epoch_cost_reg  = 0.0
    epoch_cost_snip = 0.0
    total_iter = max(1, len(train_dataset) // opt['batch_size'])

    pbar = tqdm(train_loader, desc="  Train", leave=False,
                dynamic_ncols=True, mininterval=1.0,
                bar_format="{l_bar}{bar:30}{r_bar}",
                disable=not is_main_process())

    for n_iter, (input_data, cls_label, reg_label, snip_label) in enumerate(pbar):
        if warmup:
            for g in optimizer.param_groups:
                g['lr'] = (n_iter + 1) * opt['lr'] / total_iter

        input_data = input_data.cuda(non_blocking=True)

        # Forward (3 outputs)
        act_cls, act_reg, snip_cls = model(input_data)

        # ── Register gradient hooks for adaptive focal loss ───────────
        act_cls.register_hook(partial(cls_loss_fn.collect_grad,  cls_label))
        snip_cls.register_hook(partial(snip_loss_fn.collect_grad, snip_label))

        # ── Classification loss ───────────────────────────────────────
        cost_cls  = cls_loss_func(cls_label, act_cls, loss_fn=cls_loss_fn)
        epoch_cost_cls += cost_cls.detach().cpu().item()

        # ── Regression loss (L1 + DIoU) ──────────────────────────────
        cost_reg  = regress_loss_func(reg_label, act_reg,
                                      use_diou=opt.get('use_diou', True),
                                      diou_weight=opt.get('diou_weight', 1.0))
        epoch_cost_reg += cost_reg.detach().cpu().item()

        # ── Snippet classification loss (auxiliary) ───────────────────
        cost_snip = cls_loss_func(snip_label, snip_cls,
                                  loss_fn=snip_loss_fn)
        epoch_cost_snip += cost_snip.detach().cpu().item()

        # ── Total loss ────────────────────────────────────────────────
        cost = (opt['alpha'] * cost_cls
                + opt['beta']  * cost_reg
                + opt['gamma'] * cost_snip)
        epoch_cost += cost.detach().cpu().item()

        optimizer.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if is_main_process():
            pbar.set_postfix({
                'loss': f"{cost.item():.4f}",
                'cls':  f"{cost_cls.item():.4f}",
                'reg':  f"{cost_reg.item():.4f}",
                'snip': f"{cost_snip.item():.4f}",
                'lr':   f"{optimizer.param_groups[-1]['lr']:.1e}",
            })

    return n_iter, epoch_cost, epoch_cost_cls, epoch_cost_reg, epoch_cost_snip


# ---------------------------------------------------------------------------
# Evaluation — one epoch (no grad, 3-output model)
# ---------------------------------------------------------------------------
def eval_one_epoch(opt, model, test_dataset):
    (cls_loss, reg_loss, tot_loss,
     output_cls, output_reg,
     labels_cls, labels_reg,
     working_time, total_frames) = eval_frame(opt, model, test_dataset)

    result_dict = eval_map_nms(opt, test_dataset, output_cls, output_reg,
                                labels_cls, labels_reg)
    output_dict = {"version": "VERSION 1.3", "results": result_dict,
                   "external_data": {}}
    with open(opt['result_file'], 'w') as f:
        json.dump(output_dict, f, indent=2, cls=_NumpyEncoder)

    IoUmAP   = evaluation_detection(opt, verbose=False)
    IoUmAP_5 = sum(IoUmAP) / len(IoUmAP)
    return cls_loss, reg_loss, tot_loss, IoUmAP_5


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(opt):
    # ── DDP vs single-GPU setup ────────────────────────────────────────
    use_ddp = 'LOCAL_RANK' in os.environ

    if use_ddp:
        local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process():
            tqdm.write(f"\n🚀  DDP — rank {local_rank} / {get_world_size()} GPUs")
    else:
        _n_gpus = torch.cuda.device_count()
        local_rank = 0
        device = torch.device('cuda:0')
        if _n_gpus > 1:
            tqdm.write(f"\n⚠️  {_n_gpus} GPUs found but launched without torchrun.")
            tqdm.write("    For best performance run:")
            tqdm.write("    torchrun --nproc_per_node=2 main.py [args]")
            tqdm.write("    Falling back to DataParallel for now.\n")
        else:
            tqdm.write("\nℹ️  Single GPU detected")

    writer = SummaryWriter() if is_main_process() else None

    # ── Build model ───────────────────────────────────────────────────
    raw_model = MYNET(opt).to(device)

    if use_ddp:
        model = DDP(raw_model, device_ids=[local_rank],
                    output_device=local_rank, find_unused_parameters=False)
    elif torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(raw_model)
        opt   = dict(opt)
        opt['batch_size'] = opt['batch_size'] * torch.cuda.device_count()
    else:
        model = raw_model

    # ── Differential LR: history unit gets a much lower LR ───────────
    hist_params  = list(raw_model.history_unit.parameters())
    hist_ids     = set(id(p) for p in hist_params)
    other_params = [p for p in raw_model.parameters() if id(p) not in hist_ids]

    optimizer = optim.Adam([
        {'params': hist_params,  'lr': opt.get('lr_hist', 1e-6)},
        {'params': other_params}
    ], lr=opt['lr'], weight_decay=opt['weight_decay'])

    # ── Scheduler ────────────────────────────────────────────────────
    if opt.get('use_cosine_lr', True):
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0     = opt.get('lr_T0',   opt['epoch']),
            T_up    = opt.get('lr_Tup',  1),
            eta_max = opt['lr'],
            gamma   = opt.get('lr_gamma', 0.5)
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt['lr_step']
        )

    # ── Adaptive focal loss objects ───────────────────────────────────
    n_class      = opt['num_of_class']
    cls_loss_fn  = AdaptiveFocalLoss(num_classes=n_class).to(device)
    snip_loss_fn = AdaptiveFocalLoss(num_classes=n_class).to(device)

    train_dataset = VideoDataSet(opt, subset='train')
    test_dataset  = VideoDataSet(opt, subset=opt['inference_subset'])

    # ── DDP sampler (ensures each rank sees a different shard) ────────
    train_sampler = (DistributedSampler(train_dataset,
                                        num_replicas=get_world_size(),
                                        rank=get_rank(),
                                        shuffle=True)
                     if use_ddp else None)

    epoch_bar = tqdm(range(opt['epoch']), desc="Epochs", unit="ep",
                     bar_format="{l_bar}{bar:20}{r_bar}",
                     disable=not is_main_process())

    for n_epoch in epoch_bar:
        warmup = (n_epoch == 0)

        # DDP sampler must be told the epoch so shuffling differs each time
        if train_sampler is not None:
            train_sampler.set_epoch(n_epoch)

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        (n_iter, epoch_cost, epoch_cost_cls,
         epoch_cost_reg, epoch_cost_snip) = train_one_epoch(
            opt, model, train_dataset, optimizer,
            cls_loss_fn, snip_loss_fn,
            warmup=warmup, sampler=train_sampler
        )

        avg_loss = epoch_cost      / (n_iter + 1)
        avg_cls  = epoch_cost_cls  / (n_iter + 1)
        avg_reg  = epoch_cost_reg  / (n_iter + 1)
        avg_snip = epoch_cost_snip / (n_iter + 1)
        cur_lr   = optimizer.param_groups[-1]['lr']

        if writer:
            writer.add_scalars('data/cost', {'train': avg_loss}, n_epoch)

        # ── Eval (only on rank 0 to avoid duplicate JSON writes) ──────
        scheduler.step()
        IoUmAP_5 = 0.0
        if is_main_process():
            model.eval()
            with torch.no_grad():
                cls_loss, reg_loss, tot_loss, IoUmAP_5 = eval_one_epoch(
                    opt, model, test_dataset
                )
            if writer:
                writer.add_scalars('data/mAP', {'test': IoUmAP_5}, n_epoch)

        # Broadcast best mAP to all ranks so checkpoint logic is consistent
        if use_ddp:
            t = torch.tensor([IoUmAP_5], device=device)
            dist.broadcast(t, src=0)
            IoUmAP_5 = t.item()

        # ── Checkpoint (rank 0 only) ───────────────────────────────────
        if is_main_process():
            state   = {'epoch': n_epoch + 1, 'state_dict': raw_model.state_dict()}
            torch.save(state, opt['checkpoint_path'] + '/checkpoint.pth.tar')
            is_best = IoUmAP_5 > raw_model.best_map
            if is_best:
                raw_model.best_map = IoUmAP_5
                torch.save(state, opt['checkpoint_path'] + '/ckp_best.pth.tar')

            epoch_bar.set_postfix({
                'loss':  f"{avg_loss:.4f}",
                'cls':   f"{avg_cls:.4f}",
                'reg':   f"{avg_reg:.4f}",
                'snip':  f"{avg_snip:.4f}",
                'eloss': f"{tot_loss:.4f}",
                'mAP':   f"{IoUmAP_5:.4f}" + (" ✓best" if is_best else ""),
                'lr':    f"{cur_lr:.1e}",
            })
            tqdm.write(
                f"[Epoch {n_epoch:02d}] "
                f"train={avg_loss:.4f}  cls={avg_cls:.4f}  reg={avg_reg:.4f}  snip={avg_snip:.4f} | "
                f"eval={tot_loss:.4f}  mAP={IoUmAP_5:.4f}"
                f"{'  ← best' if is_best else ''}  lr={cur_lr:.1e}"
            )

    if writer:
        writer.close()

    cleanup_ddp()
    return raw_model.best_map


# ---------------------------------------------------------------------------
# Frame-level evaluation (no grad)
# ---------------------------------------------------------------------------
def eval_frame(opt, model, dataset):
    num_workers = opt.get('num_workers', 8)
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    labels_cls = {v: [] for v in dataset.video_list}
    labels_reg = {v: [] for v in dataset.video_list}
    output_cls  = {v: [] for v in dataset.video_list}
    output_reg  = {v: [] for v in dataset.video_list}

    start_time   = time.time()
    total_frames = 0
    epoch_cost = epoch_cost_cls = epoch_cost_reg = 0.0

    n_class     = opt['num_of_class']
    eval_cls_fn = AdaptiveFocalLoss(num_classes=n_class).cuda()

    eval_bar = tqdm(test_loader, desc="  Eval ", leave=False,
                    dynamic_ncols=True, mininterval=1.0,
                    bar_format="{l_bar}{bar:30}{r_bar}",
                    disable=not is_main_process())

    for n_iter, (input_data, cls_label, reg_label, _) in enumerate(eval_bar):
        input_data = input_data.cuda(non_blocking=True)
        with torch.no_grad():
            act_cls, act_reg, _ = model(input_data)

        cost_cls = cls_loss_func(cls_label, act_cls, loss_fn=eval_cls_fn)
        cost_reg = regress_loss_func(reg_label, act_reg,
                                     use_diou=opt.get('use_diou', True))
        cost     = opt['alpha'] * cost_cls + opt['beta'] * cost_reg

        epoch_cost_cls += cost_cls.detach().cpu().item()
        epoch_cost_reg += cost_reg.detach().cpu().item()
        epoch_cost     += cost.detach().cpu().item()

        act_cls      = torch.softmax(act_cls, dim=-1)
        total_frames += input_data.size(0)

        for b in range(input_data.size(0)):
            idx = n_iter * opt['batch_size'] + b
            if idx >= len(dataset.inputs):
                break
            video_name, st, ed, data_idx = dataset.inputs[idx]
            output_cls[video_name].append(act_cls[b].detach().cpu().numpy())
            output_reg[video_name].append(act_reg[b].detach().cpu().numpy())
            labels_cls[video_name].append(cls_label[b].numpy())
            labels_reg[video_name].append(reg_label[b].numpy())

        if is_main_process():
            eval_bar.set_postfix(loss=f"{cost.item():.4f}")

    end_time     = time.time()
    working_time = end_time - start_time

    for v in dataset.video_list:
        if labels_cls[v]:
            labels_cls[v] = np.stack(labels_cls[v], axis=0)
            labels_reg[v] = np.stack(labels_reg[v], axis=0)
            output_cls[v] = np.stack(output_cls[v], axis=0)
            output_reg[v] = np.stack(output_reg[v], axis=0)

    n = max(1, n_iter)
    return (epoch_cost_cls / n, epoch_cost_reg / n, epoch_cost / n,
            output_cls, output_reg, labels_cls, labels_reg,
            working_time, total_frames)


# ---------------------------------------------------------------------------
# NMS-based proposal generation
# ---------------------------------------------------------------------------
def eval_map_nms(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    result_dict   = {}
    proposal_dict = []
    anchors       = opt['anchors']

    for video_name in dataset.video_list:
        duration   = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]['duration'])
        frame_to_time = 100.0 * video_time / duration

        for idx in range(duration):
            cls_anc = output_cls[video_name][idx]   # (A, C)
            reg_anc = output_reg[video_name][idx]   # (A, 2)

            prop_list = []
            for anc_idx, anc in enumerate(anchors):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)
                if len(cls) == 0:
                    continue
                ed     = idx + anc * reg_anc[anc_idx][0]
                length = anc * np.exp(reg_anc[anc_idx][1])
                st     = ed - length
                for label in cls:
                    prop_list.append({
                        'segment': [float(st * frame_to_time / 100.0),
                                    float(ed * frame_to_time / 100.0)],
                        'score':   float(cls_anc[anc_idx][label]),
                        'label':   dataset.label_name[int(label)],
                        'gentime': float(idx * frame_to_time / 100.0),
                    })

            proposal_dict += prop_list

        proposal_dict = non_max_suppression(proposal_dict,
                                            overlapThresh=opt['soft_nms'])
        result_dict[video_name] = proposal_dict
        proposal_dict = []

    return result_dict


# ---------------------------------------------------------------------------
# SuppressNet-based post-processing
# ---------------------------------------------------------------------------
def eval_map_supnet(opt, dataset, output_cls, output_reg, labels_cls, labels_reg):
    model = SuppressNet(opt).cuda()
    ckp   = torch.load(opt['checkpoint_path'] + '/ckp_best_suppress.pth.tar')
    model.load_state_dict(ckp['state_dict'])
    model.eval()

    result_dict   = {}
    proposal_dict = []
    anchors       = opt['anchors']
    num_class     = opt['num_of_class']
    unit_size     = opt['segment_size']

    for video_name in dataset.video_list:
        duration   = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]['duration'])
        frame_to_time = 100.0 * video_time / duration
        conf_queue = torch.zeros((unit_size, num_class - 1))

        for idx in range(duration):
            cls_anc = output_cls[video_name][idx]
            reg_anc = output_reg[video_name][idx]

            prop_list = []
            for anc_idx, anc in enumerate(anchors):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)
                if len(cls) == 0:
                    continue
                ed     = idx + anc * reg_anc[anc_idx][0]
                length = anc * np.exp(reg_anc[anc_idx][1])
                st     = ed - length
                for label in cls:
                    prop_list.append({
                        'segment': [float(st * frame_to_time / 100.0),
                                    float(ed * frame_to_time / 100.0)],
                        'score':   float(cls_anc[anc_idx][label]),
                        'label':   dataset.label_name[int(label)],
                        'gentime': float(idx * frame_to_time / 100.0),
                    })

            prop_list = non_max_suppression(prop_list, overlapThresh=opt['soft_nms'])

            # ── Use torch.roll for O(1) queue shift ───────────────────
            conf_queue = torch.roll(conf_queue, -1, dims=0)
            conf_queue[-1, :] = 0
            for proposal in prop_list:
                cidx = dataset.label_name.index(proposal['label'])
                conf_queue[-1, cidx] = proposal['score']

            with torch.no_grad():
                suppress_conf = model(conf_queue.unsqueeze(0).cuda())
                suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()

            for cls in range(num_class - 1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for p in prop_list:
                        if p['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(proposal_dict, p,
                                                       overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(p)

        result_dict[video_name] = proposal_dict
        proposal_dict = []

    return result_dict


# ---------------------------------------------------------------------------
# Test / inference helpers
# ---------------------------------------------------------------------------
def test_frame(opt):
    model = MYNET(opt).cuda()
    ckp   = torch.load(opt['checkpoint_path'] + '/ckp_best.pth.tar')
    model.load_state_dict(ckp['state_dict'])
    model.eval()

    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    outfile = h5py.File(opt['frame_result_file'], 'w')

    (cls_loss, reg_loss, tot_loss,
     output_cls, output_reg,
     labels_cls, labels_reg,
     working_time, total_frames) = eval_frame(opt, model, dataset)

    print("test loss: %.4f  cls: %.4f  reg: %.4f" % (tot_loss, cls_loss, reg_loss))
    for vn in dataset.video_list:
        for key, arr in [('pred_cls',   output_cls[vn]),
                         ('pred_reg',   output_reg[vn]),
                         ('label_cls',  labels_cls[vn]),
                         ('label_reg',  labels_reg[vn])]:
            ds = outfile.create_dataset(vn + '/' + key, arr.shape,
                                        maxshape=arr.shape, chunks=True,
                                        dtype=np.float32)
            ds[:] = arr
    outfile.close()
    print("speed: %.1f fps  (%d frames in %.1fs)" % (
        total_frames / working_time, total_frames, working_time))


def test(opt):
    model = MYNET(opt).cuda()
    ckp   = torch.load(opt['checkpoint_path'] + '/ckp_best.pth.tar')
    model.load_state_dict(ckp['state_dict'])
    model.eval()

    dataset = VideoDataSet(opt, subset=opt['inference_subset'])
    (cls_loss, reg_loss, tot_loss,
     output_cls, output_reg,
     labels_cls, labels_reg, _, _) = eval_frame(opt, model, dataset)

    if opt['pptype'] == 'nms':
        result_dict = eval_map_nms(opt, dataset, output_cls, output_reg,
                                   labels_cls, labels_reg)
    elif opt['pptype'] == 'net':
        result_dict = eval_map_supnet(opt, dataset, output_cls, output_reg,
                                      labels_cls, labels_reg)
    else:
        raise ValueError("Unknown pptype: %s" % opt['pptype'])

    output_dict = {"version": "VERSION 1.3", "results": result_dict,
                   "external_data": {}}
    with open(opt['result_file'], 'w') as f:
        json.dump(output_dict, f, indent=2, cls=_NumpyEncoder)
    evaluation_detection(opt)


def test_online(opt):
    """True online inference: one frame at a time, memory carried across frames."""
    model = MYNET(opt).cuda()
    ckp   = torch.load(opt['checkpoint_path'] + '/ckp_best.pth.tar')
    model.load_state_dict(ckp['state_dict'])
    model.eval()
    model.reset_memory()

    sup_model = SuppressNet(opt).cuda()
    ckp_sup   = torch.load(opt['checkpoint_path'] + '/ckp_best_suppress.pth.tar')
    sup_model.load_state_dict(ckp_sup['state_dict'])
    sup_model.eval()

    dataset   = VideoDataSet(opt, subset=opt['inference_subset'])
    num_class = opt['num_of_class']
    unit_size = opt['segment_size']
    anchors   = opt['anchors']

    result_dict  = {}
    start_time   = time.time()
    total_frames = 0

    for video_name in dataset.video_list:
        model.reset_memory()
        input_queue = torch.zeros((unit_size, opt['feat_dim']))
        sup_queue   = torch.zeros((unit_size, num_class - 1))
        proposal_dict = []

        duration   = dataset.video_len[video_name]
        video_time = float(dataset.video_dict[video_name]['duration'])
        frame_to_time = 100.0 * video_time / duration

        for idx in range(duration):
            total_frames += 1

            # ── O(1) roll instead of clone-shift ─────────────────────
            input_queue = torch.roll(input_queue, -1, dims=0)
            input_queue[-1, :] = dataset._get_base_data(video_name, idx, idx + 1)

            minput = input_queue.unsqueeze(0)
            with torch.no_grad():
                act_cls, act_reg, _ = model(minput.cuda(),
                                            video_names=[video_name])
                act_cls = torch.softmax(act_cls, dim=-1)

            cls_anc = act_cls.squeeze(0).detach().cpu().numpy()
            reg_anc = act_reg.squeeze(0).detach().cpu().numpy()

            prop_list = []
            for anc_idx, anc in enumerate(anchors):
                cls = np.argwhere(cls_anc[anc_idx][:-1] > opt['threshold']).reshape(-1)
                if len(cls) == 0:
                    continue
                ed     = idx + anc * reg_anc[anc_idx][0]
                length = anc * np.exp(reg_anc[anc_idx][1])
                st     = ed - length
                for label in cls:
                    prop_list.append({
                        'segment': [float(st * frame_to_time / 100.0),
                                    float(ed * frame_to_time / 100.0)],
                        'score':   float(cls_anc[anc_idx][label]),
                        'label':   dataset.label_name[int(label)],
                        'gentime': float(idx * frame_to_time / 100.0),
                    })

            prop_list = non_max_suppression(prop_list, overlapThresh=opt['soft_nms'])

            # ── O(1) roll for sup_queue ───────────────────────────────
            sup_queue = torch.roll(sup_queue, -1, dims=0)
            sup_queue[-1, :] = 0
            for p in prop_list:
                cidx = dataset.label_name.index(p['label'])
                sup_queue[-1, cidx] = p['score']

            with torch.no_grad():
                suppress_conf = sup_model(sup_queue.unsqueeze(0).cuda())
                suppress_conf = suppress_conf.squeeze(0).detach().cpu().numpy()

            for cls in range(num_class - 1):
                if suppress_conf[cls] > opt['sup_threshold']:
                    for p in prop_list:
                        if p['label'] == dataset.label_name[cls]:
                            if check_overlap_proposal(
                                    proposal_dict, p,
                                    overlapThresh=opt['soft_nms']) is None:
                                proposal_dict.append(p)

        result_dict[video_name] = proposal_dict

    working_time = time.time() - start_time
    print("Online speed: %.1f fps  (%d frames in %.1fs)" % (
        total_frames / working_time, total_frames, working_time))

    output_dict = {"version": "VERSION 1.3", "results": result_dict,
                   "external_data": {}}
    with open(opt['result_file'], 'w') as f:
        json.dump(output_dict, f, indent=2, cls=_NumpyEncoder)
    evaluation_detection(opt)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(opt):
    max_perf = 0
    if opt['mode'] == 'train':
        max_perf = train(opt)
    elif opt['mode'] == 'test':
        test(opt)
    elif opt['mode'] == 'test_frame':
        test_frame(opt)
    elif opt['mode'] == 'test_online':
        test_online(opt)
    elif opt['mode'] == 'eval':
        evaluation_detection(opt)
    return max_perf


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    os.makedirs(opt['checkpoint_path'], exist_ok=True)

    # Save opts only on rank 0 (or when not using DDP)
    if 'LOCAL_RANK' not in os.environ or int(os.environ.get('LOCAL_RANK', 0)) == 0:
        with open(opt['checkpoint_path'] + '/opts.json', 'w') as f:
            json.dump(opt, f)

    if opt['seed'] >= 0:
        torch.manual_seed(opt['seed'])
        np.random.seed(opt['seed'])

    opt['anchors'] = [int(x) for x in opt['anchors'].split(',')]

    main(opt)
    while opt['wterm']:
        pass