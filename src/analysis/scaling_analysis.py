"""
scaling_analysis.py
===================
Data-scaling analysis for a single img→lis MEG mapping architecture.

For each held-out subject s, and for each training-set size k = 1…N-1,
we train the model on every combination of k subjects drawn from the
remaining N-1 subjects, then test on all 40 imagined trials of subject s.

Usage
-----
    python scaling_analysis.py --model RNN_full
    python scaling_analysis.py --model LinearLag_full --max_combos 200
    python scaling_analysis.py --model CNN1D_windowed --heldout sub-01
    python scaling_analysis.py --model RNN_full --save_models

Arguments
---------
  --model          Model key, e.g. RNN_full, CNN1D_windowed, LinearLag_full
  --heldout        Run for one held-out subject only (default: all 13)
  --max_combos     Max combinations to sample per k when C(N-1,k) > this
                   value.  All exact combos used when C(N-1,k) ≤ max_combos.
                   Default: 50.  Set higher (or 0 = unlimited) for ridge.
  --combo_seed     RNG seed for sampling combinations (default: 42)
  --save_models    Also save model checkpoints for each combination

Outputs (under scaling_out/{model_key}/)
-----------------------------------------
  heldout_{s}/k_{k:02d}_results.json
      List of records, one per combination:
        { combo_idx, subjects, mean_r, mean_r_per_trial: [40 floats] }
  heldout_{s}/r_arrays/k{k:02d}_c{c:04d}_r_per_trial.npy   shape (40, C)
  heldout_{s}/models/k{k:02d}_c{c:04d}_{model_key}.pt      (if --save_models)

  summary.json          mean ± std of mean_r per (s, k) across combinations
  scaling_curve.png     mean r vs k, all held-out subjects overlaid + grand mean
  scaling_heatmap.png   heatmap: rows = subjects, cols = k, cell = mean r
"""

import argparse
import json
import os
import random
import time
from copy import deepcopy
from itertools import combinations as iter_combinations
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import mne
mne.set_log_level("ERROR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
#  CONFIG  (identical to benchmark_loso.py)
# =============================================================================
BASE_PATH = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
COND_ALL = [
    "melody1lis", "melody2lis", "poem1lis", "poem2lis",
    "melody1img", "melody2img", "poem1img", "poem2img",
]
COND_BASE  = ["melody1", "melody2", "poem1", "poem2"]
N_CLASSES  = 4
N_SESSIONS = 10
DS_FACTOR  = 10
SFREQ_DS   = 100.0

WIN_MS    = 1000
STRIDE_MS = 500
WIN_T     = int(WIN_MS   * SFREQ_DS / 1000)
STRIDE_T  = int(STRIDE_MS * SFREQ_DS / 1000)

N_EPOCHS     = 80
BATCH_SIZE   = 16
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
PEARSON_LAM  = 0.5
PATIENCE     = 15

ALPHA_RIDGE   = 600.0
LAG_BEFORE_MS = 100
LAG_AFTER_MS  = 100

TARGET_HIDDEN = 64
TRAIN_SEED    = 42       # seed for val split within each training run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
#  DATA LOADING
# =============================================================================

def load_meg_subject(subject: str) -> np.ndarray:
    """Returns (n_cond=8, n_sessions=10, C, T_ds)."""
    all_data = None
    for i_cond, cond in enumerate(COND_ALL):
        for i_ses in range(N_SESSIONS):
            fname = f"{subject}_sess-{i_ses}_task-{cond}_meg-epo.fif"
            fpath = os.path.join(BASE_PATH, subject, f"ses-{i_ses}", "meg", fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(fpath)
            epochs  = mne.read_epochs(fpath, preload=True)
            data    = epochs.get_data().mean(axis=0)
            new_T   = data.shape[1] // DS_FACTOR
            data_ds = resample(data, new_T, axis=1).astype(np.float32)
            if all_data is None:
                C, Tds   = data_ds.shape
                all_data = np.zeros(
                    (len(COND_ALL), N_SESSIONS, C, Tds), dtype=np.float32
                )
            all_data[i_cond, i_ses] = data_ds
    return all_data


def remove_flash_windows(data: np.ndarray) -> np.ndarray:
    n_t  = data.shape[-1]
    fidx = np.arange(207, n_t, 207, dtype=int)
    rmv  = []
    for idx in fidx:
        rmv.extend(range(idx, min(idx + 51, n_t)))
    keep = np.setdiff1d(np.arange(n_t), sorted(set(rmv)))
    return data[..., keep].astype(np.float32)


def load_all_subjects() -> Dict[str, np.ndarray]:
    data = {}
    for subj in SUBJECTS:
        print(f"  loading {subj}...")
        data[subj] = remove_flash_windows(load_meg_subject(subj))
    return data


# =============================================================================
#  TRIAL INDEXING
# =============================================================================

TrialIdx = Tuple[str, str, int]   # (subject, cond_base, session)


def trials_for_subjects(subj_list: List[str]) -> List[TrialIdx]:
    return [
        (subj, cb, s)
        for subj in subj_list
        for cb   in COND_BASE
        for s    in range(N_SESSIONS)
    ]


def test_trials_for_subject(subj: str) -> List[TrialIdx]:
    """All 40 trials of one subject."""
    return [(subj, cb, s) for cb in COND_BASE for s in range(N_SESSIONS)]


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def zscore_ch(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=1, keepdims=True)
    sd = np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    return (x - mu) / sd


def get_xy_full(
    data_by_subj: Dict[str, np.ndarray],
    idx_list:     List[TrialIdx],
    cond_to_idx:  Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for subj, cb, s in idx_list:
        xs.append(zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}img"], s]))
        ys.append(zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}lis"], s]))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def window_trial(x: np.ndarray, win_t: int, stride_t: int) -> np.ndarray:
    C, T = x.shape
    windows, start = [], 0
    while start < T:
        chunk = x[:, start:start + win_t]
        if chunk.shape[1] < win_t:
            pad   = np.zeros((C, win_t - chunk.shape[1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=1)
        windows.append(chunk)
        start += stride_t
    return np.stack(windows)


def get_xy_windowed(
    data_by_subj: Dict[str, np.ndarray],
    idx_list:     List[TrialIdx],
    cond_to_idx:  Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for subj, cb, s in idx_list:
        x = zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}img"], s])
        y = zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}lis"], s])
        xs.append(window_trial(x, WIN_T, STRIDE_T))
        ys.append(window_trial(y, WIN_T, STRIDE_T))
    return (np.concatenate(xs, axis=0).astype(np.float32),
            np.concatenate(ys, axis=0).astype(np.float32))


# =============================================================================
#  RIDGE REGRESSION
# =============================================================================

def ms_to_samples(ms: float) -> int:
    return int(round(ms * SFREQ_DS / 1000.0))


def build_lagged_features(x: np.ndarray, lb: int, la: int) -> np.ndarray:
    C, T   = x.shape
    n_lags = lb + la + 1
    x_pad  = np.zeros((C, T + lb + la), dtype=np.float32)
    x_pad[:, lb:lb + T] = x
    return np.concatenate([x_pad[:, k:k + T] for k in range(n_lags)], axis=0).T


def fit_ridge(X_tr, Y_tr, lb, la, alpha):
    C   = X_tr.shape[1]
    p   = C * (lb + la + 1)
    XtX = np.zeros((p, p), dtype=np.float64)
    XtY = np.zeros((p, C), dtype=np.float64)
    for i in range(len(X_tr)):
        Xl   = build_lagged_features(X_tr[i], lb, la).astype(np.float64)
        XtX += Xl.T @ Xl
        XtY += Xl.T @ Y_tr[i].T.astype(np.float64)
    return np.linalg.solve(XtX + alpha * np.eye(p), XtY)


def predict_ridge(X_te, W, lb, la):
    return np.stack([(build_lagged_features(X_te[i], lb, la).astype(np.float64) @ W
                      ).T.astype(np.float32) for i in range(len(X_te))])


# =============================================================================
#  PYTORCH DATASET
# =============================================================================

class MEGDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# =============================================================================
#  NEURAL ARCHITECTURES  (unchanged from benchmark_loso.py)
# =============================================================================

class DepthwiseSepConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                            padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation=1, dropout=DROPOUT):
        super().__init__()
        self.conv = DepthwiseSepConv1d(in_ch, out_ch, kernel, dilation)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.conv(x)))


class ShallowMLP(nn.Module):
    def __init__(self, C, hidden=TARGET_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, C),
        )

    def forward(self, x):
        B, C, T = x.shape
        return self.net(x.permute(0, 2, 1).reshape(B * T, C)).reshape(B, T, C).permute(0, 2, 1)


class CNN1D(nn.Module):
    def __init__(self, C, hidden=TARGET_HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.input_proj  = nn.Conv1d(C, hidden, 1)
        self.layers      = nn.Sequential(
            ConvBlock(hidden, hidden, 7, dilation=1, dropout=dropout),
            ConvBlock(hidden, hidden, 7, dilation=2, dropout=dropout),
            ConvBlock(hidden, hidden, 5, dilation=4, dropout=dropout),
            ConvBlock(hidden, hidden, 5, dilation=8, dropout=dropout),
        )
        self.output_proj = nn.Conv1d(hidden, C, 1)

    def forward(self, x):
        return self.output_proj(self.layers(self.input_proj(x)))


class UNet1D(nn.Module):
    def __init__(self, C, hidden=TARGET_HIDDEN // 2, dropout=DROPOUT):
        super().__init__()
        h = hidden
        self.enc1       = ConvBlock(C,   h,   7, dropout=dropout)
        self.down1      = nn.Conv1d(h,   h,   3, stride=2, padding=1)
        self.enc2       = ConvBlock(h,   h*2, 5, dropout=dropout)
        self.down2      = nn.Conv1d(h*2, h*2, 3, stride=2, padding=1)
        self.bottleneck = ConvBlock(h*2, h*2, 5, dilation=2, dropout=dropout)
        self.up2        = nn.ConvTranspose1d(h*2, h*2, 4, stride=2, padding=1)
        self.dec2       = ConvBlock(h*2 + h*2, h,  5, dropout=dropout)
        self.up1        = nn.ConvTranspose1d(h,  h,  4, stride=2, padding=1)
        self.dec1       = ConvBlock(h + h,     h,  7, dropout=dropout)
        self.out        = nn.Conv1d(h, C, 1)

    def forward(self, x):
        e1  = self.enc1(x)
        e1d = self.down1(e1)
        e2  = self.enc2(e1d)
        e2d = self.down2(e2)
        b   = self.bottleneck(e2d)
        d2  = _cat_skip(self.up2(b), e2)
        d2  = self.dec2(d2)
        d1  = _cat_skip(self.up1(d2), e1)
        return self.out(self.dec1(d1))


def _cat_skip(x, skip):
    if x.shape[-1] != skip.shape[-1]:
        x = x[..., :skip.shape[-1]]
    return torch.cat([x, skip], dim=1)


class RNN(nn.Module):
    def __init__(self, C, hidden=TARGET_HIDDEN, dropout=DROPOUT, n_layers=2):
        super().__init__()
        self.input_proj  = nn.Linear(C, hidden)
        self.gru         = nn.GRU(hidden, hidden // 2, num_layers=n_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden, C)
        self.drop        = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.drop(F.gelu(self.input_proj(x)))
        x, _ = self.gru(x)
        return self.output_proj(self.drop(x)).permute(0, 2, 1)


class TCNBlock(nn.Module):
    def __init__(self, channels, kernel, dilation, dropout=DROPOUT):
        super().__init__()
        self._pad  = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self._pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self._pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def _trim(self, x):
        return x[..., :-self._pad] if self._pad > 0 else x

    def forward(self, x):
        r = self.drop(F.gelu(self.bn1(self._trim(self.conv1(x)))))
        return self.drop(F.gelu(self.bn2(self._trim(self.conv2(r))))) + x


class TCN(nn.Module):
    def __init__(self, C, hidden=TARGET_HIDDEN // 2, dropout=DROPOUT):
        super().__init__()
        self.input_proj  = nn.Conv1d(C, hidden, 1)
        self.blocks      = nn.Sequential(*[
            TCNBlock(hidden, kernel=3, dilation=2**i, dropout=dropout) for i in range(5)
        ])
        self.output_proj = nn.Conv1d(hidden, C, 1)

    def forward(self, x):
        return self.output_proj(self.blocks(self.input_proj(x)))


# =============================================================================
#  LOSS + TRAINING
# =============================================================================

def pearson_r_loss(pred, target):
    pred   = pred   - pred.mean(dim=-1,   keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    r = (pred * target).sum(-1) / (pred.norm(dim=-1) * target.norm(dim=-1) + 1e-8)
    return (1.0 - r).mean()


def combined_loss(pred, target):
    return F.mse_loss(pred, target) + PEARSON_LAM * pearson_r_loss(pred, target)


def train_neural(model, X_tr, Y_tr, X_val, Y_val, tag="") -> nn.Module:
    model = model.to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    tr_dl  = DataLoader(MEGDataset(X_tr,  Y_tr),  BATCH_SIZE, shuffle=True,  drop_last=False)
    val_dl = DataLoader(MEGDataset(X_val, Y_val), BATCH_SIZE, shuffle=False, drop_last=False)

    best_val, best_wts, no_imp = float("inf"), deepcopy(model.state_dict()), 0

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb = xb + 0.02 * torch.randn_like(xb)
            opt.zero_grad()
            combined_loss(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vl = [combined_loss(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
                  for xb, yb in val_dl]
            val_loss = float(np.mean(vl))

        if val_loss < best_val - 1e-6:
            best_val, best_wts, no_imp = val_loss, deepcopy(model.state_dict()), 0
        else:
            no_imp += 1

        if no_imp >= PATIENCE:
            break

    model.load_state_dict(best_wts)
    return model


def predict_neural(model, X):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i + BATCH_SIZE]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


# =============================================================================
#  MODEL FACTORY
# =============================================================================

def make_model(model_name: str, C: int) -> nn.Module:
    registry = {
        "ShallowMLP": lambda: ShallowMLP(C, hidden=TARGET_HIDDEN),
        "CNN1D":      lambda: CNN1D(C,      hidden=TARGET_HIDDEN),
        "UNet1D":     lambda: UNet1D(C,     hidden=TARGET_HIDDEN // 2),
        "RNN":        lambda: RNN(C,        hidden=TARGET_HIDDEN),
        "TCN":        lambda: TCN(C,        hidden=TARGET_HIDDEN // 2),
    }
    if model_name not in registry:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(registry)}")
    return registry[model_name]()


# =============================================================================
#  METRICS
# =============================================================================

def pearsonr_per_trial_channel(Y_pred, Y_true):
    """Returns (N, C) per-trial per-channel Pearson r."""
    yp = Y_pred - Y_pred.mean(axis=-1, keepdims=True)
    yt = Y_true - Y_true.mean(axis=-1, keepdims=True)
    num   = (yp * yt).sum(axis=-1)
    denom = np.sqrt((yp**2).sum(axis=-1) * (yt**2).sum(axis=-1)) + 1e-12
    return (num / denom).astype(np.float32)   # (N, C)


def mean_pearsonr(Y_pred, Y_true):
    """Scalar: mean r across channels, concatenating trials per channel."""
    N, C, T = Y_pred.shape
    rs = []
    for c in range(C):
        yp = Y_pred[:, c, :].ravel();  yp = yp - yp.mean()
        yt = Y_true[:, c, :].ravel();  yt = yt - yt.mean()
        rs.append(float((yp * yt).sum() /
                        (np.sqrt((yp**2).sum() * (yt**2).sum()) + 1e-12)))
    return float(np.mean(rs))


# =============================================================================
#  COMBINATION SAMPLING
# =============================================================================

def get_combinations(
    pool:       List[str],
    k:          int,
    max_combos: int,
    rng:        random.Random,
) -> List[Tuple[str, ...]]:
    """Return all C(len(pool), k) combos, or a random subsample if > max_combos."""
    all_combos = list(iter_combinations(pool, k))
    if max_combos <= 0 or len(all_combos) <= max_combos:
        return all_combos
    rng.shuffle(all_combos)
    return all_combos[:max_combos]


# =============================================================================
#  SINGLE COMBINATION TRAINING + EVALUATION
# =============================================================================

def run_combination(
    train_subjects: Tuple[str, ...],
    heldout_subj:   str,
    mode:           str,         # "full" or "windowed"
    model_name:     str,         # e.g. "RNN"
    data_by_subj:   Dict[str, np.ndarray],
    cond_to_idx:    Dict[str, int],
    C:              int,
    save_model_path: str = None,
) -> Tuple[np.ndarray, float]:
    """
    Train model_name (mode) on train_subjects, test on heldout_subj.
    Returns:
      r_tc  : (40, C) per-trial per-channel Pearson r
      mean_r: scalar mean r across channels and trials
    """
    train_idx = trials_for_subjects(list(train_subjects))
    test_idx  = test_trials_for_subject(heldout_subj)

    # val = 10% of training trials
    rng = random.Random(TRAIN_SEED)
    rng.shuffle(train_idx)
    n_val   = max(4, int(0.10 * len(train_idx)))
    val_idx = train_idx[-n_val:]
    tr_idx  = train_idx[:-n_val]

    # test arrays (always full trial)
    X_te, Y_te = get_xy_full(data_by_subj, test_idx, cond_to_idx)

    if model_name == "LinearLag":
        lb = ms_to_samples(LAG_BEFORE_MS)
        la = ms_to_samples(LAG_AFTER_MS)
        # ridge uses all train+val (no early stopping needed)
        X_tr_all, Y_tr_all = get_xy_full(
            data_by_subj, train_idx, cond_to_idx
        )
        W  = fit_ridge(X_tr_all, Y_tr_all, lb, la, ALPHA_RIDGE)
        Yp = predict_ridge(X_te, W, lb, la)
        if save_model_path:
            np.save(save_model_path, W)
    else:
        if mode == "full":
            X_tr, Y_tr   = get_xy_full(data_by_subj, tr_idx,  cond_to_idx)
            X_val, Y_val = get_xy_full(data_by_subj, val_idx, cond_to_idx)
        else:
            X_tr, Y_tr   = get_xy_windowed(data_by_subj, tr_idx,  cond_to_idx)
            X_val, Y_val = get_xy_windowed(data_by_subj, val_idx, cond_to_idx)

        model   = make_model(model_name, C).to(DEVICE)
        trained = train_neural(model, X_tr, Y_tr, X_val, Y_val, tag="")
        Yp      = predict_neural(trained, X_te)
        if save_model_path:
            torch.save(trained.state_dict(), save_model_path)
        del trained
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    r_tc   = pearsonr_per_trial_channel(Yp, Y_te)   # (40, C)
    mean_r = float(r_tc.mean())
    return r_tc, mean_r


# =============================================================================
#  SCALING LOOP FOR ONE HELD-OUT SUBJECT
# =============================================================================

def run_scaling_for_subject(
    heldout_subj:  str,
    model_name:    str,
    mode:          str,
    model_key:     str,
    data_by_subj:  Dict[str, np.ndarray],
    cond_to_idx:   Dict[str, int],
    C:             int,
    out_dir:       str,
    max_combos:    int,
    combo_rng:     random.Random,
    save_models:   bool,
) -> Dict[int, List[Dict]]:
    """
    Returns {k: [{"combo_idx", "subjects", "mean_r", "mean_r_per_trial"}, ...]}
    """
    pool = [s for s in SUBJECTS if s != heldout_subj]   # 12 subjects
    k_results: Dict[int, List[Dict]] = {}

    r_arr_dir = os.path.join(out_dir, "r_arrays")
    mdl_dir   = os.path.join(out_dir, "models") if save_models else None
    os.makedirs(r_arr_dir, exist_ok=True)
    if mdl_dir:
        os.makedirs(mdl_dir, exist_ok=True)

    for k in range(1, len(pool) + 1):
        result_file = os.path.join(out_dir, f"k{k:02d}_results.json")

        # ---- resume: skip if already done ----
        if os.path.exists(result_file):
            with open(result_file) as f:
                k_results[k] = json.load(f)
            n_done = len(k_results[k])
            print(f"    [resume] k={k:2d}: found {n_done} combos in {result_file}")
            continue

        combos = get_combinations(pool, k, max_combos, combo_rng)
        total_c = len(list(iter_combinations(pool, k)))
        print(f"    k={k:2d}: {len(combos)}/{total_c} combos, "
              f"each trains on {k * 40} trials")

        k_records = []
        for c_idx, combo in enumerate(combos):
            t0 = time.time()

            model_path = None
            if save_models:
                ext = ".npy" if model_name == "LinearLag" else ".pt"
                model_path = os.path.join(
                    mdl_dir, f"k{k:02d}_c{c_idx:04d}_{model_key}{ext}"
                )

            r_tc, mean_r = run_combination(
                train_subjects  = combo,
                heldout_subj    = heldout_subj,
                mode            = mode,
                model_name      = model_name,
                data_by_subj    = data_by_subj,
                cond_to_idx     = cond_to_idx,
                C               = C,
                save_model_path = model_path,
            )

            # save r array
            r_path = os.path.join(
                r_arr_dir, f"k{k:02d}_c{c_idx:04d}_r_per_trial.npy"
            )
            np.save(r_path, r_tc)

            rec = {
                "combo_idx":       c_idx,
                "subjects":        list(combo),
                "mean_r":          mean_r,
                "mean_r_per_trial": r_tc.mean(axis=1).tolist(),  # (40,)
            }
            k_records.append(rec)

            elapsed = time.time() - t0
            if (c_idx + 1) % 5 == 0 or c_idx == 0:
                print(f"      combo {c_idx+1:3d}/{len(combos)}"
                      f"  subjs={list(combo)}"
                      f"  mean_r={mean_r:.4f}  ({elapsed:.1f}s)")

        k_results[k] = k_records
        # save immediately (enables resume)
        with open(result_file, "w") as f:
            json.dump(k_records, f, indent=2)
        print(f"      [saved] {result_file}")

    return k_results


# =============================================================================
#  PLOTTING
# =============================================================================

def plot_scaling(
    all_k_results: Dict[str, Dict[int, List[Dict]]],
    model_key:     str,
    out_dir:       str,
) -> None:
    """
    all_k_results: {heldout_subj: {k: [records]}}
    """
    k_values = sorted(next(iter(all_k_results.values())).keys())
    cmap     = matplotlib.colormaps["tab20"]
    subj_colors = {s: cmap(i / max(len(SUBJECTS) - 1, 1))
                   for i, s in enumerate(SUBJECTS)}

    # ---- 1. Scaling curve: mean r vs k, all subjects + grand mean ----
    fig, ax = plt.subplots(figsize=(10, 5))

    grand_means, grand_stds = [], []
    for k in k_values:
        per_subj_means = []
        for subj, k_res in all_k_results.items():
            if k not in k_res or not k_res[k]:
                continue
            per_subj_means.append(np.mean([r["mean_r"] for r in k_res[k]]))
        grand_means.append(float(np.mean(per_subj_means)))
        grand_stds.append(float(np.std(per_subj_means)))

    # per-subject lines (thin, coloured)
    for subj, k_res in all_k_results.items():
        subj_means = []
        for k in k_values:
            if k not in k_res or not k_res[k]:
                subj_means.append(float("nan"))
            else:
                subj_means.append(float(np.mean([r["mean_r"] for r in k_res[k]])))
        ax.plot(k_values, subj_means, marker="o", ms=4, lw=1,
                color=subj_colors[subj], alpha=0.55, label=subj)

    # grand mean ± std (thick black)
    ax.errorbar(k_values, grand_means, yerr=grand_stds,
                color="black", lw=2.5, marker="D", ms=6, capsize=4,
                label="Grand mean ± std", zorder=5)

    ax.axhline(0, color="grey", lw=0.7, linestyle="--")
    ax.set_xlabel("Number of training subjects (k)", fontsize=12)
    ax.set_ylabel("Mean Pearson r on held-out subject", fontsize=12)
    ax.set_title(f"Data scaling: {model_key}", fontsize=13)
    ax.set_xticks(k_values)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    plt.tight_layout()
    out = os.path.join(out_dir, "scaling_curve.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")

    # ---- 2. Heatmap: subjects × k, cell = mean r (mean over combos) ----
    subj_list = list(all_k_results.keys())
    heat = np.full((len(subj_list), len(k_values)), float("nan"))
    for i, subj in enumerate(subj_list):
        for j, k in enumerate(k_values):
            k_res = all_k_results[subj]
            if k in k_res and k_res[k]:
                heat[i, j] = float(np.mean([r["mean_r"] for r in k_res[k]]))

    vmax = float(np.nanquantile(np.abs(heat), 0.98))
    fig, ax = plt.subplots(figsize=(max(10, len(k_values) * 0.7), 6))
    im = ax.imshow(heat, aspect="auto", cmap="RdYlGn",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(range(len(k_values)))
    ax.set_xticklabels([str(k) for k in k_values], fontsize=9)
    ax.set_yticks(range(len(subj_list)))
    ax.set_yticklabels(subj_list, fontsize=9)
    ax.set_xlabel("k (training subjects)", fontsize=11)
    ax.set_title(f"Mean Pearson r per held-out subject × k  —  {model_key}",
                 fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="mean r")
    for i in range(len(subj_list)):
        for j in range(len(k_values)):
            v = heat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=6, color="black")
    plt.tight_layout()
    out = os.path.join(out_dir, "scaling_heatmap.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")

    # ---- 3. Distribution of combo r-values per k (violin / box) ----
    fig, ax = plt.subplots(figsize=(max(10, len(k_values) * 0.8), 5))
    all_vals_per_k = []
    for k in k_values:
        vals = []
        for subj, k_res in all_k_results.items():
            if k in k_res:
                vals.extend([r["mean_r"] for r in k_res[k]])
        all_vals_per_k.append(vals)

    positions = list(range(len(k_values)))
    ax.violinplot(all_vals_per_k, positions=positions, showmedians=True,
                  showextrema=True)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(k) for k in k_values], fontsize=9)
    ax.axhline(0, color="grey", lw=0.7, linestyle="--")
    ax.set_xlabel("k (training subjects)", fontsize=12)
    ax.set_ylabel("Mean Pearson r (all combos × all subjects)", fontsize=11)
    ax.set_title(f"Distribution of per-combination r by k  —  {model_key}",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, "scaling_distribution.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")


# =============================================================================
#  MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Data scaling analysis")
    parser.add_argument("--model", required=True,
                        help="Model key, e.g. RNN_full, CNN1D_windowed, LinearLag_full")
    parser.add_argument("--heldout", default=None,
                        help="Run for a single held-out subject (default: all)")
    parser.add_argument("--max_combos", type=int, default=50,
                        help="Max combinations per k (0 = all). Default: 50")
    parser.add_argument("--combo_seed", type=int, default=42,
                        help="RNG seed for subsampling combinations")
    parser.add_argument("--save_models", action="store_true",
                        help="Save model checkpoint for every combination")
    return parser.parse_args()


def main():
    args = parse_args()

    # parse model key → name + mode
    parts = args.model.rsplit("_", 1)
    if len(parts) != 2 or parts[1] not in ("full", "windowed"):
        raise ValueError(
            f"--model must be <ArchName>_full or <ArchName>_windowed, got '{args.model}'"
        )
    model_name, mode = parts[0], parts[1]
    model_key        = args.model

    if model_name != "LinearLag":
        try:
            make_model(model_name, C=10)   # dummy check
        except ValueError as e:
            raise SystemExit(str(e))

    subjects_to_run = [args.heldout] if args.heldout else SUBJECTS
    if args.heldout and args.heldout not in SUBJECTS:
        raise SystemExit(f"Unknown subject '{args.heldout}'. Valid: {SUBJECTS}")

    out_root = os.path.join("scaling_out", model_key)
    os.makedirs(out_root, exist_ok=True)

    combo_rng = random.Random(args.combo_seed)

    print(f"Device     : {DEVICE}")
    print(f"Model      : {model_key}  ({model_name}, {mode})")
    print(f"Subjects   : {subjects_to_run}")
    print(f"Max combos : {args.max_combos if args.max_combos > 0 else 'all'}")
    print(f"Output     : {out_root}\n")

    # ---- load data ----
    print("Loading all subjects...")
    data_by_subj = load_all_subjects()
    cond_to_idx  = {c: i for i, c in enumerate(COND_ALL)}
    C = data_by_subj[SUBJECTS[0]].shape[2]
    T = data_by_subj[SUBJECTS[0]].shape[3]
    print(f"C={C}  T={T}\n")

    all_k_results: Dict[str, Dict[int, List[Dict]]] = {}

    for subj in subjects_to_run:
        print(f"\n{'='*65}")
        print(f"  HELD-OUT: {subj}")
        print(f"{'='*65}")
        subj_out = os.path.join(out_root, f"heldout_{subj}")
        os.makedirs(subj_out, exist_ok=True)

        k_results = run_scaling_for_subject(
            heldout_subj = subj,
            model_name   = model_name,
            mode         = mode,
            model_key    = model_key,
            data_by_subj = data_by_subj,
            cond_to_idx  = cond_to_idx,
            C            = C,
            out_dir      = subj_out,
            max_combos   = args.max_combos,
            combo_rng    = combo_rng,
            save_models  = args.save_models,
        )
        all_k_results[subj] = k_results

        # print per-subject scaling summary
        print(f"\n  k   n_combos   mean_r_mean   mean_r_std")
        for k in sorted(k_results):
            rs = [r["mean_r"] for r in k_results[k]]
            print(f"  {k:2d}   {len(rs):6d}     {np.mean(rs):10.5f}   {np.std(rs):10.5f}")

    # ---- aggregate summary ----
    k_values = sorted(next(iter(all_k_results.values())).keys())
    summary: Dict[str, Dict] = {}
    for subj, k_res in all_k_results.items():
        summary[subj] = {}
        for k in k_values:
            if k not in k_res or not k_res[k]:
                continue
            rs = [r["mean_r"] for r in k_res[k]]
            summary[subj][str(k)] = {
                "n_combos": len(rs),
                "mean_r_mean": float(np.mean(rs)),
                "mean_r_std":  float(np.std(rs)),
                "mean_r_min":  float(np.min(rs)),
                "mean_r_max":  float(np.max(rs)),
            }

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {os.path.join(out_root, 'summary.json')}")

    # ---- plots (only if we ran more than one subject) ----
    if len(all_k_results) > 1:
        plot_scaling(all_k_results, model_key, out_root)
    else:
        # single-subject quick plot
        subj = list(all_k_results.keys())[0]
        k_res = all_k_results[subj]
        fig, ax = plt.subplots(figsize=(9, 4))
        ks    = sorted(k_res.keys())
        means = [float(np.mean([r["mean_r"] for r in k_res[k]])) for k in ks]
        stds  = [float(np.std( [r["mean_r"] for r in k_res[k]])) for k in ks]
        ax.errorbar(ks, means, yerr=stds, marker="o", capsize=4, color="steelblue")
        ax.axhline(0, color="grey", lw=0.7, linestyle="--")
        ax.set_xlabel("k (training subjects)"); ax.set_ylabel("Mean Pearson r")
        ax.set_title(f"Scaling — {model_key}, heldout={subj}")
        ax.set_xticks(ks)
        plt.tight_layout()
        out = os.path.join(out_root, "scaling_curve.png")
        plt.savefig(out, dpi=200); plt.close()
        print(f"[saved] {out}")

    print(f"\nDone. All outputs in {out_root}/")


if __name__ == "__main__":
    main()
