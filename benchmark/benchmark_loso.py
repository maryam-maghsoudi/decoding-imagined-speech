"""
benchmark_loso.py
=================
Leave-One-Subject-Out (LOSO) cross-validation of the img→lis MEG mapping
architectures defined in benchmark_archs_v2.py.

For each of the 13 subjects:
  Train  : all 40 trials (4 conds × 10 sessions) from the other 12 subjects
  Val    : random 10% of training trials (early-stopping only)
  Test   : all 40 imagined trials of the held-out subject
             → predict listened, evaluate against ground-truth listened

Classification pool  : all 40 listened signals of the held-out subject
Training modes       : "full" (whole trial) and "windowed" (1s/50%-overlap)

Saved per held-out subject
--------------------------
  loso_out/heldout_{subj}_results.json
  loso_out/heldout_{subj}_test_idx.json
  loso_out/models/heldout_{subj}/{model_key}.pt
  loso_out/models/heldout_{subj}/LinearLag_W.npy
  loso_out/per_trial_r/heldout_{subj}_{model_key}_r_per_trial.npy
      shape: (N_test=40, C)  — per-trial per-channel Pearson r

Summary
-------
  loso_out/summary_metrics.json
  loso_out/comparison_barplot.png
  loso_out/per_subject_lineplots.png
  loso_out/per_subject_acc_heatmap.png
"""

import os
import json
import random
import time
from copy import deepcopy
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
#  CONFIG
# =============================================================================
BASE_PATH    = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
LOSO_OUT_DIR = "./loso_out"
os.makedirs(LOSO_OUT_DIR, exist_ok=True)

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

# ---- windowing ----
WIN_MS    = 1000
STRIDE_MS = 500
WIN_T     = int(WIN_MS   * SFREQ_DS / 1000)
STRIDE_T  = int(STRIDE_MS * SFREQ_DS / 1000)

# ---- neural training ----
N_EPOCHS     = 80
BATCH_SIZE   = 16
LR           = 3e-4
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
PEARSON_LAM  = 0.5
PATIENCE     = 15

# ---- ridge ----
ALPHA_RIDGE   = 600.0
LAG_BEFORE_MS = 100
LAG_AFTER_MS  = 100

TARGET_HIDDEN = 64
SEED          = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NEURAL_NAMES = ["ShallowMLP", "CNN1D", "UNet1D", "RNN", "TCN"]
ALL_MODES    = ["full", "windowed"]


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
#  TRIAL INDEXING AND SPLITTING
# =============================================================================

TrialIdx = Tuple[str, str, int]   # (subject, cond_base, session)


def loso_split(heldout_subj: str) -> Tuple[List[TrialIdx], List[TrialIdx]]:
    """
    Train : all 40 trials from every subject except heldout_subj.
    Test  : all 40 trials (4 conds × 10 sessions) of heldout_subj.
    """
    train_idx = [
        (subj, cb, s)
        for subj in SUBJECTS if subj != heldout_subj
        for cb   in COND_BASE
        for s    in range(N_SESSIONS)
    ]
    test_idx = [
        (heldout_subj, cb, s)
        for cb in COND_BASE
        for s  in range(N_SESSIONS)
    ]
    return train_idx, test_idx


# =============================================================================
#  FEATURE EXTRACTION
# =============================================================================

def zscore_ch(x: np.ndarray) -> np.ndarray:
    """x: (C, T) — z-score each channel over time."""
    mu = x.mean(axis=1, keepdims=True)
    sd = np.maximum(x.std(axis=1, keepdims=True), 1e-12)
    return (x - mu) / sd


def get_xy_full(
    data_by_subj: Dict[str, np.ndarray],
    idx_list:     List[TrialIdx],
    cond_to_idx:  Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns X (N, C, T) imagined and Y (N, C, T) listened, z-scored."""
    xs, ys = [], []
    for subj, cb, s in idx_list:
        x = data_by_subj[subj][cond_to_idx[f"{cb}img"], s]
        y = data_by_subj[subj][cond_to_idx[f"{cb}lis"], s]
        xs.append(zscore_ch(x))
        ys.append(zscore_ch(y))
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def window_trial(x: np.ndarray, win_t: int, stride_t: int) -> np.ndarray:
    """x: (C, T) → (n_win, C, win_t), last window zero-padded if needed."""
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
    win_t:        int = WIN_T,
    stride_t:     int = STRIDE_T,
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for subj, cb, s in idx_list:
        x = zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}img"], s])
        y = zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}lis"], s])
        xs.append(window_trial(x, win_t, stride_t))
        ys.append(window_trial(y, win_t, stride_t))
    return (np.concatenate(xs, axis=0).astype(np.float32),
            np.concatenate(ys, axis=0).astype(np.float32))


# =============================================================================
#  LINEAR LAG MODEL (ridge regression)
# =============================================================================

def ms_to_samples(ms: float) -> int:
    return int(round(ms * SFREQ_DS / 1000.0))


def build_lagged_features(x: np.ndarray, lb: int, la: int) -> np.ndarray:
    C, T   = x.shape
    n_lags = lb + la + 1
    x_pad  = np.zeros((C, T + lb + la), dtype=np.float32)
    x_pad[:, lb:lb + T] = x
    slices = [x_pad[:, k:k + T] for k in range(n_lags)]
    return np.concatenate(slices, axis=0).T   # (T, C*n_lags)


def fit_ridge(
    X_tr: np.ndarray, Y_tr: np.ndarray,
    lb: int, la: int, alpha: float,
) -> np.ndarray:
    C   = X_tr.shape[1]
    p   = C * (lb + la + 1)
    XtX = np.zeros((p, p), dtype=np.float64)
    XtY = np.zeros((p, C), dtype=np.float64)
    for i in range(len(X_tr)):
        Xl   = build_lagged_features(X_tr[i], lb, la).astype(np.float64)
        Yr   = Y_tr[i].T.astype(np.float64)
        XtX += Xl.T @ Xl
        XtY += Xl.T @ Yr
    return np.linalg.solve(XtX + alpha * np.eye(p), XtY)


def predict_ridge(X_te: np.ndarray, W: np.ndarray, lb: int, la: int) -> np.ndarray:
    preds = []
    for i in range(len(X_te)):
        Xl = build_lagged_features(X_te[i], lb, la).astype(np.float64)
        preds.append((Xl @ W).T.astype(np.float32))
    return np.stack(preds)


def ridge_param_count(C: int, lb: int, la: int) -> int:
    return C * (lb + la + 1) * C


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
#  NEURAL ARCHITECTURES
# =============================================================================

class DepthwiseSepConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, dilation=dilation,
                            padding=pad, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return self.bn(self.pw(self.dw(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int = 1, dropout: float = DROPOUT):
        super().__init__()
        self.conv = DepthwiseSepConv1d(in_ch, out_ch, kernel, dilation)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.act(self.conv(x)))


class ShallowMLP(nn.Module):
    def __init__(self, C: int, hidden: int = TARGET_HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, C),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        x = x.permute(0, 2, 1).reshape(B * T, C)
        return self.net(x).reshape(B, T, C).permute(0, 2, 1)


class CNN1D(nn.Module):
    def __init__(self, C: int, hidden: int = TARGET_HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj  = nn.Conv1d(C, hidden, 1)
        self.layers      = nn.Sequential(
            ConvBlock(hidden, hidden, 7, dilation=1, dropout=dropout),
            ConvBlock(hidden, hidden, 7, dilation=2, dropout=dropout),
            ConvBlock(hidden, hidden, 5, dilation=4, dropout=dropout),
            ConvBlock(hidden, hidden, 5, dilation=8, dropout=dropout),
        )
        self.output_proj = nn.Conv1d(hidden, C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.layers(self.input_proj(x)))


class UNet1D(nn.Module):
    def __init__(self, C: int, hidden: int = TARGET_HIDDEN // 2, dropout: float = DROPOUT):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1  = self.enc1(x)
        e1d = self.down1(e1)
        e2  = self.enc2(e1d)
        e2d = self.down2(e2)
        b   = self.bottleneck(e2d)
        d2  = _cat_skip(self.up2(b), e2)
        d2  = self.dec2(d2)
        d1  = _cat_skip(self.up1(d2), e1)
        return self.out(self.dec1(d1))


def _cat_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] != skip.shape[-1]:
        x = x[..., :skip.shape[-1]]
    return torch.cat([x, skip], dim=1)


class RNN(nn.Module):
    def __init__(self, C: int, hidden: int = TARGET_HIDDEN,
                 dropout: float = DROPOUT, n_layers: int = 2):
        super().__init__()
        self.input_proj  = nn.Linear(C, hidden)
        self.gru         = nn.GRU(hidden, hidden // 2, num_layers=n_layers,
                                  batch_first=True, bidirectional=True,
                                  dropout=dropout if n_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden, C)
        self.drop        = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.drop(F.gelu(self.input_proj(x)))
        x, _ = self.gru(x)
        return self.output_proj(self.drop(x)).permute(0, 2, 1)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, dropout: float = DROPOUT):
        super().__init__()
        self._pad  = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self._pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation, padding=self._pad)
        self.bn1   = nn.BatchNorm1d(channels)
        self.bn2   = nn.BatchNorm1d(channels)
        self.drop  = nn.Dropout(dropout)

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :-self._pad] if self._pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = self.drop(F.gelu(self.bn1(self._trim(self.conv1(x)))))
        return self.drop(F.gelu(self.bn2(self._trim(self.conv2(r))))) + x


class TCN(nn.Module):
    def __init__(self, C: int, hidden: int = TARGET_HIDDEN // 2, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj  = nn.Conv1d(C, hidden, 1)
        self.blocks      = nn.Sequential(*[
            TCNBlock(hidden, kernel=3, dilation=2**i, dropout=dropout) for i in range(5)
        ])
        self.output_proj = nn.Conv1d(hidden, C, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_proj(self.blocks(self.input_proj(x)))


# =============================================================================
#  LOSS
# =============================================================================

def pearson_r_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred   = pred   - pred.mean(dim=-1,   keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    r = (pred * target).sum(-1) / (pred.norm(dim=-1) * target.norm(dim=-1) + 1e-8)
    return (1.0 - r).mean()


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target) + PEARSON_LAM * pearson_r_loss(pred, target)


# =============================================================================
#  TRAINING LOOP
# =============================================================================

def train_neural(
    model: nn.Module,
    X_tr:  np.ndarray,
    Y_tr:  np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    tag:   str = "",
) -> nn.Module:
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

        if epoch % 10 == 0:
            print(f"    [{tag}] epoch {epoch:3d}/{N_EPOCHS}"
                  f"  val={val_loss:.4f}  best={best_val:.4f}  no_imp={no_imp}")

        if no_imp >= PATIENCE:
            print(f"    [{tag}] early stop at epoch {epoch}")
            break

    model.load_state_dict(best_wts)
    return model


# =============================================================================
#  PREDICTION
# =============================================================================

def predict_neural(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), BATCH_SIZE):
            xb = torch.from_numpy(X[i:i + BATCH_SIZE]).to(DEVICE)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


# =============================================================================
#  EVALUATION METRICS
# =============================================================================

def pearsonr_per_trial_channel(
    Y_pred: np.ndarray,   # (N, C, T)
    Y_true: np.ndarray,   # (N, C, T)
) -> np.ndarray:
    """
    Returns r_tc of shape (N, C): Pearson r between predicted and true
    for every trial and every channel independently.
    """
    N, C, T = Y_pred.shape
    yp = Y_pred - Y_pred.mean(axis=-1, keepdims=True)
    yt = Y_true - Y_true.mean(axis=-1, keepdims=True)
    num   = (yp * yt).sum(axis=-1)                                    # (N, C)
    denom = np.sqrt((yp**2).sum(axis=-1) * (yt**2).sum(axis=-1)) + 1e-12
    return (num / denom).astype(np.float32)                           # (N, C)


def pearsonr_mean(Y_pred: np.ndarray, Y_true: np.ndarray) -> Tuple[float, float]:
    """Mean and median r across channels, concatenating all trials per channel."""
    N, C, T = Y_pred.shape
    rs = []
    for c in range(C):
        yp = Y_pred[:, c, :].ravel();  yp = yp - yp.mean()
        yt = Y_true[:, c, :].ravel();  yt = yt - yt.mean()
        rs.append(float((yp * yt).sum() /
                        (np.sqrt((yp**2).sum() * (yt**2).sum()) + 1e-12)))
    return float(np.mean(rs)), float(np.median(rs))


def mse_metric(Y_pred: np.ndarray, Y_true: np.ndarray) -> float:
    return float(np.mean((Y_pred - Y_true) ** 2))


# =============================================================================
#  CORRELATION-BASED 4-CLASS CLASSIFICATION
# =============================================================================

def corr_clf(
    Y_pred:      np.ndarray,   # (N_test, C, T)
    Y_lis_pool:  np.ndarray,   # (N_pool, C, T)
    test_labels: np.ndarray,   # (N_test,)
    pool_labels: np.ndarray,   # (N_pool,)
) -> Tuple[float, np.ndarray]:
    eps = 1e-8

    def znorm(x):
        return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)

    Pz = znorm(Y_pred)
    Lz = znorm(Y_lis_pool)
    T  = Pz.shape[-1]

    R      = np.einsum("ict,jct->ijc", Pz, Lz) / T   # (N_pred, N_pool, C)
    R      = R.mean(axis=2)                            # (N_pred, N_pool)
    scores = np.zeros((len(Y_pred), N_CLASSES), dtype=np.float32)
    for k in range(N_CLASSES):
        scores[:, k] = R[:, pool_labels == k].mean(axis=1)

    pred_labels = scores.argmax(axis=1)
    acc = float((pred_labels == test_labels).mean())
    cm  = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(test_labels, pred_labels):
        cm[t, p] += 1
    return acc, cm


def make_lis_pool_subject(
    data_by_subj: Dict[str, np.ndarray],
    subj:         str,
    cond_to_idx:  Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """All 40 listened signals for one subject — used as classification pool."""
    ys, labels = [], []
    for cb in COND_BASE:
        for s in range(N_SESSIONS):
            ys.append(zscore_ch(data_by_subj[subj][cond_to_idx[f"{cb}lis"], s]))
            labels.append(COND_BASE.index(cb))
    return np.stack(ys).astype(np.float32), np.array(labels)


# =============================================================================
#  MODEL FACTORY
# =============================================================================

def make_models(C: int) -> Dict[str, nn.Module]:
    return {
        "ShallowMLP": ShallowMLP(C, hidden=TARGET_HIDDEN),
        "CNN1D":      CNN1D(C,      hidden=TARGET_HIDDEN),
        "UNet1D":     UNet1D(C,     hidden=TARGET_HIDDEN // 2),
        "RNN":        RNN(C,        hidden=TARGET_HIDDEN),
        "TCN":        TCN(C,        hidden=TARGET_HIDDEN // 2),
    }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
#  CHECKPOINT HELPERS
# =============================================================================

def _model_dir(heldout_subj: str) -> str:
    d = os.path.join(LOSO_OUT_DIR, "models", f"heldout_{heldout_subj}")
    os.makedirs(d, exist_ok=True)
    return d


def save_neural_checkpoint(heldout_subj: str, model_key: str, model: nn.Module) -> None:
    path = os.path.join(_model_dir(heldout_subj), f"{model_key}.pt")
    torch.save(model.state_dict(), path)


def save_ridge_checkpoint(heldout_subj: str, W: np.ndarray) -> None:
    path = os.path.join(_model_dir(heldout_subj), "LinearLag_W.npy")
    np.save(path, W)


def save_r_per_trial(
    heldout_subj: str,
    model_key:    str,
    r_tc:         np.ndarray,   # (N_test, C)
) -> None:
    """Save per-trial per-channel r values as a .npy file."""
    d = os.path.join(LOSO_OUT_DIR, "per_trial_r")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"heldout_{heldout_subj}_{model_key}_r_per_trial.npy")
    np.save(path, r_tc)


# =============================================================================
#  SINGLE LOSO SUBJECT EVALUATION
# =============================================================================

def eval_metrics(
    Y_pred:      np.ndarray,
    Y_true:      np.ndarray,
    Y_lis_pool:  np.ndarray,
    pool_labels: np.ndarray,
    test_labels: np.ndarray,
) -> Dict:
    mean_r, median_r = pearsonr_mean(Y_pred, Y_true)
    mse              = mse_metric(Y_pred, Y_true)
    acc, cm          = corr_clf(Y_pred, Y_lis_pool, test_labels, pool_labels)
    return {
        "mean_r":   mean_r,
        "median_r": median_r,
        "mse":      mse,
        "clf_acc":  acc,
        "cm":       cm.tolist(),
    }


def run_loso_subject(
    heldout_subj: str,
    data_by_subj: Dict[str, np.ndarray],
    cond_to_idx:  Dict[str, int],
    C:            int,
) -> Dict:
    train_idx, test_idx = loso_split(heldout_subj)

    # save test indices
    with open(os.path.join(LOSO_OUT_DIR,
                           f"heldout_{heldout_subj}_test_idx.json"), "w") as f:
        json.dump(test_idx, f, indent=2)

    # carve 10% of training trials for validation (early stopping)
    rng = random.Random(SEED)
    rng.shuffle(train_idx)
    n_val   = max(8, int(0.10 * len(train_idx)))
    val_idx = train_idx[-n_val:]
    tr_idx  = train_idx[:-n_val]

    print(f"  train={len(tr_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ---- full-trial arrays ----
    X_tr_f,  Y_tr_f  = get_xy_full(data_by_subj, tr_idx,  cond_to_idx)
    X_val_f, Y_val_f = get_xy_full(data_by_subj, val_idx, cond_to_idx)
    X_te,    Y_te    = get_xy_full(data_by_subj, test_idx, cond_to_idx)

    # ---- windowed arrays (train + val only; test always full) ----
    X_tr_w,  Y_tr_w  = get_xy_windowed(data_by_subj, tr_idx,  cond_to_idx)
    X_val_w, Y_val_w = get_xy_windowed(data_by_subj, val_idx, cond_to_idx)

    print(f"  full train: {X_tr_f.shape}  windowed train: {X_tr_w.shape}")

    # ---- classification pool: all 40 listened of heldout subject ----
    Y_lis_pool, pool_labels = make_lis_pool_subject(
        data_by_subj, heldout_subj, cond_to_idx
    )
    # test labels: cond repeats 10 times each  (order matches loso_split)
    test_labels = np.array([COND_BASE.index(cb) for cb in COND_BASE
                             for _ in range(N_SESSIONS)])

    subj_res: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    #  LinearLag  (ridge; uses all train+val, no early stopping)
    # ------------------------------------------------------------------
    print("\n  [LinearLag] fitting ridge...")
    t0 = time.time()
    lb = ms_to_samples(LAG_BEFORE_MS)
    la = ms_to_samples(LAG_AFTER_MS)
    X_tr_all = np.concatenate([X_tr_f, X_val_f], axis=0)
    Y_tr_all = np.concatenate([Y_tr_f, Y_val_f], axis=0)
    W  = fit_ridge(X_tr_all, Y_tr_all, lb, la, ALPHA_RIDGE)
    Yp = predict_ridge(X_te, W, lb, la)

    key = "LinearLag_full"
    subj_res[key] = eval_metrics(Yp, Y_te, Y_lis_pool, pool_labels, test_labels)
    subj_res[key]["time_s"] = time.time() - t0

    r_tc = pearsonr_per_trial_channel(Yp, Y_te)   # (40, C)
    subj_res[key]["mean_r_per_trial"] = r_tc.mean(axis=1).tolist()   # (40,)
    save_r_per_trial(heldout_subj, key, r_tc)
    save_ridge_checkpoint(heldout_subj, W)
    _print_metrics(key, subj_res[key])

    # ------------------------------------------------------------------
    #  Neural models × {full, windowed}
    # ------------------------------------------------------------------
    for mode in ALL_MODES:
        X_tr  = X_tr_f  if mode == "full" else X_tr_w
        Y_tr  = Y_tr_f  if mode == "full" else Y_tr_w
        X_val = X_val_f if mode == "full" else X_val_w
        Y_val = Y_val_f if mode == "full" else Y_val_w

        for model_name, model in make_models(C).items():
            key = f"{model_name}_{mode}"
            print(f"\n  [{key}] training ({count_params(model):,} params,"
                  f" train_samples={len(X_tr)})...")
            t0 = time.time()

            trained = train_neural(model, X_tr, Y_tr, X_val, Y_val, tag=key)
            Yp      = predict_neural(trained, X_te)

            subj_res[key] = eval_metrics(Yp, Y_te, Y_lis_pool, pool_labels, test_labels)
            subj_res[key]["time_s"] = time.time() - t0

            r_tc = pearsonr_per_trial_channel(Yp, Y_te)   # (40, C)
            subj_res[key]["mean_r_per_trial"] = r_tc.mean(axis=1).tolist()
            save_r_per_trial(heldout_subj, key, r_tc)
            save_neural_checkpoint(heldout_subj, key, trained)
            _print_metrics(key, subj_res[key])

            del trained
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return subj_res


def _print_metrics(name: str, m: Dict) -> None:
    print(f"    → {name}: mean_r={m['mean_r']:.4f}  mse={m['mse']:.4f}"
          f"  acc={m['clf_acc']:.3f}  ({m['time_s']:.1f}s)")


# =============================================================================
#  PLOTTING
# =============================================================================

def _display_name(key: str) -> str:
    name, mode = key.rsplit("_", 1)
    return f"{name}\n({'full' if mode == 'full' else 'win'})"


def plot_results(summary: Dict, model_keys: List[str], subjects: List[str]) -> None:
    cmap    = matplotlib.colormaps["tab20"]
    colors  = [cmap(i / max(len(model_keys) - 1, 1)) for i in range(len(model_keys))]
    xlabels = [_display_name(k) for k in model_keys]
    M       = len(model_keys)
    N_SUBJ  = len(subjects)

    bench_metrics = [
        ("mean_r_mean",  "mean_r_std",  "per_subject_mean_r",  "Mean Pearson r ↑"),
        ("mse_mean",     "mse_std",     "per_subject_mse",     "MSE ↓"),
        ("clf_acc_mean", "clf_acc_std", "per_subject_clf_acc", "4-class accuracy ↑"),
    ]

    # ---- 1. Bar chart: mean ± std across subjects ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    for ax, (mean_k, std_k, _, ylabel) in zip(axes, bench_metrics):
        means = [summary[k][mean_k] for k in model_keys]
        stds  = [summary[k][std_k]  for k in model_keys]
        bars  = ax.bar(range(M), means, yerr=stds, capsize=4, color=colors, alpha=0.85)
        ax.set_xticks(range(M))
        ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11)
        ax.axhline(0, color="black", lw=0.5, linestyle="--")
        if "acc" in mean_k:
            ax.axhline(1 / N_CLASSES, color="red", lw=1, linestyle=":",
                       label=f"chance={1/N_CLASSES:.2f}")
            ax.legend(fontsize=8)
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.001,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=7)

    plt.suptitle(
        f"MEG img→lis LOSO benchmark  ({N_SUBJ} held-out subjects, "
        f"win={WIN_MS}ms stride={STRIDE_MS}ms)",
        fontsize=11,
    )
    plt.tight_layout()
    out = os.path.join(LOSO_OUT_DIR, "comparison_barplot.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")

    # ---- 2. Per-subject line plots ----
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    linestyles = ["-", "--"]

    for ax, (_, _, subj_key, ylabel) in zip(axes, bench_metrics):
        for i, key in enumerate(model_keys):
            mode = key.rsplit("_", 1)[1]
            ls   = linestyles[0] if mode == "full" else linestyles[1]
            vals = summary[key][subj_key]
            ax.plot(range(N_SUBJ), vals, marker="o",
                    label=_display_name(key), color=colors[i],
                    linestyle=ls, alpha=0.85)
        ax.set_xticks(range(N_SUBJ))
        ax.set_xticklabels(subjects, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Held-out subject")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Per-subject {ylabel}")
        ax.legend(fontsize=6, ncol=2)
        if "acc" in subj_key:
            ax.axhline(1 / N_CLASSES, color="red", lw=0.8, linestyle=":")

    plt.suptitle("Per-held-out-subject metrics: solid=full, dashed=windowed",
                 fontsize=11)
    plt.tight_layout()
    out = os.path.join(LOSO_OUT_DIR, "per_subject_lineplots.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")

    # ---- 3. Accuracy heatmap: subjects × models ----
    accs = np.array([[summary[k]["per_subject_clf_acc"][i]
                      for k in model_keys]
                     for i in range(N_SUBJ)])   # (N_SUBJ, M)

    fig, ax = plt.subplots(figsize=(max(10, M * 0.8), 6))
    im = ax.imshow(accs, aspect="auto", vmin=0, vmax=1,
                   cmap="RdYlGn", interpolation="nearest")
    ax.set_xticks(range(M))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(N_SUBJ))
    ax.set_yticklabels(subjects, fontsize=9)
    ax.set_title("Per-subject 4-class accuracy (LOSO)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    for i in range(N_SUBJ):
        for j in range(M):
            ax.text(j, i, f"{accs[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")

    plt.tight_layout()
    out = os.path.join(LOSO_OUT_DIR, "per_subject_acc_heatmap.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")

    # ---- 4. Mean r-per-trial across subjects (box per model) ----
    # r_per_trial values are (40,) per subject per model — pool across subjects
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, key in enumerate(model_keys):
        vals = summary[key]["all_trial_mean_r"]   # list of 40*N_SUBJ values
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=colors[i], alpha=0.3, s=8)
        parts = ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
                           boxprops=dict(facecolor=colors[i], alpha=0.5),
                           medianprops=dict(color="black", lw=1.5),
                           whiskerprops=dict(color=colors[i]),
                           capprops=dict(color=colors[i]),
                           flierprops=dict(marker="", markersize=0))
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xticks(range(M))
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Mean r over channels (per trial)")
    ax.set_title("Distribution of per-trial mean Pearson r across all held-out subjects")
    plt.tight_layout()
    out = os.path.join(LOSO_OUT_DIR, "r_per_trial_distribution.png")
    plt.savefig(out, dpi=200); plt.close()
    print(f"[saved] {out}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    print(f"Device: {DEVICE}")
    print(f"LOSO over {len(SUBJECTS)} subjects")
    print(f"Epochs={N_EPOCHS}  LR={LR}  Batch={BATCH_SIZE}  Dropout={DROPOUT}")
    print(f"Window={WIN_MS}ms  Stride={STRIDE_MS}ms  ({WIN_T} / {STRIDE_T} samples)")
    print(f"Ridge alpha={ALPHA_RIDGE}  Lag ±{LAG_BEFORE_MS}ms\n")

    print("Loading all subjects...")
    data_by_subj = load_all_subjects()
    cond_to_idx  = {c: i for i, c in enumerate(COND_ALL)}
    C = data_by_subj[SUBJECTS[0]].shape[2]
    T = data_by_subj[SUBJECTS[0]].shape[3]
    print(f"C={C}  T={T}")

    lb = ms_to_samples(LAG_BEFORE_MS)
    la = ms_to_samples(LAG_AFTER_MS)
    print("\n=== Parameter counts ===")
    print(f"  {'LinearLag':20s}: {ridge_param_count(C, lb, la):>10,}  (effective)")
    for mname, m in make_models(C).items():
        print(f"  {mname:20s}: {count_params(m):>10,}")
    print()

    model_keys = ["LinearLag_full"] + [
        f"{m}_{mode}" for m in NEURAL_NAMES for mode in ALL_MODES
    ]

    all_results: Dict[str, Dict] = {}   # keyed by heldout subject

    for subj in SUBJECTS:
        print(f"\n{'='*65}")
        print(f"  HELD-OUT: {subj}  (train on {len(SUBJECTS)-1} subjects)")
        print(f"{'='*65}")

        subj_res = run_loso_subject(subj, data_by_subj, cond_to_idx, C)
        all_results[subj] = subj_res

        with open(os.path.join(LOSO_OUT_DIR,
                               f"heldout_{subj}_results.json"), "w") as f:
            json.dump(subj_res, f, indent=2)

        print(f"\n  --- {subj} summary ---")
        print(f"  {'Model':30s}  {'mean_r':>8}  {'MSE':>8}  {'acc':>6}")
        for key in model_keys:
            if key not in subj_res:
                continue
            r = subj_res[key]
            print(f"  {key:30s}  {r['mean_r']:8.4f}  {r['mse']:8.4f}  {r['clf_acc']:6.3f}")

    # ---- summary across held-out subjects ----
    print(f"\n{'='*65}")
    print("  SUMMARY ACROSS ALL HELD-OUT SUBJECTS")
    print(f"{'='*65}")

    summary: Dict[str, Dict] = {}
    for key in model_keys:
        mean_rs = [all_results[s][key]["mean_r"]  for s in SUBJECTS]
        mses    = [all_results[s][key]["mse"]     for s in SUBJECTS]
        accs    = [all_results[s][key]["clf_acc"] for s in SUBJECTS]
        # per-trial mean-r values pooled across all subjects: (40 * N_SUBJ,)
        all_trial_r = [v for s in SUBJECTS
                       for v in all_results[s][key]["mean_r_per_trial"]]

        summary[key] = {
            "mean_r_mean":         float(np.mean(mean_rs)),
            "mean_r_std":          float(np.std(mean_rs)),
            "mse_mean":            float(np.mean(mses)),
            "mse_std":             float(np.std(mses)),
            "clf_acc_mean":        float(np.mean(accs)),
            "clf_acc_std":         float(np.std(accs)),
            "per_subject_mean_r":  mean_rs,
            "per_subject_mse":     mses,
            "per_subject_clf_acc": accs,
            "all_trial_mean_r":    all_trial_r,
        }
        print(
            f"  {key:30s}  "
            f"r={np.mean(mean_rs):.4f}±{np.std(mean_rs):.4f}  "
            f"mse={np.mean(mses):.4f}±{np.std(mses):.4f}  "
            f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}"
        )

    with open(os.path.join(LOSO_OUT_DIR, "summary_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(summary, model_keys, SUBJECTS)
    print(f"\nAll outputs saved to {LOSO_OUT_DIR}")


if __name__ == "__main__":
    main()
