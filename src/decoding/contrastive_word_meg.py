"""
contrastive_word_meg.py
=======================
Contrastive learning pipeline to decode words from MEG.

Architecture
------------
Phase 1 — train MEG encoder on LISTENED MEG:

    LISTENED MEG  →  window [-200ms, +800ms] around word onset
                  →  MEGWordEncoder (small CNN)  →  128-d embedding
                  ↕  NT-Xent contrastive loss
    word text     →  text encoder (BERT / GloVe, frozen)
                  →  learned projection  →  128-d embedding

Phase 2 — evaluate on IMAGINED MEG:

    IMAGINED MEG  →  img→lis mapping model (from benchmark)
                  →  predicted listened MEG
                  →  same windowing
                  →  frozen MEGWordEncoder  →  128-d embedding
                  →  cosine similarity against word embedding bank
                  →  rank-k accuracy

Notes on flash removal
-----------------------
The benchmark removes ~51 samples after each flash event (every 207 samples at
100 Hz), which eliminates ~25 % of time points.  With a 100-sample word window
this causes ~91 % of word windows to overlap a removed region, leaving only ~5
words per poem.

This pipeline therefore defaults to REMOVE_FLASHES = False: the flash artifact
is temporally fixed (constant across all words) so it cannot systematically
confound word identity decoding.  The contrastive loss will push apart word
representations regardless of the shared flash background.

Set REMOVE_FLASHES = True to run on the flash-removed signal that the img→lis
benchmark models were trained on (useful for Phase 2 evaluation consistency).

Usage
-----
  # train MEG encoder on all subjects' listened MEG
  python contrastive_word_meg.py --phase train

  # evaluate on listened MEG (upper bound)
  python contrastive_word_meg.py --phase eval_lis

  # evaluate on imagined MEG (requires benchmark img→lis model checkpoint)
  python contrastive_word_meg.py --phase eval_img \\
      --img_lis_ckpt loso_out/models/heldout_sub-01/CNN1D_full.pt \\
      --img_lis_arch CNN1D --heldout_subject sub-01

Requirements
------------
  pip install transformers   (for BERT text encoder; already available)
  pip install gensim         (optional, for GloVe fallback)
"""

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample
from torch.utils.data import DataLoader, Dataset, random_split

import mne
mne.set_log_level("ERROR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
#  CONFIG — edit these
# =============================================================================
BASE_PATH    = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
ONSET_DIR    = "./onset_out"
OUT_DIR      = "./contrastive_out"
os.makedirs(OUT_DIR, exist_ok=True)

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEM_KEYS = ["poem1", "poem2"]

# MEG preprocessing
DS_FACTOR    = 10
SFREQ_DS     = 100.0      # Hz (after downsampling)
N_SESSIONS   = 10
EPOCH_TMIN_S = 0.0        # epoch t=0 aligned to stimulus onset (verified from data)

# Word window: [-200ms, +800ms] around onset
WIN_PRE_MS  = 200
WIN_POST_MS = 800
WIN_PRE     = int(WIN_PRE_MS  * SFREQ_DS / 1000)   # 20 samples
WIN_POST    = int(WIN_POST_MS * SFREQ_DS / 1000)   # 80 samples
WIN_SIZE    = WIN_PRE + WIN_POST                    # 100 samples

# Flash removal (see module docstring)
# False = keep all words (recommended for contrastive training)
# True  = skip words whose window overlaps a flash region (~91% loss)
REMOVE_FLASHES = False

# Contrastive model
EMB_DIM      = 128
TEMPERATURE  = 0.07
DROPOUT      = 0.3

# MODEL_SIZE: "full" (~544k params, ~550ms/batch CPU) or
#             "small" (~143k params, ~170ms/batch CPU — 3× faster, use on CPU)
MODEL_SIZE   = "small" # modify at run: python contrastive_word_meg.py --phase train --model_size full

# Training
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 100
PATIENCE     = 15
VAL_FRAC     = 0.15
SEED         = 42

# Text encoder: "bert" | "glove" | "random"
TEXT_ENCODER = "bert"
BERT_MODEL   = "bert-base-uncased"
GLOVE_PATH   = None   # path to glove.6B.100d.txt (only needed if TEXT_ENCODER="glove")

def _get_device() -> torch.device:
    """
    Return CUDA device only if the GPU supports the current PyTorch build.
    PyTorch ≥2.0 requires compute capability ≥7.0 (Volta+).
    GTX TITAN X / GTX 9xx / earlier Maxwell/Kepler cards are sm_52 and will
    raise 'no kernel image available' at runtime — fall back to CPU instead.
    """
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 7:
            return torch.device("cuda")
        print(
            f"  INFO: GPU has compute capability {cap[0]}.{cap[1]} "
            f"(need ≥7.0 for PyTorch {torch.__version__}) — using CPU"
        )
    return torch.device("cpu")


DEVICE = _get_device()


# =============================================================================
#  FLASH REMOVAL UTILITIES
# =============================================================================

def build_keep_map(n_t: int) -> np.ndarray:
    """Return sorted array of original sample indices kept after flash removal."""
    fidx = np.arange(207, n_t, 207, dtype=int)
    rmv: set = set()
    for idx in fidx:
        rmv.update(range(idx, min(idx + 51, n_t)))
    return np.setdiff1d(np.arange(n_t), sorted(rmv))   # shape (n_kept,)


def onset_to_window_flash_removed(
    onset_s: float,
    keep: np.ndarray,
    pre: int = WIN_PRE,
    post: int = WIN_POST,
) -> Optional[Tuple[int, int]]:
    """
    Map an audio onset (seconds) to (start, end) in the flash-removed time axis.
    Returns None if the window spans a removed region or is out of bounds.
    """
    orig_onset = int(round((onset_s - EPOCH_TMIN_S) * SFREQ_DS))
    orig_start = orig_onset - pre
    orig_end   = orig_onset + post  # exclusive

    if orig_start < 0 or orig_end > int(keep[-1]) + 1:
        return None

    # Build reverse map (O(log n) per sample via searchsorted)
    orig_range = np.arange(orig_start, orig_end)
    pos = np.searchsorted(keep, orig_range)
    valid = (pos < len(keep)) & (keep[pos] == orig_range)
    if not np.all(valid):
        return None   # overlaps a removed flash region

    return int(pos[0]), int(pos[-1]) + 1


def onset_to_window_raw(
    onset_s: float,
    n_t: int,
    pre: int = WIN_PRE,
    post: int = WIN_POST,
) -> Optional[Tuple[int, int]]:
    """Map audio onset to (start, end) on the raw (no flash removal) time axis."""
    orig_onset = int(round((onset_s - EPOCH_TMIN_S) * SFREQ_DS))
    orig_start = orig_onset - pre
    orig_end   = orig_onset + post
    if orig_start < 0 or orig_end > n_t:
        return None
    return orig_start, orig_end


# =============================================================================
#  MEG DATA LOADING
# =============================================================================

def load_meg_trial(
    subject: str,
    cond: str,
    session: int,
    apply_flash_removal: bool = REMOVE_FLASHES,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Load, downsample, z-score one MEG trial.

    Returns
    -------
    data : (C, T) float32   — T depends on apply_flash_removal
    keep : (T,) int | None  — original-index map (None if no removal)
    """
    fname = (f"{subject}_sess-{session}_task-{cond}_meg-epo.fif")
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
    epochs  = mne.read_epochs(fpath, preload=True)
    raw     = epochs.get_data().mean(axis=0)           # (C, T_raw)
    new_T   = raw.shape[1] // DS_FACTOR
    data_ds = resample(raw, new_T, axis=1).astype(np.float32)  # (C, T_ds)

    # z-score per channel
    mu  = data_ds.mean(axis=1, keepdims=True)
    sd  = np.maximum(data_ds.std(axis=1, keepdims=True), 1e-12)
    data_ds = (data_ds - mu) / sd

    if apply_flash_removal:
        keep    = build_keep_map(data_ds.shape[-1])
        data_ds = data_ds[:, keep]
        return data_ds, keep

    return data_ds, None


# =============================================================================
#  DATASET
# =============================================================================

class MEGWordDataset(Dataset):
    """
    Each item: (meg_window, word_idx)
      meg_window : (C, WIN_SIZE) float32
      word_idx   : int — index into self.vocab (word → int)

    self.vocab   : dict  word_str → int
    self.words   : list  word_str in vocab-index order
    self.pairs   : list  (meg_window_ndarray, word_str)
    """

    def __init__(
        self,
        subjects: List[str],
        poem_keys: List[str],
        onset_dir: str,
        cond_suffix: str = "lis",          # "lis" for listened, "img" for imagined
        remove_flashes: bool = REMOVE_FLASHES,
        min_word_len: int = 1,
    ):
        self.pairs: List[Tuple[np.ndarray, str]] = []
        self.vocab: Dict[str, int] = {}

        for poem_key in poem_keys:
            onset_file = os.path.join(onset_dir, f"{poem_key}_word_onsets.json")
            if not os.path.exists(onset_file):
                print(f"  WARNING: {onset_file} not found — skipping {poem_key}")
                continue
            with open(onset_file) as f:
                word_onsets = json.load(f)

            cond = f"{poem_key}{cond_suffix}"

            for subject in subjects:
                for session in range(N_SESSIONS):
                    try:
                        data, keep = load_meg_trial(
                            subject, cond, session,
                            apply_flash_removal=remove_flashes,
                        )
                    except Exception as e:
                        print(f"  WARNING: {subject}/{cond}/ses-{session}: {e}")
                        continue

                    n_t = data.shape[-1]
                    n_added = 0

                    for w in word_onsets:
                        word = w["word"].strip().lower()
                        if len(word) < min_word_len:
                            continue

                        if remove_flashes and keep is not None:
                            idx = onset_to_window_flash_removed(w["start"], keep)
                        else:
                            idx = onset_to_window_raw(w["start"], n_t)

                        if idx is None:
                            continue

                        start, end = idx
                        window = data[:, start:end]    # (C, WIN_SIZE)
                        if window.shape[-1] != WIN_SIZE:
                            continue

                        self.pairs.append((window.copy(), word))
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
                        n_added += 1

        self.words = sorted(self.vocab, key=self.vocab.get)   # idx → word
        print(
            f"  MEGWordDataset ({cond_suffix}): "
            f"{len(self.pairs)} windows, "
            f"{len(self.vocab)} unique words, "
            f"{len(subjects)} subjects"
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        window, word = self.pairs[idx]
        return torch.from_numpy(window), self.vocab[word]


# =============================================================================
#  MODEL COMPONENTS
# =============================================================================

class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int = 1, dropout: float = DROPOUT):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MEGWordEncoder(nn.Module):
    """
    Input : (B, C, WIN_SIZE)  — C MEG channels, 100 time samples
    Output: (B, EMB_DIM)      — L2-normalised embedding
    """

    def __init__(
        self,
        n_channels: int,
        win_size: int = WIN_SIZE,
        emb_dim: int = EMB_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        # Spatial compression: C → 64 (one 1×1 conv per channel)
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        # Temporal feature extraction with increasing dilation
        self.temporal = nn.Sequential(
            _ConvBlock(64,  128, 7, dilation=1, dropout=dropout),
            _ConvBlock(128, 128, 5, dilation=2, dropout=dropout),
            _ConvBlock(128, 256, 3, dilation=4, dropout=dropout),
            _ConvBlock(256, 256, 3, dilation=8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)        # (B, 64, T)
        x = self.temporal(x)       # (B, 256, T)
        x = self.pool(x).squeeze(-1)  # (B, 256)
        return F.normalize(self.proj(x), dim=-1)  # (B, emb_dim)


class MEGWordEncoderSmall(nn.Module):
    """
    Lighter encoder for CPU training (~143k params, ~3× faster than full).
    Compresses to 32 spatial features instead of 64, 3 temporal layers,
    max 128 hidden units.  Recommended when running without a Volta+ GPU.
    """

    def __init__(
        self,
        n_channels: int,
        win_size: int = WIN_SIZE,
        emb_dim: int = EMB_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.temporal = nn.Sequential(
            _ConvBlock(32,  64,  7, dilation=1, dropout=dropout),
            _ConvBlock(64,  128, 5, dilation=2, dropout=dropout),
            _ConvBlock(128, 128, 3, dilation=4, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


def make_meg_encoder(
    n_channels: int,
    model_size: str = MODEL_SIZE,
    dropout: float = DROPOUT,
) -> nn.Module:
    """Factory: returns full or small encoder based on MODEL_SIZE config."""
    if model_size == "small":
        return MEGWordEncoderSmall(n_channels, dropout=dropout)
    return MEGWordEncoder(n_channels, dropout=dropout)


class TextEncoder(nn.Module):
    """
    Frozen word embeddings (BERT/GloVe) + learned projection to EMB_DIM.
    Only the projection is trained; base embeddings are fixed.
    """

    def __init__(
        self,
        raw_embeddings: torch.Tensor,   # (V, raw_dim)
        emb_dim: int = EMB_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.register_buffer("embeddings", raw_embeddings)
        raw_dim = raw_embeddings.shape[1]
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )

    def forward(self, word_indices: torch.Tensor) -> torch.Tensor:
        """word_indices: (B,) → (B, emb_dim) normalised"""
        raw = self.embeddings[word_indices]      # (B, raw_dim)
        return F.normalize(self.proj(raw), dim=-1)

    @torch.no_grad()
    def get_all(self) -> torch.Tensor:
        """(V, emb_dim) normalised embeddings for the full vocabulary."""
        return F.normalize(self.proj(self.embeddings), dim=-1)


# =============================================================================
#  TEXT EMBEDDING EXTRACTION
# =============================================================================

def build_text_embeddings_bert(
    words: List[str],
    model_name: str = BERT_MODEL,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Returns (V, 768) float32 — mean last-layer embedding (no special tokens).
    Batch-processes all words for efficiency.
    """
    if device is None:
        device = DEVICE
    from transformers import AutoModel, AutoTokenizer
    print(f"  Loading BERT tokenizer/model: {model_name} (device={device})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device).eval()

    embeddings = []
    CHUNK = 64
    with torch.no_grad():
        for i in range(0, len(words), CHUNK):
            batch = words[i : i + CHUNK]
            enc   = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=8,
            ).to(device)
            out   = model(**enc).last_hidden_state   # (B, seq_len, 768)
            # Mean over non-padding tokens (exclude [CLS] and [SEP])
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            emb    = (summed / counts).cpu()          # (B, 768)
            embeddings.append(emb)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    result = torch.cat(embeddings, dim=0).float()   # (V, 768)
    print(f"  BERT embeddings: {result.shape}")
    return result


def build_text_embeddings_glove(
    words: List[str],
    glove_path: str,
) -> torch.Tensor:
    """
    Returns (V, glove_dim) float32. Unknown words get a zero vector.
    """
    import gensim.downloader as api
    if glove_path and os.path.exists(glove_path):
        from gensim.models import KeyedVectors
        kv = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
    else:
        print("  Downloading GloVe (glove-wiki-gigaword-100) via gensim...")
        kv = api.load("glove-wiki-gigaword-100")

    dim = kv.vector_size
    out = np.zeros((len(words), dim), dtype=np.float32)
    missing = 0
    for i, w in enumerate(words):
        if w in kv:
            out[i] = kv[w]
        else:
            missing += 1
    print(f"  GloVe embeddings: {out.shape}  missing={missing}/{len(words)}")
    return torch.from_numpy(out)


def build_text_embeddings_random(
    words: List[str],
    dim: int = 300,
    seed: int = SEED,
) -> torch.Tensor:
    """Unit-sphere random vectors — for debugging / ablation."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((len(words), dim)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True) + 1e-12
    print(f"  Random embeddings: {raw.shape}  (debugging only)")
    return torch.from_numpy(raw)


def build_text_embeddings(
    words: List[str],
    method: str = TEXT_ENCODER,
) -> torch.Tensor:
    if method == "bert":
        return build_text_embeddings_bert(words)
    elif method == "glove":
        return build_text_embeddings_glove(words, GLOVE_PATH or "")
    elif method == "random":
        return build_text_embeddings_random(words)
    else:
        raise ValueError(f"Unknown TEXT_ENCODER: {method!r}")


# =============================================================================
#  CONTRASTIVE LOSS
# =============================================================================

def nt_xent_loss(
    z_meg:  torch.Tensor,   # (N, D) normalised
    z_text: torch.Tensor,   # (N, D) normalised
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    """
    Symmetric NT-Xent (InfoNCE) cross-modal loss.
    Diagonal entries are positives; off-diagonal are negatives.

    Note: if the same word appears multiple times in the batch, those
    duplicate entries are treated as negatives — a known limitation for
    high-frequency words ("the", "a").  For a batch of 64 drawn from a
    ~117-word poem vocabulary this is acceptable.
    """
    N = z_meg.shape[0]
    sim = z_meg @ z_text.T / temperature        # (N, N)
    labels = torch.arange(N, device=z_meg.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss


# =============================================================================
#  TRAINING
# =============================================================================

def train(
    meg_encoder:  MEGWordEncoder,
    text_encoder: TextEncoder,
    train_set:    MEGWordDataset,
    val_set:      MEGWordDataset,
    out_dir:      str = OUT_DIR,
) -> Tuple[MEGWordEncoder, TextEncoder]:
    """
    Train MEGWordEncoder and TextEncoder.proj jointly with NT-Xent loss.
    """
    meg_encoder  = meg_encoder.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)

    # Only train MEG encoder + text projection (base embeddings are frozen)
    params = list(meg_encoder.parameters()) + list(text_encoder.proj.parameters())
    opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    tr_dl  = DataLoader(train_set, BATCH_SIZE, shuffle=True,  drop_last=True,
                        num_workers=0)
    val_dl = DataLoader(val_set,  BATCH_SIZE, shuffle=False, drop_last=False,
                        num_workers=0)

    best_val   = float("inf")
    best_meg_w = deepcopy(meg_encoder.state_dict())
    best_txt_w = deepcopy(text_encoder.state_dict())
    no_imp     = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, N_EPOCHS + 1):
        meg_encoder.train()
        text_encoder.train()
        tr_losses = []

        for meg_win, word_idx in tr_dl:
            meg_win  = meg_win.to(DEVICE)
            word_idx = word_idx.to(DEVICE)

            # Light augmentation: channel-independent Gaussian noise
            meg_win = meg_win + 0.02 * torch.randn_like(meg_win)

            z_meg  = meg_encoder(meg_win)
            z_text = text_encoder(word_idx)
            loss   = nt_xent_loss(z_meg, z_text)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_losses.append(loss.item())

        sched.step()

        # Validation
        meg_encoder.eval()
        text_encoder.eval()
        val_losses = []
        with torch.no_grad():
            for meg_win, word_idx in val_dl:
                meg_win  = meg_win.to(DEVICE)
                word_idx = word_idx.to(DEVICE)
                z_meg    = meg_encoder(meg_win)
                z_text   = text_encoder(word_idx)
                val_losses.append(nt_xent_loss(z_meg, z_text).item())

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        history["train"].append(tr_loss)
        history["val"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{N_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  no_imp={no_imp}")

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_meg_w = deepcopy(meg_encoder.state_dict())
            best_txt_w = deepcopy(text_encoder.state_dict())
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"  early stop at epoch {epoch}")
                break

    # Restore best weights
    meg_encoder.load_state_dict(best_meg_w)
    text_encoder.load_state_dict(best_txt_w)

    # Save checkpoints
    torch.save(meg_encoder.state_dict(),
               os.path.join(out_dir, "meg_encoder.pt"))
    torch.save(text_encoder.state_dict(),
               os.path.join(out_dir, "text_encoder.pt"))
    print(f"  [saved] checkpoints in {out_dir}/")

    _plot_training_curve(history, out_dir)
    return meg_encoder, text_encoder


def _plot_training_curve(history: dict, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"],   label="val")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent loss")
    ax.set_title("Contrastive training curve"); ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# =============================================================================
#  EVALUATION — ranking metrics
# =============================================================================

@torch.no_grad()
def evaluate_ranking(
    meg_encoder:  MEGWordEncoder,
    text_encoder: TextEncoder,
    dataset:      MEGWordDataset,
    tag:          str = "",
) -> Dict:
    """
    For each MEG window, rank the correct word by cosine similarity against
    all word embeddings in the vocabulary.

    Returns dict with R@1, R@5, R@10, MRR, median_rank.
    """
    meg_encoder.eval()
    text_encoder.eval()

    # Pre-compute all word embeddings once
    all_text = text_encoder.get_all().to(DEVICE)   # (V, D)
    V = all_text.shape[0]

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    ranks  = []

    for meg_win, word_idx in loader:
        meg_win  = meg_win.to(DEVICE)
        word_idx = word_idx.to(DEVICE)

        z_meg = meg_encoder(meg_win)              # (B, D)
        sim   = z_meg @ all_text.T                # (B, V) cosine similarities

        for i in range(len(z_meg)):
            true_idx = word_idx[i].item()
            s = sim[i]
            # rank: 1 = best (number of words with higher similarity + 1)
            rank = int((s > s[true_idx]).sum().item()) + 1
            ranks.append(rank)

    ranks = np.array(ranks, dtype=np.int32)
    metrics = {
        "tag":         tag,
        "n_samples":   int(len(ranks)),
        "vocab_size":  int(V),
        "R@1":         float((ranks <= 1).mean()),
        "R@5":         float((ranks <= 5).mean()),
        "R@10":        float((ranks <= 10).mean()),
        "MRR":         float((1.0 / ranks).mean()),
        "median_rank": int(np.median(ranks)),
        "chance_R@1":  float(1.0 / V),
    }

    print(
        f"  [{tag}] R@1={metrics['R@1']:.3f}  R@5={metrics['R@5']:.3f}  "
        f"R@10={metrics['R@10']:.3f}  MRR={metrics['MRR']:.3f}  "
        f"median_rank={metrics['median_rank']}/{V}  "
        f"(chance R@1={metrics['chance_R@1']:.3f})"
    )
    return metrics


# =============================================================================
#  PHASE 2 — evaluation with imagined MEG via img→lis mapping
# =============================================================================

def _load_img_lis_model(
    arch:      str,
    ckpt_path: str,
    n_channels: int,
) -> nn.Module:
    """
    Load a trained img→lis mapping model from the benchmark.
    Only CNN1D is implemented here as an example; extend for others.
    """
    # Import architectures from the benchmark (must be on sys.path or same dir)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark"))
    from benchmark_loso import CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN

    arch_map = {
        "CNN1D":      lambda: CNN1D(n_channels, TARGET_HIDDEN),
        "ShallowMLP": lambda: ShallowMLP(n_channels, TARGET_HIDDEN),
        "UNet1D":     lambda: UNet1D(n_channels, TARGET_HIDDEN // 2),
        "RNN":        lambda: RNN(n_channels, TARGET_HIDDEN),
        "TCN":        lambda: TCN(n_channels, TARGET_HIDDEN // 2),
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch {arch!r}; choose from {list(arch_map)}")

    model = arch_map[arch]()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval()


def build_imagined_dataset(
    meg_encoder:     MEGWordEncoder,
    text_encoder:    TextEncoder,
    heldout_subject: str,
    img_lis_ckpt:    str,
    img_lis_arch:    str,
    n_channels:      int,
) -> MEGWordDataset:
    """
    Apply img→lis mapping to imagined MEG, return dataset ready for ranking.
    The returned dataset wraps the *predicted* listened windows.
    """
    img_lis = _load_img_lis_model(img_lis_arch, img_lis_ckpt, n_channels).to(DEVICE)

    # Build a dataset from the imagined MEG of the held-out subject
    img_ds = MEGWordDataset(
        subjects=[heldout_subject],
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="img",
        remove_flashes=REMOVE_FLASHES,
    )

    # Replace each MEG window with the img→lis model's prediction
    mapped_pairs = []
    with torch.no_grad():
        for i in range(len(img_ds)):
            window, word_str = img_ds.pairs[i]
            x     = torch.from_numpy(window).unsqueeze(0).to(DEVICE)  # (1, C, T)
            x_hat = img_lis(x).squeeze(0).cpu().numpy()               # (C, T)
            mapped_pairs.append((x_hat, word_str))

    # Patch the dataset's pairs with mapped windows
    img_ds.pairs = mapped_pairs
    print(f"  Applied {img_lis_arch} img→lis mapping: {len(mapped_pairs)} windows")
    return img_ds


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Contrastive MEG word decoding")
    parser.add_argument(
        "--phase",
        choices=["train", "eval_lis", "eval_img"],
        default="train",
        help=(
            "train: train MEG encoder on listened MEG\n"
            "eval_lis: evaluate encoder on listened MEG (upper bound)\n"
            "eval_img: evaluate on imagined MEG via img→lis mapping"
        ),
    )
    parser.add_argument("--img_lis_ckpt",  default=None,
                        help="Path to benchmark img→lis model .pt checkpoint")
    parser.add_argument("--img_lis_arch",  default="CNN1D",
                        help="Architecture name (CNN1D | ShallowMLP | UNet1D | RNN | TCN)")
    parser.add_argument("--heldout_subject", default=None,
                        help="Subject to evaluate on (e.g. sub-01)")
    parser.add_argument("--text_encoder",
                        choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    parser.add_argument("--remove_flashes", action="store_true",
                        default=REMOVE_FLASHES,
                        help="Apply flash removal (severely reduces word count)")
    parser.add_argument("--model_size", choices=["small", "full"],
                        default=MODEL_SIZE,
                        help="small=~143k params ~3x faster on CPU; full=~544k params")
    args = parser.parse_args()

    print(f"Device       : {DEVICE}")
    print(f"Phase        : {args.phase}")
    print(f"Model size   : {args.model_size}")
    print(f"Text encoder : {args.text_encoder}")
    print(f"Remove flash : {args.remove_flashes}")
    print(f"Window       : [{WIN_PRE_MS}ms, +{WIN_POST_MS}ms] = {WIN_SIZE} samples")
    print(f"Out dir      : {OUT_DIR}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------
    #  Phase: train
    # ------------------------------------------------------------------
    if args.phase == "train":

        print("Building dataset from listened MEG...")
        full_ds = MEGWordDataset(
            subjects=SUBJECTS,
            poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR,
            cond_suffix="lis",
            remove_flashes=args.remove_flashes,
        )

        n_val = max(1, int(VAL_FRAC * len(full_ds)))
        n_tr  = len(full_ds) - n_val
        train_ds, val_ds = random_split(
            full_ds, [n_tr, n_val],
            generator=torch.Generator().manual_seed(SEED),
        )
        # Share vocab/words from full_ds (random_split wraps items only)
        train_ds.vocab = full_ds.vocab
        val_ds.vocab   = full_ds.vocab
        print(f"  train={n_tr}  val={n_val}")

        print(f"\nBuilding text embeddings ({args.text_encoder})...")
        raw_emb = build_text_embeddings(full_ds.words, method=args.text_encoder)

        n_channels = full_ds.pairs[0][0].shape[0]
        print(f"\nInitialising models  (C={n_channels}, size={args.model_size})...")
        meg_enc  = make_meg_encoder(n_channels, args.model_size)
        txt_enc  = TextEncoder(raw_emb)
        n_meg    = sum(p.numel() for p in meg_enc.parameters())
        n_proj   = sum(p.numel() for p in txt_enc.proj.parameters())
        print(f"  MEGWordEncoder ({args.model_size}): {n_meg:,} params")
        print(f"  TextEncoder.proj: {n_proj:,} params  "
              f"(base {raw_emb.shape[1]}d embeddings frozen)")

        print(f"\nTraining...")
        meg_enc, txt_enc = train(meg_enc, txt_enc, train_ds, val_ds)

        print(f"\nEvaluating on validation set...")
        metrics = evaluate_ranking(meg_enc, txt_enc, val_ds,
                                   tag="val (listened)")
        with open(os.path.join(OUT_DIR, "val_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    #  Phase: eval_lis  (upper bound)
    # ------------------------------------------------------------------
    elif args.phase == "eval_lis":

        print("Loading dataset (listened MEG)...")
        ds = MEGWordDataset(
            subjects=SUBJECTS if args.heldout_subject is None
                     else [args.heldout_subject],
            poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR,
            cond_suffix="lis",
            remove_flashes=args.remove_flashes,
        )

        raw_emb    = build_text_embeddings(ds.words, method=args.text_encoder)
        n_channels = ds.pairs[0][0].shape[0]

        meg_enc = make_meg_encoder(n_channels, args.model_size)
        txt_enc = TextEncoder(raw_emb)

        ckpt_meg = os.path.join(OUT_DIR, "meg_encoder.pt")
        ckpt_txt = os.path.join(OUT_DIR, "text_encoder.pt")
        if not os.path.exists(ckpt_meg):
            raise FileNotFoundError(f"{ckpt_meg} not found — run --phase train first")

        meg_enc.load_state_dict(torch.load(ckpt_meg, map_location="cpu"))
        txt_enc.load_state_dict(torch.load(ckpt_txt, map_location="cpu"))
        meg_enc = meg_enc.to(DEVICE)
        txt_enc = txt_enc.to(DEVICE)

        metrics = evaluate_ranking(meg_enc, txt_enc, ds, tag="eval_lis")
        with open(os.path.join(OUT_DIR, "eval_lis_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    #  Phase: eval_img  (imagined MEG via img→lis mapping)
    # ------------------------------------------------------------------
    elif args.phase == "eval_img":
        if args.heldout_subject is None:
            parser.error("--heldout_subject required for eval_img")
        if args.img_lis_ckpt is None:
            parser.error("--img_lis_ckpt required for eval_img")

        # Use all other subjects' data to build vocab + text embeddings
        other_subjects = [s for s in SUBJECTS if s != args.heldout_subject]
        train_ds = MEGWordDataset(
            subjects=other_subjects,
            poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR,
            cond_suffix="lis",
            remove_flashes=args.remove_flashes,
        )
        raw_emb    = build_text_embeddings(train_ds.words, method=args.text_encoder)
        n_channels = train_ds.pairs[0][0].shape[0]

        meg_enc = make_meg_encoder(n_channels, args.model_size).to(DEVICE)
        txt_enc = TextEncoder(raw_emb).to(DEVICE)
        meg_enc.load_state_dict(
            torch.load(os.path.join(OUT_DIR, "meg_encoder.pt"), map_location="cpu"))
        txt_enc.load_state_dict(
            torch.load(os.path.join(OUT_DIR, "text_encoder.pt"), map_location="cpu"))

        print(f"\nBuilding imagined MEG dataset for {args.heldout_subject}...")
        img_ds = build_imagined_dataset(
            meg_enc, txt_enc,
            heldout_subject=args.heldout_subject,
            img_lis_ckpt=args.img_lis_ckpt,
            img_lis_arch=args.img_lis_arch,
            n_channels=n_channels,
        )
        # Patch vocab into imagined dataset so ranking uses the same indices
        img_ds.vocab = train_ds.vocab
        img_ds.words = train_ds.words

        metrics = evaluate_ranking(
            meg_enc, txt_enc, img_ds,
            tag=f"eval_img ({args.heldout_subject})",
        )
        out_path = os.path.join(
            OUT_DIR, f"eval_img_{args.heldout_subject}_metrics.json"
        )
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    main()
