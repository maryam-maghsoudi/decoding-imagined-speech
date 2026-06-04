"""
contrastive_word_meg_compare.py
================================
Multi-encoder comparison pipeline for MEG word decoding.

Trains and evaluates 4 text-encoder variants on listened MEG, then
applies the full img→lis→decoder pipeline on imagined MEG:

    1. BERT        — semantic/syntactic (bert-base-uncased)
    2. Whisper     — acoustic (openai/whisper-base encoder)
    3. Wav2Vec2    — phonetic/acoustic (facebook/wav2vec2-base-960h)
    4. BERT+Wav2Vec — concatenated (semantic + acoustic)

Outputs (all in OUT_DIR)
------------------------
  models/
    {encoder_name}/
      meg_encoder.pt
      text_encoder.pt
      training_curve.png

  results/
    {encoder_name}/
      val_metrics.json
      eval_lis_metrics.json
      eval_img_{subject}_metrics.json
      ranks_val.npy          — raw rank array (N,)
      ranks_eval_lis.npy
      ranks_eval_img.npy
      per_word_stats.json    — per-word {word: {mean_rank, n, R@1, R@5}}

  comparison/
    rank_cdf_all.png         — CDF of ranks for all 4 encoders
    rank_cdf_top_words.png   — same, restricted to top-K words
    bar_metrics.png          — R@1/R@5/R@10/MRR bar chart
    word_heatmap.png         — per-word median rank heatmap across encoders
    summary.json

Usage
-----
  # Train + eval all encoders on listened MEG
  python contrastive_word_meg_compare.py --phase train_eval_lis

  # Full pipeline: listened + imagined
  python contrastive_word_meg_compare.py --phase full \\
      --img_lis_ckpt loso_out/models/heldout_sub-01/CNN1D_full.pt \\
      --img_lis_arch CNN1D --heldout_subject sub-01

  # Only run specific encoders
  python contrastive_word_meg_compare.py --phase train_eval_lis \\
      --encoders bert wav2vec

Requirements
------------
  pip install transformers torchaudio librosa
  pip install gensim   (only if using glove fallback)
"""

import argparse
import json
import os
import sys
import warnings
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
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
#  CONFIG
# =============================================================================
BASE_PATH  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
ONSET_DIR  = "./onset_out"
AUDIO_DIR  = "./audio"          # directory with word audio segments (wav files)
OUT_DIR    = "./compare_out"

SUBJECTS = [
    "sub-01", "sub-03", "sub-04", "sub-05", "sub-06", "sub-09", "sub-10",
    "sub-11", "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
]
POEM_KEYS = ["poem1", "poem2"]

# MEG preprocessing
DS_FACTOR    = 10
SFREQ_DS     = 100.0
N_SESSIONS   = 10
EPOCH_TMIN_S = 0.0

# Word window
WIN_PRE_MS  = 200
WIN_POST_MS = 800
WIN_PRE     = int(WIN_PRE_MS  * SFREQ_DS / 1000)
WIN_POST    = int(WIN_POST_MS * SFREQ_DS / 1000)
WIN_SIZE    = WIN_PRE + WIN_POST

REMOVE_FLASHES = False

# Model
EMB_DIM      = 128
TEMPERATURE  = 0.07
DROPOUT      = 0.3
MODEL_SIZE   = "small"

# Training
BATCH_SIZE   = 64
LR           = 3e-4
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 100
PATIENCE     = 15
VAL_FRAC     = 0.15
SEED         = 42

# Per-word analysis
TOP_K_WORDS  = 20   # "top words" = those with best median rank across encoders

ENCODER_NAMES = ["bert", "whisper", "wav2vec", "bert_wav2vec"]
ENCODER_COLORS = {
    "bert":         "#4C72B0",
    "whisper":      "#DD8452",
    "wav2vec":      "#55A868",
    "bert_wav2vec": "#C44E52",
}
ENCODER_LABELS = {
    "bert":         "BERT (semantic)",
    "whisper":      "Whisper (acoustic)",
    "wav2vec":      "Wav2Vec2 (phonetic)",
    "bert_wav2vec": "BERT + Wav2Vec2",
}

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "comparison"), exist_ok=True)


# =============================================================================
#  DEVICE
# =============================================================================

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap[0] >= 7:
            return torch.device("cuda")
        print(f"  INFO: GPU compute capability {cap[0]}.{cap[1]} < 7.0 — using CPU")
    return torch.device("cpu")

DEVICE = _get_device()


# =============================================================================
#  FLASH REMOVAL UTILITIES
# =============================================================================

def build_keep_map(n_t: int) -> np.ndarray:
    fidx = np.arange(207, n_t, 207, dtype=int)
    rmv: set = set()
    for idx in fidx:
        rmv.update(range(idx, min(idx + 51, n_t)))
    return np.setdiff1d(np.arange(n_t), sorted(rmv))


def onset_to_window_flash_removed(
    onset_s: float, keep: np.ndarray,
    pre: int = WIN_PRE, post: int = WIN_POST,
) -> Optional[Tuple[int, int]]:
    orig_onset = int(round((onset_s - EPOCH_TMIN_S) * SFREQ_DS))
    orig_start = orig_onset - pre
    orig_end   = orig_onset + post
    if orig_start < 0 or orig_end > int(keep[-1]) + 1:
        return None
    orig_range = np.arange(orig_start, orig_end)
    pos = np.searchsorted(keep, orig_range)
    valid = (pos < len(keep)) & (keep[pos] == orig_range)
    if not np.all(valid):
        return None
    return int(pos[0]), int(pos[-1]) + 1


def onset_to_window_raw(
    onset_s: float, n_t: int,
    pre: int = WIN_PRE, post: int = WIN_POST,
) -> Optional[Tuple[int, int]]:
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
    subject: str, cond: str, session: int,
    apply_flash_removal: bool = REMOVE_FLASHES,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
    epochs  = mne.read_epochs(fpath, preload=True)
    raw     = epochs.get_data().mean(axis=0)
    new_T   = raw.shape[1] // DS_FACTOR
    data_ds = resample(raw, new_T, axis=1).astype(np.float32)
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
    self.pairs  : list of (ndarray (C, WIN_SIZE), word_str)
    self.vocab  : dict word_str → int
    self.words  : list word_str in index order
    self.word_audio : dict word_str → path to audio segment (optional)
    """

    def __init__(
        self,
        subjects: List[str],
        poem_keys: List[str],
        onset_dir: str,
        cond_suffix: str = "lis",
        remove_flashes: bool = REMOVE_FLASHES,
        min_word_len: int = 1,
    ):
        self.pairs: List[Tuple[np.ndarray, str]] = []
        self.vocab: Dict[str, int] = {}
        # Store audio paths keyed by word for Whisper/Wav2Vec
        # Each word may have multiple segments; we keep the first one found.
        self.word_audio: Dict[str, str] = {}

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
                        window = data[:, start:end]
                        if window.shape[-1] != WIN_SIZE:
                            continue

                        self.pairs.append((window.copy(), word))
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)

                        # Store audio segment path if available
                        if word not in self.word_audio:
                            audio_path = os.path.join(
                                AUDIO_DIR, poem_key, f"{word}.wav"
                            )
                            if os.path.exists(audio_path):
                                self.word_audio[word] = audio_path

        self.words = sorted(self.vocab, key=self.vocab.get)
        print(
            f"  MEGWordDataset ({cond_suffix}): "
            f"{len(self.pairs)} windows, "
            f"{len(self.vocab)} unique words, "
            f"{len(subjects)} subjects, "
            f"{len(self.word_audio)} words with audio"
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
    def __init__(self, in_ch, out_ch, kernel, dilation=1, dropout=DROPOUT):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.block(x)


class MEGWordEncoderSmall(nn.Module):
    def __init__(self, n_channels, win_size=WIN_SIZE, emb_dim=EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, 32, 1, bias=False),
            nn.BatchNorm1d(32), nn.GELU(),
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
    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


class MEGWordEncoder(nn.Module):
    def __init__(self, n_channels, win_size=WIN_SIZE, emb_dim=EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, 64, 1, bias=False),
            nn.BatchNorm1d(64), nn.GELU(),
        )
        self.temporal = nn.Sequential(
            _ConvBlock(64,  128, 7, dilation=1, dropout=dropout),
            _ConvBlock(128, 128, 5, dilation=2, dropout=dropout),
            _ConvBlock(128, 256, 3, dilation=4, dropout=dropout),
            _ConvBlock(256, 256, 3, dilation=8, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )
    def forward(self, x):
        x = self.spatial(x)
        x = self.temporal(x)
        x = self.pool(x).squeeze(-1)
        return F.normalize(self.proj(x), dim=-1)


def make_meg_encoder(n_channels, model_size=MODEL_SIZE, dropout=DROPOUT):
    if model_size == "small":
        return MEGWordEncoderSmall(n_channels, dropout=dropout)
    return MEGWordEncoder(n_channels, dropout=dropout)


class TextEncoder(nn.Module):
    def __init__(self, raw_embeddings: torch.Tensor, emb_dim=EMB_DIM, dropout=DROPOUT):
        super().__init__()
        self.register_buffer("embeddings", raw_embeddings)
        raw_dim = raw_embeddings.shape[1]
        self.proj = nn.Sequential(
            nn.Linear(raw_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, emb_dim),
        )
    def forward(self, word_indices):
        raw = self.embeddings[word_indices]
        return F.normalize(self.proj(raw), dim=-1)
    @torch.no_grad()
    def get_all(self):
        return F.normalize(self.proj(self.embeddings), dim=-1)


# =============================================================================
#  TEXT / AUDIO EMBEDDING BUILDERS
# =============================================================================

def build_embeddings_bert(words: List[str], model_name="bert-base-uncased") -> torch.Tensor:
    """(V, 768) — BERT mean last-layer embedding."""
    from transformers import AutoModel, AutoTokenizer
    print(f"  [BERT] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(DEVICE).eval()
    embeddings = []
    CHUNK = 64
    with torch.no_grad():
        for i in range(0, len(words), CHUNK):
            batch = words[i:i+CHUNK]
            enc   = tokenizer(batch, return_tensors="pt", padding=True,
                              truncation=True, max_length=8).to(DEVICE)
            out   = model(**enc).last_hidden_state
            mask  = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            embeddings.append((summed / counts).cpu())
    del model
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    result = torch.cat(embeddings, dim=0).float()
    print(f"  [BERT] shape={result.shape}")
    return result


def build_embeddings_whisper(
    words: List[str],
    word_audio: Dict[str, str],
    model_name: str = "openai/whisper-base",
) -> torch.Tensor:
    """
    (V, 512) — Whisper encoder mean-pool over audio segment.
    For words without an audio file, falls back to a zero vector
    (the projection head will learn to handle these).
    """
    from transformers import WhisperProcessor, WhisperModel
    import librosa
    print(f"  [Whisper] Loading {model_name}...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model     = WhisperModel.from_pretrained(model_name).to(DEVICE).eval()
    dim       = model.config.d_model   # 512 for whisper-base

    embeddings = []
    missing = 0
    with torch.no_grad():
        for word in words:
            audio_path = word_audio.get(word)
            if audio_path and os.path.exists(audio_path):
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                inputs = processor(
                    audio, sampling_rate=16000,
                    return_tensors="pt",
                ).input_features.to(DEVICE)
                # Use encoder only
                enc_out = model.encoder(inputs).last_hidden_state  # (1, T, D)
                emb = enc_out.mean(dim=1).squeeze(0).cpu()         # (D,)
            else:
                emb = torch.zeros(dim)
                missing += 1
            embeddings.append(emb)

    del model
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    result = torch.stack(embeddings, dim=0).float()
    print(f"  [Whisper] shape={result.shape}  missing={missing}/{len(words)}")
    return result


def build_embeddings_wav2vec(
    words: List[str],
    word_audio: Dict[str, str],
    model_name: str = "facebook/wav2vec2-base-960h",
) -> torch.Tensor:
    """
    (V, 768) — Wav2Vec2 mean-pool last hidden state.
    Falls back to zero vector for words without audio.
    """
    from transformers import Wav2Vec2Processor, Wav2Vec2Model
    import librosa
    print(f"  [Wav2Vec2] Loading {model_name}...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2Model.from_pretrained(model_name).to(DEVICE).eval()
    dim       = model.config.hidden_size   # 768

    embeddings = []
    missing = 0
    with torch.no_grad():
        for word in words:
            audio_path = word_audio.get(word)
            if audio_path and os.path.exists(audio_path):
                audio, sr = librosa.load(audio_path, sr=16000, mono=True)
                inputs = processor(
                    audio, sampling_rate=16000,
                    return_tensors="pt", padding=True,
                ).input_values.to(DEVICE)
                out = model(inputs).last_hidden_state  # (1, T, D)
                emb = out.mean(dim=1).squeeze(0).cpu()
            else:
                emb = torch.zeros(dim)
                missing += 1
            embeddings.append(emb)

    del model
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    result = torch.stack(embeddings, dim=0).float()
    print(f"  [Wav2Vec2] shape={result.shape}  missing={missing}/{len(words)}")
    return result


def build_embeddings_combined(
    bert_emb: torch.Tensor,
    wav2vec_emb: torch.Tensor,
) -> torch.Tensor:
    """Concatenate BERT + Wav2Vec2 → (V, 768+768=1536)."""
    result = torch.cat([bert_emb, wav2vec_emb], dim=1).float()
    print(f"  [BERT+Wav2Vec2] shape={result.shape}")
    return result


def build_embeddings_for_encoder(
    encoder_name: str,
    words: List[str],
    word_audio: Dict[str, str],
    cache: dict,
) -> torch.Tensor:
    """
    Build (and cache) raw embeddings for a given encoder name.
    cache is a shared dict to avoid re-computing BERT/wav2vec for combined.
    """
    if encoder_name == "bert":
        if "bert" not in cache:
            cache["bert"] = build_embeddings_bert(words)
        return cache["bert"]

    elif encoder_name == "whisper":
        if "whisper" not in cache:
            cache["whisper"] = build_embeddings_whisper(words, word_audio)
        return cache["whisper"]

    elif encoder_name == "wav2vec":
        if "wav2vec" not in cache:
            cache["wav2vec"] = build_embeddings_wav2vec(words, word_audio)
        return cache["wav2vec"]

    elif encoder_name == "bert_wav2vec":
        if "bert" not in cache:
            cache["bert"] = build_embeddings_bert(words)
        if "wav2vec" not in cache:
            cache["wav2vec"] = build_embeddings_wav2vec(words, word_audio)
        return build_embeddings_combined(cache["bert"], cache["wav2vec"])

    else:
        raise ValueError(f"Unknown encoder: {encoder_name!r}")


# =============================================================================
#  CONTRASTIVE LOSS
# =============================================================================

def nt_xent_loss(z_meg, z_text, temperature=TEMPERATURE):
    N   = z_meg.shape[0]
    sim = z_meg @ z_text.T / temperature
    labels = torch.arange(N, device=z_meg.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


# =============================================================================
#  TRAINING
# =============================================================================

def train_one_encoder(
    encoder_name: str,
    meg_encoder:  nn.Module,
    text_encoder: TextEncoder,
    train_set:    Dataset,
    val_set:      Dataset,
    out_dir:      str,
) -> Tuple[nn.Module, TextEncoder, dict]:
    """Train and return (meg_encoder, text_encoder, history)."""
    meg_encoder  = meg_encoder.to(DEVICE)
    text_encoder = text_encoder.to(DEVICE)

    params = list(meg_encoder.parameters()) + list(text_encoder.proj.parameters())
    opt   = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    tr_dl  = DataLoader(train_set, BATCH_SIZE, shuffle=True,  drop_last=True,  num_workers=0)
    val_dl = DataLoader(val_set,   BATCH_SIZE, shuffle=False, drop_last=False, num_workers=0)

    best_val   = float("inf")
    best_meg_w = deepcopy(meg_encoder.state_dict())
    best_txt_w = deepcopy(text_encoder.state_dict())
    no_imp     = 0
    history    = {"train": [], "val": []}

    print(f"\n  Training [{encoder_name}]...")
    for epoch in range(1, N_EPOCHS + 1):
        meg_encoder.train(); text_encoder.train()
        tr_losses = []
        for meg_win, word_idx in tr_dl:
            meg_win  = meg_win.to(DEVICE)
            word_idx = word_idx.to(DEVICE)
            meg_win  = meg_win + 0.02 * torch.randn_like(meg_win)
            z_meg    = meg_encoder(meg_win)
            z_text   = text_encoder(word_idx)
            loss     = nt_xent_loss(z_meg, z_text)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        meg_encoder.eval(); text_encoder.eval()
        val_losses = []
        with torch.no_grad():
            for meg_win, word_idx in val_dl:
                meg_win  = meg_win.to(DEVICE)
                word_idx = word_idx.to(DEVICE)
                val_losses.append(
                    nt_xent_loss(meg_encoder(meg_win), text_encoder(word_idx)).item()
                )

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        history["train"].append(tr_loss); history["val"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:4d}/{N_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  no_imp={no_imp}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_meg_w = deepcopy(meg_encoder.state_dict())
            best_txt_w = deepcopy(text_encoder.state_dict())
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    early stop at epoch {epoch}")
                break

    meg_encoder.load_state_dict(best_meg_w)
    text_encoder.load_state_dict(best_txt_w)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(meg_encoder.state_dict(),  os.path.join(out_dir, "meg_encoder.pt"))
    torch.save(text_encoder.state_dict(), os.path.join(out_dir, "text_encoder.pt"))
    _plot_training_curve(history, out_dir, encoder_name)
    return meg_encoder, text_encoder, history


def _plot_training_curve(history: dict, out_dir: str, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train", color=ENCODER_COLORS.get(title, "#333"))
    ax.plot(history["val"],   label="val",   color=ENCODER_COLORS.get(title, "#333"), linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"Training curve — {ENCODER_LABELS.get(title, title)}")
    ax.legend(); plt.tight_layout()
    path = os.path.join(out_dir, "training_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
#  EVALUATION — ranking with per-word breakdown
# =============================================================================

@torch.no_grad()
def evaluate_ranking(
    meg_encoder:  nn.Module,
    text_encoder: TextEncoder,
    dataset:      MEGWordDataset,
    tag:          str = "",
    words_list:   Optional[List[str]] = None,
) -> Tuple[Dict, np.ndarray, List[str]]:
    """
    Returns
    -------
    metrics : dict with R@1, R@5, R@10, MRR, median_rank
    ranks   : (N,) int32 array — one entry per MEG window
    word_labels : (N,) list of word strings corresponding to ranks
    """
    meg_encoder.eval(); text_encoder.eval()
    all_text = text_encoder.get_all().to(DEVICE)
    V = all_text.shape[0]

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    ranks_list  = []
    word_labels = []

    for meg_win, word_idx in loader:
        meg_win  = meg_win.to(DEVICE)
        word_idx = word_idx.to(DEVICE)
        z_meg    = meg_encoder(meg_win)
        sim      = z_meg @ all_text.T

        for i in range(len(z_meg)):
            true_idx = word_idx[i].item()
            s    = sim[i]
            rank = int((s > s[true_idx]).sum().item()) + 1
            ranks_list.append(rank)
            if words_list:
                word_labels.append(words_list[true_idx])
            else:
                word_labels.append(str(true_idx))

    ranks = np.array(ranks_list, dtype=np.int32)
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
        f"(chance={metrics['chance_R@1']:.4f})"
    )
    return metrics, ranks, word_labels


def compute_per_word_stats(
    ranks: np.ndarray,
    word_labels: List[str],
) -> Dict[str, Dict]:
    """
    For each unique word, compute:
      mean_rank, median_rank, std_rank, n, R@1, R@5, R@10
    """
    from collections import defaultdict
    word_ranks: Dict[str, List[int]] = defaultdict(list)
    for r, w in zip(ranks, word_labels):
        word_ranks[w].append(int(r))

    stats = {}
    for word, rs in word_ranks.items():
        rs_arr = np.array(rs)
        stats[word] = {
            "n":           int(len(rs_arr)),
            "mean_rank":   float(rs_arr.mean()),
            "median_rank": float(np.median(rs_arr)),
            "std_rank":    float(rs_arr.std()),
            "R@1":         float((rs_arr <= 1).mean()),
            "R@5":         float((rs_arr <= 5).mean()),
            "R@10":        float((rs_arr <= 10).mean()),
        }
    return stats


def get_top_words(per_word_stats_all_encoders: Dict[str, Dict[str, Dict]],
                  top_k: int = TOP_K_WORDS) -> List[str]:
    """
    Return the top_k words with the best (lowest) median rank averaged
    across all encoders.
    """
    all_words = set()
    for stats in per_word_stats_all_encoders.values():
        all_words.update(stats.keys())

    word_scores = {}
    for word in all_words:
        scores = []
        for enc_stats in per_word_stats_all_encoders.values():
            if word in enc_stats:
                scores.append(enc_stats[word]["median_rank"])
        word_scores[word] = float(np.mean(scores))

    sorted_words = sorted(word_scores, key=lambda w: word_scores[w])
    return sorted_words[:top_k]


# =============================================================================
#  IMG → LIS MAPPING
# =============================================================================

def _load_img_lis_model(arch: str, ckpt_path: str, n_channels: int) -> nn.Module:
    # Handle LinearLag special case
    if arch == "LinearLag":
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"LinearLag checkpoint not found: {ckpt_path}")
        W = np.load(ckpt_path)
        return LinearLagModule(W).eval().to(DEVICE)
    
    # Handle neural architectures
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "benchmark" / "no_flash_removal"))
    try:
        from benchmark_loso import CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN
    except ImportError:
        raise ImportError("Could not import benchmark architectures. Make sure benchmark_loso.py is available.")
    
    arch_map = {
        "CNN1D":      lambda: CNN1D(n_channels, TARGET_HIDDEN),
        "ShallowMLP": lambda: ShallowMLP(n_channels, TARGET_HIDDEN),
        "UNet1D":     lambda: UNet1D(n_channels, TARGET_HIDDEN // 2),
        "RNN":        lambda: RNN(n_channels, TARGET_HIDDEN),
        "TCN":        lambda: TCN(n_channels, TARGET_HIDDEN // 2),
    }
    if arch not in arch_map:
        raise ValueError(f"Unknown arch {arch!r}; choose from {list(arch_map)} or LinearLag")
    model = arch_map[arch]()
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval()


def build_imagined_dataset(
    heldout_subject: str,
    img_lis_ckpt:    str,
    img_lis_arch:    str,
    n_channels:      int,
) -> MEGWordDataset:
    """Apply img→lis mapping and return dataset with mapped MEG windows."""
    img_lis = _load_img_lis_model(img_lis_arch, img_lis_ckpt, n_channels).to(DEVICE)

    img_ds = MEGWordDataset(
        subjects=[heldout_subject],
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="img",
        remove_flashes=REMOVE_FLASHES,
    )

    mapped_pairs = []
    with torch.no_grad():
        for i in range(len(img_ds)):
            window, word_str = img_ds.pairs[i]
            x     = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            x_hat = img_lis(x).squeeze(0).cpu().numpy()
            mapped_pairs.append((x_hat, word_str))

    img_ds.pairs = mapped_pairs
    print(f"  Applied {img_lis_arch} img→lis mapping: {len(mapped_pairs)} windows")
    return img_ds


# =============================================================================
#  COMPARISON PLOTS
# =============================================================================

def plot_rank_cdf(
    ranks_dict: Dict[str, np.ndarray],
    vocab_size: int,
    title: str,
    save_path: str,
) -> None:
    """CDF of rank (lower = better) for each encoder."""
    fig, ax = plt.subplots(figsize=(9, 5))
    max_rank_plot = min(vocab_size, 50)
    max_rank_plot  = 76
    x = np.arange(1, max_rank_plot + 1)

    for enc_name, ranks in ranks_dict.items():
        if ranks is None or len(ranks) == 0:
            continue
        cdf = np.array([(ranks <= r).mean() for r in x])
        ax.plot(x, cdf,
                label=ENCODER_LABELS.get(enc_name, enc_name),
                color=ENCODER_COLORS.get(enc_name, "#888"),
                linewidth=2.5)

    # Chance line
    chance = x / vocab_size
    ax.plot(x, chance, "k--", linewidth=1, alpha=0.5, label=f"Chance (V={vocab_size})")

    ax.set_xlabel("Rank k"); ax.set_ylabel("P(rank ≤ k)")
    ax.set_title(title)
    ax.set_xlim(1, max_rank_plot)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def plot_bar_metrics(
    metrics_dict: Dict[str, Dict],   # enc_name → metrics
    title: str,
    save_path: str,
) -> None:
    metric_keys = ["R@1", "R@5", "R@10", "MRR"]
    enc_names   = [e for e in ENCODER_NAMES if e in metrics_dict]
    n_enc       = len(enc_names)
    n_met       = len(metric_keys)

    fig, axes = plt.subplots(1, n_met, figsize=(4 * n_met, 4), sharey=False)
    for j, mkey in enumerate(metric_keys):
        ax  = axes[j]
        vals = [metrics_dict[e].get(mkey, 0) for e in enc_names]
        colors = [ENCODER_COLORS.get(e, "#888") for e in enc_names]
        bars = ax.bar(range(n_enc), vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
        ax.set_xticks(range(n_enc))
        ax.set_xticklabels(
            [ENCODER_LABELS.get(e, e).replace(" ", "\n") for e in enc_names],
            fontsize=8,
        )
        ax.set_title(mkey, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(vals) * 1.3 + 0.05))
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def plot_word_heatmap(
    per_word_all: Dict[str, Dict[str, Dict]],  # enc_name → word → stats
    words_to_show: List[str],
    title: str,
    save_path: str,
    metric: str = "median_rank",
) -> None:
    """Heatmap: rows=words, cols=encoders, cell=median_rank."""
    enc_names = [e for e in ENCODER_NAMES if e in per_word_all]
    n_words   = len(words_to_show)
    n_enc     = len(enc_names)

    data = np.full((n_words, n_enc), np.nan)
    for j, enc in enumerate(enc_names):
        stats = per_word_all.get(enc, {})
        for i, word in enumerate(words_to_show):
            if word in stats:
                data[i, j] = stats[word][metric]

    fig, ax = plt.subplots(figsize=(max(6, n_enc * 2), max(5, n_words * 0.45)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=1)
    ax.set_xticks(range(n_enc))
    ax.set_xticklabels([ENCODER_LABELS.get(e, e) for e in enc_names], fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(n_words))
    ax.set_yticklabels(words_to_show, fontsize=9)
    ax.set_title(title)

    for i in range(n_words):
        for j in range(n_enc):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.0f}", ha="center", va="center",
                        fontsize=8, color="black")

    plt.colorbar(im, ax=ax, label=metric.replace("_", " "))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


def plot_per_word_rank_box(
    ranks_dict: Dict[str, np.ndarray],
    word_labels_dict: Dict[str, List[str]],
    top_words: List[str],
    save_path: str,
    tag: str = "",
) -> None:
    """
    Side-by-side boxplots comparing rank distributions for:
      (a) all words  (b) top-K words only
    for each encoder.
    """
    enc_names = [e for e in ENCODER_NAMES if e in ranks_dict]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (subset_label, use_top) in enumerate([("All words", False), (f"Top {TOP_K_WORDS} words", True)]):
        ax = axes[ax_idx]
        data_to_plot = []
        tick_labels  = []
        for enc in enc_names:
            ranks = ranks_dict[enc]
            wlabs = word_labels_dict[enc]
            if ranks is None:
                continue
            if use_top:
                mask  = np.array([w in top_words for w in wlabs])
                ranks = ranks[mask]
            data_to_plot.append(ranks)
            tick_labels.append(ENCODER_LABELS.get(enc, enc))

        parts = ax.violinplot(data_to_plot, showmedians=True, showextrema=True)
        for i, (pc, enc) in enumerate(zip(parts["bodies"], enc_names)):
            pc.set_facecolor(ENCODER_COLORS.get(enc, "#888"))
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(tick_labels) + 1))
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Rank")
        ax.set_title(f"{tag} — {subset_label}")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# =============================================================================
#  SAVE / LOAD HELPERS
# =============================================================================

def save_results(
    encoder_name: str,
    phase: str,
    metrics: Dict,
    ranks: np.ndarray,
    per_word_stats: Dict,
    out_dir: str,
    word_labels: List[str] = None,
    top_words: List[str] = None,
) -> None:
    enc_dir = os.path.join(out_dir, "results", encoder_name)
    os.makedirs(enc_dir, exist_ok=True)
    
    # Save all-words data (original functionality)
    with open(os.path.join(enc_dir, f"{phase}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(enc_dir, f"ranks_{phase}.npy"), ranks)
    with open(os.path.join(enc_dir, f"per_word_stats_{phase}.json"), "w") as f:
        json.dump(per_word_stats, f, indent=2)
    
    # Save top-20 words data if available
    if word_labels is not None and top_words is not None:
        mask = np.array([w in top_words for w in word_labels])
        top_ranks = ranks[mask]
        top_labels = np.array(word_labels)[mask]
        
        # Save top-20 ranks and word labels
        np.save(os.path.join(enc_dir, f"ranks_{phase}_top20.npy"), top_ranks)
        np.save(os.path.join(enc_dir, f"word_labels_{phase}_top20.npy"), top_labels)
        
        # Save top-20 list for reference
        with open(os.path.join(enc_dir, f"top20_words_{phase}.json"), "w") as f:
            json.dump(top_words, f, indent=2)
    
    print(f"  [saved] results for {encoder_name}/{phase}")


def load_encoder_checkpoint(
    encoder_name: str,
    n_channels: int,
    raw_emb: torch.Tensor,
    out_dir: str,
    model_size: str = MODEL_SIZE,
) -> Tuple[nn.Module, TextEncoder]:
    model_dir = os.path.join(out_dir, "models", encoder_name)
    meg_enc   = make_meg_encoder(n_channels, model_size).to(DEVICE)
    txt_enc   = TextEncoder(raw_emb).to(DEVICE)
    meg_enc.load_state_dict(
        torch.load(os.path.join(model_dir, "meg_encoder.pt"), map_location="cpu"))
    txt_enc.load_state_dict(
        torch.load(os.path.join(model_dir, "text_encoder.pt"), map_location="cpu"))
    return meg_enc, txt_enc


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def extract_mapping_mode_from_path(ckpt_path: str) -> str:
    """Extract mapping mode (full/windowed) from checkpoint path."""
    if "LinearLag" in ckpt_path:
        return ""  # LinearLag doesn't have a mode
    if "windowed" in ckpt_path:
        return "windowed"
    return "full"


class LinearLagModule(nn.Module):
    """
    Wraps the ridge-regression LinearLag model as a nn.Module so it fits
    the same load/forward interface as the neural mapping models.

    Input : (B, C, T) tensor
    Output: (B, C, T) tensor  — predicted listened MEG
    """
    def __init__(self, W: np.ndarray):
        super().__init__()
        # Import LAG constants and functions from benchmark
        import sys
        from pathlib import Path
        benchmark_path = Path(__file__).parent.parent / "benchmark" / "no_flash_removal"
        sys.path.insert(0, str(benchmark_path))
        
        try:
            from benchmark_loso import build_lagged_features, ms_to_samples, LAG_BEFORE_MS, LAG_AFTER_MS
            self.lb = ms_to_samples(LAG_BEFORE_MS)
            self.la = ms_to_samples(LAG_AFTER_MS)
            self.build_lagged_features = build_lagged_features
            print(f"  LinearLag: lb={self.lb}, la={self.la}, weights shape={W.shape}")
        except ImportError as e:
            raise ImportError(f"Could not import LinearLag dependencies from benchmark: {e}")
        
        # Store weights as a buffer (float64 — ridge was solved in float64)
        self.register_buffer("W", torch.from_numpy(W.astype(np.float64)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_np = self.W.cpu().numpy()
        results = []
        for i in range(x.shape[0]):
            xi   = x[i].cpu().numpy()                             # (C, T)
            Xl   = self.build_lagged_features(xi, self.lb, self.la).astype(np.float64)
            pred = (Xl @ W_np).T.astype(np.float32)              # (C, T)
            results.append(torch.from_numpy(pred))
        return torch.stack(results).to(x.device)




# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_train_eval_lis(
    encoders:   List[str],
    model_size: str,
    remove_flashes: bool,
) -> None:
    """Train all requested encoders on listened MEG and evaluate."""
    print("\n" + "="*60)
    print("  Building listened MEG dataset...")
    print("="*60)
    full_ds = MEGWordDataset(
        subjects=SUBJECTS, poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, cond_suffix="lis",
        remove_flashes=remove_flashes,
    )
    n_val  = max(1, int(VAL_FRAC * len(full_ds)))
    n_tr   = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_ds.vocab = full_ds.vocab
    val_ds.vocab   = full_ds.vocab
    n_channels = full_ds.pairs[0][0].shape[0]
    print(f"  train={n_tr}  val={n_val}  channels={n_channels}")

    # Build all needed raw embeddings once (cached)
    emb_cache: dict = {}

    # Collect results for comparison
    val_metrics_all:    Dict[str, Dict]           = {}
    val_ranks_all:      Dict[str, np.ndarray]     = {}
    val_word_labs_all:  Dict[str, List[str]]      = {}
    per_word_stats_all: Dict[str, Dict[str, Dict]] = {}

    for enc_name in encoders:
        print(f"\n{'='*60}")
        print(f"  Encoder: {ENCODER_LABELS.get(enc_name, enc_name)}")
        print(f"{'='*60}")

        raw_emb = build_embeddings_for_encoder(
            enc_name, full_ds.words, full_ds.word_audio, emb_cache
        )

        meg_enc = make_meg_encoder(n_channels, model_size)
        txt_enc = TextEncoder(raw_emb)

        n_meg  = sum(p.numel() for p in meg_enc.parameters())
        n_proj = sum(p.numel() for p in txt_enc.proj.parameters())
        print(f"  MEG encoder: {n_meg:,} params | Text proj: {n_proj:,} params")

        model_dir = os.path.join(OUT_DIR, "models", enc_name)
        meg_enc, txt_enc, _ = train_one_encoder(
            enc_name, meg_enc, txt_enc,
            train_ds, val_ds,
            out_dir=model_dir,
        )

        print(f"\n  Evaluating on validation set...")
        val_metrics, val_ranks, val_wlabs = evaluate_ranking(
            meg_enc, txt_enc, val_ds,
            tag=f"val ({enc_name})",
            words_list=full_ds.words,
        )

        per_word = compute_per_word_stats(val_ranks, val_wlabs)

        val_metrics_all[enc_name]   = val_metrics
        val_ranks_all[enc_name]     = val_ranks
        val_word_labs_all[enc_name] = val_wlabs
        per_word_stats_all[enc_name] = per_word

    # Calculate top words after all encoders processed
    top_words = get_top_words(per_word_stats_all, top_k=TOP_K_WORDS)
    
    # Save results with top words info
    for enc_name in encoders:
        save_results(enc_name, "val", val_metrics_all[enc_name], val_ranks_all[enc_name], 
                    per_word_stats_all[enc_name], OUT_DIR, 
                    word_labels=val_word_labs_all[enc_name], top_words=top_words)

    # ---- Comparison plots ----
    print("\n  Generating comparison plots...")
    comp_dir  = os.path.join(OUT_DIR, "comparison")
    vocab_size = len(full_ds.vocab)
    top_ranks_all = {}
    for enc in encoders:
        ranks = val_ranks_all[enc]
        wlabs = val_word_labs_all[enc]
        mask  = np.array([w in top_words for w in wlabs])
        top_ranks_all[enc] = ranks[mask]

    plot_rank_cdf(
        val_ranks_all, vocab_size,
        title="Rank CDF — Validation (All Words)",
        save_path=os.path.join(comp_dir, "rank_cdf_val_all.png"),
    )
    plot_rank_cdf(
        top_ranks_all, vocab_size,
        title=f"Rank CDF — Validation (Top {TOP_K_WORDS} Words)",
        save_path=os.path.join(comp_dir, "rank_cdf_val_top.png"),
    )
    plot_bar_metrics(
        val_metrics_all,
        title="Metric Comparison — Validation (Listened MEG)",
        save_path=os.path.join(comp_dir, "bar_metrics_val.png"),
    )
    plot_word_heatmap(
        per_word_stats_all, top_words,
        title=f"Per-Word Median Rank — Top {TOP_K_WORDS} Words (Validation)",
        save_path=os.path.join(comp_dir, "word_heatmap_val.png"),
    )
    plot_per_word_rank_box(
        val_ranks_all, val_word_labs_all, top_words,
        save_path=os.path.join(comp_dir, "violin_val.png"),
        tag="Validation",
    )

    summary = {
        "phase":    "train_eval_lis",
        "encoders": encoders,
        "top_words": top_words,
        "val":       val_metrics_all,
    }
    with open(os.path.join(comp_dir, "summary_lis.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {os.path.join(comp_dir, 'summary_lis.json')}")


def run_eval_img(
    encoders:        List[str],
    heldout_subject: str,
    img_lis_ckpt:    str,
    img_lis_arch:    str,
    model_size:      str,
    remove_flashes:  bool,
) -> None:
    """Load trained encoders and evaluate on imagined MEG via img→lis mapping."""
    print("\n" + "="*60)
    print(f"  Eval Imagined MEG — {heldout_subject} — {img_lis_arch}")
    print("="*60)

    # Build vocab from other subjects' listened data
    other_subs = [s for s in SUBJECTS if s != heldout_subject]
    ref_ds = MEGWordDataset(
        subjects=other_subs, poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, cond_suffix="lis",
        remove_flashes=remove_flashes,
    )
    n_channels = ref_ds.pairs[0][0].shape[0]
    emb_cache: dict = {}

    # Build imagined MEG dataset (shared across encoders — mapping is fixed)
    print(f"\n  Building imagined MEG dataset ({img_lis_arch})...")
    img_ds = build_imagined_dataset(
        heldout_subject, img_lis_ckpt, img_lis_arch, n_channels
    )
    img_ds.vocab = ref_ds.vocab
    img_ds.words = ref_ds.words

    img_metrics_all:    Dict[str, Dict]           = {}
    img_ranks_all:      Dict[str, np.ndarray]     = {}
    img_word_labs_all:  Dict[str, List[str]]      = {}
    per_word_stats_all: Dict[str, Dict[str, Dict]] = {}

    for enc_name in encoders:
        print(f"\n  Loading [{enc_name}] checkpoint...")
        raw_emb = build_embeddings_for_encoder(
            enc_name, ref_ds.words, ref_ds.word_audio, emb_cache
        )
        meg_enc, txt_enc = load_encoder_checkpoint(
            enc_name, n_channels, raw_emb, OUT_DIR, model_size
        )

        img_metrics, img_ranks, img_wlabs = evaluate_ranking(
            meg_enc, txt_enc, img_ds,
            tag=f"eval_img/{heldout_subject} ({enc_name})",
            words_list=ref_ds.words,
        )

        per_word = compute_per_word_stats(img_ranks, img_wlabs)

        img_metrics_all[enc_name]    = img_metrics
        img_ranks_all[enc_name]      = img_ranks
        img_word_labs_all[enc_name]  = img_wlabs
        per_word_stats_all[enc_name] = per_word

    # Calculate top words after all encoders processed
    top_words = get_top_words(per_word_stats_all, top_k=TOP_K_WORDS)
    
    # Save results with top words info
    for enc_name in encoders:
        phase_key = f"eval_img_{heldout_subject}"
        save_results(enc_name, phase_key, img_metrics_all[enc_name], img_ranks_all[enc_name], 
                    per_word_stats_all[enc_name], OUT_DIR,
                    word_labels=img_word_labs_all[enc_name], top_words=top_words)

    # ---- Comparison plots (imagined) ----
    # Create architecture-specific comparison directory to avoid overwriting
    mapping_mode = extract_mapping_mode_from_path(img_lis_ckpt)
    if img_lis_arch == "LinearLag":
        arch_tag = "LinearLag"
    else:
        arch_tag = f"{img_lis_arch}_{mapping_mode}"
    comp_dir = os.path.join(OUT_DIR, "comparison", arch_tag)
    os.makedirs(comp_dir, exist_ok=True)
    
    vocab_size = len(ref_ds.vocab)

    top_ranks_all = {}
    for enc in encoders:
        ranks = img_ranks_all[enc]
        wlabs = img_word_labs_all[enc]
        mask  = np.array([w in top_words for w in wlabs])
        top_ranks_all[enc] = ranks[mask]

    tag = heldout_subject
    plot_rank_cdf(
        img_ranks_all, vocab_size,
        title=f"Rank CDF — Imagined MEG / {heldout_subject} / {arch_tag} (All Words)",
        save_path=os.path.join(comp_dir, f"rank_cdf_img_{tag}_all.png"),
    )
    plot_rank_cdf(
        top_ranks_all, vocab_size,
        title=f"Rank CDF — Imagined MEG / {heldout_subject} / {arch_tag} (Top {TOP_K_WORDS} Words)",
        save_path=os.path.join(comp_dir, f"rank_cdf_img_{tag}_top.png"),
    )
    plot_bar_metrics(
        img_metrics_all,
        title=f"Metric Comparison — Imagined MEG ({heldout_subject}) / {arch_tag}",
        save_path=os.path.join(comp_dir, f"bar_metrics_img_{tag}.png"),
    )
    plot_word_heatmap(
        per_word_stats_all, top_words,
        title=f"Per-Word Median Rank — Imagined MEG / {heldout_subject} / {arch_tag}",
        save_path=os.path.join(comp_dir, f"word_heatmap_img_{tag}.png"),
    )
    plot_per_word_rank_box(
        img_ranks_all, img_word_labs_all, top_words,
        save_path=os.path.join(comp_dir, f"violin_img_{tag}.png"),
        tag=f"Imagined ({heldout_subject}) / {arch_tag}",
    )

    summary = {
        "phase":           "eval_img",
        "heldout_subject": heldout_subject,
        "img_lis_arch":    img_lis_arch,
        "img_lis_ckpt":    img_lis_ckpt,
        "mapping_mode":    mapping_mode,
        "arch_tag":        arch_tag,
        "encoders":        encoders,
        "top_words":       top_words,
        "img":             img_metrics_all,
    }
    with open(os.path.join(comp_dir, f"summary_img_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {os.path.join(comp_dir, f'summary_img_{tag}.json')}")
    print(f"  Architecture-specific results in: {comp_dir}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-encoder MEG word decoding comparison",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=["train_eval_lis", "eval_img", "full"],
        default="train_eval_lis",
        help=(
            "train_eval_lis : train all encoders on listened MEG and compare\n"
            "eval_img       : eval imagined MEG using already-trained encoders\n"
            "full           : both phases"
        ),
    )
    parser.add_argument(
        "--encoders", nargs="+",
        choices=ENCODER_NAMES, default=ENCODER_NAMES,
        help="Which encoders to run (default: all four)",
    )
    parser.add_argument("--img_lis_ckpt",     default=None)
    parser.add_argument("--img_lis_arch",     default="CNN1D")
    parser.add_argument("--heldout_subject",  default=None)
    parser.add_argument("--model_size",       choices=["small", "full"], default=MODEL_SIZE)
    parser.add_argument("--remove_flashes",   action="store_true", default=REMOVE_FLASHES)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\nDevice       : {DEVICE}")
    print(f"Phase        : {args.phase}")
    print(f"Encoders     : {args.encoders}")
    print(f"Model size   : {args.model_size}")
    print(f"Remove flash : {args.remove_flashes}")
    print(f"Window       : [-{WIN_PRE_MS}ms, +{WIN_POST_MS}ms] = {WIN_SIZE} samples")
    if args.phase in ("eval_img", "full") and args.img_lis_arch:
        mapping_mode = extract_mapping_mode_from_path(args.img_lis_ckpt or "")
        if args.img_lis_arch == "LinearLag":
            print(f"Img→lis arch : LinearLag")
        else:
            print(f"Img→lis arch : {args.img_lis_arch}_{mapping_mode}")
    print(f"Out dir      : {OUT_DIR}\n")

    if args.phase in ("train_eval_lis", "full"):
        run_train_eval_lis(
            encoders=args.encoders,
            model_size=args.model_size,
            remove_flashes=args.remove_flashes,
        )

    if args.phase in ("eval_img", "full"):
        if args.heldout_subject is None:
            parser.error("--heldout_subject required for eval_img / full")
        if args.img_lis_ckpt is None:
            parser.error("--img_lis_ckpt required for eval_img / full")
        run_eval_img(
            encoders=args.encoders,
            heldout_subject=args.heldout_subject,
            img_lis_ckpt=args.img_lis_ckpt,
            img_lis_arch=args.img_lis_arch,
            model_size=args.model_size,
            remove_flashes=args.remove_flashes,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()