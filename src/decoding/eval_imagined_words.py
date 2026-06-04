"""
eval_imagined_words.py
======================
End-to-end evaluation: imagined MEG → img→lis mapping → word decoding.

Pipeline
--------
  imagined MEG (C, T)
      → mapping model (img→lis)
      → predicted listened MEG (C, T)
      → word windowing at onset timestamps  →  (N_words, C, WIN_SIZE)
      → contrastive MEG encoder             →  (N_words, 128)
      → cosine similarity vs text embeddings →  R@1 / R@5 / R@10 / MRR

Conditions (--condition)
------------------------
  A          Full LOSO       mapping=LOSO  decoder=LOSO        neither model saw heldout
  A_ft       LOSO + FT       mapping=LOSO  decoder=LOSO-FT     FT on heldout real listened
  A_ft_pred  LOSO + FT-pred  mapping=LOSO  decoder=LOSO-FT     FT on predicted-listened windows
                              (run finetune_decoder_on_predicted.py first)
  B          Decoder leakage  mapping=LOSO  decoder=global      global decoder saw heldout's listened
  C          Seen mapping     mapping=ensemble of all 12 seen folds  decoder=global
             For test subject S, averages predicted-listened signals from all
             mapping models trained WITH S (i.e. heldout=T for every T≠S).
             Uses the global decoder (same as B) to isolate the mapping gain.

Usage
-----
  python eval_imagined_words.py --condition A
  python eval_imagined_words.py --condition A --heldout_subject sub-01
  python eval_imagined_words.py --condition A_ft --mapping_arch RNN
  python eval_imagined_words.py --condition B
  python eval_imagined_words.py --condition C
  python eval_imagined_words.py --condition A --mapping_arch CNN1D --mapping_mode windowed
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample

import mne
mne.set_log_level("ERROR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
#  sys.path — make both local packages importable
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).parent.resolve()
_BENCH   = _HERE.parent / "benchmark" / "no_flash_removal"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH))

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE, SEED,
    BASE_PATH, SFREQ_DS, DS_FACTOR, N_SESSIONS, EPOCH_TMIN_S,
    WIN_PRE, WIN_POST, WIN_SIZE,
    MEGWordDataset, TextEncoder, make_meg_encoder,
    build_text_embeddings, onset_to_window_raw,
    TEXT_ENCODER, MODEL_SIZE,
)
from benchmark_loso import (
    CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN,
    build_lagged_features, ms_to_samples, LAG_BEFORE_MS, LAG_AFTER_MS,
)

# ---------------------------------------------------------------------------
#  PATHS
# ---------------------------------------------------------------------------
_LOSO_OUT   = str(_BENCH / "loso_out")
MAPPING_DIR  = os.path.join(_LOSO_OUT, "models")

LOSO_DEC_DIR      = str(_HERE / "contrastive_loso_out" / "models")
FINETUNE_DIR      = str(_HERE / "contrastive_loso_out" / "finetune" / "models")
FINETUNE_PRED_DIR = str(_HERE / "contrastive_loso_out" / "finetune_predicted" / "models")
GLOBAL_DEC_DIR    = str(_HERE / "contrastive_out")
OUT_DIR         = str(_HERE / "eval_imagined_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
#  DEFAULTS
# ---------------------------------------------------------------------------
DEFAULT_MAPPING_ARCH = "RNN"
DEFAULT_MAPPING_MODE = "full"

_ARCH_FACTORY = {
    "CNN1D":      lambda C: CNN1D(C,      hidden=TARGET_HIDDEN),
    "ShallowMLP": lambda C: ShallowMLP(C, hidden=TARGET_HIDDEN),
    "UNet1D":     lambda C: UNet1D(C,     hidden=TARGET_HIDDEN // 2),
    "RNN":        lambda C: RNN(C,        hidden=TARGET_HIDDEN),
    "TCN":        lambda C: TCN(C,        hidden=TARGET_HIDDEN // 2),
}

_LINEAR_LB = ms_to_samples(LAG_BEFORE_MS)
_LINEAR_LA = ms_to_samples(LAG_AFTER_MS)


class LinearLagModule(nn.Module):
    """
    Wraps the ridge-regression LinearLag model as a nn.Module so it fits
    the same load/forward interface as the neural mapping models.

    Input : (B, C, T) tensor
    Output: (B, C, T) tensor  — predicted listened MEG
    """
    def __init__(self, W: np.ndarray, lb: int = _LINEAR_LB, la: int = _LINEAR_LA):
        super().__init__()
        self.lb = lb
        self.la = la
        # Store weights as a buffer (float64 — ridge was solved in float64)
        self.register_buffer("W", torch.from_numpy(W.astype(np.float64)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_np = self.W.cpu().numpy()
        results = []
        for i in range(x.shape[0]):
            xi   = x[i].cpu().numpy()                             # (C, T)
            Xl   = build_lagged_features(xi, self.lb, self.la).astype(np.float64)
            pred = (Xl @ W_np).T.astype(np.float32)              # (C, T)
            results.append(torch.from_numpy(pred))
        return torch.stack(results).to(x.device)


# ---------------------------------------------------------------------------
#  MAPPING MODEL
# ---------------------------------------------------------------------------

def load_mapping_model(
    heldout_subj: str,
    n_channels:   int,
    arch:         str,
    mode:         str,
) -> nn.Module:
    fold_dir = os.path.join(MAPPING_DIR, f"heldout_{heldout_subj}")
    if arch == "LinearLag":
        npy_path = os.path.join(fold_dir, "LinearLag_W.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Mapping checkpoint not found: {npy_path}")
        W = np.load(npy_path)
        return LinearLagModule(W).eval().to(DEVICE)
    key  = f"{arch}_{mode}"
    ckpt = os.path.join(fold_dir, f"{key}.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Mapping checkpoint not found: {ckpt}")
    model = _ARCH_FACTORY[arch](n_channels)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval().to(DEVICE)


# ---------------------------------------------------------------------------
#  CONTRASTIVE DECODER
# ---------------------------------------------------------------------------

def load_decoder(
    condition:      str,
    heldout_subj:   str,
    n_channels:     int,
    combined_words: List[str],
    model_size:     str,
    text_method:    str,
) -> Tuple[nn.Module, nn.Module]:
    """
    Load (meg_encoder, text_encoder) for the requested condition.
    Returns both in eval mode on DEVICE.
    """
    raw_emb = build_text_embeddings(combined_words, method=text_method)

    # Which saved checkpoint to load for the meg_encoder weights
    if condition in ("A", "A_ft", "A_ft_pred"):
        dec_dir = os.path.join(LOSO_DEC_DIR, f"heldout_{heldout_subj}")
    else:   # B / C — global decoder
        dec_dir = GLOBAL_DEC_DIR

    ckpt_meg = os.path.join(dec_dir, "meg_encoder.pt")
    ckpt_txt = os.path.join(dec_dir, "text_encoder.pt")

    if not os.path.exists(ckpt_meg):
        raise FileNotFoundError(
            f"Decoder checkpoint not found: {ckpt_meg}\n"
            f"Run contrastive_loso.py (conditions A/A_ft/A_ft_pred) or "
            f"contrastive_word_meg.py --phase train (condition B) first."
        )

    # Infer model size from the checkpoint so --model_size never has to match manually.
    # "full" has 256 units in temporal.2; "small" has 128.
    _sd = torch.load(ckpt_meg, map_location="cpu")
    model_size = "full" if _sd.get("temporal.2.block.1.bias", torch.zeros(1)).shape[0] == 256 else "small"

    meg_enc = make_meg_encoder(n_channels, model_size).to(DEVICE)
    txt_enc = TextEncoder(raw_emb).to(DEVICE)

    meg_enc.load_state_dict(torch.load(ckpt_meg, map_location="cpu"))
    txt_enc.load_state_dict(torch.load(ckpt_txt, map_location="cpu"))

    # For A_ft: overwrite meg_encoder weights with the fine-tuned version (real listened)
    if condition == "A_ft":
        ft_ckpt = os.path.join(FINETUNE_DIR, f"heldout_{heldout_subj}", "meg_encoder_ft.pt")
        if not os.path.exists(ft_ckpt):
            raise FileNotFoundError(
                f"Fine-tuned checkpoint not found: {ft_ckpt}\n"
                f"Run contrastive_loso_finetune.py first."
            )
        meg_enc.load_state_dict(torch.load(ft_ckpt, map_location="cpu"))

    # For A_ft_pred: overwrite with the decoder fine-tuned on predicted-listened MEG
    if condition == "A_ft_pred":
        ft_ckpt = os.path.join(
            FINETUNE_PRED_DIR, f"heldout_{heldout_subj}", "meg_encoder_ft_pred.pt"
        )
        if not os.path.exists(ft_ckpt):
            raise FileNotFoundError(
                f"Fine-tuned (predicted) checkpoint not found: {ft_ckpt}\n"
                f"Run finetune_decoder_on_predicted.py first."
            )
        meg_enc.load_state_dict(torch.load(ft_ckpt, map_location="cpu"))

    meg_enc.eval()
    txt_enc.eval()
    return meg_enc, txt_enc


# ---------------------------------------------------------------------------
#  IMAGINED MEG LOADING + MAPPING
# ---------------------------------------------------------------------------

def load_and_map_session(
    subject:       str,
    poem_key:      str,
    session:       int,
    mapping_model: nn.Module,
) -> Optional[np.ndarray]:
    """
    Load one imagined MEG session, z-score per channel, apply img→lis mapping.

    Returns predicted listened (C, T) float32, or None on failure.

    The z-scoring mirrors contrastive_word_meg.load_meg_trial so the mapping
    model receives the same normalised input it was trained on.  The output is
    left as-is: the mapping model was trained to predict z-scored listened, so
    its output is already in the domain the contrastive decoder expects.
    """
    cond  = f"{poem_key}img"
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)

    try:
        epochs = mne.read_epochs(fpath, preload=True)
    except Exception as e:
        print(f"    WARNING: {subject}/{cond}/ses-{session}: {e}")
        return None

    raw  = epochs.get_data().mean(axis=0)                  # (C, T_raw)
    new_T = raw.shape[1] // DS_FACTOR
    data  = resample(raw, new_T, axis=1).astype(np.float32) # (C, T_ds)

    mu   = data.mean(axis=1, keepdims=True)
    sd   = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    data = (data - mu) / sd

    with torch.no_grad():
        x      = torch.from_numpy(data).unsqueeze(0).to(DEVICE)  # (1, C, T)
        x_pred = mapping_model(x).squeeze(0).cpu().numpy()       # (C, T)

    return x_pred.astype(np.float32)


# ---------------------------------------------------------------------------
#  SEEN-MAPPING ENSEMBLE  (condition C)
# ---------------------------------------------------------------------------

def score_avg_ensemble_session(
    subject:    str,
    poem_key:   str,
    session:    int,
    seen_folds: List[str],
    n_channels: int,
    arch:       str,
    mode:       str,
    meg_enc:    nn.Module,
    all_text:   torch.Tensor,   # (V, D) on DEVICE — pre-computed once
    word_onsets: List[dict],
    vocab:      Dict[str, int],
) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    """
    Score-averaging ensemble for condition C.

    For one (subject, poem, session):
      1. Load imagined MEG once and z-score.
      2. For each of the 12 seen mapping models (loaded then freed one at a time):
           predict listened → extract word windows → encode → cosine sim scores (N, V)
      3. Average the similarity score matrices across all models.
      4. Return (mean_sim, word_strings) for ranking downstream.

    Averaging in score space is more principled than averaging in signal space:
    each model votes on which text embedding each MEG window resembles, and we
    tally those votes before making the final ranking decision.

    Returns (mean_sim (N_valid, V) float32, word_strings) or (None, None).
    """
    cond  = f"{poem_key}img"
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)

    try:
        epochs = mne.read_epochs(fpath, preload=True)
    except Exception as e:
        print(f"    WARNING: {subject}/{cond}/ses-{session}: {e}")
        return None, None

    raw   = epochs.get_data().mean(axis=0)
    new_T = raw.shape[1] // DS_FACTOR
    data  = resample(raw, new_T, axis=1).astype(np.float32)
    mu    = data.mean(axis=1, keepdims=True)
    sd    = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    data  = (data - mu) / sd

    x_img = torch.from_numpy(data).unsqueeze(0).to(DEVICE)  # (1, C, T) — reused

    acc_sim:   Optional[np.ndarray] = None
    ref_words: Optional[List[str]]  = None
    n_counted = 0

    for fold_T in seen_folds:
        try:
            model = load_mapping_model(fold_T, n_channels, arch, mode)
        except FileNotFoundError:
            continue

        with torch.no_grad():
            x_pred = model(x_img).squeeze(0).cpu().numpy()   # (C, T)

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        wins, wds = extract_word_windows(x_pred, word_onsets)
        valid = [(w, ws) for w, ws in zip(wins, wds) if ws in vocab]
        if not valid:
            continue
        wins_v, wds_v = zip(*valid)
        wds_v = list(wds_v)

        # First model sets the reference word order for this session
        if ref_words is None:
            ref_words = wds_v
        elif wds_v != ref_words:
            # Should not happen (same onset times → same windows), but guard
            print(f"    WARNING: fold {fold_T} word alignment mismatch — skipped")
            continue

        x_t = torch.from_numpy(np.stack(wins_v)).to(DEVICE)
        with torch.no_grad():
            z_chunks = []
            for i in range(0, len(x_t), 256):
                z_chunks.append(meg_enc(x_t[i:i + 256]))
            z   = torch.cat(z_chunks)              # (N, D) on DEVICE
            sim = (z @ all_text.T).cpu().numpy()   # (N, V)

        acc_sim = sim if acc_sim is None else acc_sim + sim
        n_counted += 1

    if acc_sim is None or n_counted == 0:
        return None, None

    return (acc_sim / n_counted).astype(np.float32), ref_words


# ---------------------------------------------------------------------------
#  WORD WINDOW EXTRACTION
# ---------------------------------------------------------------------------

def extract_word_windows(
    predicted:   np.ndarray,      # (C, T)
    word_onsets: List[dict],
    min_word_len: int = 1,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Cut [-WIN_PRE, +WIN_POST] windows from the predicted listened signal.
    Returns parallel lists of windows (each C × WIN_SIZE) and word strings.
    """
    n_t = predicted.shape[-1]
    windows, words = [], []

    for w in word_onsets:
        word = w["word"].strip().lower()
        if len(word) < min_word_len:
            continue
        idx = onset_to_window_raw(w["start"], n_t)
        if idx is None:
            continue
        start, end = idx
        win = predicted[:, start:end]
        if win.shape[-1] != WIN_SIZE:
            continue
        windows.append(win.copy())
        words.append(word)

    return windows, words


# ---------------------------------------------------------------------------
#  RANKING
# ---------------------------------------------------------------------------

@torch.no_grad()
def ranking_metrics(
    meg_enc:      nn.Module,
    txt_enc:      nn.Module,
    windows:      List[np.ndarray],
    word_strings: List[str],
    vocab:        Dict[str, int],
) -> Optional[Dict]:
    """
    Encode word windows and rank each against the full text embedding bank.
    Windows with words outside the vocab are silently skipped.
    Returns None if no valid windows remain.
    """
    if not windows:
        return None

    valid_pairs = [(w, ws) for w, ws in zip(windows, word_strings) if ws in vocab]
    if not valid_pairs:
        return None
    win_arrays, words_filt = zip(*valid_pairs)

    all_text = txt_enc.get_all().to(DEVICE)   # (V, D)
    V = all_text.shape[0]

    x = torch.from_numpy(np.stack(win_arrays)).to(DEVICE)  # (N, C, WIN_SIZE)

    # Chunk to avoid OOM on large N
    z_chunks = []
    for i in range(0, len(x), 256):
        z_chunks.append(meg_enc(x[i:i + 256]))
    z_meg = torch.cat(z_chunks, dim=0)   # (N, D)

    sim   = z_meg @ all_text.T           # (N, V)
    ranks = []
    for i, word in enumerate(words_filt):
        true_idx = vocab[word]
        s        = sim[i]
        ranks.append(int((s > s[true_idx]).sum().item()) + 1)

    ranks = np.array(ranks, dtype=np.int32)
    return {
        "n_samples":   int(len(ranks)),
        "vocab_size":  int(V),
        "R@1":         float((ranks <= 1).mean()),
        "R@5":         float((ranks <= 5).mean()),
        "R@10":        float((ranks <= 10).mean()),
        "MRR":         float((1.0 / ranks).mean()),
        "median_rank": int(np.median(ranks)),
        "chance_R@1":  float(1.0 / V),
    }


# ---------------------------------------------------------------------------
#  VOCAB HELPERS
# ---------------------------------------------------------------------------

def build_combined_vocab(
    train_subjects: List[str],
    heldout_subj:   str,
) -> Tuple[Dict[str, int], List[str]]:
    """
    Reconstruct the same combined vocab used in contrastive_loso.py so that
    text embedding indices are consistent with the saved decoder checkpoints.
    """
    train_ds = MEGWordDataset(
        subjects=train_subjects, poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, cond_suffix="lis", remove_flashes=False,
    )
    val_ds = MEGWordDataset(
        subjects=[heldout_subj], poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, cond_suffix="lis", remove_flashes=False,
    )
    vocab = dict(train_ds.vocab)
    for w in val_ds.vocab:
        if w not in vocab:
            vocab[w] = len(vocab)
    words = sorted(vocab, key=vocab.get)
    return vocab, words


def _probe_n_channels(subject: str) -> int:
    """Peek at one imagined session to determine channel count."""
    for poem_key in POEM_KEYS:
        for session in range(N_SESSIONS):
            cond  = f"{poem_key}img"
            fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
            fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
            try:
                return mne.read_epochs(fpath, preload=False).get_data().shape[1]
            except Exception:
                continue
    raise RuntimeError(f"Could not read any imagined MEG file for {subject}")


# ---------------------------------------------------------------------------
#  PER-FOLD EVALUATION
# ---------------------------------------------------------------------------

def run_fold(
    heldout_subj: str,
    condition:    str,
    mapping_arch: str,
    mapping_mode: str,
    model_size:   str,
    text_method:  str,
) -> Dict:
    print(f"\n{'='*62}")
    print(f"  Heldout : {heldout_subj}  condition={condition!r}  "
          f"mapping={mapping_arch}_{mapping_mode}")
    print(f"{'='*62}")

    train_subjects = [s for s in SUBJECTS if s != heldout_subj]
    vocab, words   = build_combined_vocab(train_subjects, heldout_subj)
    print(f"  Combined vocab: {len(vocab)} words")

    n_channels = _probe_n_channels(heldout_subj)

    # Condition C uses the global decoder (same as B) and score-averaging over
    # all 12 seen mapping models; the decoder_condition alias is kept generic.
    decoder_condition = "B" if condition == "C" else condition
    meg_enc, txt_enc  = load_decoder(
        decoder_condition, heldout_subj, n_channels, words, model_size, text_method,
    )

    seen_folds = [s for s in SUBJECTS if s != heldout_subj]

    # ------------------------------------------------------------------
    #  Condition C — score-averaging ensemble path
    # ------------------------------------------------------------------
    if condition == "C":
        print(f"  Score-avg ensemble: {len(seen_folds)} seen folds "
              f"({mapping_arch}_{mapping_mode})")

        with torch.no_grad():
            all_text = txt_enc.get_all().to(DEVICE)   # (V, D) — computed once

        all_mean_sims: List[np.ndarray] = []
        all_word_strs: List[str]        = []

        for poem_key in POEM_KEYS:
            onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
            if not os.path.exists(onset_file):
                print(f"  WARNING: {onset_file} not found — skipping {poem_key}")
                continue
            with open(onset_file) as f:
                word_onsets = json.load(f)

            for session in range(N_SESSIONS):
                mean_sim, ref_words = score_avg_ensemble_session(
                    heldout_subj, poem_key, session,
                    seen_folds, n_channels, mapping_arch, mapping_mode,
                    meg_enc, all_text, word_onsets, vocab,
                )
                if mean_sim is None:
                    continue
                all_mean_sims.append(mean_sim)
                all_word_strs.extend(ref_words)

        print(f"  Word windows: {len(all_word_strs)}")
        if not all_mean_sims:
            print("  WARNING: no valid windows")
            return {}

        sim_all = np.concatenate(all_mean_sims, axis=0)   # (N_total, V)
        V = sim_all.shape[1]
        ranks = []
        for i, word in enumerate(all_word_strs):
            true_idx = vocab[word]
            s        = sim_all[i]
            ranks.append(int((s > s[true_idx]).sum()) + 1)
        ranks = np.array(ranks, dtype=np.int32)

        metrics = {
            "n_samples":   int(len(ranks)),
            "vocab_size":  int(V),
            "R@1":         float((ranks <= 1).mean()),
            "R@5":         float((ranks <= 5).mean()),
            "R@10":        float((ranks <= 10).mean()),
            "MRR":         float((1.0 / ranks).mean()),
            "median_rank": int(np.median(ranks)),
            "chance_R@1":  float(1.0 / V),
        }

    # ------------------------------------------------------------------
    #  Conditions A / A_ft / B — single mapping model path
    # ------------------------------------------------------------------
    else:
        mapping_model = load_mapping_model(
            heldout_subj, n_channels, mapping_arch, mapping_mode,
        )
        all_windows:   List[np.ndarray] = []
        all_word_strs: List[str]        = []

        for poem_key in POEM_KEYS:
            onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
            if not os.path.exists(onset_file):
                print(f"  WARNING: {onset_file} not found — skipping {poem_key}")
                continue
            with open(onset_file) as f:
                word_onsets = json.load(f)

            for session in range(N_SESSIONS):
                predicted = load_and_map_session(
                    heldout_subj, poem_key, session, mapping_model,
                )
                if predicted is None:
                    continue
                wins, wds = extract_word_windows(predicted, word_onsets)
                all_windows.extend(wins)
                all_word_strs.extend(wds)

        print(f"  Word windows extracted: {len(all_windows)}")
        metrics = ranking_metrics(meg_enc, txt_enc, all_windows, all_word_strs, vocab)
        if metrics is None:
            print("  WARNING: no valid windows — check onset files and vocab overlap")
            return {}

    metrics.update({
        "heldout_subject": heldout_subj,
        "condition":       condition,
        "mapping":         f"{mapping_arch}_{mapping_mode}",
    })
    print(
        f"  R@1={metrics['R@1']:.3f}  R@5={metrics['R@5']:.3f}  "
        f"R@10={metrics['R@10']:.3f}  MRR={metrics['MRR']:.3f}  "
        f"median_rank={metrics['median_rank']}/{metrics['vocab_size']}  "
        f"chance={metrics['chance_R@1']:.3f}"
    )
    return metrics


# ---------------------------------------------------------------------------
#  SUMMARY PLOT
# ---------------------------------------------------------------------------

def plot_summary(
    all_results:  Dict[str, Dict],
    condition:    str,
    mapping_tag:  str,
) -> None:
    valid_subjs = [s for s in SUBJECTS if all_results.get(s)]
    if not valid_subjs:
        return

    metrics_to_plot = ["R@1", "R@5", "R@10", "MRR"]
    chance = all_results[valid_subjs[0]].get("chance_R@1")

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric in zip(axes, metrics_to_plot):
        vals   = [all_results[s][metric] for s in valid_subjs]
        mean_v = float(np.mean(vals))
        colors = ["#2ECC71" if v >= (chance or 0) else "#E74C3C" for v in vals]

        ax.bar(range(len(valid_subjs)), vals, color=colors, alpha=0.85)
        ax.axhline(mean_v, color="navy", lw=1.5, linestyle="--",
                   label=f"mean={mean_v:.3f}")
        if metric == "R@1" and chance:
            ax.axhline(chance, color="grey", lw=1.0, linestyle=":",
                       label=f"chance={chance:.3f}")
        ax.set_xticks(range(len(valid_subjs)))
        ax.set_xticklabels(
            [s.replace("sub-", "") for s in valid_subjs],
            rotation=45, ha="right", fontsize=8,
        )
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"Imagined MEG → word decoding   [condition={condition!r}  mapping={mapping_tag}]\n"
        f"imagined → {mapping_tag} → predicted listened → contrastive decoder → ranking",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"summary_{condition}_{mapping_tag}.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate imagined MEG word decoding via img→lis mapping.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--condition", choices=["A", "A_ft", "A_ft_pred", "B", "C"], default="A",
        help=(
            "A         : LOSO mapping + LOSO decoder        (both unseen)\n"
            "A_ft      : LOSO mapping + LOSO decoder FT on heldout real listened\n"
            "A_ft_pred : LOSO mapping + LOSO decoder FT on predicted-listened windows\n"
            "B         : LOSO mapping + global decoder       (global saw heldout listened)\n"
            "C         : Seen mapping (ensemble of 12) + global decoder"
        ),
    )
    parser.add_argument("--heldout_subject", default=None,
                        help="Single fold (e.g. sub-01). Omit for all subjects.")
    parser.add_argument("--mapping_arch",
                        choices=["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN", "LinearLag"],
                        default=DEFAULT_MAPPING_ARCH,
                        help="img→lis model architecture (default: RNN)")
    parser.add_argument("--mapping_mode", choices=["full", "windowed"],
                        default=DEFAULT_MAPPING_MODE)
    parser.add_argument("--model_size", choices=["small", "full"], default=MODEL_SIZE)
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds       = [args.heldout_subject] if args.heldout_subject else SUBJECTS
    mapping_tag = f"{args.mapping_arch}_{args.mapping_mode}"

    print(f"Device       : {DEVICE}")
    print(f"Condition    : {args.condition}")
    print(f"Mapping      : {mapping_tag}")
    print(f"Model size   : {args.model_size}")
    print(f"Text encoder : {args.text_encoder}")
    print(f"Folds        : {folds}")
    print(f"Out dir      : {OUT_DIR}\n")

    all_results: Dict[str, Dict] = {}

    for subj in folds:
        metrics = run_fold(
            heldout_subj=subj,
            condition=args.condition,
            mapping_arch=args.mapping_arch,
            mapping_mode=args.mapping_mode,
            model_size=args.model_size,
            text_method=args.text_encoder,
        )
        all_results[subj] = metrics

        if metrics:
            fname    = f"{args.condition}_{mapping_tag}_{subj}.json"
            out_path = os.path.join(OUT_DIR, fname)
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

    # Aggregate over all completed folds
    valid = {s: m for s, m in all_results.items() if m}
    if not valid:
        print("\nNo results to aggregate.")
        return

    print(f"\n{'='*62}")
    print(f"  SUMMARY   condition={args.condition!r}   mapping={mapping_tag}")
    print(f"{'='*62}")
    print(f"  {'subject':10s}  {'R@1':>5}  {'R@5':>5}  {'R@10':>5}  "
          f"{'MRR':>5}  median/vocab")
    for subj in SUBJECTS:
        if subj not in valid:
            continue
        m = valid[subj]
        print(f"  {subj:10s}  {m['R@1']:.3f}  {m['R@5']:.3f}  "
              f"{m['R@10']:.3f}  {m['MRR']:.3f}  "
              f"{m['median_rank']}/{m['vocab_size']}")

    for metric in ["R@1", "R@5", "R@10", "MRR"]:
        vals = [valid[s][metric] for s in valid]
        print(f"  {'MEAN':10s}  {metric}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    if len(valid) > 1:
        agg = {
            metric: {
                "mean": float(np.mean([valid[s][metric] for s in valid])),
                "std":  float(np.std( [valid[s][metric] for s in valid])),
                "per_subject": {s: valid[s][metric] for s in valid},
            }
            for metric in ["R@1", "R@5", "R@10", "MRR", "median_rank"]
        }
        agg_path = os.path.join(OUT_DIR, f"summary_{args.condition}_{mapping_tag}.json")
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n[saved] {agg_path}")

        plot_summary(all_results, args.condition, mapping_tag)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
