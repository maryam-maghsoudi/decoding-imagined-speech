"""
finetune_decoder_on_predicted.py
=================================
Fine-tune the LOSO contrastive decoder on PREDICTED-listened MEG.

Motivation
----------
Conditions A and A_ft both use a decoder trained on REAL listened MEG.
When applied to PREDICTED listened (img→lis mapping model output) there is a
domain gap: the mapping model output has lower SNR and different spectral
characteristics.  This script closes that gap by adapting the decoder to the
mapping model's output distribution.

Pipeline (one LOSO fold, heldout = S_test)
-------------------------------------------
  1. Load the LOSO decoder for S_test
     (contrastive_loso_out/models/heldout_{S_test}/).
  2. Collect training windows:
     For each training subject S_train  (SUBJECTS − {S_test}):
       a. Load S_train's imagined MEG.
       b. Apply mapping model  heldout_{S_test}
          (trained on all subjects EXCEPT S_test, so S_train was seen).
       c. Extract word windows at onset timestamps.
  3. Fine-tune the MEG encoder on these synthetic windows via NT-Xent loss.
  4. Evaluate on S_test's imagined MEG through the same mapping model.
  5. Save checkpoint →
       contrastive_loso_out/finetune_predicted/models/heldout_{S_test}/
         meg_encoder_ft_pred.pt

This checkpoint can be used as condition "A_ft_pred" in eval_imagined_words.py.

Usage
-----
  cd /fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning
  python finetune_decoder_on_predicted.py --heldout_subject sub-01
  python finetune_decoder_on_predicted.py                    # all folds
  python finetune_decoder_on_predicted.py --layers spatial+proj
  python finetune_decoder_on_predicted.py --mapping_arch CNN1D
"""

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
mne.set_log_level("ERROR")

from scipy.signal import resample

# ---------------------------------------------------------------------------
#  sys.path
# ---------------------------------------------------------------------------
_HERE  = Path(__file__).parent.resolve()
_BENCH = _HERE.parent / "benchmark" / "no_flash_removal"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH))

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE, SEED,
    BASE_PATH, SFREQ_DS, DS_FACTOR, N_SESSIONS,
    WIN_SIZE, MEGWordDataset, TextEncoder, make_meg_encoder,
    build_text_embeddings, onset_to_window_raw,
    nt_xent_loss, evaluate_ranking,
    TEXT_ENCODER, MODEL_SIZE,
)
from benchmark_loso import CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN
from eval_imagined_words import (
    MAPPING_DIR, LOSO_DEC_DIR, GLOBAL_DEC_DIR,
    _ARCH_FACTORY, DEFAULT_MAPPING_ARCH, DEFAULT_MAPPING_MODE,
    build_combined_vocab, _probe_n_channels, extract_word_windows,
    load_and_map_session, ranking_metrics,
)

# ---------------------------------------------------------------------------
#  PATHS
# ---------------------------------------------------------------------------
FTPRED_OUT = str(_HERE / "contrastive_loso_out" / "finetune_predicted")
os.makedirs(FTPRED_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
#  HYPER-PARAMETERS
# ---------------------------------------------------------------------------
FT_LR       = 3e-5
FT_EPOCHS   = 40
FT_PATIENCE = 8
FT_BATCH    = 32
FT_VAL_FRAC = 0.15
FT_NOISE    = 0.02   # Gaussian noise std added during training


# ---------------------------------------------------------------------------
#  PREDICTED-WINDOW DATASET
# ---------------------------------------------------------------------------

class PredictedWindowDataset(Dataset):
    """
    Holds pre-computed (C, WIN_SIZE) windows from predicted-listened MEG
    together with their vocab indices.
    """

    def __init__(
        self,
        windows:    List[np.ndarray],   # each (C, WIN_SIZE)
        word_idxs:  List[int],
    ):
        assert len(windows) == len(word_idxs)
        self.windows   = [w.astype(np.float32) for w in windows]
        self.word_idxs = word_idxs

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.windows[idx]),
            torch.tensor(self.word_idxs[idx], dtype=torch.long),
        )


# ---------------------------------------------------------------------------
#  DATA COLLECTION
# ---------------------------------------------------------------------------

def collect_predicted_windows(
    train_subjects: List[str],
    heldout_subj:   str,
    vocab:          Dict[str, int],
    arch:           str,
    mode:           str,
    n_channels:     int,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    For each training subject load their imagined MEG, map it with the
    mapping model  heldout_{heldout_subj}  (trained on all except heldout,
    so all training subjects were seen), and extract word windows.

    Returns parallel lists (windows, word_idxs).
    """
    print(f"\n  Collecting predicted-listened windows from {len(train_subjects)} "
          f"training subjects using mapping model heldout_{heldout_subj} …")

    ckpt_map = os.path.join(
        MAPPING_DIR, f"heldout_{heldout_subj}", f"{arch}_{mode}.pt"
    )
    if not os.path.exists(ckpt_map):
        raise FileNotFoundError(f"Mapping checkpoint not found: {ckpt_map}")

    mapping_model = _ARCH_FACTORY[arch](n_channels)
    mapping_model.load_state_dict(torch.load(ckpt_map, map_location="cpu"))
    mapping_model = mapping_model.eval().to(DEVICE)

    all_windows: List[np.ndarray] = []
    all_idxs:    List[int]        = []

    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
            print(f"    WARNING: {onset_file} not found — skipping {poem_key}")
            continue
        with open(onset_file) as f:
            word_onsets = json.load(f)

        for train_subj in train_subjects:
            for session in range(N_SESSIONS):
                predicted = load_and_map_session(
                    train_subj, poem_key, session, mapping_model,
                )
                if predicted is None:
                    continue

                wins, wds = extract_word_windows(predicted, word_onsets)
                for win, word in zip(wins, wds):
                    if word in vocab:
                        all_windows.append(win)
                        all_idxs.append(vocab[word])

    del mapping_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    print(f"  Collected {len(all_windows)} windows "
          f"({len(set(all_idxs))} unique words)")
    return all_windows, all_idxs


# ---------------------------------------------------------------------------
#  LAYER SELECTION  (mirrors contrastive_loso_finetune.py)
# ---------------------------------------------------------------------------

def set_trainable_layers(meg_enc: nn.Module, layers: str) -> List[nn.Parameter]:
    for p in meg_enc.parameters():
        p.requires_grad = False

    if layers == "all":
        for p in meg_enc.parameters():
            p.requires_grad = True
    elif layers == "spatial":
        for p in meg_enc.spatial.parameters():
            p.requires_grad = True
    elif layers == "proj":
        for p in meg_enc.proj.parameters():
            p.requires_grad = True
    elif layers == "spatial+proj":
        for p in meg_enc.spatial.parameters():
            p.requires_grad = True
        for p in meg_enc.proj.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown --layers: {layers!r}. "
                         "Choose from: all, spatial, proj, spatial+proj")

    trainable = [p for p in meg_enc.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    print(f"  Fine-tuning layers={layers!r}  trainable params={n:,}")
    return trainable


# ---------------------------------------------------------------------------
#  FINE-TUNING LOOP
# ---------------------------------------------------------------------------

def finetune_on_predicted(
    meg_enc: nn.Module,
    txt_enc: nn.Module,
    dataset: PredictedWindowDataset,
    layers:  str,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Fine-tune meg_enc on predicted-listened windows.
    Returns (best_meg_enc, train_loss_history, val_loss_history).
    """
    n_val  = max(1, int(FT_VAL_FRAC * len(dataset)))
    n_tr   = len(dataset) - n_val
    g      = torch.Generator().manual_seed(SEED)
    tr_ds, val_ds = random_split(dataset, [n_tr, n_val], generator=g)

    tr_dl  = DataLoader(tr_ds,  FT_BATCH, shuffle=True,  drop_last=True,  num_workers=0)
    val_dl = DataLoader(val_ds, FT_BATCH, shuffle=False, drop_last=False, num_workers=0)

    print(f"  Fine-tune split: train={n_tr}  val={n_val}")

    trainable = set_trainable_layers(meg_enc, layers)
    meg_enc   = meg_enc.to(DEVICE)
    txt_enc   = txt_enc.to(DEVICE).eval()
    for p in txt_enc.parameters():
        p.requires_grad = False

    opt   = torch.optim.AdamW(trainable, lr=FT_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FT_EPOCHS)

    best_val = float("inf")
    best_wts = deepcopy(meg_enc.state_dict())
    no_imp   = 0
    tr_hist, val_hist = [], []

    for epoch in range(1, FT_EPOCHS + 1):
        meg_enc.train()
        tr_losses = []
        for meg_win, word_idx in tr_dl:
            meg_win  = meg_win.to(DEVICE)
            word_idx = word_idx.to(DEVICE)
            if FT_NOISE > 0:
                meg_win = meg_win + FT_NOISE * torch.randn_like(meg_win)

            z_meg = meg_enc(meg_win)
            with torch.no_grad():
                z_txt = txt_enc(word_idx)
            loss  = nt_xent_loss(z_meg, z_txt)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        meg_enc.eval()
        val_losses = []
        with torch.no_grad():
            for meg_win, word_idx in val_dl:
                meg_win  = meg_win.to(DEVICE)
                word_idx = word_idx.to(DEVICE)
                z_meg    = meg_enc(meg_win)
                z_txt    = txt_enc(word_idx)
                val_losses.append(nt_xent_loss(z_meg, z_txt).item())

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        tr_hist.append(tr_loss)
        val_hist.append(val_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{FT_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  no_imp={no_imp}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_wts = deepcopy(meg_enc.state_dict())
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= FT_PATIENCE:
                print(f"    early stop at epoch {epoch}")
                break

    meg_enc.load_state_dict(best_wts)
    return meg_enc, tr_hist, val_hist


# ---------------------------------------------------------------------------
#  TEST EVALUATION  (condition A_ft_pred equivalent)
# ---------------------------------------------------------------------------

def evaluate_on_heldout(
    meg_enc:     nn.Module,
    vocab:       Dict[str, int],
    words:       List[str],
    heldout_subj: str,
    arch:         str,
    mode:         str,
    n_channels:   int,
    label:        str,
) -> Optional[Dict]:
    """
    Evaluate meg_enc on heldout_subj's imagined MEG using mapping model
    heldout_{heldout_subj} (condition A pipeline).
    """
    print(f"\n  [{label}] evaluating on {heldout_subj} imagined MEG …")

    ckpt_map = os.path.join(
        MAPPING_DIR, f"heldout_{heldout_subj}", f"{arch}_{mode}.pt"
    )
    if not os.path.exists(ckpt_map):
        print(f"  WARNING: mapping checkpoint not found: {ckpt_map} — skipping eval")
        return None

    mapping_model = _ARCH_FACTORY[arch](n_channels)
    mapping_model.load_state_dict(torch.load(ckpt_map, map_location="cpu"))
    mapping_model = mapping_model.eval().to(DEVICE)

    txt_enc_eval = TextEncoder(build_text_embeddings(words)).to(DEVICE)

    # We need the text encoder weights that go with this meg_enc.
    # They come from the LOSO decoder checkpoint.
    ckpt_txt = os.path.join(LOSO_DEC_DIR, f"heldout_{heldout_subj}", "text_encoder.pt")
    if os.path.exists(ckpt_txt):
        txt_enc_eval.load_state_dict(torch.load(ckpt_txt, map_location="cpu"))
    txt_enc_eval.eval()

    all_wins, all_wds = [], []

    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
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
            all_wins.extend(wins)
            all_wds.extend(wds)

    del mapping_model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    metrics = ranking_metrics(meg_enc, txt_enc_eval, all_wins, all_wds, vocab)
    if metrics is None:
        print(f"    No valid windows for {heldout_subj}")
        return None

    print(f"    n={metrics['n_samples']}  V={metrics['vocab_size']}  "
          f"R@1={metrics['R@1']:.3f}  R@5={metrics['R@5']:.3f}  "
          f"R@10={metrics['R@10']:.3f}  MRR={metrics['MRR']:.3f}  "
          f"median_rank={metrics['median_rank']}")
    return metrics


# ---------------------------------------------------------------------------
#  PER-FOLD RUNNER
# ---------------------------------------------------------------------------

def run_fold(
    heldout_subj: str,
    layers:       str,
    arch:         str,
    mode:         str,
    model_size:   str,
    text_method:  str,
) -> Dict:
    print(f"\n{'='*62}")
    print(f"  Heldout : {heldout_subj}  layers={layers!r}  "
          f"mapping={arch}_{mode}")
    print(f"{'='*62}")

    train_subjects = [s for s in SUBJECTS if s != heldout_subj]
    vocab, words   = build_combined_vocab(train_subjects, heldout_subj)
    print(f"  Combined vocab: {len(vocab)} words")

    n_channels = _probe_n_channels(heldout_subj)

    # Load base LOSO decoder for this fold
    ckpt_meg = os.path.join(LOSO_DEC_DIR, f"heldout_{heldout_subj}", "meg_encoder.pt")
    ckpt_txt = os.path.join(LOSO_DEC_DIR, f"heldout_{heldout_subj}", "text_encoder.pt")
    if not os.path.exists(ckpt_meg):
        raise FileNotFoundError(
            f"{ckpt_meg} not found — run contrastive_loso.py first"
        )

    raw_emb = build_text_embeddings(words, method=text_method)
    meg_enc = make_meg_encoder(n_channels, model_size)
    meg_enc.load_state_dict(torch.load(ckpt_meg, map_location="cpu"))
    txt_enc = TextEncoder(raw_emb)
    txt_enc.load_state_dict(torch.load(ckpt_txt, map_location="cpu"))

    # ---- Baseline eval (no fine-tuning) ----
    meg_enc_eval = meg_enc.eval().to(DEVICE)
    baseline = evaluate_on_heldout(
        meg_enc_eval, vocab, words, heldout_subj, arch, mode, n_channels,
        label="Baseline (no FT)",
    )

    # ---- Collect training windows from predicted-listened ----
    windows, word_idxs = collect_predicted_windows(
        train_subjects, heldout_subj, vocab, arch, mode, n_channels,
    )
    if len(windows) < FT_BATCH * 2:
        print(f"  WARNING: only {len(windows)} windows — skipping fine-tuning")
        return {"baseline": baseline, "finetuned": None, "delta": None}

    dataset = PredictedWindowDataset(windows, word_idxs)

    # ---- Fine-tuning ----
    print(f"\n  [Fine-tune] adapting decoder to predicted-listened domain …")
    meg_enc_ft = deepcopy(meg_enc)
    meg_enc_ft, tr_hist, val_hist = finetune_on_predicted(
        meg_enc_ft, txt_enc, dataset, layers=layers,
    )
    meg_enc_ft.eval()

    # ---- Fine-tuned eval ----
    finetuned = evaluate_on_heldout(
        meg_enc_ft, vocab, words, heldout_subj, arch, mode, n_channels,
        label="Fine-tuned on predicted-listened",
    )

    # ---- Save checkpoint ----
    ft_dir = os.path.join(FTPRED_OUT, "models", f"heldout_{heldout_subj}")
    os.makedirs(ft_dir, exist_ok=True)
    ckpt_out = os.path.join(ft_dir, "meg_encoder_ft_pred.pt")
    torch.save(meg_enc_ft.state_dict(), ckpt_out)
    print(f"  [saved] {ckpt_out}")

    # ---- Loss curves ----
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tr_hist,  label="train loss", color="#E74C3C")
    ax.plot(val_hist, label="val loss",   color="#2ECC71")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"Fine-tune on predicted-listened — {heldout_subj}\n"
                 f"layers={layers!r}  mapping={arch}_{mode}")
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(ft_dir, "loss_curve.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    # ---- Delta ----
    delta = None
    if baseline is not None and finetuned is not None:
        delta = {
            m: finetuned[m] - baseline[m]
            for m in ["R@1", "R@5", "R@10", "MRR"]
        }
        delta["median_rank"] = baseline["median_rank"] - finetuned["median_rank"]
        print(f"\n  Δ (finetuned_pred - baseline):  "
              f"R@1={delta['R@1']:+.3f}  R@5={delta['R@5']:+.3f}  "
              f"R@10={delta['R@10']:+.3f}  MRR={delta['MRR']:+.3f}  "
              f"Δmedian_rank={delta['median_rank']:+d} (positive=better)")

    return {"baseline": baseline, "finetuned": finetuned, "delta": delta}


# ---------------------------------------------------------------------------
#  SUMMARY PLOT
# ---------------------------------------------------------------------------

def plot_summary(all_results: Dict[str, Dict], layers: str, arch: str, mode: str) -> None:
    subjects = [s for s in all_results if all_results[s].get("finetuned") is not None]
    if not subjects:
        return
    metrics = ["R@1", "R@5", "R@10", "MRR"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, metric in zip(axes, metrics):
        base_vals = [all_results[s]["baseline"][metric] for s in subjects
                     if all_results[s]["baseline"] is not None]
        ft_vals   = [all_results[s]["finetuned"][metric] for s in subjects]
        chance    = 1.0 / all_results[subjects[0]]["baseline"]["vocab_size"]

        x = np.arange(len(subjects))
        ax.bar(x - 0.2, base_vals, 0.35, label="Baseline",    color="#E74C3C", alpha=0.8)
        ax.bar(x + 0.2, ft_vals,   0.35, label="FT-predicted", color="#3498DB", alpha=0.8)
        if metric == "R@1":
            ax.axhline(chance, color="grey", lw=1, linestyle=":",
                       label=f"chance={chance:.3f}")
        ax.axhline(np.mean(base_vals), color="#E74C3C", lw=1.2, linestyle="--", alpha=0.6)
        ax.axhline(np.mean(ft_vals),   color="#3498DB", lw=1.2, linestyle="--", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("sub-", "") for s in subjects],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)

        mean_b = np.mean(base_vals)
        mean_f = np.mean(ft_vals)
        ax.text(0.02, 0.97,
                f"mean base={mean_b:.3f}\nmean FT={mean_f:.3f}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    tag = layers.replace("+", "_")
    plt.suptitle(
        f"Fine-tune on predicted-listened MEG  (layers={layers!r}  mapping={arch}_{mode})\n"
        f"Red=baseline  |  Blue=after FT on predicted-listened from training subjects",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(FTPRED_OUT, f"summary_{tag}_{arch}_{mode}.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune contrastive decoder on predicted-listened MEG.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--heldout_subject", default=None,
                        help="Single fold (e.g. sub-01). Omit for all subjects.")
    parser.add_argument("--layers",
                        choices=["all", "spatial", "proj", "spatial+proj"],
                        default="all",
                        help="Which layers to fine-tune")
    parser.add_argument("--mapping_arch",
                        choices=["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN"],
                        default=DEFAULT_MAPPING_ARCH)
    parser.add_argument("--mapping_mode", choices=["full", "windowed"],
                        default=DEFAULT_MAPPING_MODE)
    parser.add_argument("--model_size", choices=["small", "full"],
                        default=MODEL_SIZE)
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds = [args.heldout_subject] if args.heldout_subject else SUBJECTS

    print(f"Device       : {DEVICE}")
    print(f"Model size   : {args.model_size}")
    print(f"Layers       : {args.layers}")
    print(f"Mapping      : {args.mapping_arch}_{args.mapping_mode}")
    print(f"LR={FT_LR}  epochs={FT_EPOCHS}  patience={FT_PATIENCE}  "
          f"batch={FT_BATCH}  noise={FT_NOISE}")
    print(f"Folds        : {folds}\n")

    all_results = {}

    for subj in folds:
        fold_res = run_fold(
            subj, args.layers, args.mapping_arch, args.mapping_mode,
            args.model_size, args.text_encoder,
        )
        all_results[subj] = fold_res

        out_json = os.path.join(FTPRED_OUT, f"ftpred_{subj}.json")
        with open(out_json, "w") as f:
            json.dump(fold_res, f, indent=2)

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print(f"  SUMMARY  (layers={args.layers!r}  mapping={args.mapping_arch}_{args.mapping_mode})")
    print(f"{'='*70}")
    hdr = (f"  {'subject':10s}  {'base R@1':>8}  {'ft R@1':>8}  {'ΔR@1':>7}  "
           f"{'base R@10':>9}  {'ft R@10':>8}  {'ΔMRR':>7}")
    print(hdr)
    for subj, res in all_results.items():
        b = res.get("baseline")
        ft = res.get("finetuned")
        if b is None or ft is None:
            print(f"  {subj:10s}  (skipped)")
            continue
        print(f"  {subj:10s}  {b['R@1']:8.3f}  {ft['R@1']:8.3f}  "
              f"{ft['R@1']-b['R@1']:+7.3f}  "
              f"{b['R@10']:9.3f}  {ft['R@10']:8.3f}  "
              f"{ft['MRR']-b['MRR']:+7.3f}")

    valid = {s: r for s, r in all_results.items()
             if r.get("baseline") and r.get("finetuned")}
    if len(valid) > 1:
        for metric in ["R@1", "R@5", "R@10", "MRR"]:
            bm = np.mean([valid[s]["baseline"][metric] for s in valid])
            fm = np.mean([valid[s]["finetuned"][metric] for s in valid])
            print(f"  {'MEAN':10s}  {metric}: {bm:.3f} → {fm:.3f}  ({fm-bm:+.3f})")

        summary_path = os.path.join(
            FTPRED_OUT,
            f"summary_{args.layers.replace('+','_')}_{args.mapping_arch}_{args.mapping_mode}.json",
        )
        with open(summary_path, "w") as f:
            json.dump({s: all_results[s] for s in valid}, f, indent=2)
        print(f"\n[saved] {summary_path}")

        plot_summary(all_results, args.layers, args.mapping_arch, args.mapping_mode)

    print(f"\nDone. Results in {FTPRED_OUT}/")


if __name__ == "__main__":
    main()
