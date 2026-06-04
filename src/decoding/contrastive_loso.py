"""
contrastive_loso.py
===================
Subject-wise Leave-One-Subject-Out (LOSO) version of the contrastive word
decoding pipeline.

Key differences from contrastive_word_meg.py
---------------------------------------------
1. LOSO split — val/test = one held-out subject's listened MEG; train = all
   others.  This gives an honest measure of cross-subject generalisation
   rather than the inflated score from a random within-subject split.

2. Stronger augmentation — three independent transforms applied at training
   time only:
     · Channel dropout  : zero out a random subset of channels (p per channel)
     · Time masking     : zero out a random contiguous block of time steps
     · Amplitude jitter : scale each channel by a random scalar ≈ 1

3. Larger batch (128) — NT-Xent benefits strongly from more in-batch
   negatives.  With B=128 each sample sees 127 negatives vs 63 with B=64.

4. All outputs go to ./contrastive_loso_out/ so they never overwrite the
   first run.

Usage
-----
  # full LOSO loop (one fold per subject)
  python contrastive_loso.py

  # single fold — useful for quick testing
  python contrastive_loso.py --heldout_subject sub-01

  # choose model size
  python contrastive_loso.py --model_size full   # GPU recommended
  python contrastive_loso.py --model_size small  # CPU-friendly
"""

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-use all shared components from the first script
from contrastive_word_meg import (
    # config
    SUBJECTS, POEM_KEYS, ONSET_DIR, SFREQ_DS, EMB_DIM, TEMPERATURE,
    REMOVE_FLASHES, SEED, TEXT_ENCODER, DEVICE,
    # data
    MEGWordDataset,
    # models
    make_meg_encoder, TextEncoder,
    # loss / eval
    nt_xent_loss, evaluate_ranking,
    # text embeddings
    build_text_embeddings,
)


# =============================================================================
#  LOSO CONFIG  — override only what differs from the base script
# =============================================================================
OUT_DIR      = "./contrastive_loso_out"
os.makedirs(OUT_DIR, exist_ok=True)

BATCH_SIZE   = 128      # larger → more NT-Xent negatives per step
LR           = 3e-4
WEIGHT_DECAY = 1e-4
N_EPOCHS     = 120
PATIENCE     = 20       # slightly more patience for noisier LOSO curves
DROPOUT      = 0.4      # increased from 0.3

# Augmentation strengths
AUG_NOISE_STD      = 0.03   # additive Gaussian noise per sample
AUG_CHAN_DROP_P    = 0.10   # probability of zeroing each channel independently
AUG_TIME_MASK_MAX  = 15     # max time-mask width in samples (0 to disable)
AUG_AMP_JITTER_STD = 0.10   # std of multiplicative amplitude jitter per channel


# =============================================================================
#  AUGMENTATION
# =============================================================================

def augment(x: torch.Tensor) -> torch.Tensor:
    """
    Apply three independent augmentations to a batch of MEG windows.

    x : (B, C, T) — assumed already on the correct device
    Returns a new tensor of the same shape (original is not modified).
    """
    x = x.clone()
    B, C, T = x.shape

    # 1. Additive Gaussian noise
    x = x + AUG_NOISE_STD * torch.randn_like(x)

    # 2. Channel dropout — zero entire channels independently
    if AUG_CHAN_DROP_P > 0:
        # mask shape (B, C, 1): broadcast over time
        mask = (torch.rand(B, C, 1, device=x.device) > AUG_CHAN_DROP_P).float()
        x = x * mask

    # 3. Time masking — one random contiguous block per sample
    if AUG_TIME_MASK_MAX > 0:
        width = torch.randint(1, AUG_TIME_MASK_MAX + 1, (B,))
        start = torch.randint(0, max(1, T - AUG_TIME_MASK_MAX), (B,))
        for i in range(B):
            x[i, :, start[i]: start[i] + width[i]] = 0.0

    # 4. Per-channel amplitude jitter
    if AUG_AMP_JITTER_STD > 0:
        scale = 1.0 + AUG_AMP_JITTER_STD * torch.randn(B, C, 1, device=x.device)
        x = x * scale

    return x


# =============================================================================
#  LOSO TRAINING
# =============================================================================

def train_one_fold(
    heldout_subj: str,
    model_size:   str,
    text_method:  str,
) -> Dict:
    """
    Train on all subjects except heldout_subj, validate on heldout_subj.
    Returns ranking metrics on the held-out subject.
    """
    train_subjects = [s for s in SUBJECTS if s != heldout_subj]

    print(f"\n  Building train dataset ({len(train_subjects)} subjects)...")
    train_ds = MEGWordDataset(
        subjects=train_subjects,
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="lis",
        remove_flashes=REMOVE_FLASHES,
    )

    print(f"  Building val dataset ({heldout_subj})...")
    val_ds = MEGWordDataset(
        subjects=[heldout_subj],
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="lis",
        remove_flashes=REMOVE_FLASHES,
    )

    # Build a shared vocabulary across train + val so the ranking pool is consistent
    combined_vocab = dict(train_ds.vocab)
    for w in val_ds.vocab:
        if w not in combined_vocab:
            combined_vocab[w] = len(combined_vocab)
    combined_words = sorted(combined_vocab, key=combined_vocab.get)

    # Patch vocab into both datasets
    train_ds.vocab  = combined_vocab
    train_ds.words  = combined_words
    val_ds.vocab    = combined_vocab
    val_ds.words    = combined_words

    print(f"  Combined vocab: {len(combined_vocab)} words  "
          f"train={len(train_ds)}  val={len(val_ds)}")

    # Text embeddings (one per vocab word)
    print(f"  Building text embeddings ({text_method})...")
    raw_emb = build_text_embeddings(combined_words, method=text_method)

    n_channels = train_ds.pairs[0][0].shape[0]
    meg_enc = make_meg_encoder(n_channels, model_size, dropout=DROPOUT).to(DEVICE)
    txt_enc = TextEncoder(raw_emb, dropout=DROPOUT).to(DEVICE)

    n_meg  = sum(p.numel() for p in meg_enc.parameters())
    n_proj = sum(p.numel() for p in txt_enc.proj.parameters())
    print(f"  MEGWordEncoder ({model_size}): {n_meg:,}  "
          f"TextEncoder.proj: {n_proj:,}  (BERT frozen)")

    params = list(meg_enc.parameters()) + list(txt_enc.proj.parameters())
    opt    = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)

    tr_dl  = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  drop_last=True,
                        num_workers=0)
    val_dl = DataLoader(val_ds,  BATCH_SIZE, shuffle=False, drop_last=False,
                        num_workers=0)

    best_val   = float("inf")
    best_meg_w = deepcopy(meg_enc.state_dict())
    best_txt_w = deepcopy(txt_enc.state_dict())
    no_imp     = 0
    history    = {"train": [], "val": []}

    for epoch in range(1, N_EPOCHS + 1):
        meg_enc.train(); txt_enc.train()
        tr_losses = []

        for meg_win, word_idx in tr_dl:
            meg_win  = augment(meg_win.to(DEVICE))
            word_idx = word_idx.to(DEVICE)

            z_meg  = meg_enc(meg_win)
            z_text = txt_enc(word_idx)
            loss   = nt_xent_loss(z_meg, z_text)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_losses.append(loss.item())

        sched.step()

        meg_enc.eval(); txt_enc.eval()
        val_losses = []
        with torch.no_grad():
            for meg_win, word_idx in val_dl:
                meg_win  = meg_win.to(DEVICE)
                word_idx = word_idx.to(DEVICE)
                val_losses.append(
                    nt_xent_loss(meg_enc(meg_win), txt_enc(word_idx)).item()
                )

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        history["train"].append(tr_loss)
        history["val"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:4d}/{N_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  no_imp={no_imp}")

        if val_loss < best_val - 1e-6:
            best_val   = val_loss
            best_meg_w = deepcopy(meg_enc.state_dict())
            best_txt_w = deepcopy(txt_enc.state_dict())
            no_imp     = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"    early stop at epoch {epoch}")
                break

    meg_enc.load_state_dict(best_meg_w)
    txt_enc.load_state_dict(best_txt_w)

    # Save per-fold checkpoints
    fold_dir = os.path.join(OUT_DIR, "models", f"heldout_{heldout_subj}")
    os.makedirs(fold_dir, exist_ok=True)
    torch.save(meg_enc.state_dict(), os.path.join(fold_dir, "meg_encoder.pt"))
    torch.save(txt_enc.state_dict(), os.path.join(fold_dir, "text_encoder.pt"))

    # Evaluate on the held-out subject
    metrics = evaluate_ranking(meg_enc, txt_enc, val_ds,
                               tag=f"heldout={heldout_subj}")

    _save_curve(history, heldout_subj)
    return metrics


def _save_curve(history: dict, heldout_subj: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"],   label=f"val ({heldout_subj})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"LOSO fold: held-out {heldout_subj}"); ax.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"curve_heldout_{heldout_subj}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
#  SUMMARY PLOT
# =============================================================================

def plot_summary(all_metrics: Dict[str, Dict]) -> None:
    """Bar chart of per-subject R@1/R@5/R@10/MRR + mean across folds."""
    subjects = list(all_metrics.keys())
    metrics  = ["R@1", "R@5", "R@10", "MRR"]
    chance   = list(all_metrics.values())[0]["chance_R@1"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)

    for ax, m in zip(axes, metrics):
        vals = [all_metrics[s][m] for s in subjects]
        mean = float(np.mean(vals))
        colors = ["#E74C3C" if v < chance else "#2ECC71" for v in vals]
        ax.bar(range(len(subjects)), vals, color=colors, alpha=0.85)
        ax.axhline(mean,   color="navy",   lw=1.5, linestyle="--", label=f"mean={mean:.3f}")
        if m == "R@1":
            ax.axhline(chance, color="grey", lw=1.0, linestyle=":",  label=f"chance={chance:.3f}")
        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([s.replace("sub-","") for s in subjects],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(m, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)

    plt.suptitle(
        "LOSO contrastive word decoding — listened MEG (upper bound)\n"
        f"Val = held-out subject  |  vocab≈76 words  |  green > chance",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "loso_summary.png")
    plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
    print(f"[saved] {path}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO contrastive MEG word decoding")
    parser.add_argument("--heldout_subject", default=None,
                        help="Run a single fold (e.g. sub-01). Omit for all subjects.")
    parser.add_argument("--model_size", choices=["small", "full"], default="full",
                        help="full=~544k params (GPU recommended); small=~143k (CPU ok)")
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds = [args.heldout_subject] if args.heldout_subject else SUBJECTS

    print("=" * 65)
    print(f"  Contrastive LOSO  —  {len(folds)} fold(s)")
    print(f"  Device      : {DEVICE}")
    print(f"  Model size  : {args.model_size}")
    print(f"  Text encoder: {args.text_encoder}")
    print(f"  Batch size  : {BATCH_SIZE}  (vs 64 in base script)")
    print(f"  Dropout     : {DROPOUT}  (vs 0.3)")
    print(f"  Aug: noise={AUG_NOISE_STD}  chan_drop={AUG_CHAN_DROP_P}"
          f"  time_mask={AUG_TIME_MASK_MAX}  amp_jitter={AUG_AMP_JITTER_STD}")
    print(f"  Out dir     : {OUT_DIR}")
    print("=" * 65)

    all_metrics: Dict[str, Dict] = {}

    for subj in folds:
        print(f"\n{'='*65}")
        print(f"  HELD-OUT: {subj}")
        print(f"{'='*65}")

        metrics = train_one_fold(
            heldout_subj=subj,
            model_size=args.model_size,
            text_method=args.text_encoder,
        )
        all_metrics[subj] = metrics

        with open(os.path.join(OUT_DIR, f"metrics_{subj}.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    # Summary across all completed folds
    print(f"\n{'='*65}")
    print("  LOSO SUMMARY")
    print(f"{'='*65}")
    print(f"  {'subject':10s}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'MRR':>6}  {'med_rank':>9}")
    for subj, m in all_metrics.items():
        print(f"  {subj:10s}  {m['R@1']:6.3f}  {m['R@5']:6.3f}  "
              f"{m['R@10']:6.3f}  {m['MRR']:6.3f}  {m['median_rank']:>5}/{m['vocab_size']}")

    if len(all_metrics) > 1:
        for metric in ["R@1", "R@5", "R@10", "MRR"]:
            vals = [m[metric] for m in all_metrics.values()]
            print(f"  {'MEAN':10s}  " + "       " * ["R@1","R@5","R@10","MRR"].index(metric)
                  + f"{np.mean(vals):.3f}±{np.std(vals):.3f}")

        summary = {
            metric: {
                "mean": float(np.mean([m[metric] for m in all_metrics.values()])),
                "std":  float(np.std( [m[metric] for m in all_metrics.values()])),
                "per_subject": {s: all_metrics[s][metric] for s in all_metrics},
            }
            for metric in ["R@1", "R@5", "R@10", "MRR", "median_rank"]
        }
        with open(os.path.join(OUT_DIR, "loso_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {OUT_DIR}/loso_summary.json")

        plot_summary(all_metrics)


if __name__ == "__main__":
    main()
