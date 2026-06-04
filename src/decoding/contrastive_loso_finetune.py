"""
contrastive_loso_finetune.py
=============================
Option 2: subject-specific fine-tuning on listened MEG.

For each LOSO fold:
  1. Load the pre-trained encoder from contrastive_loso_out/models/heldout_{subj}/
  2. Fine-tune selected layers on the held-out subject's OWN listened MEG
     (using NT-Xent loss, small LR, few epochs).
  3. Evaluate and compare baseline vs fine-tuned R@1/R@5/R@10/MRR.

Fine-tuning layers (--layers):
  all          — fine-tune everything (spatial + temporal + proj)
  spatial      — only the spatial compression layer (most subject-specific)
  proj         — only the projection head
  spatial+proj — spatial + projection, freeze temporal

Usage
-----
  cd /fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning
  python contrastive_loso_finetune.py [--layers all] [--model_size small|full]
  python contrastive_loso_finetune.py --heldout_subject sub-01  # single fold
"""

import argparse
import json
import os
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE, SEED,
    MEGWordDataset, TextEncoder,
    make_meg_encoder, build_text_embeddings,
    nt_xent_loss, evaluate_ranking,
    TEXT_ENCODER, MODEL_SIZE,
)

LOSO_OUT    = "./contrastive_loso_out"
FINETUNE_OUT = os.path.join(LOSO_OUT, "finetune")
os.makedirs(FINETUNE_OUT, exist_ok=True)

# Fine-tuning hyperparameters — conservative to avoid overfitting
FINETUNE_LR       = 5e-5
FINETUNE_EPOCHS   = 30
FINETUNE_PATIENCE = 7
FINETUNE_BATCH    = 32
FINETUNE_VAL_FRAC = 0.20


# =============================================================================
#  LAYER SELECTION
# =============================================================================

def set_trainable_layers(meg_enc: nn.Module, layers: str) -> List[nn.Parameter]:
    """
    Freeze/unfreeze layers according to --layers flag.
    Returns the list of parameters that will be optimised.
    """
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
        raise ValueError(f"Unknown --layers value: {layers!r}. "
                         "Choose from: all, spatial, proj, spatial+proj")

    trainable = [p for p in meg_enc.parameters() if p.requires_grad]
    n = sum(p.numel() for p in trainable)
    print(f"  Fine-tuning layers={layers!r}  trainable params={n:,}")
    return trainable


# =============================================================================
#  FINE-TUNING LOOP
# =============================================================================

def finetune(
    meg_enc:      nn.Module,
    txt_enc:      nn.Module,
    dataset:      MEGWordDataset,
    layers:       str,
) -> nn.Module:
    """
    Fine-tune meg_enc on the held-out subject's listened MEG.
    txt_enc is frozen throughout (we only adapt the MEG encoder).
    Returns the fine-tuned encoder (best checkpoint by val loss).
    """
    n_val  = max(1, int(FINETUNE_VAL_FRAC * len(dataset)))
    n_tr   = len(dataset) - n_val
    g      = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    tr_idx, val_idx = indices[:n_tr], indices[n_tr:]

    tr_dl  = DataLoader(Subset(dataset, tr_idx),  FINETUNE_BATCH,
                        shuffle=True,  drop_last=True,  num_workers=0)
    val_dl = DataLoader(Subset(dataset, val_idx), FINETUNE_BATCH,
                        shuffle=False, drop_last=False, num_workers=0)

    print(f"  Fine-tune split: train={n_tr}  val={n_val}")

    trainable = set_trainable_layers(meg_enc, layers)
    meg_enc   = meg_enc.to(DEVICE)
    txt_enc   = txt_enc.to(DEVICE)
    txt_enc.eval()
    for p in txt_enc.parameters():
        p.requires_grad = False

    opt   = torch.optim.AdamW(trainable, lr=FINETUNE_LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)

    best_val  = float("inf")
    best_wts  = deepcopy(meg_enc.state_dict())
    no_imp    = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        meg_enc.train()
        tr_losses = []
        for meg_win, word_idx in tr_dl:
            meg_win  = meg_win.to(DEVICE)
            word_idx = word_idx.to(DEVICE)
            meg_win  = meg_win + 0.02 * torch.randn_like(meg_win)

            z_meg  = meg_enc(meg_win)
            with torch.no_grad():
                z_txt = txt_enc(word_idx)
            loss   = nt_xent_loss(z_meg, z_txt)

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

        if epoch % 5 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}/{FINETUNE_EPOCHS}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"best={best_val:.4f}  no_imp={no_imp}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_wts = deepcopy(meg_enc.state_dict())
            no_imp   = 0
        else:
            no_imp += 1
            if no_imp >= FINETUNE_PATIENCE:
                print(f"    early stop at epoch {epoch}")
                break

    meg_enc.load_state_dict(best_wts)
    return meg_enc


# =============================================================================
#  PER-FOLD EVALUATION
# =============================================================================

def run_fold(heldout_subj: str, model_size: str,
             layers: str, text_method: str) -> Dict:
    """
    Returns dict with keys "baseline" and "finetuned", each containing
    ranking metrics for the held-out subject.
    """
    print(f"\n{'='*60}")
    print(f"  Held-out: {heldout_subj}  layers={layers!r}")
    print(f"{'='*60}")

    train_subjects = [s for s in SUBJECTS if s != heldout_subj]

    # Rebuild the same combined vocab used during LOSO training
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
    train_ds.vocab = vocab; train_ds.words = words
    val_ds.vocab   = vocab; val_ds.words   = words

    # Load fold checkpoints
    fold_dir = os.path.join(LOSO_OUT, "models", f"heldout_{heldout_subj}")
    ckpt_meg = os.path.join(fold_dir, "meg_encoder.pt")
    ckpt_txt = os.path.join(fold_dir, "text_encoder.pt")
    if not os.path.exists(ckpt_meg):
        raise FileNotFoundError(
            f"{ckpt_meg} not found — run contrastive_loso.py first"
        )

    n_channels = train_ds.pairs[0][0].shape[0]
    raw_emb    = build_text_embeddings(words, method=text_method)

    meg_enc = make_meg_encoder(n_channels, model_size)
    meg_enc.load_state_dict(torch.load(ckpt_meg, map_location="cpu"))

    txt_enc = TextEncoder(raw_emb)
    txt_enc.load_state_dict(torch.load(ckpt_txt, map_location="cpu"))

    meg_enc = meg_enc.to(DEVICE)
    txt_enc = txt_enc.to(DEVICE)

    # ---- Baseline (no fine-tuning) ----
    print("\n  [Baseline] evaluating without fine-tuning...")
    baseline_metrics = evaluate_ranking(
        meg_enc, txt_enc, val_ds, tag=f"baseline {heldout_subj}"
    )

    # ---- Fine-tuning ----
    print(f"\n  [Fine-tune] adapting to {heldout_subj} listened MEG...")
    meg_enc_ft = deepcopy(meg_enc)
    meg_enc_ft = finetune(meg_enc_ft, txt_enc, val_ds, layers=layers)

    print("\n  [Fine-tuned] evaluating after fine-tuning...")
    ft_metrics = evaluate_ranking(
        meg_enc_ft, txt_enc, val_ds, tag=f"finetuned {heldout_subj}"
    )

    # Save fine-tuned checkpoint
    ft_dir = os.path.join(FINETUNE_OUT, "models", f"heldout_{heldout_subj}")
    os.makedirs(ft_dir, exist_ok=True)
    torch.save(meg_enc_ft.state_dict(), os.path.join(ft_dir, "meg_encoder_ft.pt"))

    delta = {
        m: ft_metrics[m] - baseline_metrics[m]
        for m in ["R@1", "R@5", "R@10", "MRR"]
    }
    delta["median_rank"] = baseline_metrics["median_rank"] - ft_metrics["median_rank"]
    print(f"\n  Δ (finetuned - baseline):  "
          f"R@1={delta['R@1']:+.3f}  R@5={delta['R@5']:+.3f}  "
          f"R@10={delta['R@10']:+.3f}  MRR={delta['MRR']:+.3f}  "
          f"Δmedian_rank={delta['median_rank']:+d} (positive=better)")

    return {"baseline": baseline_metrics, "finetuned": ft_metrics, "delta": delta}


# =============================================================================
#  SUMMARY PLOT
# =============================================================================

def plot_summary(all_results: Dict[str, Dict], layers: str) -> None:
    subjects = list(all_results.keys())
    metrics  = ["R@1", "R@5", "R@10", "MRR"]
    n_s      = len(subjects)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, metric in zip(axes, metrics):
        base_vals = [all_results[s]["baseline"][metric] for s in subjects]
        ft_vals   = [all_results[s]["finetuned"][metric] for s in subjects]
        chance    = all_results[subjects[0]]["baseline"]["chance_R@1"]

        x = np.arange(n_s)
        ax.bar(x - 0.2, base_vals, 0.35, label="Baseline",    color="#E74C3C", alpha=0.8)
        ax.bar(x + 0.2, ft_vals,   0.35, label="Fine-tuned",  color="#2ECC71", alpha=0.8)

        if metric == "R@1":
            ax.axhline(chance, color="grey", lw=1, linestyle=":",
                       label=f"chance={chance:.3f}")

        ax.axhline(np.mean(base_vals), color="#E74C3C", lw=1.2, linestyle="--", alpha=0.6)
        ax.axhline(np.mean(ft_vals),   color="#2ECC71", lw=1.2, linestyle="--", alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("sub-", "") for s in subjects],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7)

        mean_b = np.mean(base_vals)
        mean_f = np.mean(ft_vals)
        ax.text(0.02, 0.97,
                f"mean baseline={mean_b:.3f}\nmean finetuned={mean_f:.3f}",
                transform=ax.transAxes, va="top", fontsize=7,
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.suptitle(
        f"Subject-specific fine-tuning on listened MEG  (layers={layers!r})\n"
        f"Red=baseline (no FT)  |  Green=after fine-tuning on subject's own MEG",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(FINETUNE_OUT, f"finetune_summary_{layers.replace('+','_')}.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[saved] {path}")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=["small", "full"], default=MODEL_SIZE)
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    parser.add_argument("--layers",
                        choices=["all", "spatial", "proj", "spatial+proj"],
                        default="all",
                        help="Which layers of the MEG encoder to fine-tune")
    parser.add_argument("--heldout_subject", default=None,
                        help="Single fold (e.g. sub-01). Omit for all subjects.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds = [args.heldout_subject] if args.heldout_subject else SUBJECTS

    print(f"Device        : {DEVICE}")
    print(f"Model size    : {args.model_size}")
    print(f"Layers        : {args.layers}")
    print(f"FT LR         : {FINETUNE_LR}  epochs={FINETUNE_EPOCHS}  patience={FINETUNE_PATIENCE}")
    print(f"Folds         : {folds}\n")

    all_results = {}

    for subj in folds:
        fold_res = run_fold(subj, args.model_size, args.layers, args.text_encoder)
        all_results[subj] = fold_res
        with open(os.path.join(FINETUNE_OUT, f"finetune_{subj}.json"), "w") as f:
            json.dump(fold_res, f, indent=2)

    # Summary table
    print(f"\n{'='*65}")
    print(f"  SUMMARY  (layers={args.layers!r})")
    print(f"{'='*65}")
    header = (f"  {'subject':10s}  {'base R@1':>8}  {'ft R@1':>8}  {'ΔR@1':>7}  "
              f"{'base R@10':>9}  {'ft R@10':>8}  {'ΔMRR':>7}")
    print(header)
    for subj, res in all_results.items():
        b, f = res["baseline"], res["finetuned"]
        print(f"  {subj:10s}  {b['R@1']:8.3f}  {f['R@1']:8.3f}  "
              f"{f['R@1']-b['R@1']:+7.3f}  "
              f"{b['R@10']:9.3f}  {f['R@10']:8.3f}  "
              f"{f['MRR']-b['MRR']:+7.3f}")

    if len(all_results) > 1:
        for metric in ["R@1", "R@5", "R@10", "MRR"]:
            base_mean = np.mean([all_results[s]["baseline"][metric] for s in all_results])
            ft_mean   = np.mean([all_results[s]["finetuned"][metric] for s in all_results])
            print(f"  {'MEAN':10s}  {metric}: {base_mean:.3f} → {ft_mean:.3f}  "
                  f"({ft_mean - base_mean:+.3f})")

        summary = {
            subj: {
                "baseline": all_results[subj]["baseline"],
                "finetuned": all_results[subj]["finetuned"],
                "delta": all_results[subj]["delta"],
            }
            for subj in all_results
        }
        out_path = os.path.join(FINETUNE_OUT, f"summary_{args.layers.replace('+','_')}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {out_path}")

        plot_summary(all_results, args.layers)

    print(f"\nDone. Results in {FINETUNE_OUT}/")


if __name__ == "__main__":
    main()
