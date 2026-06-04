"""
contrastive_imagined_direct.py
===============================
Modified version of contrastive_word_meg_compare.py that trains the contrastive
decoder directly on imagined MEG responses instead of listened responses.

This eliminates the img→lis mapping step and allows us to see how well we can
decode words directly from imagined neural activity.

Key differences from original:
- Trains on imagined MEG data (suffix="img") 
- Uses leave-one-subject-out (LOSO) approach
- Evaluates on the same imagined data for comparison
- Focuses on top-20 words for cleaner analysis

Usage:
    # Train on imagined MEG for one heldout subject
    python contrastive_imagined_direct.py --heldout_subject sub-03 --encoders bert

    # Train on all subjects' imagined data (not LOSO)
    python contrastive_imagined_direct.py --mode all_subjects --encoders bert

    Single encoder:
    python contrastive_imagined_direct.py --mode all_subjects --encoders bert
    python contrastive_imagined_direct.py --mode all_subjects --encoders whisper
    python contrastive_imagined_direct.py --mode all_subjects --encoders wav2vec
    python contrastive_imagined_direct.py --mode all_subjects --encoders bert_wav2vec

    Multiple encoders at once:
    # Train all 4 encoders in one run
    python contrastive_imagined_direct.py --mode all_subjects --encoders bert whisper
    wav2vec bert_wav2vec
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

warnings.filterwarnings("ignore", category=UserWarning)

# Import components from the original script
import sys
sys.path.insert(0, '.')
from contrastive_word_meg_compare import (
    # Constants
    BASE_PATH, ONSET_DIR, SUBJECTS, POEM_KEYS, DS_FACTOR, SFREQ_DS, N_SESSIONS,
    EPOCH_TMIN_S, WIN_PRE_MS, WIN_POST_MS, WIN_PRE, WIN_POST, WIN_SIZE, 
    REMOVE_FLASHES, EMB_DIM, TEMPERATURE, DROPOUT, MODEL_SIZE, BATCH_SIZE,
    LR, WEIGHT_DECAY, N_EPOCHS, PATIENCE, VAL_FRAC, SEED, ENCODER_NAMES,
    ENCODER_COLORS, ENCODER_LABELS, DEVICE,
    
    # Functions and classes
    MEGWordDataset, make_meg_encoder, TextEncoder, build_embeddings_for_encoder,
    nt_xent_loss, evaluate_ranking, compute_per_word_stats, get_top_words,
    save_results, plot_rank_cdf, plot_bar_metrics, plot_word_heatmap
)

# =============================================================================
#  CONFIG
# =============================================================================
OUT_DIR = "./imagined_direct_out"
TOP_K_WORDS = 20  # Focus on top words for cleaner analysis

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "models"), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "results"), exist_ok=True) 
os.makedirs(os.path.join(OUT_DIR, "comparison"), exist_ok=True)

# =============================================================================
#  TRAINING FUNCTION (modified for imagined MEG)
# =============================================================================

def train_one_encoder_imagined(
    encoder_name: str,
    meg_encoder:  nn.Module,
    text_encoder: TextEncoder,
    train_set:    Dataset,
    val_set:      Dataset,
    out_dir:      str,
) -> Tuple[nn.Module, TextEncoder, dict]:
    """Train encoder on imagined MEG data."""
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

    print(f"\n  Training [{encoder_name}] on IMAGINED MEG...")
    for epoch in range(1, N_EPOCHS + 1):
        meg_encoder.train(); text_encoder.train()
        tr_losses = []
        for meg_win, word_idx in tr_dl:
            meg_win  = meg_win.to(DEVICE)
            word_idx = word_idx.to(DEVICE)
            # Note: Less/no noise augmentation for imagined data since signal may be weaker
            meg_win  = meg_win + 0.01 * torch.randn_like(meg_win)  # Reduced from 0.02
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
    
    # Plot training curve
    _plot_training_curve_imagined(history, out_dir, encoder_name)
    return meg_encoder, text_encoder, history


def _plot_training_curve_imagined(history: dict, out_dir: str, title: str = "") -> None:
    """Plot training curve for imagined MEG training."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train"], label="train", color=ENCODER_COLORS.get(title, "#333"))
    ax.plot(history["val"],   label="val",   color=ENCODER_COLORS.get(title, "#333"), linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent loss")
    ax.set_title(f"Training curve (Imagined MEG) — {ENCODER_LABELS.get(title, title)}")
    ax.legend(); plt.tight_layout()
    path = os.path.join(out_dir, "training_curve_imagined.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
#  FILTERED EVALUATION (Top-K words only)
# =============================================================================

def create_top_words_dataset(
    dataset: MEGWordDataset,
    top_words: List[str]
) -> MEGWordDataset:
    """Create a new dataset containing only samples from top_words."""
    filtered_pairs = []
    for window, word_str in dataset.pairs:
        if word_str in top_words:
            filtered_pairs.append((window, word_str))
    
    # Create new dataset with filtered pairs
    filtered_ds = MEGWordDataset.__new__(MEGWordDataset)
    filtered_ds.pairs = filtered_pairs
    
    # Build filtered vocabulary
    filtered_vocab = {}
    for _, word_str in filtered_pairs:
        if word_str not in filtered_vocab:
            filtered_vocab[word_str] = len(filtered_vocab)
    
    filtered_ds.vocab = filtered_vocab
    filtered_ds.words = sorted(filtered_vocab, key=filtered_vocab.get)
    filtered_ds.word_audio = {w: dataset.word_audio.get(w, "") for w in filtered_ds.words}
    
    print(f"  Filtered dataset: {len(filtered_pairs)} windows, {len(filtered_vocab)} words")
    return filtered_ds


@torch.no_grad()
def evaluate_ranking_top_words(
    meg_encoder:  nn.Module,
    text_encoder: TextEncoder,
    dataset:      MEGWordDataset,
    top_words:    List[str],
    tag:          str = "",
) -> Tuple[Dict, np.ndarray, List[str]]:
    """Evaluate ranking performance on top words only - but rank against full vocabulary."""
    meg_encoder.eval(); text_encoder.eval()
    
    # Get ALL text embeddings (full vocabulary)
    all_text = text_encoder.get_all().to(DEVICE)
    full_vocab_size = all_text.shape[0]  # Should be 76

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    ranks_list = []
    word_labels = []
    
    # Process all samples, but only keep results for top words
    for meg_win, word_idx in loader:
        meg_win = meg_win.to(DEVICE)
        word_idx = word_idx.to(DEVICE)
        z_meg = meg_encoder(meg_win)
        sim = z_meg @ all_text.T  # Similarity against ALL 76 words

        for i in range(len(z_meg)):
            true_idx = word_idx[i].item()
            word_str = dataset.words[true_idx]
            
            # Only keep samples from top words
            if word_str in top_words:
                s = sim[i]
                rank = int((s > s[true_idx]).sum().item()) + 1
                ranks_list.append(rank)
                word_labels.append(word_str)

    if len(ranks_list) == 0:
        print(f"  WARNING: No samples found for top words in {tag}")
        return {}, np.array([]), []

    ranks = np.array(ranks_list, dtype=np.int32)
    metrics = {
        "tag": tag,
        "n_samples": int(len(ranks)),
        "vocab_size": int(full_vocab_size),  # Report full vocab size
        "R@1": float((ranks <= 1).mean()),
        "R@5": float((ranks <= 5).mean()),
        "R@10": float((ranks <= 10).mean()),
        "MRR": float((1.0 / ranks).mean()),
        "median_rank": int(np.median(ranks)),
        "chance_R@1": float(1.0 / full_vocab_size),
    }

    print(
        f"  [{tag}] (Top-{len(top_words)}) R@1={metrics['R@1']:.3f}  "
        f"R@5={metrics['R@5']:.3f}  R@10={metrics['R@10']:.3f}  "
        f"MRR={metrics['MRR']:.3f}  median_rank={metrics['median_rank']}/{full_vocab_size}"
    )
    return metrics, ranks, word_labels


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_imagined_direct_training(
    encoders: List[str],
    heldout_subject: str,
    mode: str = "loso",
    model_size: str = MODEL_SIZE,
) -> None:
    """Train decoders directly on imagined MEG data."""
    print("\n" + "="*60)
    print(f"  DIRECT IMAGINED MEG TRAINING — {mode.upper()}")
    if mode == "loso":
        print(f"  Heldout subject: {heldout_subject}")
    print("="*60)

    # Build dataset based on mode
    if mode == "loso":
        # Leave-one-subject-out: train on other subjects' imagined data
        other_subjects = [s for s in SUBJECTS if s != heldout_subject]
        train_subjects = other_subjects
        eval_subjects = [heldout_subject]
        print(f"  Training on {len(train_subjects)} subjects, evaluating on {heldout_subject}")
    else:  # mode == "all_subjects"
        # Train on all subjects' imagined data
        train_subjects = SUBJECTS
        eval_subjects = SUBJECTS
        print(f"  Training on all {len(train_subjects)} subjects")

    # Build training dataset from imagined MEG
    print(f"  Building imagined MEG training dataset...")
    full_ds = MEGWordDataset(
        subjects=train_subjects, 
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, 
        cond_suffix="img",  # Key difference: use imagined data
        remove_flashes=REMOVE_FLASHES,
    )

    if len(full_ds.pairs) == 0:
        print(f"  ERROR: No imagined MEG data found for subjects {train_subjects}")
        return

    # Split for training/validation
    n_val  = max(1, int(VAL_FRAC * len(full_ds)))
    n_tr   = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_tr, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_ds.vocab = full_ds.vocab
    val_ds.vocab   = full_ds.vocab
    n_channels = full_ds.pairs[0][0].shape[0]
    print(f"  train={n_tr}  val={n_val}  channels={n_channels}  vocab={len(full_ds.vocab)}")

    # Build embeddings cache
    emb_cache: dict = {}

    # Storage for results  
    val_metrics_all = {}
    val_ranks_all = {}
    val_word_labs_all = {}
    per_word_stats_all = {}

    # Train each encoder
    for enc_name in encoders:
        print(f"\n{'='*60}")
        print(f"  Encoder: {ENCODER_LABELS.get(enc_name, enc_name)} (IMAGINED)")
        print(f"{'='*60}")

        # Build embeddings
        raw_emb = build_embeddings_for_encoder(
            enc_name, full_ds.words, full_ds.word_audio, emb_cache
        )

        # Create models
        meg_enc = make_meg_encoder(n_channels, model_size)
        txt_enc = TextEncoder(raw_emb)

        n_meg  = sum(p.numel() for p in meg_enc.parameters())
        n_proj = sum(p.numel() for p in txt_enc.proj.parameters())
        print(f"  MEG encoder: {n_meg:,} params | Text proj: {n_proj:,} params")

        # Train
        model_dir = os.path.join(OUT_DIR, "models", f"{enc_name}_{mode}")
        if heldout_subject and mode == "loso":
            model_dir = os.path.join(OUT_DIR, "models", f"{enc_name}_{heldout_subject}")
        
        meg_enc, txt_enc, _ = train_one_encoder_imagined(
            enc_name, meg_enc, txt_enc,
            train_ds, val_ds, 
            out_dir=model_dir,
        )

        # Evaluate on validation set (from training data)
        print(f"\n  Evaluating on validation set...")
        val_metrics, val_ranks, val_wlabs = evaluate_ranking(
            meg_enc, txt_enc, val_ds,
            tag=f"val_imagined ({enc_name})",
            words_list=full_ds.words,
        )

        # Store results
        per_word = compute_per_word_stats(val_ranks, val_wlabs)
        val_metrics_all[enc_name] = val_metrics
        val_ranks_all[enc_name] = val_ranks
        val_word_labs_all[enc_name] = val_wlabs
        per_word_stats_all[enc_name] = per_word

    # Get top words based on validation performance
    top_words = get_top_words(per_word_stats_all, top_k=TOP_K_WORDS)
    print(f"\nTop {TOP_K_WORDS} words: {top_words}")

    # Re-evaluate on top words only for cleaner analysis
    print(f"\nRe-evaluating on top {TOP_K_WORDS} words only...")
    top_metrics_all = {}
    top_ranks_all = {}
    top_word_labs_all = {}

    for enc_name in encoders:
        # Load trained model
        model_dir = os.path.join(OUT_DIR, "models", f"{enc_name}_{mode}")
        if heldout_subject and mode == "loso":
            model_dir = os.path.join(OUT_DIR, "models", f"{enc_name}_{heldout_subject}")
            
        raw_emb = build_embeddings_for_encoder(
            enc_name, full_ds.words, full_ds.word_audio, emb_cache
        )
        meg_enc = make_meg_encoder(n_channels, model_size).to(DEVICE)
        txt_enc = TextEncoder(raw_emb).to(DEVICE)
        
        meg_enc.load_state_dict(torch.load(os.path.join(model_dir, "meg_encoder.pt"), map_location="cpu"))
        txt_enc.load_state_dict(torch.load(os.path.join(model_dir, "text_encoder.pt"), map_location="cpu"))

        # Evaluate on top words only - use full_ds instead of val_ds subset
        top_metrics, top_ranks, top_wlabs = evaluate_ranking_top_words(
            meg_enc, txt_enc, full_ds, top_words,
            tag=f"top{TOP_K_WORDS}_val_imagined ({enc_name})"
        )
        
        top_metrics_all[enc_name] = top_metrics
        top_ranks_all[enc_name] = top_ranks
        top_word_labs_all[enc_name] = top_wlabs

    # Save results
    suffix = f"_{mode}" if mode == "all_subjects" else f"_{heldout_subject}" 
    for enc_name in encoders:
        save_results(
            enc_name, f"val_imagined{suffix}", val_metrics_all[enc_name], 
            val_ranks_all[enc_name], per_word_stats_all[enc_name], OUT_DIR,
            word_labels=val_word_labs_all[enc_name], top_words=top_words
        )
        
        # Save top-words-only results
        save_results(
            enc_name, f"top{TOP_K_WORDS}_val_imagined{suffix}", top_metrics_all[enc_name],
            top_ranks_all[enc_name], {}, OUT_DIR  # No per-word stats needed for top-only
        )

    # Generate comparison plots (focus on top words)
    print("\n  Generating comparison plots...")
    comp_dir = os.path.join(OUT_DIR, "comparison")
    
    plot_rank_cdf(
        top_ranks_all, len(full_ds.vocab),  # Use full vocabulary size (76), not TOP_K_WORDS (20)
        title=f"Rank CDF — Direct Imagined MEG Training (Top {TOP_K_WORDS} Words){suffix}",
        save_path=os.path.join(comp_dir, f"rank_cdf_imagined_direct_top{TOP_K_WORDS}{suffix}.png"),
    )
    
    plot_bar_metrics(
        top_metrics_all,
        title=f"Metrics — Direct Imagined MEG (Top {TOP_K_WORDS} Words){suffix}",
        save_path=os.path.join(comp_dir, f"bar_metrics_imagined_direct_top{TOP_K_WORDS}{suffix}.png"),
    )

    # Save summary
    summary = {
        "mode": mode,
        "heldout_subject": heldout_subject if mode == "loso" else None,
        "training_approach": "direct_imagined_meg",
        "encoders": encoders,
        "top_words": top_words,
        "top_k": TOP_K_WORDS,
        "val_metrics_all_words": val_metrics_all,
        "val_metrics_top_words": top_metrics_all,
        "n_training_subjects": len(train_subjects),
        "training_subjects": train_subjects,
    }
    
    with open(os.path.join(comp_dir, f"summary_imagined_direct{suffix}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved: {os.path.join(comp_dir, f'summary_imagined_direct{suffix}.json')}")
    print(f"Results directory: {OUT_DIR}")


# =============================================================================
#  MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train contrastive decoder directly on imagined MEG responses",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--encoders", nargs="+",
        choices=ENCODER_NAMES, default=["bert"],
        help="Which encoders to train (default: bert only)",
    )
    parser.add_argument(
        "--heldout_subject", default="sub-03",
        help="Subject to hold out for LOSO evaluation (default: sub-03)"
    )
    parser.add_argument(
        "--mode", choices=["loso", "all_subjects"], default="loso",
        help="loso: leave-one-subject-out, all_subjects: train on all subjects"
    )
    parser.add_argument("--model_size", choices=["small", "full"], default=MODEL_SIZE)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"\nDevice           : {DEVICE}")
    print(f"Mode             : {args.mode}")
    if args.mode == "loso":
        print(f"Heldout subject  : {args.heldout_subject}")
    print(f"Encoders         : {args.encoders}")
    print(f"Model size       : {args.model_size}")
    print(f"Top-K words      : {TOP_K_WORDS}")
    print(f"Window           : [-{WIN_PRE_MS}ms, +{WIN_POST_MS}ms] = {WIN_SIZE} samples")
    print(f"Out dir          : {OUT_DIR}\n")

    run_imagined_direct_training(
        encoders=args.encoders,
        heldout_subject=args.heldout_subject,
        mode=args.mode,
        model_size=args.model_size,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()