"""
subject_subspace_correction.py
================================
Diagnostic: does projecting out the subject subspace from MEG embeddings
improve LOSO word decoding?

For each LOSO fold:
  1. Load the fold's trained MEG encoder + text encoder.
  2. Extract embeddings for the TRAIN subjects → estimate subject subspace
     via PCA on per-subject mean embeddings (top-k components).
  3. Project the HELD-OUT subject's embeddings onto the orthogonal complement.
  4. Re-run word ranking; compare R@1/R@5/R@10/MRR vs. uncorrected.

The sweep over k (1..12) shows how many subject dimensions need to be removed.

Usage
-----
  cd /fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning
  python subject_subspace_correction.py [--model_size small|full] [--n_components 1,2,3,5,8,12]
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE,
    MEGWordDataset, TextEncoder,
    make_meg_encoder, build_text_embeddings,
    TEXT_ENCODER, MODEL_SIZE,
)

LOSO_OUT = "./contrastive_loso_out"
OUT_DIR  = os.path.join(LOSO_OUT, "subject_corrected")
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
#  HELPERS
# =============================================================================

def _build_combined_vocab(train_subjects, heldout_subj):
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
    return train_ds, val_ds, vocab, words


def _load_fold_models(heldout_subj, n_channels, combined_words, model_size, text_method):
    fold_dir = os.path.join(LOSO_OUT, "models", f"heldout_{heldout_subj}")
    ckpt_meg = os.path.join(fold_dir, "meg_encoder.pt")
    ckpt_txt = os.path.join(fold_dir, "text_encoder.pt")
    if not os.path.exists(ckpt_meg):
        raise FileNotFoundError(f"No checkpoint for {heldout_subj}: run contrastive_loso.py first")

    meg_enc = make_meg_encoder(n_channels, model_size).to(DEVICE)
    meg_enc.load_state_dict(torch.load(ckpt_meg, map_location="cpu"))
    meg_enc.eval()

    raw_emb = build_text_embeddings(combined_words, method=text_method)
    txt_enc = TextEncoder(raw_emb).to(DEVICE)
    txt_enc.load_state_dict(torch.load(ckpt_txt, map_location="cpu"))
    txt_enc.eval()

    return meg_enc, txt_enc


def _extract_meg_embeddings(meg_enc, ds):
    """Returns embs (N, D) float32, word_indices (N,) int."""
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
    embs, widx = [], []
    with torch.no_grad():
        for meg_win, word_idx in loader:
            z = meg_enc(meg_win.to(DEVICE))
            embs.append(z.cpu().numpy())
            widx.extend(word_idx.numpy().tolist())
    return np.concatenate(embs, axis=0).astype(np.float32), np.array(widx, dtype=np.int32)


# =============================================================================
#  SUBJECT SUBSPACE ESTIMATION & PROJECTION
# =============================================================================

def estimate_subject_subspace(embs: np.ndarray,
                               subj_labels: np.ndarray,
                               n_components: int):
    """
    Estimate the linear subspace in embedding space that encodes subject identity.

    Strategy: compute per-subject mean embeddings, subtract the global mean,
    then take the top-k left singular vectors via SVD.  These k directions
    span the subspace most responsible for separating subjects.

    Returns
    -------
    basis       : (D, k) orthonormal basis of the subject subspace
    global_mean : (D,)   mean of per-subject means (used to centre embeddings)
    """
    subjects = np.unique(subj_labels)
    per_subj_means = np.stack([
        embs[subj_labels == s].mean(axis=0) for s in subjects
    ])                                              # (S, D)
    global_mean = per_subj_means.mean(axis=0)       # (D,)
    centered    = per_subj_means - global_mean      # (S, D)

    # SVD on (S, D): right singular vectors (rows of Vt) are directions in D-space
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    k    = min(n_components, Vt.shape[0])
    basis = Vt[:k].T                                # (D, k)
    return basis, global_mean


def project_out_subspace(embs: np.ndarray,
                          basis: np.ndarray,
                          global_mean: np.ndarray) -> np.ndarray:
    """
    Remove the subject subspace from embeddings, then re-normalise to unit sphere.

      z_clean = z - basis @ basis^T @ (z - global_mean)
    """
    centered   = embs - global_mean                 # (N, D)
    projection = centered @ basis @ basis.T         # (N, D)
    cleaned    = embs - projection                  # (N, D)
    norms      = np.linalg.norm(cleaned, axis=1, keepdims=True)
    return cleaned / np.maximum(norms, 1e-12)


# =============================================================================
#  RANKING
# =============================================================================

def ranking_metrics(meg_embs: np.ndarray,
                    word_indices: np.ndarray,
                    text_embs_all: np.ndarray) -> dict:
    """
    meg_embs      : (N, D) L2-normalised
    word_indices  : (N,)   int indices into text_embs_all
    text_embs_all : (V, D) L2-normalised
    """
    sim   = meg_embs @ text_embs_all.T              # (N, V)
    ranks = []
    for i in range(len(meg_embs)):
        s    = sim[i]
        rank = int((s > s[word_indices[i]]).sum()) + 1
        ranks.append(rank)
    ranks = np.array(ranks, dtype=np.int32)
    V     = text_embs_all.shape[0]
    return {
        "R@1":         float((ranks <= 1).mean()),
        "R@5":         float((ranks <= 5).mean()),
        "R@10":        float((ranks <= 10).mean()),
        "MRR":         float((1.0 / ranks).mean()),
        "median_rank": int(np.median(ranks)),
        "vocab_size":  int(V),
        "chance_R@1":  float(1.0 / V),
    }


# =============================================================================
#  PER-FOLD EVALUATION
# =============================================================================

def run_fold(heldout_subj: str, model_size: str,
             k_values: list, text_method: str) -> dict:
    """
    Returns dict: k → {"raw": metrics, "corrected": metrics}
    """
    print(f"\n{'='*60}")
    print(f"  Held-out: {heldout_subj}")
    print(f"{'='*60}")

    train_subjects = [s for s in SUBJECTS if s != heldout_subj]

    train_ds, val_ds, vocab, words = _build_combined_vocab(train_subjects, heldout_subj)
    train_ds.vocab = vocab; train_ds.words = words
    val_ds.vocab   = vocab; val_ds.words   = words

    n_channels = train_ds.pairs[0][0].shape[0]
    meg_enc, txt_enc = _load_fold_models(
        heldout_subj, n_channels, words, model_size, text_method
    )

    with torch.no_grad():
        text_embs_all = txt_enc.get_all().cpu().numpy()  # (V, D) L2-normed

    # Held-out subject embeddings (uncorrected)
    print(f"  Extracting held-out embeddings...")
    val_embs, val_widx = _extract_meg_embeddings(meg_enc, val_ds)

    # Train subject embeddings (for subspace estimation)
    print(f"  Extracting train embeddings ({len(train_subjects)} subjects)...")
    all_train_embs, all_train_subj = [], []
    for s_idx, subj in enumerate(train_subjects):
        ds = MEGWordDataset(
            subjects=[subj], poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR, cond_suffix="lis", remove_flashes=False,
        )
        ds.vocab = vocab; ds.words = words
        e, _ = _extract_meg_embeddings(meg_enc, ds)
        all_train_embs.append(e)
        all_train_subj.extend([s_idx] * len(e))

    train_embs  = np.concatenate(all_train_embs, axis=0)
    train_subj  = np.array(all_train_subj, dtype=np.int32)

    # Uncorrected baseline
    raw_metrics = ranking_metrics(val_embs, val_widx, text_embs_all)
    print(f"  [k=0 / uncorrected]  "
          f"R@1={raw_metrics['R@1']:.3f}  R@5={raw_metrics['R@5']:.3f}  "
          f"R@10={raw_metrics['R@10']:.3f}  MRR={raw_metrics['MRR']:.3f}  "
          f"median={raw_metrics['median_rank']}/{raw_metrics['vocab_size']}")

    fold_results = {0: {"corrected": raw_metrics}}

    for k in k_values:
        basis, global_mean = estimate_subject_subspace(train_embs, train_subj, k)
        val_corr = project_out_subspace(val_embs, basis, global_mean)
        m = ranking_metrics(val_corr, val_widx, text_embs_all)
        fold_results[k] = {"corrected": m}
        print(f"  [k={k:2d}]  "
              f"R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  "
              f"R@10={m['R@10']:.3f}  MRR={m['MRR']:.3f}  "
              f"median={m['median_rank']}/{m['vocab_size']}")

    return fold_results


# =============================================================================
#  SUMMARY PLOT
# =============================================================================

def plot_summary(all_results: dict, k_values: list) -> None:
    """
    For each metric, plot mean±std across subjects as a function of k.
    k=0 = uncorrected baseline.
    """
    ks      = [0] + k_values
    metrics = ["R@1", "R@5", "R@10", "MRR"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    for ax, metric in zip(axes, metrics):
        means, stds = [], []
        for k in ks:
            vals = [all_results[subj][k]["corrected"][metric]
                    for subj in all_results]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        means = np.array(means)
        stds  = np.array(stds)
        xs    = np.arange(len(ks))

        ax.plot(xs, means, "o-", color="#2ECC71", lw=2)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.2, color="#2ECC71")
        ax.axhline(means[0], color="grey", lw=1, linestyle="--", label="uncorrected")

        # chance line for R@1
        if metric == "R@1":
            chance = list(all_results.values())[0][0]["corrected"]["chance_R@1"]
            ax.axhline(chance, color="red", lw=1, linestyle=":", label=f"chance={chance:.3f}")
            ax.legend(fontsize=8)

        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k (subject dimensions removed)")
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(bottom=0)

    plt.suptitle(
        "Effect of subject subspace removal on LOSO word decoding\n"
        "k=0 = uncorrected  |  mean ± std across all subjects",
        fontsize=10,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "subspace_correction_summary.png")
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
    parser.add_argument("--n_components", default="1,2,3,5,8,12",
                        help="Comma-separated list of k values to sweep")
    parser.add_argument("--heldout_subject", default=None,
                        help="Run a single fold (e.g. sub-01). Omit for all subjects.")
    args = parser.parse_args()

    k_values = [int(k) for k in args.n_components.split(",")]
    folds    = [args.heldout_subject] if args.heldout_subject else SUBJECTS

    print(f"Device     : {DEVICE}")
    print(f"Model size : {args.model_size}")
    print(f"k sweep    : {k_values}")
    print(f"Folds      : {folds}\n")

    all_results = {}

    for subj in folds:
        fold_results = run_fold(subj, args.model_size, k_values, args.text_encoder)
        all_results[subj] = fold_results
        with open(os.path.join(OUT_DIR, f"correction_{subj}.json"), "w") as f:
            json.dump({str(k): v for k, v in fold_results.items()}, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY  (mean across subjects)")
    print(f"{'='*60}")
    header = f"  {'k':>3}  {'R@1':>6}  {'R@5':>6}  {'R@10':>6}  {'MRR':>6}  {'med_rank':>9}"
    print(header)

    for k in [0] + k_values:
        r1   = np.mean([all_results[s][k]["corrected"]["R@1"]  for s in all_results])
        r5   = np.mean([all_results[s][k]["corrected"]["R@5"]  for s in all_results])
        r10  = np.mean([all_results[s][k]["corrected"]["R@10"] for s in all_results])
        mrr  = np.mean([all_results[s][k]["corrected"]["MRR"]  for s in all_results])
        med  = np.mean([all_results[s][k]["corrected"]["median_rank"] for s in all_results])
        print(f"  {k:>3}  {r1:6.3f}  {r5:6.3f}  {r10:6.3f}  {mrr:6.3f}  {med:9.1f}")

    if len(all_results) > 1:
        summary = {}
        for k in [0] + k_values:
            summary[str(k)] = {
                metric: {
                    "mean": float(np.mean([all_results[s][k]["corrected"][metric]
                                           for s in all_results])),
                    "std":  float(np.std( [all_results[s][k]["corrected"][metric]
                                           for s in all_results])),
                    "per_subject": {s: all_results[s][k]["corrected"][metric]
                                    for s in all_results},
                }
                for metric in ["R@1", "R@5", "R@10", "MRR", "median_rank"]
            }
        out_path = os.path.join(OUT_DIR, "correction_summary.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {out_path}")

        plot_summary(all_results, k_values)

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
