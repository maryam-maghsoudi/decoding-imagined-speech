"""
probe_subject_specificity.py
============================
Diagnostic: does the trained MEGWordEncoder encode subject identity?

Two analyses:
  1. Linear probe — logistic regression on subject ID from 128-d embeddings.
     If accuracy >> chance (1/13 ≈ 7.7%), the encoder captures subject identity.

  2. t-SNE visualisation — two plots: colored by subject, colored by word
     (top-20 most frequent words).  Visual clustering by subject = bad.

  3. Intra- vs inter-subject cosine similarity — mean cosine sim within the
     same subject vs across subjects.  Large gap → subject-specific geometry.

Outputs → ./contrastive_out/subject_probe/
  linear_probe_results.json
  tsne_by_subject.png
  tsne_by_word.png
  cosine_sim_matrix.png

Usage
-----
  cd /fs/nexus-projects/brain_project/maryam_meg_dataset/imgtolis/contrastive_learning
  python probe_subject_specificity.py [--model_size small|full]
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Import everything from contrastive_word_meg (same directory)
from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, OUT_DIR,
    MEGWordDataset, TextEncoder,
    make_meg_encoder, build_text_embeddings,
    DEVICE, WIN_SIZE, TEXT_ENCODER, MODEL_SIZE,
)

PROBE_OUT = os.path.join(OUT_DIR, "subject_probe")
os.makedirs(PROBE_OUT, exist_ok=True)


# =============================================================================
#  EMBEDDING EXTRACTION — subject by subject so we have labels
# =============================================================================

def extract_embeddings(meg_encoder, model_size: str):
    """
    Build MEGWordDataset one subject at a time, encode with the trained encoder.

    Returns
    -------
    embs       : (N, 128) float32 numpy  — L2-normalised MEG embeddings
    subj_labels: (N,)     int            — subject index 0..12
    word_labels : (N,)    str            — word string
    vocab       : list[str]              — full vocabulary (from all subjects)
    """
    # First pass: get full vocab from all subjects together (needed for TextEncoder)
    print("Building full vocabulary from all subjects...")
    full_ds = MEGWordDataset(
        subjects=SUBJECTS,
        poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR,
        cond_suffix="lis",
        remove_flashes=False,
    )
    vocab = full_ds.words  # list of words in vocab-index order

    meg_encoder.eval()
    meg_encoder = meg_encoder.to(DEVICE)

    all_embs   = []
    all_subj   = []
    all_words  = []

    for s_idx, subject in enumerate(SUBJECTS):
        print(f"  Extracting embeddings: {subject}...")
        ds = MEGWordDataset(
            subjects=[subject],
            poem_keys=POEM_KEYS,
            onset_dir=ONSET_DIR,
            cond_suffix="lis",
            remove_flashes=False,
        )
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

        with torch.no_grad():
            for meg_win, word_idx in loader:
                meg_win = meg_win.to(DEVICE)
                z = meg_encoder(meg_win)          # (B, 128) L2-normed
                all_embs.append(z.cpu().numpy())
                all_subj.extend([s_idx] * len(z))

        # Collect word strings for this subject's pairs
        all_words.extend([w for _, w in ds.pairs])

    embs        = np.concatenate(all_embs, axis=0).astype(np.float32)
    subj_labels = np.array(all_subj, dtype=np.int32)
    word_labels = np.array(all_words)

    print(f"\nTotal embeddings: {len(embs)}  "
          f"subjects: {len(SUBJECTS)}  vocab: {len(vocab)}")
    return embs, subj_labels, word_labels, vocab


# =============================================================================
#  1. LINEAR PROBE — subject identity
# =============================================================================

def linear_probe(embs: np.ndarray, subj_labels: np.ndarray) -> dict:
    """
    Fit a logistic regression to predict subject ID from 128-d embeddings.
    Uses 5-fold stratified split; reports mean ± std accuracy.
    """
    print("\n--- Linear probe: subject identity ---")
    chance = 1.0 / len(SUBJECTS)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    accs = []

    for fold, (tr_idx, te_idx) in enumerate(sss.split(embs, subj_labels)):
        clf = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            multi_class="multinomial", random_state=42,
        )
        clf.fit(embs[tr_idx], subj_labels[tr_idx])
        pred = clf.predict(embs[te_idx])
        acc  = accuracy_score(subj_labels[te_idx], pred)
        accs.append(acc)
        print(f"  fold {fold+1}: acc={acc:.3f}")

    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    print(f"  Mean acc = {mean_acc:.3f} ± {std_acc:.3f}  "
          f"(chance = {chance:.3f},  ratio = {mean_acc/chance:.1f}×)")

    return {
        "mean_accuracy":    mean_acc,
        "std_accuracy":     std_acc,
        "chance_accuracy":  chance,
        "accuracy_ratio":   mean_acc / chance,
        "per_fold":         accs,
        "n_subjects":       len(SUBJECTS),
        "n_samples":        int(len(embs)),
    }


# =============================================================================
#  2. t-SNE VISUALISATION
# =============================================================================

def plot_tsne(embs: np.ndarray,
              subj_labels: np.ndarray,
              word_labels: np.ndarray) -> None:
    print("\n--- t-SNE projection ---")

    # Subsample for speed: max 5000 points
    rng = np.random.default_rng(0)
    N   = len(embs)
    if N > 5000:
        idx = rng.choice(N, 5000, replace=False)
        e   = embs[idx]; sl = subj_labels[idx]; wl = word_labels[idx]
    else:
        e, sl, wl = embs, subj_labels, word_labels

    print(f"  Running TSNE on {len(e)} points...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, n_jobs=1)
    coords = tsne.fit_transform(e)       # (N, 2)

    # ---- Panel A: colored by subject ----
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = cm.get_cmap("tab20", len(SUBJECTS))
    for s_idx, subject in enumerate(SUBJECTS):
        mask = sl == s_idx
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap(s_idx)], s=6, alpha=0.5, label=subject,
        )
    ax.legend(markerscale=3, fontsize=7, ncol=2, loc="best")
    ax.set_title("t-SNE of MEG embeddings — colored by SUBJECT\n"
                 "(clustering by subject = encoder is subject-specific)", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out = os.path.join(PROBE_OUT, "tsne_by_subject.png")
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  [saved] {out}")

    # ---- Panel B: colored by word (top-20 most frequent) ----
    counts = Counter(wl)
    top20  = [w for w, _ in counts.most_common(20)]
    cmap2  = cm.get_cmap("tab20", 20)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Grey background for all other words
    other = ~np.isin(wl, top20)
    ax.scatter(coords[other, 0], coords[other, 1],
               c="lightgrey", s=4, alpha=0.3, label="other")
    for w_idx, word in enumerate(top20):
        mask = wl == word
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[cmap2(w_idx)], s=10, alpha=0.7, label=f"{word} ({counts[word]})",
        )
    ax.legend(markerscale=2, fontsize=6, ncol=2, loc="best")
    ax.set_title("t-SNE of MEG embeddings — colored by WORD (top-20)\n"
                 "(clustering by word = encoder captures semantics)", fontsize=11)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out = os.path.join(PROBE_OUT, "tsne_by_word.png")
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  [saved] {out}")


# =============================================================================
#  3. INTRA- vs INTER-SUBJECT COSINE SIMILARITY
# =============================================================================

def cosine_similarity_analysis(embs: np.ndarray,
                                subj_labels: np.ndarray) -> dict:
    """
    For each subject, compute mean cosine similarity of its embeddings
    to (a) other embeddings of the SAME subject and (b) other subjects.
    """
    print("\n--- Intra- vs inter-subject cosine similarity ---")

    # Subsample per subject for speed: max 200 per subject
    rng = np.random.default_rng(1)
    sel_embs  = []
    sel_subjs = []
    for s in range(len(SUBJECTS)):
        idx = np.where(subj_labels == s)[0]
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        sel_embs.append(embs[idx])
        sel_subjs.extend([s] * len(idx))

    E = np.concatenate(sel_embs, axis=0)   # (M, 128)
    S = np.array(sel_subjs)

    # Full pairwise cosine sim matrix (embeddings are already L2-normed)
    sim = E @ E.T                           # (M, M)

    intra, inter = [], []
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            if S[i] == S[j]:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])

    intra_mean = float(np.mean(intra))
    inter_mean = float(np.mean(inter))
    print(f"  Intra-subject cosine sim: {intra_mean:.4f}")
    print(f"  Inter-subject cosine sim: {inter_mean:.4f}")
    print(f"  Gap (intra - inter)     : {intra_mean - inter_mean:.4f}")

    # Subject-mean similarity matrix (S x S)
    n_s = len(SUBJECTS)
    mat = np.zeros((n_s, n_s), dtype=np.float32)
    cnt = np.zeros((n_s, n_s), dtype=np.int32)
    for i in range(len(E)):
        for j in range(len(E)):
            if i != j:
                mat[S[i], S[j]] += sim[i, j]
                cnt[S[i], S[j]] += 1
    with np.errstate(invalid="ignore"):
        mat = np.where(cnt > 0, mat / cnt, 0.0)

    # Plot similarity matrix
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap="coolwarm", vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(n_s)); ax.set_yticks(range(n_s))
    ax.set_xticklabels(SUBJECTS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SUBJECTS, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mean cosine similarity")
    ax.set_title(
        f"Subject-pair mean cosine similarity in embedding space\n"
        f"Intra-subject diagonal vs off-diagonal: "
        f"{intra_mean:.3f} vs {inter_mean:.3f}  (gap={intra_mean-inter_mean:.3f})",
        fontsize=10,
    )
    plt.tight_layout()
    out = os.path.join(PROBE_OUT, "cosine_sim_matrix.png")
    plt.savefig(out, dpi=180, bbox_inches="tight"); plt.close()
    print(f"  [saved] {out}")

    return {
        "intra_subject_cosine_sim": intra_mean,
        "inter_subject_cosine_sim": inter_mean,
        "gap": intra_mean - inter_mean,
    }


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", choices=["small", "full"], default=MODEL_SIZE)
    args = parser.parse_args()

    print(f"Device     : {DEVICE}")
    print(f"Model size : {args.model_size}")
    print(f"Output dir : {PROBE_OUT}\n")

    # Load checkpoint
    ckpt_path = os.path.join(OUT_DIR, "meg_encoder.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found — run contrastive_word_meg.py --phase train first"
        )

    # Need n_channels: build a tiny one-subject dataset just to get shape
    print("Probing n_channels from one subject...")
    _tmp = MEGWordDataset(
        subjects=[SUBJECTS[0]], poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, cond_suffix="lis", remove_flashes=False,
    )
    n_channels = _tmp.pairs[0][0].shape[0]
    del _tmp
    print(f"  n_channels = {n_channels}")

    meg_enc = make_meg_encoder(n_channels, args.model_size)
    meg_enc.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    meg_enc.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Extract embeddings
    embs, subj_labels, word_labels, vocab = extract_embeddings(meg_enc, args.model_size)

    # 1. Linear probe
    probe_results = linear_probe(embs, subj_labels)

    # 2. t-SNE
    plot_tsne(embs, subj_labels, word_labels)

    # 3. Cosine similarity analysis
    cosim_results = cosine_similarity_analysis(embs, subj_labels)

    # Save summary
    summary = {**probe_results, **cosim_results}
    out_path = os.path.join(PROBE_OUT, "linear_probe_results.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[saved] {out_path}")

    # Print interpretation
    ratio = probe_results["accuracy_ratio"]
    gap   = cosim_results["gap"]
    print(f"\n=== Interpretation ===")
    print(f"  Subject probe accuracy ratio : {ratio:.1f}× chance")
    print(f"  Intra/inter cosine sim gap   : {gap:.4f}")
    if ratio > 3:
        print("  >> Embeddings strongly encode subject identity (subject-specific encoder)")
    elif ratio > 1.5:
        print("  >> Embeddings partially encode subject identity")
    else:
        print("  >> Embeddings do NOT strongly encode subject identity")
    if gap > 0.05:
        print("  >> Large intra-subject similarity gap confirms subject geometry")
    print(f"\nDone. Check {PROBE_OUT}/")


if __name__ == "__main__":
    main()
