"""
analyze_imagined_embeddings.py
================================
Four diagnostic analyses on imagined MEG → predicted-listened → encoder embeddings.

Analyses
--------
1. RSA            Spearman ρ between 76×76 MEG-space and text-space cosine
                  similarity matrices.  Measures semantic geometry preservation
                  independently of exact word identity.

2. Category acc.  Syntactic-category soft accuracy (noun/verb/adj/adv/function).
                  Asks: even when R@1 is wrong, does the nearest neighbour share
                  the same grammatical category?

3. t-SNE          MEG embeddings colored by word label (top-N most frequent),
                  mirroring the probe_subject_specificity.py style.

4. Per-word rank  Mean rank per unique word across all occurrences, sorted bar
                  chart; reveals which words are consistently decoded vs. chance.

Usage
-----
  python analyze_imagined_embeddings.py --condition B --heldout_subject sub-01
  python analyze_imagined_embeddings.py --condition C --heldout_subject sub-01
  python analyze_imagined_embeddings.py --condition B          # all subjects

Outputs → eval_imagined_out/analysis/{condition}_{mapping}_{subject}/
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from scipy.signal import resample
from sklearn.manifold import TSNE

import mne
mne.set_log_level("ERROR")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
    TEXT_ENCODER, MODEL_SIZE,
)
from benchmark_loso import CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN
from eval_imagined_words import (
    MAPPING_DIR, LOSO_DEC_DIR, FINETUNE_DIR, GLOBAL_DEC_DIR,
    _ARCH_FACTORY, DEFAULT_MAPPING_ARCH, DEFAULT_MAPPING_MODE,
    load_mapping_model, load_decoder,
    build_combined_vocab, _probe_n_channels,
    extract_word_windows, load_and_map_session,
)

ANALYSIS_OUT = str(_HERE / "eval_imagined_out" / "analysis")
os.makedirs(ANALYSIS_OUT, exist_ok=True)

# ---------------------------------------------------------------------------
#  POS TAGGING  (syntactic categories)
# ---------------------------------------------------------------------------

def assign_pos_categories(words: List[str]) -> Dict[str, str]:
    """
    Assign each word to: noun / verb / adjective / adverb / function.
    Uses NLTK pos_tag; falls back to 'function' for unknowns.
    """
    try:
        import nltk
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger", quiet=True)
        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)

        tagged = nltk.pos_tag(words)
    except Exception:
        return {w: "function" for w in words}

    tag_map = {
        "NN": "noun",  "NNS": "noun",  "NNP": "noun",  "NNPS": "noun",
        "VB": "verb",  "VBD": "verb",  "VBG": "verb",
        "VBN": "verb", "VBP": "verb",  "VBZ": "verb",
        "JJ": "adjective", "JJR": "adjective", "JJS": "adjective",
        "RB": "adverb",    "RBR": "adverb",    "RBS": "adverb",
    }
    return {w: tag_map.get(tag, "function") for w, tag in tagged}


CAT_COLORS = {
    "noun":      "#2196F3",
    "verb":      "#E91E63",
    "adjective": "#4CAF50",
    "adverb":    "#FF9800",
    "function":  "#9E9E9E",
}
CAT_ORDER = ["noun", "verb", "adjective", "adverb", "function"]


# ---------------------------------------------------------------------------
#  EMBEDDING EXTRACTION
# ---------------------------------------------------------------------------

def _encode_windows(meg_enc, windows: List[np.ndarray]) -> np.ndarray:
    """(N, C, WIN_SIZE) list → (N, D) L2-normalised numpy."""
    if not windows:
        return np.empty((0, 128), dtype=np.float32)
    x = torch.from_numpy(np.stack(windows)).to(DEVICE)
    chunks = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            chunks.append(meg_enc(x[i:i + 256]))
    return torch.cat(chunks).cpu().numpy()


def extract_embeddings(
    heldout_subj: str,
    condition:    str,
    mapping_arch: str,
    mapping_mode: str,
    model_size:   str,
    text_method:  str,
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, int], List[str]]:
    """
    Run the full imagined→predicted-listened→encoder pipeline and collect:

    Returns
    -------
    meg_embs   : (N, D) float32  L2-normalised MEG embeddings
    word_labels: (N,)   str      word string for each embedding
    text_embs  : (V, D) float32  L2-normalised text embeddings (full vocab)
    vocab      : dict word→int
    words      : list of vocab words in index order
    """
    train_subjects = [s for s in SUBJECTS if s != heldout_subj]
    vocab, words   = build_combined_vocab(train_subjects, heldout_subj)
    n_channels     = _probe_n_channels(heldout_subj)

    decoder_cond  = "B" if condition == "C" else condition
    meg_enc, txt_enc = load_decoder(
        decoder_cond, heldout_subj, n_channels, words, model_size, text_method,
    )
    meg_enc.eval(); txt_enc.eval()

    with torch.no_grad():
        text_embs = txt_enc.get_all().cpu().numpy()   # (V, D)

    seen_folds = [s for s in SUBJECTS if s != heldout_subj]

    all_embs:  List[np.ndarray] = []
    all_words: List[str]        = []

    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
            continue
        with open(onset_file) as f:
            word_onsets = json.load(f)

        for session in range(N_SESSIONS):

            if condition != "C":
                mapping_model = load_mapping_model(
                    heldout_subj, n_channels, mapping_arch, mapping_mode,
                )
                predicted = load_and_map_session(
                    heldout_subj, poem_key, session, mapping_model,
                )
                if predicted is None:
                    continue
                wins, wds = extract_word_windows(predicted, word_onsets)
                valid     = [(w, ws) for w, ws in zip(wins, wds) if ws in vocab]
                if not valid:
                    continue
                wins_v, wds_v = zip(*valid)
                embs = _encode_windows(meg_enc, list(wins_v))
                all_embs.append(embs)
                all_words.extend(wds_v)

            else:
                # Condition C: average MEG embeddings across 12 seen models
                # (embedding-space averaging, then re-normalise)
                cond_str = f"{poem_key}img"
                fname    = (f"{heldout_subj}_sess-{session}_task-"
                            f"{cond_str}_meg-epo.fif")
                fpath    = os.path.join(
                    BASE_PATH, heldout_subj, f"ses-{session}", "meg", fname,
                )
                try:
                    epochs = mne.read_epochs(fpath, preload=True)
                except Exception:
                    continue

                raw    = epochs.get_data().mean(axis=0)
                new_T  = raw.shape[1] // DS_FACTOR
                data   = resample(raw, new_T, axis=1).astype(np.float32)
                mu     = data.mean(axis=1, keepdims=True)
                sd     = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
                data   = (data - mu) / sd
                x_img  = torch.from_numpy(data).unsqueeze(0).to(DEVICE)

                acc_emb:   Optional[np.ndarray] = None
                ref_words: Optional[List[str]]  = None
                n_seen = 0

                for fold_T in seen_folds:
                    ckpt = os.path.join(
                        MAPPING_DIR,
                        f"heldout_{fold_T}",
                        f"{mapping_arch}_{mapping_mode}.pt",
                    )
                    if not os.path.exists(ckpt):
                        continue
                    model = _ARCH_FACTORY[mapping_arch](n_channels)
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model = model.eval().to(DEVICE)

                    with torch.no_grad():
                        x_pred = model(x_img).squeeze(0).cpu().numpy()

                    del model
                    if DEVICE.type == "cuda":
                        torch.cuda.empty_cache()

                    wins, wds = extract_word_windows(x_pred, word_onsets)
                    valid = [(w, ws) for w, ws in zip(wins, wds) if ws in vocab]
                    if not valid:
                        continue
                    wins_v, wds_v = zip(*valid)
                    wds_v = list(wds_v)

                    if ref_words is None:
                        ref_words = wds_v
                    elif wds_v != ref_words:
                        continue

                    embs = _encode_windows(meg_enc, list(wins_v))  # (N, D)
                    acc_emb = embs if acc_emb is None else acc_emb + embs
                    n_seen += 1

                if acc_emb is None or n_seen == 0:
                    continue

                mean_emb = acc_emb / n_seen
                norms    = np.linalg.norm(mean_emb, axis=1, keepdims=True)
                mean_emb = mean_emb / np.maximum(norms, 1e-12)
                all_embs.append(mean_emb)
                all_words.extend(ref_words)

    if not all_embs:
        raise RuntimeError("No embeddings extracted — check data paths and onset files.")

    meg_embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    print(f"  Extracted {len(meg_embs)} embeddings  ({len(set(all_words))} unique words)")
    return meg_embs, all_words, text_embs, vocab, words


# ---------------------------------------------------------------------------
#  ANALYSIS 1 — RSA
# ---------------------------------------------------------------------------

def analysis_rsa(
    meg_embs:   np.ndarray,   # (N, D)
    word_labels: List[str],
    text_embs:  np.ndarray,   # (V, D)
    vocab:      Dict[str, int],
    words:      List[str],
    pos_map:    Dict[str, str],
    out_dir:    str,
) -> Dict:
    print("\n--- Analysis 1: RSA ---")

    # Per-word mean MEG embedding (re-normalised)
    per_word_meg = {}
    for emb, w in zip(meg_embs, word_labels):
        per_word_meg.setdefault(w, []).append(emb)

    # Only include words that appear in both meg and vocab
    common_words = sorted(
        [w for w in per_word_meg if w in vocab],
        key=lambda w: pos_map.get(w, "function"),  # group by POS for matrix sorting
    )
    if len(common_words) < 2:
        print("  Not enough words for RSA — skipping")
        return {}

    meg_centroids = np.stack([
        (lambda x: x / np.maximum(np.linalg.norm(x), 1e-12))(
            np.mean(per_word_meg[w], axis=0)
        )
        for w in common_words
    ])                                              # (K, D)
    txt_centroids = np.stack([
        text_embs[vocab[w]] for w in common_words
    ])                                              # (K, D)

    # K×K cosine similarity matrices (embeddings already L2-normed)
    meg_sim  = meg_centroids @ meg_centroids.T      # (K, K)
    text_sim = txt_centroids @ txt_centroids.T      # (K, K)

    K = len(common_words)
    triu_idx = np.triu_indices(K, k=1)
    rho, pval = spearmanr(meg_sim[triu_idx], text_sim[triu_idx])
    print(f"  RSA Spearman ρ = {rho:.4f}  (p = {pval:.4e}  n_pairs = {len(triu_idx[0])})")

    # Sort words by category then alphabetically for a cleaner matrix
    cat_order_key = {c: i for i, c in enumerate(CAT_ORDER)}
    sort_idx = sorted(
        range(K),
        key=lambda i: (cat_order_key.get(pos_map.get(common_words[i], "function"), 99),
                       common_words[i]),
    )
    common_words_s = [common_words[i] for i in sort_idx]
    meg_sim_s      = meg_sim[np.ix_(sort_idx, sort_idx)]
    text_sim_s     = text_sim[np.ix_(sort_idx, sort_idx)]

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    label_fs  = max(4, min(7, 140 // K))

    for ax, mat, title in zip(axes,
                               [meg_sim_s, text_sim_s],
                               ["MEG embedding similarity\n(imagined→predicted→encoder)",
                                "Text embedding similarity\n(BERT)"]):
        im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels(common_words_s, rotation=90, fontsize=label_fs)
        ax.set_yticklabels(common_words_s, fontsize=label_fs)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
        ax.set_title(title, fontsize=11)

        # Draw category dividers
        counts = Counter(pos_map.get(w, "function") for w in common_words_s)
        pos_order = [c for c in CAT_ORDER if counts.get(c, 0) > 0]
        cum = 0
        for cat in pos_order[:-1]:
            cum += counts[cat]
            ax.axhline(cum - 0.5, color="black", lw=0.8)
            ax.axvline(cum - 0.5, color="black", lw=0.8)

    plt.suptitle(
        f"Representational Similarity Analysis\n"
        f"Spearman ρ = {rho:.3f}  (p = {pval:.3e})  "
        f"  words sorted by syntactic category",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "rsa.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    return {"rsa_spearman_rho": float(rho), "rsa_pval": float(pval),
            "n_words": K, "n_pairs": len(triu_idx[0])}


# ---------------------------------------------------------------------------
#  ANALYSIS 2 — CATEGORY-LEVEL ACCURACY
# ---------------------------------------------------------------------------

def analysis_category_accuracy(
    meg_embs:    np.ndarray,
    word_labels: List[str],
    text_embs:   np.ndarray,
    vocab:       Dict[str, int],
    words:       List[str],
    pos_map:     Dict[str, str],
    out_dir:     str,
) -> Dict:
    print("\n--- Analysis 2: Category-level accuracy ---")

    valid = [(e, w) for e, w in zip(meg_embs, word_labels) if w in vocab]
    if not valid:
        return {}
    embs_v, words_v = zip(*valid)
    embs_v  = np.stack(embs_v)
    words_v = list(words_v)

    txt_all = text_embs                           # (V, D)
    sim     = embs_v @ txt_all.T                  # (N, V)
    nn_idx  = sim.argmax(axis=1)                  # (N,) nearest neighbour word index
    nn_words = [words[i] for i in nn_idx]

    # Exact R@1
    r1 = float(np.mean([w == nn for w, nn in zip(words_v, nn_words)]))

    # Category accuracy
    cat_match = [
        pos_map.get(w, "function") == pos_map.get(nn, "function")
        for w, nn in zip(words_v, nn_words)
    ]
    cat_acc = float(np.mean(cat_match))
    print(f"  R@1 (exact)          : {r1:.3f}")
    print(f"  Category accuracy    : {cat_acc:.3f}")

    # Per-category breakdown
    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for w, match in zip(words_v, cat_match):
        by_cat[pos_map.get(w, "function")].append(match)
    cat_detail = {c: float(np.mean(v)) for c, v in by_cat.items()}
    for c in CAT_ORDER:
        if c in cat_detail:
            n = len(by_cat[c])
            print(f"    {c:12s}: {cat_detail[c]:.3f}  (n={n})")

    # ---- Plot ----
    cats_present = [c for c in CAT_ORDER if c in cat_detail]
    accs = [cat_detail[c] for c in cats_present]
    colors = [CAT_COLORS[c] for c in cats_present]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(cats_present)), accs, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(r1,     color="navy", lw=1.5, linestyle="--",
               label=f"exact R@1 = {r1:.3f}")
    ax.axhline(cat_acc, color="black", lw=1.5, linestyle="-",
               label=f"overall cat acc = {cat_acc:.3f}")
    ax.set_xticks(range(len(cats_present)))
    ax.set_xticklabels(cats_present, fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.set_title("Category-level accuracy: nearest neighbour in same syntactic class?",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(out_dir, "category_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    return {"r1_exact": r1, "category_accuracy": cat_acc, "per_category": cat_detail}


# ---------------------------------------------------------------------------
#  ANALYSIS 3 — t-SNE
# ---------------------------------------------------------------------------

def analysis_tsne(
    meg_embs:    np.ndarray,
    word_labels: List[str],
    pos_map:     Dict[str, str],
    out_dir:     str,
    top_n:       int = 20,
) -> None:
    print(f"\n--- Analysis 3: t-SNE (top-{top_n} words) ---")

    counts  = Counter(word_labels)
    top_words = [w for w, _ in counts.most_common(top_n)]

    # Subsample for speed if needed
    rng = np.random.default_rng(SEED)
    N   = len(meg_embs)
    if N > 5000:
        idx      = rng.choice(N, 5000, replace=False)
        embs_s   = meg_embs[idx]
        labels_s = [word_labels[i] for i in idx]
    else:
        embs_s, labels_s = meg_embs, word_labels

    print(f"  Running t-SNE on {len(embs_s)} points…")
    tsne   = TSNE(n_components=2, perplexity=40, random_state=SEED, n_jobs=1)
    coords = tsne.fit_transform(embs_s)   # (N, 2)

    cmap_tab20 = cm.get_cmap("tab20", top_n)

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Grey background: words not in top-N
    other_mask = np.array([w not in top_words for w in labels_s])
    if other_mask.any():
        ax.scatter(
            coords[other_mask, 0], coords[other_mask, 1],
            c="lightgrey", s=5, alpha=0.3, label="other", rasterized=True,
        )

    # Coloured foreground: top-N words
    for w_idx, word in enumerate(top_words):
        mask = np.array([w == word for w in labels_s])
        if not mask.any():
            continue
        color = cmap_tab20(w_idx)
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], s=18, alpha=0.75,
            label=f"{word} ({counts[word]})",
            rasterized=True,
        )

    ax.legend(
        markerscale=2.5, fontsize=7, ncol=2,
        loc="upper right",
        framealpha=0.9, edgecolor="grey",
    )
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(
        f"t-SNE of MEG embeddings — colored by WORD (top-{top_n})\n"
        f"(imagined → predicted listened → contrastive encoder)",
        fontsize=12,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "tsne_by_word.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
#  ANALYSIS 4 — PER-WORD RANK BREAKDOWN
# ---------------------------------------------------------------------------

def analysis_per_word(
    meg_embs:    np.ndarray,
    word_labels: List[str],
    text_embs:   np.ndarray,
    vocab:       Dict[str, int],
    words:       List[str],
    pos_map:     Dict[str, str],
    out_dir:     str,
) -> Dict:
    print("\n--- Analysis 4: Per-word rank breakdown ---")

    valid = [(e, w) for e, w in zip(meg_embs, word_labels) if w in vocab]
    if not valid:
        return {}
    embs_v, words_v = zip(*valid)
    embs_v  = np.stack(embs_v)
    words_v = list(words_v)

    V   = text_embs.shape[0]
    sim = embs_v @ text_embs.T   # (N, V)

    per_word_ranks: Dict[str, List[int]] = defaultdict(list)
    for i, w in enumerate(words_v):
        true_idx = vocab[w]
        rank     = int((sim[i] > sim[i, true_idx]).sum()) + 1
        per_word_ranks[w].append(rank)

    # Compute summary per word
    summary = {
        w: {
            "mean_rank": float(np.mean(ranks)),
            "r1":        float(np.mean(np.array(ranks) <= 1)),
            "r5":        float(np.mean(np.array(ranks) <= 5)),
            "r10":       float(np.mean(np.array(ranks) <= 10)),
            "n":         len(ranks),
            "pos":       pos_map.get(w, "function"),
        }
        for w, ranks in per_word_ranks.items()
    }

    # Sort by mean rank (ascending = better decoded first)
    sorted_words = sorted(summary, key=lambda w: summary[w]["mean_rank"])

    mean_ranks = [summary[w]["mean_rank"] for w in sorted_words]
    colors     = [CAT_COLORS[summary[w]["pos"]] for w in sorted_words]
    chance     = V / 2.0

    print(f"  Vocab size: {V}  |  chance median rank: {V//2}")
    print(f"  Best decoded : {sorted_words[:5]}")
    print(f"  Worst decoded: {sorted_words[-5:]}")

    # ---- Plot — horizontal bar chart ----
    n_words = len(sorted_words)
    fig_h   = max(6, n_words * 0.22)
    fig, ax = plt.subplots(figsize=(10, fig_h))

    ax.barh(range(n_words), mean_ranks, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(chance, color="grey", lw=1.2, linestyle="--",
               label=f"chance = {chance:.0f}")
    ax.axvline(1, color="gold", lw=1.0, linestyle=":",
               label="rank 1")

    ax.set_yticks(range(n_words))
    ax.set_yticklabels(
        [f"{w}  (n={summary[w]['n']})" for w in sorted_words],
        fontsize=7,
    )
    ax.set_xlabel("Mean rank (lower = better decoded)", fontsize=10)
    ax.set_xlim(0, V + 2)
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")

    # Category legend
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(color=CAT_COLORS[c], label=c)
        for c in CAT_ORDER if any(summary[w]["pos"] == c for w in sorted_words)
    ]
    ax.legend(
        handles=legend_handles,
        fontsize=8, loc="lower right",
        title="POS", title_fontsize=8,
    )

    ax.set_title(
        f"Per-word mean rank  (vocab={V},  chance={chance:.0f})\n"
        f"sorted best→worst decoded  |  color = syntactic category",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(out_dir, "per_word_ranks.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    # Also print a table of top-10 and bottom-10
    print(f"\n  {'word':15s} {'pos':10s} {'mean_rank':>10} {'R@1':>6} {'R@5':>6} {'R@10':>6}")
    print(f"  {'─'*55}")
    for w in sorted_words[:10]:
        s = summary[w]
        print(f"  {w:15s} {s['pos']:10s} {s['mean_rank']:10.1f} "
              f"{s['r1']:6.3f} {s['r5']:6.3f} {s['r10']:6.3f}")
    print(f"  {'  ...':15s}")
    for w in sorted_words[-5:]:
        s = summary[w]
        print(f"  {w:15s} {s['pos']:10s} {s['mean_rank']:10.1f} "
              f"{s['r1']:6.3f} {s['r5']:6.3f} {s['r10']:6.3f}")

    return summary


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnostic analyses on imagined MEG word embeddings.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--condition", choices=["A", "A_ft", "B", "C"], default="B")
    parser.add_argument("--heldout_subject", default=None,
                        help="Single subject (e.g. sub-01). Omit for all.")
    parser.add_argument("--mapping_arch",
                        choices=["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN"],
                        default=DEFAULT_MAPPING_ARCH)
    parser.add_argument("--mapping_mode", choices=["full", "windowed"],
                        default=DEFAULT_MAPPING_MODE)
    parser.add_argument("--model_size", choices=["small", "full"], default=MODEL_SIZE)
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    parser.add_argument("--top_n", type=int, default=20,
                        help="Number of top words to highlight in t-SNE (default 20)")
    args = parser.parse_args()

    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds       = [args.heldout_subject] if args.heldout_subject else SUBJECTS
    mapping_tag = f"{args.mapping_arch}_{args.mapping_mode}"

    print(f"Condition    : {args.condition}")
    print(f"Mapping      : {mapping_tag}")
    print(f"Model size   : {args.model_size}")
    print(f"Folds        : {folds}\n")

    all_rsa_results = {}

    for subj in folds:
        print(f"\n{'='*60}")
        print(f"  Subject: {subj}")
        print(f"{'='*60}")

        tag     = f"{args.condition}_{mapping_tag}_{subj}"
        out_dir = os.path.join(ANALYSIS_OUT, tag)
        os.makedirs(out_dir, exist_ok=True)

        # Extract embeddings
        meg_embs, word_labels, text_embs, vocab, words = extract_embeddings(
            heldout_subj=subj,
            condition=args.condition,
            mapping_arch=args.mapping_arch,
            mapping_mode=args.mapping_mode,
            model_size=args.model_size,
            text_method=args.text_encoder,
        )

        # POS tags (same for all subjects — vocab is the same)
        pos_map = assign_pos_categories(words)
        pos_counts = Counter(pos_map.values())
        print(f"  POS distribution: " +
              "  ".join(f"{c}={pos_counts[c]}" for c in CAT_ORDER if c in pos_counts))

        # Run all four analyses
        rsa_res  = analysis_rsa(
            meg_embs, word_labels, text_embs, vocab, words, pos_map, out_dir,
        )
        cat_res  = analysis_category_accuracy(
            meg_embs, word_labels, text_embs, vocab, words, pos_map, out_dir,
        )
        analysis_tsne(meg_embs, word_labels, pos_map, out_dir, top_n=args.top_n)
        word_res = analysis_per_word(
            meg_embs, word_labels, text_embs, vocab, words, pos_map, out_dir,
        )

        # Save summary JSON
        summary = {
            "subject":   subj,
            "condition": args.condition,
            "mapping":   mapping_tag,
            "rsa":       rsa_res,
            "category":  cat_res,
            "per_word":  word_res,
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  [saved] {out_dir}/summary.json")

        all_rsa_results[subj] = rsa_res

    # Cross-subject RSA summary
    if len(all_rsa_results) > 1:
        rhos = [v["rsa_spearman_rho"] for v in all_rsa_results.values() if v]
        if rhos:
            print(f"\n{'='*60}")
            print(f"  RSA across subjects: ρ = {np.mean(rhos):.3f} ± {np.std(rhos):.3f}")

    print(f"\nDone. Results in {ANALYSIS_OUT}/")


if __name__ == "__main__":
    main()
