"""
tsne_compare.py
===============
Side-by-side t-SNE of real listened vs predicted listened MEG embeddings.

For each condition the SAME decoder encodes both real and predicted signals,
so differences in cluster structure reflect mapping quality, not the decoder.

Both point clouds are embedded JOINTLY (concatenated before t-SNE) so the
coordinate spaces are directly comparable within each figure.

Conditions
----------
  A         LOSO decoder (no FT)
  A_ft      LOSO decoder FT on real listened
  A_ft_pred LOSO decoder FT on predicted listened
  B         Global decoder
  C         Global decoder + score-avg ensemble (12 mapping models)

Usage
-----
  python tsne_compare.py --heldout_subject sub-01
  python tsne_compare.py --heldout_subject sub-01 --conditions A B C
  python tsne_compare.py                          # all subjects, all conditions
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from scipy.signal import resample
from sklearn.manifold import TSNE

import mne
mne.set_log_level("ERROR")

_HERE  = Path(__file__).parent.resolve()
_BENCH = _HERE.parent / "benchmark" / "no_flash_removal"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH))

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE, SEED,
    BASE_PATH, DS_FACTOR, N_SESSIONS, WIN_SIZE,
    MEGWordDataset, TextEncoder, make_meg_encoder,
    build_text_embeddings, onset_to_window_raw,
    TEXT_ENCODER, MODEL_SIZE,
)
from benchmark_loso import CNN1D, ShallowMLP, UNet1D, RNN, TCN, TARGET_HIDDEN
from eval_imagined_words import (
    MAPPING_DIR, LOSO_DEC_DIR, FINETUNE_DIR, FINETUNE_PRED_DIR, GLOBAL_DEC_DIR,
    _ARCH_FACTORY, DEFAULT_MAPPING_ARCH, DEFAULT_MAPPING_MODE,
    build_combined_vocab, _probe_n_channels, load_decoder,
    extract_word_windows, load_and_map_session, score_avg_ensemble_session,
)

OUT_ROOT = str(_HERE / "eval_imagined_out" / "tsne_compare")
os.makedirs(OUT_ROOT, exist_ok=True)

ALL_CONDITIONS = ["A", "A_ft", "A_ft_pred", "B", "C"]


# ---------------------------------------------------------------------------
#  ENCODING HELPERS
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_windows(meg_enc: torch.nn.Module, windows: List[np.ndarray]) -> np.ndarray:
    if not windows:
        return np.empty((0, 128), dtype=np.float32)
    x = torch.from_numpy(np.stack(windows)).to(DEVICE)
    chunks = []
    for i in range(0, len(x), 256):
        chunks.append(meg_enc(x[i:i + 256]))
    return torch.cat(chunks).cpu().numpy()


# ---------------------------------------------------------------------------
#  REAL LISTENED EMBEDDINGS  (heldout subject, same decoder as condition)
# ---------------------------------------------------------------------------

def extract_real_embeddings(
    meg_enc:      torch.nn.Module,
    heldout_subj: str,
    vocab:        Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Load heldout subject's real listened MEG windows and encode them.
    Uses MEGWordDataset (no flash removal, same as training).
    """
    ds = MEGWordDataset(
        subjects=[heldout_subj], poem_keys=POEM_KEYS,
        onset_dir=ONSET_DIR, cond_suffix="lis", remove_flashes=False,
    )
    if not ds.pairs:
        return np.empty((0, 128), dtype=np.float32), []

    windows   = [p[0] for p in ds.pairs if p[1] in vocab]
    word_strs = [p[1] for p in ds.pairs if p[1] in vocab]

    embs = encode_windows(meg_enc, windows)
    return embs, word_strs


# ---------------------------------------------------------------------------
#  PREDICTED LISTENED EMBEDDINGS  (imagined → mapping → decoder)
# ---------------------------------------------------------------------------

def extract_predicted_embeddings(
    meg_enc:      torch.nn.Module,
    heldout_subj: str,
    condition:    str,
    arch:         str,
    mode:         str,
    n_channels:   int,
    vocab:        Dict[str, int],
) -> Tuple[np.ndarray, List[str]]:
    """
    Map heldout subject's imagined MEG to predicted listened, then encode.
    Handles conditions A/A_ft/A_ft_pred (single mapping model) and
    C (score-avg ensemble → we take the single-model path here for embeddings;
    for pure visualization consistency we average embedding vectors across models).
    """
    seen_folds = [s for s in SUBJECTS if s != heldout_subj]
    all_embs, all_words = [], []

    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
            continue
        with open(onset_file) as f:
            word_onsets = json.load(f)

        for session in range(N_SESSIONS):

            if condition != "C":
                ckpt_map = os.path.join(
                    MAPPING_DIR, f"heldout_{heldout_subj}", f"{arch}_{mode}.pt"
                )
                if not os.path.exists(ckpt_map):
                    continue
                map_model = _ARCH_FACTORY[arch](n_channels)
                map_model.load_state_dict(torch.load(ckpt_map, map_location="cpu"))
                map_model = map_model.eval().to(DEVICE)

                predicted = load_and_map_session(
                    heldout_subj, poem_key, session, map_model,
                )
                del map_model
                if predicted is None:
                    continue

                wins, wds = extract_word_windows(predicted, word_onsets)
                valid = [(w, ws) for w, ws in zip(wins, wds) if ws in vocab]
                if not valid:
                    continue
                wins_v, wds_v = zip(*valid)
                embs = encode_windows(meg_enc, list(wins_v))
                all_embs.append(embs)
                all_words.extend(wds_v)

            else:
                # Condition C: average MEG embeddings across 12 seen mapping models
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

                raw   = epochs.get_data().mean(axis=0)
                new_T = raw.shape[1] // DS_FACTOR
                data  = resample(raw, new_T, axis=1).astype(np.float32)
                mu    = data.mean(axis=1, keepdims=True)
                sd    = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
                data  = (data - mu) / sd
                x_img = torch.from_numpy(data).unsqueeze(0).to(DEVICE)

                acc_emb, ref_words, n_seen = None, None, 0
                for fold_T in seen_folds:
                    ckpt = os.path.join(
                        MAPPING_DIR, f"heldout_{fold_T}", f"{arch}_{mode}.pt"
                    )
                    if not os.path.exists(ckpt):
                        continue
                    model = _ARCH_FACTORY[arch](n_channels)
                    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                    model = model.eval().to(DEVICE)
                    with torch.no_grad():
                        x_pred = model(x_img).squeeze(0).cpu().numpy()
                    del model

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

                    embs = encode_windows(meg_enc, list(wins_v))
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
        return np.empty((0, 128), dtype=np.float32), []

    return np.concatenate(all_embs, axis=0).astype(np.float32), list(all_words)


# ---------------------------------------------------------------------------
#  PLOT
# ---------------------------------------------------------------------------

def plot_tsne_compare(
    real_embs:   np.ndarray,   # (N_r, D)
    real_words:  List[str],
    pred_embs:   np.ndarray,   # (N_p, D)
    pred_words:  List[str],
    condition:   str,
    heldout_subj: str,
    arch:        str,
    mode:        str,
    out_dir:     str,
    top_n:       int = 20,
) -> None:
    """
    Run t-SNE jointly on real + predicted embeddings, then plot side-by-side.
    Both panels share the same top-N word coloring derived from the combined
    word frequency across real and predicted.
    """
    if len(real_embs) == 0 or len(pred_embs) == 0:
        print(f"  [{condition}] skipping — no embeddings")
        return

    # Subsample independently if too large
    rng = np.random.default_rng(SEED)
    MAX_PER_SIDE = 3000

    def _subsample(embs, words, n):
        if len(embs) <= n:
            return embs, words
        idx = rng.choice(len(embs), n, replace=False)
        return embs[idx], [words[i] for i in idx]

    real_embs, real_words = _subsample(real_embs, real_words, MAX_PER_SIDE)
    pred_embs, pred_words = _subsample(pred_embs, pred_words, MAX_PER_SIDE)

    n_real = len(real_embs)
    n_pred = len(pred_embs)

    # Top-N words from combined frequency
    combined_counts = Counter(real_words + pred_words)
    top_words = [w for w, _ in combined_counts.most_common(top_n)]

    print(f"  [{condition}] t-SNE on {n_real} real + {n_pred} predicted points…")
    all_embs = np.concatenate([real_embs, pred_embs], axis=0)
    tsne     = TSNE(n_components=2, perplexity=40, random_state=SEED, n_jobs=1)
    coords   = tsne.fit_transform(all_embs)
    real_xy  = coords[:n_real]
    pred_xy  = coords[n_real:]

    cmap = cm.get_cmap("tab20", top_n)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.patch.set_facecolor("white")

    titles = [
        f"Real listened MEG\n({heldout_subj})",
        f"Predicted listened MEG\n({heldout_subj}  cond={condition}  {arch}_{mode})",
    ]

    for ax, xy, words, title in zip(
        axes,
        [real_xy, pred_xy],
        [real_words, pred_words],
        titles,
    ):
        ax.set_facecolor("white")
        ax.set_title(title, fontsize=12)

        # Grey background — words not in top-N
        other_mask = np.array([w not in top_words for w in words])
        if other_mask.any():
            ax.scatter(
                xy[other_mask, 0], xy[other_mask, 1],
                c="lightgrey", s=6, alpha=0.3, rasterized=True, label="other",
            )

        # Colored foreground — top-N words
        word_counts = Counter(words)
        for w_idx, word in enumerate(top_words):
            mask = np.array([w == word for w in words])
            if not mask.any():
                continue
            ax.scatter(
                xy[mask, 0], xy[mask, 1],
                c=[cmap(w_idx)], s=20, alpha=0.8, rasterized=True,
                label=f"{word} ({combined_counts[word]})",
            )

        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)

    # Shared legend on the right panel
    axes[1].legend(
        markerscale=2.5, fontsize=7, ncol=2,
        loc="upper right", framealpha=0.9, edgecolor="grey",
    )

    plt.suptitle(
        f"t-SNE: real listened vs predicted listened  —  colored by WORD (top-{top_n})\n"
        f"Jointly embedded  |  condition={condition}  |  same decoder for both panels",
        fontsize=12,
    )
    plt.tight_layout()

    path = os.path.join(out_dir, "tsne_compare.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
#  PER-CONDITION RUNNER
# ---------------------------------------------------------------------------

def run_condition(
    heldout_subj: str,
    condition:    str,
    arch:         str,
    mode:         str,
    model_size:   str,
    text_method:  str,
    top_n:        int,
) -> None:
    tag     = f"{condition}_{arch}_{mode}_{heldout_subj}"
    out_dir = os.path.join(OUT_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)

    train_subjects = [s for s in SUBJECTS if s != heldout_subj]
    vocab, words   = build_combined_vocab(train_subjects, heldout_subj)
    n_channels     = _probe_n_channels(heldout_subj)

    # Load the decoder appropriate for this condition
    # (C uses same decoder as B — global)
    decoder_condition = "B" if condition == "C" else condition
    try:
        meg_enc, _ = load_decoder(
            decoder_condition, heldout_subj, n_channels, words,
            model_size, text_method,
        )
    except FileNotFoundError as e:
        print(f"  [{condition}] skipping — {e}")
        return

    meg_enc.eval()

    print(f"\n  [{condition}] extracting real listened embeddings…")
    real_embs, real_words = extract_real_embeddings(meg_enc, heldout_subj, vocab)
    print(f"    {len(real_embs)} windows  ({len(set(real_words))} unique words)")

    print(f"  [{condition}] extracting predicted listened embeddings…")
    pred_embs, pred_words = extract_predicted_embeddings(
        meg_enc, heldout_subj, condition, arch, mode, n_channels, vocab,
    )
    print(f"    {len(pred_embs)} windows  ({len(set(pred_words))} unique words)")

    plot_tsne_compare(
        real_embs, real_words,
        pred_embs, pred_words,
        condition, heldout_subj, arch, mode,
        out_dir, top_n=top_n,
    )


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Side-by-side t-SNE: real listened vs predicted listened.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--heldout_subject", default=None,
                        help="Single subject (e.g. sub-01). Omit for all.")
    parser.add_argument("--conditions", nargs="+",
                        choices=ALL_CONDITIONS, default=ALL_CONDITIONS,
                        help="Which conditions to plot (default: all 5)")
    parser.add_argument("--mapping_arch",
                        choices=["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN"],
                        default=DEFAULT_MAPPING_ARCH)
    parser.add_argument("--mapping_mode", choices=["full", "windowed"],
                        default=DEFAULT_MAPPING_MODE)
    parser.add_argument("--model_size", choices=["small", "full"],
                        default=MODEL_SIZE)
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    parser.add_argument("--top_n", type=int, default=20,
                        help="Words to highlight in t-SNE (default 20)")
    args = parser.parse_args()

    import torch
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds = [args.heldout_subject] if args.heldout_subject else SUBJECTS

    print(f"Conditions   : {args.conditions}")
    print(f"Mapping      : {args.mapping_arch}_{args.mapping_mode}")
    print(f"Model size   : {args.model_size}")
    print(f"Folds        : {folds}\n")

    for subj in folds:
        print(f"\n{'='*60}")
        print(f"  Subject: {subj}")
        print(f"{'='*60}")
        for cond in args.conditions:
            run_condition(
                subj, cond,
                args.mapping_arch, args.mapping_mode,
                args.model_size, args.text_encoder,
                args.top_n,
            )

    print(f"\nDone. Results in {OUT_ROOT}/")


if __name__ == "__main__":
    main()
