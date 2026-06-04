"""
plot_recall_at_k.py
====================
Plot R@k vs k curves for one or more conditions, with chance overlay.

For each condition, collects raw word ranks from the full imagined MEG
pipeline across all (or selected) subjects, then plots mean R@k for
k = 1 … K_MAX alongside the chance line k/V.

Usage
-----
  python plot_recall_at_k.py --conditions B
  python plot_recall_at_k.py --conditions A B C --heldout_subject sub-01
  python plot_recall_at_k.py --conditions A A_ft_pred B C
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
mne.set_log_level("ERROR")

_HERE  = Path(__file__).parent.resolve()
_BENCH = _HERE.parent / "benchmark" / "no_flash_removal"
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH))

from contrastive_word_meg import (
    SUBJECTS, POEM_KEYS, ONSET_DIR, DEVICE, SEED,
    WIN_SIZE, TEXT_ENCODER, MODEL_SIZE,
)
from eval_imagined_words import (
    DEFAULT_MAPPING_ARCH, DEFAULT_MAPPING_MODE,
    MAPPING_DIR,
    build_combined_vocab, _probe_n_channels, load_decoder,
    load_mapping_model,
    extract_word_windows, load_and_map_session,
    score_avg_ensemble_session,
)

OUT_DIR = str(_HERE / "eval_imagined_out")
os.makedirs(OUT_DIR, exist_ok=True)

K_MAX = 76   # plot k = 1 … K_MAX

CONDITION_STYLES = {
    "A":         dict(color="#E74C3C", linestyle="-",  label="A  (LOSO dec, LOSO map)"),
    "A_ft":      dict(color="#E67E22", linestyle="--", label="A_ft  (LOSO dec FT real lis)"),
    "A_ft_pred": dict(color="#9B59B6", linestyle="-.", label="A_ft_pred  (LOSO dec FT predicted)"),
    "B":         dict(color="#2ECC71", linestyle="-",  label="B  (global dec, LOSO map)"),
    "C":         dict(color="#3498DB", linestyle="-",  label="C  (global dec, ensemble map)"),
}


# ---------------------------------------------------------------------------
#  RAW RANK COLLECTION
# ---------------------------------------------------------------------------

@torch.no_grad()
def _ranks_from_windows(
    meg_enc,
    windows:    List[np.ndarray],
    word_strs:  List[str],
    vocab:      Dict[str, int],
    all_text:   torch.Tensor,        # (V, D) on DEVICE
) -> np.ndarray:
    """Encode windows, rank each against full vocab, return int array of ranks."""
    valid = [(w, ws) for w, ws in zip(windows, word_strs) if ws in vocab]
    if not valid:
        return np.array([], dtype=np.int32)
    wins_v, wds_v = zip(*valid)

    x = torch.from_numpy(np.stack(wins_v)).to(DEVICE)
    z_chunks = []
    for i in range(0, len(x), 256):
        z_chunks.append(meg_enc(x[i:i + 256]))
    z   = torch.cat(z_chunks)          # (N, D)
    sim = z @ all_text.T               # (N, V)

    ranks = []
    for i, word in enumerate(wds_v):
        s = sim[i]
        ranks.append(int((s > s[vocab[word]]).sum().item()) + 1)
    return np.array(ranks, dtype=np.int32)


def collect_ranks_one_fold(
    heldout_subj: str,
    condition:    str,
    arch:         str,
    mode:         str,
    model_size:   str,
    text_method:  str,
) -> Optional[np.ndarray]:
    """
    Returns 1-D int array of per-word ranks for this fold/condition,
    or None if the checkpoint doesn't exist.
    """
    train_subjects = [s for s in SUBJECTS if s != heldout_subj]
    vocab, words   = build_combined_vocab(train_subjects, heldout_subj)
    n_channels     = _probe_n_channels(heldout_subj)
    seen_folds     = train_subjects

    decoder_condition = "B" if condition == "C" else condition
    try:
        meg_enc, txt_enc = load_decoder(
            decoder_condition, heldout_subj, n_channels, words,
            model_size, text_method,
        )
    except FileNotFoundError as e:
        print(f"    [{condition}] skipping {heldout_subj}: {e}")
        return None

    meg_enc.eval(); txt_enc.eval()
    all_text = txt_enc.get_all().to(DEVICE)   # (V, D)

    all_ranks = []

    for poem_key in POEM_KEYS:
        onset_file = os.path.join(ONSET_DIR, f"{poem_key}_word_onsets.json")
        if not os.path.exists(onset_file):
            continue
        with open(onset_file) as f:
            word_onsets = json.load(f)

        for session in range(
            __import__("contrastive_word_meg").N_SESSIONS
        ):
            if condition != "C":
                try:
                    map_model = load_mapping_model(
                        heldout_subj, n_channels, arch, mode,
                    )
                except FileNotFoundError:
                    continue

                predicted = load_and_map_session(
                    heldout_subj, poem_key, session, map_model,
                )
                del map_model
                if predicted is None:
                    continue

                wins, wds = extract_word_windows(predicted, word_onsets)
                r = _ranks_from_windows(meg_enc, wins, wds, vocab, all_text)

            else:
                mean_sim, ref_words = score_avg_ensemble_session(
                    heldout_subj, poem_key, session,
                    seen_folds, n_channels, arch, mode,
                    meg_enc, all_text, word_onsets, vocab,
                )
                if mean_sim is None:
                    continue
                V = all_text.shape[0]
                r = []
                for i, word in enumerate(ref_words):
                    if word not in vocab:
                        continue
                    true_idx = vocab[word]
                    r.append(int((mean_sim[i] > mean_sim[i, true_idx]).sum()) + 1)
                r = np.array(r, dtype=np.int32)

            if len(r):
                all_ranks.append(r)

    if not all_ranks:
        return None
    return np.concatenate(all_ranks)


# ---------------------------------------------------------------------------
#  RECALL @ K
# ---------------------------------------------------------------------------

def recall_at_k_curve(ranks: np.ndarray, k_max: int) -> np.ndarray:
    """Returns array of length k_max: R@k for k=1..k_max."""
    return np.array([(ranks <= k).mean() for k in range(1, k_max + 1)])


def auc(curve: np.ndarray) -> float:
    """Normalised AUC of R@k curve (trapezoid, divided by k_max so in [0,1])."""
    return float(np.trapz(curve) / len(curve))


def permutation_pvalue(
    ranks:   np.ndarray,
    k_max:   int,
    V:       int,
    n_perms: int = 200,
    rng:     Optional[np.random.Generator] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Permutation test for AUC significance.

    Null distribution: for each permutation, randomly shuffle which word label
    each window is assigned to.  Because we don't store the full (N, V)
    similarity matrix, we approximate this by sampling null ranks uniformly
    from {1..V} — the correct null when word occurrences are balanced and
    similarity scores are exchangeable under label permutation.

    Returns (real_auc, p_value, null_aucs, null_curves).
    p_value = fraction of null AUCs >= real AUC.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    real_curve = recall_at_k_curve(ranks, k_max)
    real_auc   = auc(real_curve)

    N = len(ranks)
    null_aucs = np.empty(n_perms)
    null_curves = np.empty((n_perms, k_max), dtype=np.float64)
    for i in range(n_perms):
        null_ranks = rng.integers(1, V + 1, size=N)
        null_curve = recall_at_k_curve(null_ranks, k_max)
        null_curves[i] = null_curve
        null_aucs[i] = auc(null_curve)

    p_val = float((null_aucs >= real_auc).sum() / n_perms)
    return real_auc, p_val, null_aucs, null_curves


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+",
                        choices=list(CONDITION_STYLES), default=["B"])
    parser.add_argument("--heldout_subject", default=None)
    parser.add_argument("--mapping_arch",
                        choices=["CNN1D", "ShallowMLP", "UNet1D", "RNN", "TCN", "LinearLag"],
                        default=DEFAULT_MAPPING_ARCH)
    parser.add_argument("--mapping_mode", choices=["full", "windowed"],
                        default=DEFAULT_MAPPING_MODE)
    parser.add_argument("--model_size", choices=["small", "full"],
                        default=MODEL_SIZE)
    parser.add_argument("--text_encoder", choices=["bert", "glove", "random"],
                        default=TEXT_ENCODER)
    parser.add_argument("--k_max", type=int, default=K_MAX)
    parser.add_argument("--n_perms", type=int, default=20)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    folds = [args.heldout_subject] if args.heldout_subject else SUBJECTS

    # Collect ranks per condition per subject
    # condition_ranks[cond][subj] = np.ndarray of ranks
    condition_ranks: Dict[str, Dict[str, np.ndarray]] = {}

    for cond in args.conditions:
        print(f"\n{'='*55}  condition={cond}")
        condition_ranks[cond] = {}
        for subj in folds:
            print(f"  {subj} …")
            r = collect_ranks_one_fold(
                subj, cond,
                args.mapping_arch, args.mapping_mode,
                args.model_size, args.text_encoder,
            )
            if r is not None and len(r):
                condition_ranks[cond][subj] = r
                print(f"    n={len(r)}  median={int(np.median(r))}  "
                      f"R@1={float((r<=1).mean()):.3f}  "
                      f"R@10={float((r<=10).mean()):.3f}")

    if not any(condition_ranks[c] for c in condition_ranks):
        print("Nothing to plot.")
        return

    V = 76
    if args.k_max > V:
        raise ValueError(f"--k_max ({args.k_max}) cannot exceed vocab size V={V}")

    k_vals = np.arange(1, args.k_max + 1)
    chance = k_vals / V

    rng    = np.random.default_rng(SEED)
    n_conds = len(args.conditions)
    fig, axes = plt.subplots(1, n_conds, figsize=(7 * n_conds, 6),
                             sharey=True, squeeze=False)
    curves_payload = {
        "meta": {
            "conditions": args.conditions,
            "heldout_subject": args.heldout_subject,
            "mapping_arch": args.mapping_arch,
            "mapping_mode": args.mapping_mode,
            "model_size": args.model_size,
            "text_encoder": args.text_encoder,
            "seed": SEED,
            "n_perms": args.n_perms,
            "k_max": args.k_max,
            "vocab_size": V,
        },
        "k_values": k_vals.tolist(),
        "chance_curve": chance.tolist(),
        "conditions": {},
    }

    print(f"\n{'─'*65}")
    print(f"  Permutation test  (n_perms={args.n_perms}  k=1..{args.k_max}  V={V})")
    print(f"  {'subject':10s}  {'condition':12s}  {'AUC':>7}  {'null_AUC':>10}  {'p':>6}")
    print(f"{'─'*65}")

    for ax, cond in zip(axes[0], args.conditions):
        style      = CONDITION_STYLES[cond]
        subj_ranks = condition_ranks[cond]
        cond_payload = {
            "label": style["label"],
            "subject_curves": {},
        }

        all_curves = []
        for subj, ranks in subj_ranks.items():
            curve     = recall_at_k_curve(ranks, args.k_max)
            real_auc, p_val, null_aucs, null_curves = permutation_pvalue(
                ranks, args.k_max, V, n_perms=args.n_perms, rng=rng,
            )
            all_curves.append(curve)
            cond_payload["subject_curves"][subj] = {
                "n_ranks": int(len(ranks)),
                "median_rank": float(np.median(ranks)),
                "ranks": ranks.tolist(),
                "curve": curve.tolist(),
                "auc": real_auc,
                "p_value": p_val,
                "null_aucs": null_aucs.tolist(),
                "null_curves": null_curves.tolist(),
            }

            # significance marker
            if p_val == 0:
                sig = f"p<{1/args.n_perms:.2f} ***"
            elif p_val <= 0.05:
                sig = f"p={p_val:.2f} *"
            else:
                sig = f"p={p_val:.2f}"

            print(f"  {subj:10s}  {cond:12s}  "
                  f"{real_auc:7.4f}  "
                  f"{null_aucs.mean():10.4f}  "
                  f"{p_val:6.2f}  {sig}")

            alpha = 0.55 if p_val <= 0.05 else 0.25
            ax.plot(k_vals, curve,
                    color=style["color"], alpha=alpha, linewidth=1.2)

        # Mean across subjects
        if all_curves:
            mean_curve = np.mean(all_curves, axis=0)
            ax.plot(k_vals, mean_curve,
                    color=style["color"], linewidth=2.8,
                    linestyle=style["linestyle"],
                    label=f"mean (n={len(all_curves)})")
            cond_payload["mean_curve"] = mean_curve.tolist()
        else:
            cond_payload["mean_curve"] = None

        curves_payload["conditions"][cond] = cond_payload

        ax.plot(k_vals, chance,
                color="black", linestyle=":", linewidth=1.4, label="chance")

        ax.set_title(style["label"], fontsize=11)
        ax.set_xlabel("k", fontsize=12)
        if ax is axes[0][0]:
            ax.set_ylabel("R@k", fontsize=12)
        ax.set_xlim(1, args.k_max)
        ax.set_ylim(0, None)
        ax.set_xticks([1] + list(range(5, args.k_max + 1, 5)))
        ax.grid(axis="y", alpha=0.3)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], fontsize=9, framealpha=0.9)

    tag      = "_".join(args.conditions)
    subj     = args.heldout_subject or "all"
    arch_tag = f"{args.mapping_arch}_{args.mapping_mode}" if args.mapping_arch != "LinearLag" \
               else "LinearLag"
    plt.suptitle(
        f"Recall@k — imagined MEG word decoding\n"
        f"mapping={arch_tag}  thin lines=subjects  thick=mean",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"recall_at_k_{tag}_{subj}_{arch_tag}.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {path}")

    curves_path = os.path.join(OUT_DIR, f"recall_at_k_{tag}_{subj}_{arch_tag}_curves.json")
    with open(curves_path, "w") as f:
        json.dump(curves_payload, f, indent=2)
    print(f"[saved] {curves_path}")

    # ---- Print summary table ----
    print(f"\n{'k':>4}  " + "  ".join(f"{c:>12}" for c in args.conditions) +
          f"  {'chance':>8}")
    for k in [1, 2, 3, 5, 10, 15, 20, 25, 30, 38, 50, 60, 70, 76]:
        if k > args.k_max:
            continue
        row = f"{k:>4}  "
        for cond in args.conditions:
            subj_ranks = condition_ranks[cond]
            if subj_ranks:
                all_r = np.concatenate(list(subj_ranks.values()))
                row += f"{float((all_r <= k).mean()):>12.3f}  "
            else:
                row += f"{'—':>12}  "
        row += f"{k/V:>8.3f}"
        print(row)


if __name__ == "__main__":
    main()
