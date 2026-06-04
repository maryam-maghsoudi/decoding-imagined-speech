"""
plot_scaling_results.py
=======================
Load and visualise results from scaling_analysis.py.

Outputs (saved to scaling_out/RNN_full/figures/):
  01_learning_curve.png       – mean r vs k per subject + grand mean ± std
  02_condition_curves.png     – learning curve broken down by condition
  03_trial_distributions.png  – violin plots of r across trials × combos per k
  04_topomap_mean_r.png       – per-channel mean r topomap at selected k values
  05_subject_heatmap.png      – heatmap: rows=subjects, cols=k, cell=mean r
  06_combo_variance.png       – between-combo std vs k (reliability of estimates)

Usage
-----
    python plot_scaling_results.py
    python plot_scaling_results.py --model RNN_full
    python plot_scaling_results.py --model RNN_full --results_root scaling_out
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import mne
mne.set_log_level("ERROR")


# ---------------------------------------------------------------------------
#  CONFIG
# ---------------------------------------------------------------------------
BASE_PATH   = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
COND_BASE   = ["melody1", "melody2", "poem1", "poem2"]
N_SESSIONS  = 10
# 40 trials → 4 conditions × 10 sessions
COND_SLICE  = {cb: slice(i * N_SESSIONS, (i + 1) * N_SESSIONS)
               for i, cb in enumerate(COND_BASE)}

SUBJECT_COLORS = plt.cm.tab10(np.linspace(0, 0.9, 13))


# ---------------------------------------------------------------------------
#  HELPERS
# ---------------------------------------------------------------------------

def collect_results(results_dir: str):
    """
    Returns
    -------
    subj_k_mean : dict[subj] -> dict[k] -> float   (mean r, avg over combos)
    subj_k_std  : dict[subj] -> dict[k] -> float   (std over combos)
    subj_k_trials: dict[subj] -> dict[k] -> ndarray(n_combos, 40)
    subj_k_cond : dict[subj] -> dict[k] -> dict[cond] -> float
    """
    subj_k_mean   = {}
    subj_k_std    = {}
    subj_k_trials = {}   # per combo: all 40 trial r values
    subj_k_cond   = {}

    heldout_dirs = sorted(glob.glob(os.path.join(results_dir, "heldout_*")))
    for hd in heldout_dirs:
        subj = os.path.basename(hd).replace("heldout_", "")
        subj_k_mean[subj]   = {}
        subj_k_std[subj]    = {}
        subj_k_trials[subj] = {}
        subj_k_cond[subj]   = {}

        json_files = sorted(glob.glob(os.path.join(hd, "k*_results.json")))
        for jf in json_files:
            k = int(os.path.basename(jf)[1:3])
            with open(jf) as f:
                records = json.load(f)

            all_mean_r        = np.array([r["mean_r"] for r in records])
            all_trial_r       = np.array([r["mean_r_per_trial"] for r in records])  # (n_combos, 40)

            subj_k_mean[subj][k]   = float(np.mean(all_mean_r))
            subj_k_std[subj][k]    = float(np.std(all_mean_r))
            subj_k_trials[subj][k] = all_trial_r   # (n_combos, 40)

            subj_k_cond[subj][k] = {}
            for cb, sl in COND_SLICE.items():
                subj_k_cond[subj][k][cb] = float(np.mean(all_trial_r[:, sl]))

    return subj_k_mean, subj_k_std, subj_k_trials, subj_k_cond


def load_channel_info(subject="sub-01", session=0, cond="melody1img"):
    """Load MNE Info object to get sensor positions for topomaps."""
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
    epochs = mne.read_epochs(fpath, preload=False)
    return epochs.info


def collect_channel_r(results_dir: str, k_values=None):
    """
    For each k, accumulate mean per-channel r across all subjects and combos.

    Returns
    -------
    k_channel_r : dict[k] -> ndarray(n_channels,)
    """
    heldout_dirs = sorted(glob.glob(os.path.join(results_dir, "heldout_*")))
    k_channel_r_accum = {}   # k -> list of (n_combos, 40, C) chunks
    k_channel_r_count = {}

    for hd in heldout_dirs:
        r_arr_dir = os.path.join(hd, "r_arrays")
        if not os.path.isdir(r_arr_dir):
            continue
        npy_files = sorted(glob.glob(os.path.join(r_arr_dir, "*.npy")))
        for nf in npy_files:
            bn = os.path.basename(nf)           # k01_c0000_r_per_trial.npy
            k  = int(bn[1:3])
            if k_values is not None and k not in k_values:
                continue
            arr = np.load(nf)                   # (40, C)
            if k not in k_channel_r_accum:
                k_channel_r_accum[k] = []
            k_channel_r_accum[k].append(arr)    # append (40, C)

    k_channel_r = {}
    for k, arrays in k_channel_r_accum.items():
        stacked = np.stack(arrays)              # (n_arrays, 40, C)
        k_channel_r[k] = stacked.mean(axis=(0, 1))  # (C,)

    return k_channel_r


# ---------------------------------------------------------------------------
#  PLOT 1 – LEARNING CURVE
# ---------------------------------------------------------------------------

def plot_learning_curve(subj_k_mean, subj_k_std, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    subjects = sorted(subj_k_mean.keys())
    grand_k  = sorted({k for s in subjects for k in subj_k_mean[s]})

    all_r_by_k = {k: [] for k in grand_k}
    for i, subj in enumerate(subjects):
        ks   = sorted(subj_k_mean[subj].keys())
        rs   = [subj_k_mean[subj][k] for k in ks]
        ax.plot(ks, rs, "-o", color=SUBJECT_COLORS[i], alpha=0.7,
                linewidth=1.5, markersize=5, label=subj)
        for k, r in zip(ks, rs):
            all_r_by_k[k].append(r)

    # grand mean ± 1 std
    gk   = [k for k in grand_k if all_r_by_k[k]]
    gm   = [np.mean(all_r_by_k[k]) for k in gk]
    gs   = [np.std(all_r_by_k[k])  for k in gk]
    ax.plot(gk, gm, "k-o", linewidth=2.5, markersize=7, label="Grand mean", zorder=5)
    ax.fill_between(gk, np.array(gm) - np.array(gs),
                        np.array(gm) + np.array(gs),
                    color="black", alpha=0.15, label="±1 SD")

    ax.set_xlabel("Training subjects (k)", fontsize=12)
    ax.set_ylabel("Mean Pearson r  (img→lis)", fontsize=12)
    ax.set_title("Scaling curve – RNN_full", fontsize=13)
    ax.set_xticks(grand_k)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
#  PLOT 2 – PER-CONDITION LEARNING CURVE
# ---------------------------------------------------------------------------

def plot_condition_curves(subj_k_cond, out_path):
    subjects = sorted(subj_k_cond.keys())
    grand_k  = sorted({k for s in subjects for k in subj_k_cond[s]})
    cond_colors = {"melody1": "#E63946", "melody2": "#457B9D",
                   "poem1":   "#2A9D8F", "poem2":   "#E9C46A"}

    fig, axes = plt.subplots(1, len(COND_BASE), figsize=(14, 4), sharey=True)
    for ax, cb in zip(axes, COND_BASE):
        all_r_by_k = {k: [] for k in grand_k}
        for i, subj in enumerate(subjects):
            ks = sorted(subj_k_cond[subj].keys())
            rs = [subj_k_cond[subj][k].get(cb, np.nan) for k in ks]
            ax.plot(ks, rs, "-o", color=SUBJECT_COLORS[i], alpha=0.5,
                    linewidth=1, markersize=4)
            for k, r in zip(ks, rs):
                if not np.isnan(r):
                    all_r_by_k[k].append(r)

        gk = [k for k in grand_k if all_r_by_k[k]]
        gm = [np.mean(all_r_by_k[k]) for k in gk]
        gs = [np.std(all_r_by_k[k])  for k in gk]
        ax.plot(gk, gm, "-o", color=cond_colors[cb], linewidth=2.5,
                markersize=7, zorder=5, label="Grand mean")
        ax.fill_between(gk, np.array(gm) - np.array(gs),
                            np.array(gm) + np.array(gs),
                        color=cond_colors[cb], alpha=0.2)
        ax.set_title(cb, fontsize=11)
        ax.set_xlabel("k", fontsize=10)
        ax.set_xticks(grand_k)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean Pearson r", fontsize=10)
    fig.suptitle("Per-condition scaling curves – RNN_full", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
#  PLOT 3 – VIOLIN / DISTRIBUTION PLOTS
# ---------------------------------------------------------------------------

def plot_trial_distributions(subj_k_trials, out_path):
    subjects = sorted(subj_k_trials.keys())
    grand_k  = sorted({k for s in subjects for k in subj_k_trials[s]})

    # Pool all combo×trial r values across subjects per k
    pooled = {k: [] for k in grand_k}
    for subj in subjects:
        for k in grand_k:
            if k in subj_k_trials[subj]:
                arr = subj_k_trials[subj][k]   # (n_combos, 40)
                pooled[k].extend(arr.ravel().tolist())

    fig, ax = plt.subplots(figsize=(12, 5))
    data    = [pooled[k] for k in grand_k]
    parts   = ax.violinplot(data, positions=grand_k, widths=0.7,
                            showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4C72B0")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)

    # overlay per-subject grand-mean points
    for i, subj in enumerate(subjects):
        ks = sorted(subj_k_trials[subj].keys())
        rs = [subj_k_trials[subj][k].mean() for k in ks]
        ax.scatter(ks, rs, color=SUBJECT_COLORS[i], s=40, zorder=5,
                   label=subj, edgecolors="k", linewidths=0.5)

    ax.set_xlabel("Training subjects (k)", fontsize=12)
    ax.set_ylabel("Pearson r  (img→lis)", fontsize=12)
    ax.set_title("Distribution of r across trials × combos – RNN_full", fontsize=13)
    ax.set_xticks(grand_k)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
#  PLOT 4 – TOPOMAPS
# ---------------------------------------------------------------------------

# Anatomical region definitions in 2D topomap space.
# Coordinates are in metres (same units as plot_topomap axes: head radius ≈ 0.168 m).
# Derived from CTF-155 sensor positions for this dataset (nose = top, +y).
#   x, y  = ellipse centre
#   w, h  = half-axes
_REGIONS = [
    dict(label="Occipital",   x= 0.005, y=-0.148, w=0.065, h=0.030, color="#2ECC71"),
    dict(label="L-Temporal",  x=-0.140, y= 0.000, w=0.020, h=0.075, color="#E67E22"),
    dict(label="R-Temporal",  x= 0.145, y= 0.000, w=0.020, h=0.075, color="#E67E22"),
    dict(label="Frontal",     x= 0.000, y= 0.108, w=0.065, h=0.025, color="#3498DB"),
]


def _add_region_overlays(ax):
    """Draw labelled ellipses marking anatomical regions on a topomap axis.

    Coordinates are in the same metre-scale units that mne.viz.plot_topomap uses
    (head circle radius ≈ 0.168 m for this CTF-155 dataset).
    """
    from matplotlib.patches import Ellipse
    for reg in _REGIONS:
        ell = Ellipse(
            xy=(reg["x"], reg["y"]),
            width=reg["w"] * 2,
            height=reg["h"] * 2,
            angle=0,
            linewidth=1.8,
            edgecolor=reg["color"],
            facecolor="none",
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(ell)
        # place label just outside the ellipse, still inside/on the head circle
        lx = reg["x"]
        ly = reg["y"] - (reg["h"] + 0.012) if reg["y"] < 0 else reg["y"] + (reg["h"] + 0.012)
        if reg["x"] < -0.05:   # left temporal: label to the left
            lx = reg["x"] - reg["w"] - 0.005
            ly = reg["y"]
        elif reg["x"] > 0.05:  # right temporal: label to the right
            lx = reg["x"] + reg["w"] + 0.005
            ly = reg["y"]
        ha = "center" if abs(reg["x"]) < 0.05 else ("right" if reg["x"] < 0 else "left")
        va = "top"    if reg["y"] < 0 else ("bottom" if reg["y"] > 0 else "center")
        ax.text(lx, ly, reg["label"], color=reg["color"],
                fontsize=7.5, fontweight="bold", ha=ha, va=va, zorder=6)


def plot_topomaps(k_channel_r, info, out_path, k_show=None):
    """Plot per-channel mean r as topomaps for selected k values."""
    all_k = sorted(k_channel_r.keys())
    if k_show is None:
        # pick ~4 evenly spaced k values
        step = max(1, len(all_k) // 4)
        k_show = all_k[::step][:4]
        if all_k[-1] not in k_show:
            k_show.append(all_k[-1])

    fig, axes = plt.subplots(1, len(k_show), figsize=(4 * len(k_show), 4.5))
    if len(k_show) == 1:
        axes = [axes]

    # shared colour scale
    all_vals = np.concatenate([k_channel_r[k] for k in k_show])
    vmin, vmax = np.percentile(all_vals, [2, 98])
    vlim = max(abs(vmin), abs(vmax))

    for ax, k in zip(axes, k_show):
        ch_r = k_channel_r[k]
        im, _ = mne.viz.plot_topomap(
            ch_r, info, axes=ax, show=False,
            vlim=(-vlim, vlim), cmap="RdBu_r",
            contours=6, sensors=True
        )
        _add_region_overlays(ax)
        ax.set_title(f"k={k}", fontsize=12)

    # shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                        fraction=0.03, pad=0.04)
    cbar.set_label("Mean Pearson r", fontsize=10)
    fig.suptitle("Per-channel mean r (avg over trials, combos, subjects)\n"
                 "RNN_full", fontsize=12)
    fig.subplots_adjust(top=0.85)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
#  PLOT 5 – SUBJECT × K HEATMAP
# ---------------------------------------------------------------------------

def plot_heatmap(subj_k_mean, out_path):
    subjects = sorted(subj_k_mean.keys())
    all_k    = sorted({k for s in subjects for k in subj_k_mean[s]})
    mat      = np.full((len(subjects), len(all_k)), np.nan)
    for i, subj in enumerate(subjects):
        for j, k in enumerate(all_k):
            if k in subj_k_mean[subj]:
                mat[i, j] = subj_k_mean[subj][k]

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis",
                   vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    ax.set_xticks(range(len(all_k)))
    ax.set_xticklabels([str(k) for k in all_k])
    ax.set_yticks(range(len(subjects)))
    ax.set_yticklabels(subjects, fontsize=9)
    ax.set_xlabel("Training subjects (k)", fontsize=11)
    ax.set_ylabel("Held-out subject", fontsize=11)
    ax.set_title("Mean Pearson r – RNN_full", fontsize=12)

    for i in range(len(subjects)):
        for j in range(len(all_k)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] < np.nanmean(mat) else "black")

    fig.colorbar(im, ax=ax, label="Mean r")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
#  PLOT 6 – COMBO VARIANCE (reliability of k-subject estimates)
# ---------------------------------------------------------------------------

def plot_combo_variance(subj_k_std, out_path):
    subjects = sorted(subj_k_std.keys())
    grand_k  = sorted({k for s in subjects for k in subj_k_std[s]})

    fig, ax = plt.subplots(figsize=(9, 4))
    for i, subj in enumerate(subjects):
        ks   = sorted(subj_k_std[subj].keys())
        stds = [subj_k_std[subj][k] for k in ks]
        ax.plot(ks, stds, "-o", color=SUBJECT_COLORS[i], alpha=0.7,
                linewidth=1.5, markersize=5, label=subj)

    # grand mean std
    all_std_by_k = {k: [] for k in grand_k}
    for subj in subjects:
        for k, v in subj_k_std[subj].items():
            all_std_by_k[k].append(v)
    gk  = [k for k in grand_k if all_std_by_k[k]]
    gm  = [np.mean(all_std_by_k[k]) for k in gk]
    ax.plot(gk, gm, "k-o", linewidth=2.5, markersize=7, label="Grand mean", zorder=5)

    ax.set_xlabel("Training subjects (k)", fontsize=12)
    ax.set_ylabel("Std of mean r across combos", fontsize=12)
    ax.set_title("Between-combo variance – RNN_full\n"
                 "(lower = more reliable estimates at that k)", fontsize=11)
    ax.set_xticks(grand_k)
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="RNN_full")
    parser.add_argument("--results_root", default="scaling_out")
    args = parser.parse_args()

    results_dir = os.path.join(args.results_root, args.model)
    fig_dir     = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ---- load JSON results
    print("Loading JSON results …")
    subj_k_mean, subj_k_std, subj_k_trials, subj_k_cond = collect_results(results_dir)

    subjects = sorted(subj_k_mean.keys())
    all_k    = sorted({k for s in subjects for k in subj_k_mean[s]})
    print(f"  Subjects: {subjects}")
    print(f"  k values: {all_k}")

    # ---- plot 1: learning curve
    plot_learning_curve(subj_k_mean, subj_k_std,
                        os.path.join(fig_dir, "01_learning_curve.png"))

    # ---- plot 2: per-condition curves
    plot_condition_curves(subj_k_cond,
                          os.path.join(fig_dir, "02_condition_curves.png"))

    # ---- plot 3: violin distributions
    plot_trial_distributions(subj_k_trials,
                             os.path.join(fig_dir, "03_trial_distributions.png"))

    # ---- plot 5: heatmap (before topomaps so we can fail gracefully)
    plot_heatmap(subj_k_mean,
                 os.path.join(fig_dir, "05_subject_heatmap.png"))

    # ---- plot 6: combo variance
    plot_combo_variance(subj_k_std,
                        os.path.join(fig_dir, "06_combo_variance.png"))

    # ---- plot 4: topomaps (requires .fif files → optional)
    try:
        print("Loading channel info for topomaps …")
        info = load_channel_info()
        # pick 4 k values: min, ~1/3, ~2/3, max
        n = len(all_k)
        k_show = sorted({all_k[0],
                         all_k[n // 3],
                         all_k[2 * n // 3],
                         all_k[-1]})
        print(f"  Loading r_arrays for k in {k_show} …")
        k_channel_r = collect_channel_r(results_dir, k_values=k_show)
        plot_topomaps(k_channel_r, info,
                      os.path.join(fig_dir, "04_topomap_mean_r.png"),
                      k_show=k_show)
    except Exception as e:
        print(f"  [skip topomaps] {e}")

    print(f"\nAll figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
