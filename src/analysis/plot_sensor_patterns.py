"""
plot_sensor_patterns.py
=======================
For each held-out subject, produce a figure with three sections:

  1. r-topomap  – per-channel mean Pearson r at k=12 (averaged over all
                  k=12 combinations and all 40 test trials)

  2. Imagined   – grand-average (over conditions + sessions) MEG topomap
                  at every 500 ms, for the imagined conditions
                  (melody1img, melody2img, poem1img, poem2img)

  3. Listened   – same for the listened conditions
                  (melody1lis, melody2lis, poem1lis, poem2lis)

The purpose is to check whether the spatial patterns the model is exploiting
align with known features of the raw MEG signal.

Usage
-----
    python plot_sensor_patterns.py
    python plot_sensor_patterns.py --model RNN_full --step_ms 500
    python plot_sensor_patterns.py --step_ms 1000   # fewer panels
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import resample

import mne
mne.set_log_level("ERROR")


# ---------------------------------------------------------------------------
#  CONFIG  (mirrors scaling_analysis.py)
# ---------------------------------------------------------------------------
BASE_PATH  = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
COND_ALL   = [
    "melody1lis", "melody2lis", "poem1lis", "poem2lis",   # indices 0-3  → listened
    "melody1img", "melody2img", "poem1img", "poem2img",   # indices 4-7  → imagined
]
N_SESSIONS = 10
DS_FACTOR  = 10
SFREQ_DS   = 100.0   # Hz after downsampling
N_COLS     = 9       # topomaps per row in the time-series panels

# Anatomical region overlays (CTF-155, head radius ≈ 0.168 m, nose = +y)
_REGIONS = [
    dict(label="Occ.",      x= 0.005, y=-0.148, w=0.065, h=0.030, color="#2ECC71"),
    dict(label="L-Temp.",   x=-0.140, y= 0.000, w=0.020, h=0.075, color="#E67E22"),
    dict(label="R-Temp.",   x= 0.145, y= 0.000, w=0.020, h=0.075, color="#E67E22"),
    dict(label="Front.",    x= 0.000, y= 0.108, w=0.065, h=0.025, color="#3498DB"),
]


# ---------------------------------------------------------------------------
#  DATA LOADING
# ---------------------------------------------------------------------------

def load_subject_avg(subject: str):
    """
    Load and average MEG data for one subject.

    Returns
    -------
    img_avg : ndarray (C, T_ds)   grand-average over img conditions + sessions
    lis_avg : ndarray (C, T_ds)   grand-average over lis conditions + sessions
    info    : mne.Info            sensor info for topomaps
    times   : ndarray (T_ds,)     time axis in seconds
    """
    lis_trials, img_trials = [], []
    info = None

    for i_cond, cond in enumerate(COND_ALL):
        for i_ses in range(N_SESSIONS):
            fname = f"{subject}_sess-{i_ses}_task-{cond}_meg-epo.fif"
            fpath = os.path.join(BASE_PATH, subject, f"ses-{i_ses}", "meg", fname)
            if not os.path.exists(fpath):
                print(f"    [missing] {fpath}")
                continue

            ep   = mne.read_epochs(fpath, preload=True)
            if info is None:
                info  = ep.info
                tmin  = ep.tmin
                sfreq = ep.info["sfreq"]

            data = ep.get_data().mean(axis=0)          # (C, T_orig)
            new_T = data.shape[1] // DS_FACTOR
            data_ds = resample(data, new_T, axis=1).astype(np.float32)

            if i_cond < 4:   # listened
                lis_trials.append(data_ds)
            else:             # imagined
                img_trials.append(data_ds)

    lis_avg = np.stack(lis_trials).mean(axis=0)   # (C, T_ds)
    img_avg = np.stack(img_trials).mean(axis=0)

    T_ds  = lis_avg.shape[1]
    times = np.arange(T_ds) / SFREQ_DS + tmin

    return img_avg, lis_avg, info, times


# ---------------------------------------------------------------------------
#  r PER CHANNEL  at k=12
# ---------------------------------------------------------------------------

def load_r_channel(results_dir: str, subject: str, k: int = 12):
    """
    Load all r_per_trial arrays for the given k and subject,
    average over combos and trials → per-channel mean r  (C,).

    If no files exist for the requested k, falls back to the largest
    available k (useful e.g. for sub-05 which only goes up to k=11).
    Returns (r_channel_array, k_used).
    """
    r_dir = os.path.join(results_dir, f"heldout_{subject}", "r_arrays")
    files = sorted(glob.glob(os.path.join(r_dir, f"k{k:02d}_c*_r_per_trial.npy")))
    if not files:
        # fall back to highest available k
        all_files = sorted(glob.glob(os.path.join(r_dir, "k*_c*_r_per_trial.npy")))
        if not all_files:
            raise FileNotFoundError(f"No r_arrays at all for subject={subject} in {r_dir}")
        k = max(int(os.path.basename(f)[1:3]) for f in all_files)
        files = sorted(glob.glob(os.path.join(r_dir, f"k{k:02d}_c*_r_per_trial.npy")))
        print(f"    [fallback] using k={k} (requested k not available)")
    arrays = [np.load(f) for f in files]   # each (40, C)
    return np.stack(arrays).mean(axis=(0, 1)), k   # (C,), int


# ---------------------------------------------------------------------------
#  REGION OVERLAY
# ---------------------------------------------------------------------------

def _add_region_overlays(ax, labels=True):
    from matplotlib.patches import Ellipse
    for reg in _REGIONS:
        ell = Ellipse(
            xy=(reg["x"], reg["y"]),
            width=reg["w"] * 2, height=reg["h"] * 2,
            linewidth=1.2, edgecolor=reg["color"],
            facecolor="none", linestyle="--", zorder=5,
        )
        ax.add_patch(ell)
        if labels:
            lx = reg["x"]
            ly = (reg["y"] - reg["h"] - 0.010 if reg["y"] < 0
                  else reg["y"] + reg["h"] + 0.010)
            if reg["x"] < -0.05:
                lx, ly = reg["x"] - reg["w"] - 0.004, reg["y"]
            elif reg["x"] > 0.05:
                lx, ly = reg["x"] + reg["w"] + 0.004, reg["y"]
            ha = ("center" if abs(reg["x"]) < 0.05
                  else ("right" if reg["x"] < 0 else "left"))
            va = ("top" if reg["y"] < 0
                  else ("bottom" if reg["y"] > 0 else "center"))
            ax.text(lx, ly, reg["label"], color=reg["color"],
                    fontsize=5.5, fontweight="bold", ha=ha, va=va, zorder=6)


# ---------------------------------------------------------------------------
#  FIGURE BUILDER
# ---------------------------------------------------------------------------

def make_figure(subject, img_avg, lis_avg, r_ch, info, times, step_ms, out_path,
                k_used=12):
    step_s  = step_ms / 1000.0
    step_t  = int(step_ms * SFREQ_DS / 1000)          # samples per step
    t_idxs  = np.arange(0, img_avg.shape[1], step_t)  # sample indices to plot
    n_t     = len(t_idxs)
    n_rows  = int(np.ceil(n_t / N_COLS))

    # ---- colour limits ----
    # r-map: symmetric around 0, use 2-98 percentile across channels
    r_vlim = max(abs(np.percentile(r_ch, 2)), abs(np.percentile(r_ch, 98)))

    # amplitude maps: shared scale per condition (robust percentile)
    def _vlim(arr):
        v = np.percentile(np.abs(arr), 97)
        return v

    img_vlim = _vlim(img_avg)
    lis_vlim = _vlim(lis_avg)

    # ---- layout ----
    # section heights: r-map row = 2 units, each time-series section = n_rows units
    topomap_h  = 1.6   # inches per topomap row
    fig_w      = N_COLS * topomap_h + 0.5
    # height: section label + 1 r-map row + label + n_rows img + label + n_rows lis
    fig_h      = (2 + n_rows + n_rows) * topomap_h + 1.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    fig.suptitle(f"{subject}  |  RNN_full  |  k={k_used}  |  step={step_ms} ms",
                 fontsize=13, y=1.0)

    # outer grid: 3 sections
    outer = gridspec.GridSpec(3, 1, figure=fig,
                              height_ratios=[1, n_rows, n_rows],
                              hspace=0.35)

    # ---- section 1: r-map ----
    ax_r = fig.add_subplot(outer[0])
    ax_r.axis("off")
    inner_r = gridspec.GridSpecFromSubplotSpec(
        1, N_COLS, subplot_spec=outer[0], wspace=0.05)
    # place r-topomap in first cell, leave rest empty
    ax0 = fig.add_subplot(inner_r[0, 0])
    im_r, _ = mne.viz.plot_topomap(
        r_ch, info, axes=ax0, show=False,
        vlim=(-r_vlim, r_vlim), cmap="RdBu_r",
        contours=4, sensors=True,
    )
    _add_region_overlays(ax0)
    ax0.set_title(f"Mean r\n(k={k_used})", fontsize=8)
    cb_r = fig.colorbar(im_r, ax=ax0, fraction=0.07, pad=0.04, shrink=0.8)
    cb_r.set_label("r", fontsize=7)
    cb_r.ax.tick_params(labelsize=6)
    # section label
    fig.text(0.01, _section_y(outer[0], fig), "k=12  r", fontsize=10,
             va="center", ha="left", fontweight="bold", color="#555555",
             rotation=90)

    # ---- helper to plot a time-series section ----
    def _plot_section(gs_spec, data, vlim, cmap, section_label, colorbar_label):
        inner = gridspec.GridSpecFromSubplotSpec(
            n_rows, N_COLS, subplot_spec=gs_spec, wspace=0.05, hspace=0.30)
        last_im = None
        for idx, t_idx in enumerate(t_idxs):
            row, col = divmod(idx, N_COLS)
            ax = fig.add_subplot(inner[row, col])
            t_sec = times[t_idx] if t_idx < len(times) else t_idxs[-1] / SFREQ_DS
            chan_vals = data[:, t_idx]
            im, _ = mne.viz.plot_topomap(
                chan_vals, info, axes=ax, show=False,
                vlim=(-vlim, vlim), cmap=cmap,
                contours=0, sensors=False,
            )
            _add_region_overlays(ax, labels=False)
            ax.set_title(f"{t_sec:.1f}s", fontsize=6, pad=1)
            last_im = im

        # hide unused axes in last row
        for spare in range(len(t_idxs), n_rows * N_COLS):
            row, col = divmod(spare, N_COLS)
            fig.add_subplot(inner[row, col]).axis("off")

        # shared colorbar on the right of the section
        # find the rightmost axis in the last used row
        last_row_used = (len(t_idxs) - 1) // N_COLS
        last_col_used = (len(t_idxs) - 1) % N_COLS
        ax_cb = fig.add_subplot(inner[last_row_used, -1])
        cb = fig.colorbar(last_im, ax=ax_cb, fraction=0.3, pad=0.05, shrink=0.7)
        cb.set_label(colorbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)

        fig.text(0.01, _section_y(gs_spec, fig), section_label,
                 fontsize=10, va="center", ha="left", fontweight="bold",
                 color="#555555", rotation=90)

    _plot_section(outer[1], img_avg, img_vlim, "RdBu_r",
                  "Imagined", "µT (a.u.)")
    _plot_section(outer[2], lis_avg, lis_vlim, "RdBu_r",
                  "Listened", "µT (a.u.)")

    fig.subplots_adjust(left=0.04, right=0.98, top=0.97, bottom=0.01)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")


def _section_y(subplot_spec, fig):
    """Return the vertical midpoint of a GridSpec section in figure coordinates."""
    bbox = subplot_spec.get_position(fig)
    return (bbox.y0 + bbox.y1) / 2


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default="RNN_full")
    parser.add_argument("--results_root", default="scaling_out")
    parser.add_argument("--step_ms",      default=500, type=int,
                        help="Time step between topomaps in ms (default 500)")
    parser.add_argument("--k",            default=12, type=int,
                        help="Which k to use for the r-topomap (default 12)")
    args = parser.parse_args()

    results_dir = os.path.join(args.results_root, args.model)
    fig_dir     = os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # find held-out subjects that have results
    heldout_dirs = sorted(glob.glob(os.path.join(results_dir, "heldout_*")))
    subjects = [os.path.basename(d).replace("heldout_", "") for d in heldout_dirs]
    print(f"Found subjects: {subjects}")

    for subj in subjects:
        print(f"\n=== {subj} ===")
        print("  Loading MEG data …")
        try:
            img_avg, lis_avg, info, times = load_subject_avg(subj)
        except Exception as e:
            print(f"  [skip] could not load MEG data: {e}")
            continue

        print(f"  Data shape: {img_avg.shape}  ({img_avg.shape[1]/SFREQ_DS:.1f} s)")

        print(f"  Loading r arrays (k={args.k}) …")
        try:
            r_ch, k_used = load_r_channel(results_dir, subj, k=args.k)
        except Exception as e:
            print(f"  [skip] could not load r arrays: {e}")
            continue

        out_path = os.path.join(fig_dir,
                                f"sensor_patterns_{subj}_k{k_used:02d}.png")
        print(f"  Building figure ({args.step_ms} ms steps, k={k_used}) …")
        make_figure(subj, img_avg, lis_avg, r_ch, info,
                    times, args.step_ms, out_path, k_used=k_used)

    print("\nDone.")


if __name__ == "__main__":
    main()
