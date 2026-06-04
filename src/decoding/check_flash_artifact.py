"""
check_flash_artifact.py
=======================
Quantify how large the flash-onset artifact is relative to the background
MEG signal, to decide whether training the img->lis mapping on full
(non-flash-removed) MEG is feasible.

Two analyses:
  1. Flash ERP — average signal time-locked to each flash onset (samples
     207, 414, 621, ...).  Shows shape and duration of the artifact.

  2. RMS amplitude at flash windows vs non-flash windows — gives a
     single "how much bigger is the artifact" number.

Loops over all 10 sessions per subject and averages ERP + RMS across them.
Run with more subjects by passing --subjects sub-01,sub-03,...

Usage
-----
  python check_flash_artifact.py [--subjects sub-01]
  python check_flash_artifact.py --subjects sub-01,sub-03,sub-04
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import resample
import mne
mne.set_log_level("ERROR")

BASE_PATH = "/fs/nexus-projects/brain_project/maryam_meg_dataset/icaed"
DS_FACTOR  = 10
SFREQ_DS   = 100.0
FLASH_PERIOD = 200   # samples between flash onsets (downsampled)
FLASH_DUR    = 51    # samples removed after each flash (downsampled)
OUT_DIR    = "./contrastive_out/flash_artifact"
os.makedirs(OUT_DIR, exist_ok=True)


def load_and_downsample(subject, session, cond="poem1lis"):
    fname = f"{subject}_sess-{session}_task-{cond}_meg-epo.fif"
    fpath = os.path.join(BASE_PATH, subject, f"ses-{session}", "meg", fname)
    epochs = mne.read_epochs(fpath, preload=True)
    raw    = epochs.get_data().mean(axis=0)           # (C, T_raw)
    new_T  = raw.shape[1] // DS_FACTOR
    data   = resample(raw, new_T, axis=1).astype(np.float32)
    # z-score per channel
    mu = data.mean(axis=1, keepdims=True)
    sd = np.maximum(data.std(axis=1, keepdims=True), 1e-12)
    return (data - mu) / sd


def flash_erp(data, pre=20, post=60):
    """
    Average signal around each flash onset.
    data : (C, T)
    Returns erp (C, pre+post), times_ms (pre+post,)
    """
    n_t     = data.shape[-1]
    onsets  = np.arange(FLASH_PERIOD, n_t, FLASH_PERIOD, dtype=int)
    windows = []
    for onset in onsets:
        start = onset - pre
        end   = onset + post
        if start >= 0 and end <= n_t:
            windows.append(data[:, start:end])
    if not windows:
        return None, None
    erp    = np.mean(windows, axis=0)            # (C, pre+post)
    times  = (np.arange(pre + post) - pre) / SFREQ_DS * 1000  # ms
    return erp, times


def rms_flash_vs_baseline(data):
    """
    Compare RMS amplitude inside flash windows vs outside.
    Returns (rms_flash, rms_baseline) averaged across channels.
    """
    n_t    = data.shape[-1]
    onsets = np.arange(FLASH_PERIOD, n_t, FLASH_PERIOD, dtype=int)

    flash_mask = np.zeros(n_t, dtype=bool)
    for onset in onsets:
        flash_mask[onset: min(onset + FLASH_DUR, n_t)] = True

    rms_flash    = float(np.sqrt(np.mean(data[:, flash_mask] ** 2)))
    rms_baseline = float(np.sqrt(np.mean(data[:, ~flash_mask] ** 2)))
    return rms_flash, rms_baseline


N_SESSIONS = 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", default="sub-01",
                        help="Comma-separated list of subjects")
    parser.add_argument("--cond",     default="poem1lis")
    args = parser.parse_args()

    subjects = args.subjects.split(",")
    all_erps, all_rms_flash, all_rms_base = [], [], []

    for subject in subjects:
        subj_erps, subj_rms_flash, subj_rms_base = [], [], []

        for session in range(N_SESSIONS):
            print(f"  Loading {subject} ses-{session} {args.cond}...")
            try:
                data = load_and_downsample(subject, session, args.cond)
            except Exception as e:
                print(f"    WARNING: {e}")
                continue

            erp, times = flash_erp(data)
            if erp is not None:
                subj_erps.append(erp)

            rf, rb = rms_flash_vs_baseline(data)
            subj_rms_flash.append(rf)
            subj_rms_base.append(rb)

        if not subj_erps:
            print(f"  No data for {subject}, skipping.")
            continue

        # Average across sessions for this subject
        mean_subj_erp = np.mean(subj_erps, axis=0)
        mean_rf = float(np.mean(subj_rms_flash))
        mean_rb = float(np.mean(subj_rms_base))

        all_erps.append(mean_subj_erp)
        all_rms_flash.append(mean_rf)
        all_rms_base.append(mean_rb)
        print(f"  {subject}: RMS flash={mean_rf:.4f}  baseline={mean_rb:.4f}  "
              f"ratio={mean_rf/mean_rb:.2f}x  (mean over {len(subj_erps)} sessions)")

    if not all_erps:
        print("No data loaded.")
        return

    # ---- Plot 1: Flash ERP (mean across subjects, top-10 channels by ERP power) ----
    mean_erp = np.mean(all_erps, axis=0)      # (C, T)
    erp_power = np.mean(mean_erp ** 2, axis=1)
    top_ch    = np.argsort(erp_power)[-10:]   # 10 channels with largest flash response

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    ax = axes[0]
    for ch in top_ch:
        ax.plot(times, mean_erp[ch], alpha=0.6, lw=1)
    ax.axvline(0, color="red", lw=1.5, linestyle="--", label="flash onset")
    ax.axvspan(0, FLASH_DUR / SFREQ_DS * 1000, alpha=0.15, color="red",
               label=f"removed window ({FLASH_DUR} samples = {FLASH_DUR*10}ms)")
    ax.set_xlabel("Time from flash onset (ms)")
    ax.set_ylabel("z-scored amplitude")
    ax.set_title(f"Flash ERP — top-10 channels by response power\n"
                 f"subjects: {subjects}  cond: {args.cond}  (mean over all sessions)",
                 fontsize=10)
    ax.legend(fontsize=8)

    # ---- Plot 2: RMS ratio across subjects ----
    ax = axes[1]
    x = np.arange(len(subjects[:len(all_rms_flash)]))
    ax.bar(x - 0.2, all_rms_flash, 0.35, label="Flash windows", color="#E74C3C", alpha=0.8)
    ax.bar(x + 0.2, all_rms_base,  0.35, label="Baseline",      color="#2ECC71", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects[:len(all_rms_flash)], rotation=45, ha="right")
    ax.set_ylabel("Mean RMS amplitude (z-scored)")
    ax.set_title("RMS amplitude: flash windows vs baseline", fontsize=10)
    ax.legend(fontsize=8)

    mean_rf = np.mean(all_rms_flash)
    mean_rb = np.mean(all_rms_base)
    ax.text(0.98, 0.95,
            f"Mean ratio: {mean_rf/mean_rb:.2f}×",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "flash_artifact.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\n[saved] {out}")

    print(f"\nSummary:")
    print(f"  Mean RMS flash    : {mean_rf:.4f}")
    print(f"  Mean RMS baseline : {mean_rb:.4f}")
    print(f"  Ratio             : {mean_rf/mean_rb:.2f}x")
    if mean_rf / mean_rb > 2.0:
        print("  >> Large artifact — removing flashes likely necessary for the mapping model")
    elif mean_rf / mean_rb > 1.3:
        print("  >> Moderate artifact — mapping on full MEG may work but will be noisier")
    else:
        print("  >> Small artifact — mapping on full MEG should be fine")


if __name__ == "__main__":
    main()
