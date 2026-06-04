"""
compare_results_v2.py
=====================
Analysis and visualisation of benchmark_archs_v2.py results.

Reproduces all plots from compare_r_values.py (v1) and adds v2-specific
plots for the per-subject correlation-based classification and confusion
matrices.

Outputs (saved next to this script)
------------------------------------
  --- r / MSE / accuracy (same as v1) ---
  r_comparison_boxplot.png      box+strip: per-fold r per model
  r_comparison_lineplot.png     per-fold r trajectories
  r_rankplot.png                models ranked by mean r
  r_pairwise_pvalues.png        Wilcoxon p-value heatmap
  r_stats_table.txt             ranked text table
  comparison_barplot.png        grouped bar: r, MSE, global acc, per-subj acc
  per_fold_lineplots.png        per-fold line plots (4 metrics)
  windowed_vs_full_diff.png     windowed − full Δ per neural model (4 metrics)

  --- per-subject classification (v2 additions) ---
  ps_acc_vs_global_acc.png      scatter: per-subj acc vs global acc per model
  ps_acc_heatmap.png            heatmap: subjects × models, mean acc over folds
  ps_acc_boxplot.png            box+strip: per-fold per-subj acc distribution
  agg_cm_grid.png               normalised aggregate CM for every model
  per_subj_cm_best_model.png    per-subject CMs for the best-r model
  per_subj_cm_grid_{model}.png  per-subject CM grid for every model
"""

import json
import os
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wilcoxon, ttest_rel

# ---------------------------------------------------------------------------
#  Paths & data
# ---------------------------------------------------------------------------
HERE       = os.path.dirname(os.path.abspath(__file__))
N_FOLDS    = 5
FOLD_FILES = [os.path.join(HERE, f"fold_{k}_results.json") for k in range(1, N_FOLDS + 1)]
SUMMARY_F  = os.path.join(HERE, "summary_metrics.json")

with open(SUMMARY_F) as f:
    summary = json.load(f)

fold_data = []
for fp in FOLD_FILES:
    with open(fp) as f:
        fold_data.append(json.load(f))

MODEL_KEYS   = list(summary.keys())
SUBJECTS     = list(fold_data[0][MODEL_KEYS[0]]["per_subject_clf"]["per_subject"].keys())
N_CLASSES    = 4
COND_LABELS  = ["melody1", "melody2", "poem1", "poem2"]
NEURAL_NAMES = ["ShallowMLP", "CNN1D", "UNet1D", "RNN", "TCN"]
WIN_MS, STRIDE_MS         = 1000, 500
TRAIN_SESSIONS, TEST_SESSIONS = 8, 2
M = len(MODEL_KEYS)

# ---------------------------------------------------------------------------
#  Derived arrays
# ---------------------------------------------------------------------------
# (M, N_FOLDS) r matrix
r_matrix    = np.array([summary[k]["per_fold_mean_r"]      for k in MODEL_KEYS])
mse_matrix  = np.array([summary[k]["per_fold_mse"]         for k in MODEL_KEYS])
acc_matrix  = np.array([summary[k]["per_fold_clf_acc"]     for k in MODEL_KEYS])
ps_matrix   = np.array([summary[k]["per_fold_ps_clf_acc"]  for k in MODEL_KEYS])

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def display_name(key):
    name, mode = key.rsplit("_", 1)
    return f"{name}\n({'full' if mode == 'full' else 'win'})"

arch_colors = {
    "LinearLag": "#4e79a7",
    "ShallowMLP": "#f28e2b",
    "CNN1D":      "#e15759",
    "UNet1D":     "#76b7b2",
    "RNN":        "#59a14f",
    "TCN":        "#af7aa1",
}

def key_color(key):
    return arch_colors.get(key.rsplit("_", 1)[0], "#888888")

labels = [display_name(k) for k in MODEL_KEYS]
colors = [key_color(k)    for k in MODEL_KEYS]

cmap20   = matplotlib.colormaps["tab20"]
colors20 = [cmap20(i / max(M - 1, 1)) for i in range(M)]

def save(fname):
    out = os.path.join(HERE, fname)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out}")


# ===========================================================================
#  SECTION 1 — r / MSE / accuracy  (mirrors compare_r_values.py)
# ===========================================================================

# ---------------------------------------------------------------------------
#  1a. Box + strip: per-fold r
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
for i, key in enumerate(MODEL_KEYS):
    vals = r_matrix[i]
    ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=colors[i], alpha=0.55),
               medianprops=dict(color="black", lw=1.5),
               whiskerprops=dict(color=colors[i]),
               capprops=dict(color=colors[i]),
               flierprops=dict(marker="o", markerfacecolor=colors[i], markersize=4))
    jitter = np.random.default_rng(i).uniform(-0.12, 0.12, size=len(vals))
    ax.scatter(i + jitter, vals, color=colors[i], zorder=3, s=30, alpha=0.85)
    ax.text(i, vals.max() + 3e-4, f"{vals.mean():.4f}",
            ha="center", va="bottom", fontsize=7)

ax.axhline(0, color="grey", lw=0.8, linestyle="--")
ax.set_xticks(range(M)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Mean Pearson r"); ax.set_xlim(-0.6, M - 0.4)
ax.set_title("Per-fold mean Pearson r — all models and training modes")
plt.tight_layout(); save("r_comparison_boxplot.png")

# ---------------------------------------------------------------------------
#  1b. Per-fold line plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for i, key in enumerate(MODEL_KEYS):
    mode = key.rsplit("_", 1)[1]
    ax.plot(range(1, N_FOLDS + 1), r_matrix[i],
            marker="o", linestyle="-" if mode == "full" else "--",
            color=colors[i], alpha=0.9 if mode == "full" else 0.65,
            lw=1.5, ms=5, label=f"{key.rsplit('_',1)[0]} ({mode})")
ax.axhline(0, color="grey", lw=0.8, linestyle=":")
ax.set_xticks(range(1, N_FOLDS + 1)); ax.set_xlabel("Fold")
ax.set_ylabel("Mean Pearson r")
ax.set_title("Per-fold Pearson r trajectories")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout(); save("r_comparison_lineplot.png")

# ---------------------------------------------------------------------------
#  1c. Ranked bar chart
# ---------------------------------------------------------------------------
means = r_matrix.mean(axis=1)
stds  = r_matrix.std(axis=1)
order = np.argsort(means)[::-1]

fig, ax = plt.subplots(figsize=(13, 5))
bars = ax.bar(range(M), means[order], yerr=stds[order], capsize=4,
              color=[colors[i] for i in order], alpha=0.85)
for bar, i in zip(bars, order):
    ax.text(bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + stds[i] + 3e-4,
            f"{means[i]:.4f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(range(M)); ax.set_xticklabels([labels[i] for i in order], fontsize=8)
ax.set_ylabel("Mean Pearson r (mean ± std, 5 folds)")
ax.set_title("Models ranked by mean Pearson r")
ax.axhline(0, color="grey", lw=0.8, linestyle="--")
plt.tight_layout(); save("r_rankplot.png")

# ---------------------------------------------------------------------------
#  1d. Pairwise Wilcoxon p-value heatmap
# ---------------------------------------------------------------------------
pval_matrix = np.ones((M, M))
for i, j in itertools.combinations(range(M), 2):
    a, b = r_matrix[i], r_matrix[j]
    try:
        _, p = wilcoxon(a, b) if not np.allclose(a, b) else (None, 1.0)
    except Exception:
        _, p = ttest_rel(a, b)
    pval_matrix[i, j] = pval_matrix[j, i] = p

short_labels = [k.replace("_", "\n") for k in MODEL_KEYS]
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(pval_matrix, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
plt.colorbar(im, ax=ax, fraction=0.03, label="p-value (Wilcoxon)")
ax.set_xticks(range(M)); ax.set_xticklabels(short_labels, fontsize=7, rotation=45, ha="right")
ax.set_yticks(range(M)); ax.set_yticklabels(short_labels, fontsize=7)
ax.set_title("Pairwise Wilcoxon p-values for mean Pearson r\n(5 folds — indicative only)")
for i in range(M):
    for j in range(M):
        if i == j: continue
        p = pval_matrix[i, j]
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ax.text(j, i, f"{p:.2f}\n{stars}" if stars else f"{p:.2f}",
                ha="center", va="center", fontsize=6,
                color="white" if p < 0.2 else "black")
plt.tight_layout(); save("r_pairwise_pvalues.png")

# ---------------------------------------------------------------------------
#  1e. Text summary table
# ---------------------------------------------------------------------------
ci_half = 1.96 * stds / np.sqrt(N_FOLDS)
lines = [
    f"{'Rank':<5} {'Model':<25} {'Mean r':>8} {'Std r':>8} {'95% CI':>18} {'Min r':>8} {'Max r':>8}",
    "-" * 82,
]
for rank, i in enumerate(order, 1):
    lo, hi = means[i] - ci_half[i], means[i] + ci_half[i]
    lines.append(f"{rank:<5} {MODEL_KEYS[i]:<25} {means[i]:>8.5f} {stds[i]:>8.5f} "
                 f"[{lo:>7.5f}, {hi:>7.5f}]  {r_matrix[i].min():>8.5f} {r_matrix[i].max():>8.5f}")
lines += ["", "Per-fold r values:",
          f"{'Model':<25} " + "  ".join(f"Fold{k+1:>2}" for k in range(N_FOLDS)),
          "-" * 65]
for i in order:
    lines.append(f"{MODEL_KEYS[i]:<25} " + "  ".join(f"{v:>7.5f}" for v in r_matrix[i]))
table_str = "\n".join(lines)
print("\n" + table_str)
with open(os.path.join(HERE, "r_stats_table.txt"), "w") as f:
    f.write(table_str + "\n")
print(f"[saved] {os.path.join(HERE, 'r_stats_table.txt')}")

# ---------------------------------------------------------------------------
#  1f. Grouped bar chart — 4 metrics (adds per-subj acc vs v1)
# ---------------------------------------------------------------------------
bench_metrics = [
    ("per_fold_mean_r",      "mean_r_mean",     "mean_r_std",     "Mean Pearson r ↑"),
    ("per_fold_mse",         "mse_mean",        "mse_std",        "MSE ↓"),
    ("per_fold_clf_acc",     "clf_acc_mean",     "clf_acc_std",    "Global 4-class acc ↑"),
    ("per_fold_ps_clf_acc",  "ps_clf_acc_mean",  "ps_clf_acc_std", "Per-subj 4-class acc ↑"),
]
fig, axes = plt.subplots(1, 4, figsize=(26, 5))
for ax, (_, mean_k, std_k, ylabel) in zip(axes, bench_metrics):
    ms = [summary[k][mean_k] for k in MODEL_KEYS]
    ss = [summary[k][std_k]  for k in MODEL_KEYS]
    bars = ax.bar(range(M), ms, yerr=ss, capsize=4, color=colors20, alpha=0.85)
    ax.set_xticks(range(M))
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=9); ax.set_title(ylabel, fontsize=10)
    ax.axhline(0, color="black", lw=0.5, linestyle="--")
    if "acc" in mean_k:
        ax.axhline(1 / N_CLASSES, color="red", lw=1, linestyle=":", label="chance")
        ax.legend(fontsize=7)
    for bar, m, s in zip(bars, ms, ss):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + s + 1e-3,
                f"{m:.3f}", ha="center", va="bottom", fontsize=7)
plt.suptitle(f"MEG img→lis benchmark v2  ({N_FOLDS} folds, "
             f"{TRAIN_SESSIONS}/{TEST_SESSIONS} train/test sessions, "
             f"win={WIN_MS}ms stride={STRIDE_MS}ms)", fontsize=11)
plt.tight_layout(); save("comparison_barplot.png")

# ---------------------------------------------------------------------------
#  1g. Per-fold line plots (4 metrics)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(26, 5))
for ax, (fold_key, _, _, ylabel) in zip(axes, bench_metrics):
    data_mat = {"per_fold_mean_r": r_matrix, "per_fold_mse": mse_matrix,
                "per_fold_clf_acc": acc_matrix, "per_fold_ps_clf_acc": ps_matrix}[fold_key]
    for i, key in enumerate(MODEL_KEYS):
        mode = key.rsplit("_", 1)[1]
        ax.plot(range(1, N_FOLDS + 1), data_mat[i],
                marker="o", linestyle="-" if mode == "full" else "--",
                color=colors20[i], alpha=0.85, label=display_name(key))
    ax.set_xlabel("Fold"); ax.set_ylabel(ylabel)
    ax.set_title(f"Per-fold {ylabel}"); ax.legend(fontsize=6, ncol=2)
    if "acc" in fold_key:
        ax.axhline(1 / N_CLASSES, color="red", lw=0.8, linestyle=":")
plt.suptitle("Per-fold metrics: solid=full, dashed=windowed", fontsize=11)
plt.tight_layout(); save("per_fold_lineplots.png")

# ---------------------------------------------------------------------------
#  1h. Windowed − full difference (4 metrics)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
mean_vals = {k: {metric: summary[k][mn]
                 for metric, mn in [("r", "mean_r_mean"), ("mse", "mse_mean"),
                                    ("acc", "clf_acc_mean"), ("ps_acc", "ps_clf_acc_mean")]}
             for k in MODEL_KEYS}
diff_metrics = [("r", "Mean r ↑"), ("mse", "MSE ↓"), ("acc", "Global acc ↑"), ("ps_acc", "Per-subj acc ↑")]
for ax, (metric, ylabel) in zip(axes, diff_metrics):
    diffs, names = [], []
    for mname in NEURAL_NAMES:
        diff = mean_vals[f"{mname}_windowed"][metric] - mean_vals[f"{mname}_full"][metric]
        diffs.append(diff); names.append(mname)
    ax.bar(range(len(names)), diffs,
           color=["steelblue" if d >= 0 else "tomato" for d in diffs], alpha=0.85)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Windowed − Full: {ylabel}"); ax.set_ylabel("Δ metric")
plt.suptitle("Effect of windowed vs full-trial training", fontsize=11)
plt.tight_layout(); save("windowed_vs_full_diff.png")


# ===========================================================================
#  SECTION 2 — Per-subject classification  (v2 additions)
# ===========================================================================

# Build helper structures from fold data
# ps_acc_fold[model_key][fold_idx][subj] = acc
ps_acc_fold = {k: [] for k in MODEL_KEYS}
for f_idx, fd in enumerate(fold_data):
    for k in MODEL_KEYS:
        ps_acc_fold[k].append(fd[k]["per_subject_clf"]["per_subject"])

# (M, N_FOLDS) matrix of mean-over-subjects per-subject accuracy
# (same as ps_matrix already computed from summary, cross-check)

# Per-subject mean acc over folds: summary already has ps_per_subj_mean_acc
# {model_key: {subj: mean_acc}}
ps_subj_mean = {k: summary[k]["ps_per_subj_mean_acc"] for k in MODEL_KEYS}

# ---------------------------------------------------------------------------
#  2a. Scatter: global acc vs per-subject acc (one dot per model per fold)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
for i, key in enumerate(MODEL_KEYS):
    g_vals  = acc_matrix[i]   # (N_FOLDS,) global acc
    ps_vals = ps_matrix[i]    # (N_FOLDS,) per-subj acc
    ax.scatter(g_vals, ps_vals, color=colors[i], s=60, alpha=0.8,
               label=display_name(key).replace("\n", " "))
    ax.annotate(display_name(key).replace("\n", " "),
                (g_vals.mean(), ps_vals.mean()),
                fontsize=6, ha="left", va="bottom",
                color=colors[i])

lo = min(acc_matrix.min(), ps_matrix.min()) - 0.01
hi = max(acc_matrix.max(), ps_matrix.max()) + 0.01
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
ax.axhline(1 / N_CLASSES, color="red", lw=0.8, linestyle=":", alpha=0.6)
ax.axvline(1 / N_CLASSES, color="red", lw=0.8, linestyle=":", alpha=0.6)
ax.set_xlabel("Global 4-class accuracy")
ax.set_ylabel("Per-subject 4-class accuracy")
ax.set_title("Global vs per-subject classification accuracy\n(each point = one fold)")
ax.legend(fontsize=6, ncol=2, loc="lower right")
plt.tight_layout(); save("ps_acc_vs_global_acc.png")

# ---------------------------------------------------------------------------
#  2b. Heatmap: subjects × models, cell = mean per-subject acc over folds
# ---------------------------------------------------------------------------
heat = np.array([[ps_subj_mean[k].get(s, float("nan"))
                  for k in MODEL_KEYS]
                 for s in SUBJECTS])   # (N_SUBJ, M)

fig, ax = plt.subplots(figsize=(max(12, M * 0.85), len(SUBJECTS) * 0.55 + 1.5))
im = ax.imshow(heat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
               interpolation="nearest")
ax.set_xticks(range(M)); ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
ax.set_yticks(range(len(SUBJECTS))); ax.set_yticklabels(SUBJECTS, fontsize=9)
ax.set_title("Per-subject 4-class accuracy (mean over 5 folds)", fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
ax.axhline(1 / N_CLASSES, color="red", lw=0.5, linestyle=":", alpha=0)  # kept for spacing
for i, s in enumerate(SUBJECTS):
    for j, k in enumerate(MODEL_KEYS):
        v = heat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color="black")
plt.tight_layout(); save("ps_acc_heatmap.png")

# ---------------------------------------------------------------------------
#  2c. Box + strip: per-fold per-subject accuracy distribution per model
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 5))
for i, key in enumerate(MODEL_KEYS):
    # collect per-subject accs across all folds: (N_FOLDS * N_SUBJ,)
    vals = [ps_acc_fold[key][f][s]["acc"]
            for f in range(N_FOLDS) for s in SUBJECTS
            if s in ps_acc_fold[key][f]]
    vals = np.array(vals)
    ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=colors[i], alpha=0.55),
               medianprops=dict(color="black", lw=1.5),
               whiskerprops=dict(color=colors[i]),
               capprops=dict(color=colors[i]),
               flierprops=dict(marker="", markersize=0))
    jitter = np.random.default_rng(i).uniform(-0.15, 0.15, size=len(vals))
    ax.scatter(i + jitter, vals, color=colors[i], zorder=3, s=12, alpha=0.5)
    ax.text(i, vals.max() + 0.005, f"{vals.mean():.3f}",
            ha="center", va="bottom", fontsize=7)

ax.axhline(1 / N_CLASSES, color="red", lw=1, linestyle=":", label="chance")
ax.set_xticks(range(M)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Per-subject 4-class accuracy")
ax.set_title("Per-subject classification accuracy distribution\n(each point = one subject × one fold)")
ax.set_xlim(-0.6, M - 0.4); ax.legend(fontsize=9)
plt.tight_layout(); save("ps_acc_boxplot.png")

# ---------------------------------------------------------------------------
#  2d. Aggregated (normalised) confusion matrices — one per model
# ---------------------------------------------------------------------------
def sum_cms_for_model(model_key):
    """Sum CMs across all folds and all subjects → (4, 4) int array."""
    agg = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for fd in fold_data:
        ps = fd[model_key]["per_subject_clf"]["per_subject"]
        for s in SUBJECTS:
            if s in ps:
                agg += np.array(ps[s]["cm"])
    return agg

def plot_cm(ax, cm, title, cmap="RdYlGn", normalise=True):
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot  = cm.astype(float) / row_sums
        fmt      = ".2f"
    else:
        cm_plot = cm.astype(float)
        fmt     = "d"
    # scale to the actual min/max of this CM so contrast is maximised
    vmin = cm_plot.min()
    vmax = cm_plot.max()
    if vmax == vmin:          # degenerate case: all cells identical
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(cm_plot, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(COND_LABELS, fontsize=7, rotation=30, ha="right")
    ax.set_yticklabels(COND_LABELS, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=7); ax.set_ylabel("True", fontsize=7)
    ax.set_title(title, fontsize=8)
    mid = (vmin + vmax) / 2
    for r in range(N_CLASSES):
        for c in range(N_CLASSES):
            v = cm_plot[r, c]
            txt = f"{v:{fmt}}" if fmt == ".2f" else str(int(cm[r, c]))
            ax.text(c, r, txt, ha="center", va="center", fontsize=7,
                    color="white" if v < mid else "black")
    return im

n_cols = 4
n_rows = (M + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * 3.5, n_rows * 3.5))
axes_flat = axes.flatten()
for i, key in enumerate(MODEL_KEYS):
    agg_cm = sum_cms_for_model(key)
    plot_cm(axes_flat[i], agg_cm, display_name(key).replace("\n", " "))
for j in range(M, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.suptitle("Aggregate normalised CMs (summed over 5 folds × 13 subjects)", fontsize=11)
plt.tight_layout(); save("agg_cm_grid.png")

# ---------------------------------------------------------------------------
#  2e. Per-subject CMs for the best-r model
# ---------------------------------------------------------------------------
best_model = MODEL_KEYS[int(np.argmax(means))]
print(f"\nBest model by mean r: {best_model}")

n_subj = len(SUBJECTS)
n_cols_s = 5
n_rows_s = (n_subj + n_cols_s - 1) // n_cols_s
fig, axes = plt.subplots(n_rows_s, n_cols_s,
                         figsize=(n_cols_s * 3, n_rows_s * 3))
axes_flat = axes.flatten()
for i, subj in enumerate(SUBJECTS):
    # sum CM for this subject across folds
    subj_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for fd in fold_data:
        ps = fd[best_model]["per_subject_clf"]["per_subject"]
        if subj in ps:
            subj_cm += np.array(ps[subj]["cm"])
    mean_acc = ps_subj_mean[best_model].get(subj, float("nan"))
    plot_cm(axes_flat[i], subj_cm,
            f"{subj}  (acc={mean_acc:.2f})")
for j in range(n_subj, len(axes_flat)):
    axes_flat[j].set_visible(False)
plt.suptitle(f"Per-subject normalised CMs — {best_model}\n"
             f"(summed over 5 folds, normalised per row)", fontsize=11)
plt.tight_layout(); save(f"per_subj_cm_best_model.png")

# ---------------------------------------------------------------------------
#  2f. Per-subject CM grids for every model
# ---------------------------------------------------------------------------
for key in MODEL_KEYS:
    fig, axes = plt.subplots(n_rows_s, n_cols_s,
                             figsize=(n_cols_s * 3, n_rows_s * 3))
    axes_flat = axes.flatten()
    for i, subj in enumerate(SUBJECTS):
        subj_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
        for fd in fold_data:
            ps = fd[key]["per_subject_clf"]["per_subject"]
            if subj in ps:
                subj_cm += np.array(ps[subj]["cm"])
        mean_acc = ps_subj_mean[key].get(subj, float("nan"))
        plot_cm(axes_flat[i], subj_cm, f"{subj}  (acc={mean_acc:.2f})")
    for j in range(n_subj, len(axes_flat)):
        axes_flat[j].set_visible(False)
    plt.suptitle(f"Per-subject normalised CMs — {key}", fontsize=11)
    plt.tight_layout()
    save(f"per_subj_cm_{key}.png")
