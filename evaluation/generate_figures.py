"""generate_figures.py (thesis: step11_generate_figures.py) — the ONLY plotting script for Chapter 6.

Every other script in the pipeline (step1-step10) reads/writes CSVs and
must not import matplotlib. This script reads those CSVs and produces every
figure in the thesis, for both tasks, in one place.

Methodology per section:
  - Utility (ROC-AUC / S / C): 3x2 small-multiples line grid, with an
    "Original (no privatization)" reference line per panel.
  - TRIR: grouped BAR chart, faceted by model (rows) x scenario (cols).
  - Privacy-utility trade-off: TWO figures are generated -- a heatmap
    (method x epsilon grid, colored by gamma) and a diverging bar chart
    (same data, bars left/right of zero). A third renderer (bubble
    landscape scatter) is kept in this file for ad-hoc/appendix use but
    is not called by generate_all_figures.

Changes 2026-11: inputs from common/layout.py (results/ folder), --base default ".".

Usage:
    python evaluation/generate_figures.py --task all --base .
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, trir_averaged_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Global thesis style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "normal",
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#bbbbbb",
    "grid.alpha": 0.5,
    "grid.linestyle": "-",
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.6,
    "lines.markersize": 5.5,
    "figure.constrained_layout.use": True,
})

MODEL_LABEL = {"roberta_base": "RoBERTa-base", "roberta_clinical": "RoBERTa-clinical"}

# --- seaborn "deep" palette -- used ONLY for utility-grid lines (untouched
# per this request). Bar charts (TRIR, gamma diverging bars) use VIVID below.
DEEP = {
    "blue":  "#4C72B0",
    "red":   "#C44E52",
    "green": "#55A868",
    "gold":  "#CCB974",
}

# --- vivid red/yellow/blue/green palette for bar charts, per request -----
VIVID = {
    "blue":  "#1565C0",
    "red":   "#D32F2F",
    "green": "#2E7D32",
    "gold":  "#F9A825",
}

MODEL_COLOR = {"roberta_base": DEEP["blue"], "roberta_clinical": DEEP["red"]}

METHOD_COLOR = {
    "1.0": DEEP["blue"],
    "1.5": DEEP["gold"],
    "2.0": DEEP["green"],
    "all": DEEP["red"],
}
METHOD_MARKER = {"1.0": "o", "1.5": "s", "2.0": "^", "all": "D"}
METHOD_LINESTYLE = {"1.0": "-", "1.5": "--", "2.0": "-.", "all": ":"}
METHOD_LABEL = {
    "1.0": "Selective k=1.0", "1.5": "Selective k=1.5",
    "2.0": "Selective k=2.0", "all": "Full DP-MLM",
}
SHORT_METHOD_LABEL = {"1.0": "Sel. k=1.0", "1.5": "Sel. k=1.5", "2.0": "Sel. k=2.0", "all": "Full"}
SETUP_ORDER = ["all", "1.0", "1.5", "2.0"]
EPS_ORDER = [25, 50, 75, 100]
EPS_COLORS = {
    25: VIVID["blue"],
    50: VIVID["green"],
    75: VIVID["gold"],
    100: VIVID["red"],
}
EPS_SIZES = {25: 30, 50: 100, 75: 200, 100: 340}
STD_COL = "std_TRIR"

LABEL_COLOR = "#555555"
LABEL_COLOR_BEST = "#1a1a1a"


def _print_caption(fig_kind: str, task: str, extra: str = "") -> None:
    """Print a suggested LaTeX \\caption{} to the console."""
    name = _task_name(task)
    suggestions = {
        "utility": f"Utility metrics (ROC-AUC, semantic similarity, and coherence) "
                   f"across privacy budgets for the {name.lower()} task, for both rewriting models.",
        "trir": f"Re-identification risk (TRIR, mean over five seeds) for the {name.lower()} task.",
        "gamma_heatmap": f"Privacy-utility trade-off ($\\gamma$) per setup for the {name} task, "
                         f"across both re-identification scenarios and rewriting models.",
    }
    print(f"  suggested caption: \\caption{{{suggestions.get(fig_kind, name)}{extra}}}")


def save_all(fig: plt.Figure, out_path: Path) -> None:
    """Save PNG only."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    target = out_path.with_suffix(".png")
    fig.savefig(target, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"saved: {target}")


def load_task_data(base: Path, task: str):
    L = Layout(base, task)
    util = pd.read_csv(L.utility_csv)
    gamma = pd.read_csv(L.gamma_csv)
    roc = pd.read_csv(L.classification_avg)
    trir = pd.read_csv(trir_averaged_csv(base))
    trir = trir[trir["task"] == task].copy()
    return util, gamma, roc, trir


def _setup_key(row):
    if row.get("setup_type") == "full_rewriting":
        return "all"
    sigma = row.get("sigma")
    return None if pd.isna(sigma) else f"{float(sigma):.1f}"


def _task_name(task: str) -> str:
    return "Mortality Prediction" if task == "mp" else "Procedure Prediction"


def _roc_baseline(roc: pd.DataFrame, model: str):
    row = roc[(roc["setup"] == "original") & (roc["model"] == model)]
    if row.empty:
        row = roc[(roc["setup"] == "original") & (roc["model"] == "none")]
    if row.empty:
        return None
    return float(row["roc_auc_mean"].iloc[0])


# ---------------------------------------------------------------------------
# 1. Utility small multiples (UNCHANGED per this request)
# ---------------------------------------------------------------------------
def plot_utility_grid(util: pd.DataFrame, roc: pd.DataFrame, task: str, out_path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.2), sharex=True)
    models = ["roberta_base", "roberta_clinical"]
    metrics = [
        ("roc_auc_mean", "ROC-AUC", roc, "setup"),
        ("S_avg", "Semantic similarity", util, "setup"),
        ("C", "Coherence", util, "setup"),
    ]

    print("\nplot_utility_grid — setup coverage check:")
    for row, (col, ylabel, df_metric, setup_col) in enumerate(metrics):
        row_values = []
        for c, model in enumerate(models):
            ax = axes[row, c]
            df_m = df_metric[df_metric["model"] == model]

            for setup_key in SETUP_ORDER:
                setup_name = "all_sentences" if setup_key == "all" else f"rare_sigma_{setup_key}"
                df_s = df_m[df_m[setup_col] == setup_name].sort_values("epsilon")
                if df_s.empty:
                    print(f"  MISSING: {ylabel} | {model} | {METHOD_LABEL[setup_key]} — no rows found")
                    continue
                n_missing_eps = len(EPS_ORDER) - df_s["epsilon"].nunique()
                if n_missing_eps > 0:
                    print(f"  PARTIAL: {ylabel} | {model} | {METHOD_LABEL[setup_key]} "
                          f"— {n_missing_eps}/{len(EPS_ORDER)} epsilon values missing")
                row_values.extend(df_s[col].dropna().tolist())
                ax.plot(
                    df_s["epsilon"], df_s[col],
                    color=METHOD_COLOR[setup_key],
                    marker=METHOD_MARKER[setup_key],
                    markersize=4.8,
                    linestyle=METHOD_LINESTYLE[setup_key],
                    label=METHOD_LABEL[setup_key],
                    zorder=3 if setup_key != "all" else 2,
                )

            if col == "roc_auc_mean":
                baseline_val = _roc_baseline(roc, model)
                if baseline_val is None:
                    print(f"  MISSING BASELINE: ROC-AUC | {model} — no 'original' row found at all")
            else:
                baseline_val = 1.0
            if baseline_val is not None:
                ax.axhline(baseline_val, color="#666666", linestyle=(0, (1, 1.2)),
                           linewidth=1.1, alpha=0.85, zorder=1,
                           label="Original (no privatization)" if (row == 0 and c == 0) else None)
                row_values.append(baseline_val)

            if row == 0:
                ax.set_title(MODEL_LABEL[model])
            if c == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
            if row == 2:
                ax.set_xlabel(r"Privacy budget $\varepsilon$")
            ax.set_xticks(EPS_ORDER)

        if row_values:
            lo, hi = np.nanmin(row_values), np.nanmax(row_values)
            pad = max((hi - lo) * 0.10, 0.01)
            for ax in axes[row, :]:
                ax.set_ylim(lo - pad, hi + pad)

    handles = [
        Line2D([0], [0], color=METHOD_COLOR[k], marker=METHOD_MARKER[k], markersize=4.8,
               linestyle=METHOD_LINESTYLE[k], label=METHOD_LABEL[k])
        for k in SETUP_ORDER
    ]
    handles.append(Line2D([0], [0], color="#666666", linestyle=(0, (1, 1.2)),
                           linewidth=1.1, label="Original (no privatization)"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    _print_caption("utility", task)
    save_all(fig, out_path)


# ---------------------------------------------------------------------------
# 2. TRIR — grouped BAR chart, faceted by model x scenario
# ---------------------------------------------------------------------------
def plot_trir_bars(trir: pd.DataFrame, task: str, out_path: Path, std_col: str = STD_COL):
    scenarios = ["50E", "500R"]
    models = ["roberta_base", "roberta_clinical"]

    # Wider figure gives the four epsilon bars in every method group more room.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
        sharey="col",
    )

    # Add more separation between the four method groups.
    method_spacing = 1.12
    x_positions = np.arange(len(SETUP_ORDER)) * method_spacing
    bar_width = 0.17
    group_gap = 1.10

    # Softer, publication-style, colorblind-friendly palette.
    trir_colors = {
        25:  "#4C78A8",
        50:  "#59A14F",
        75:  "#F2B447",
        100: "#E15759",
    }

    panel_data = {}
    col_max = {scenario: 0.0 for scenario in scenarios}

    # First pass: collect all bar heights and original-study baselines.
    for model in models:
        for scenario in scenarios:
            df_s = trir[
                (trir["scenario"] == scenario) &
                (trir["setup_type"] != "original") &
                (trir["model"] == model)
            ].copy()

            df_s["setup_key"] = df_s.apply(_setup_key, axis=1)

            bars_by_eps = {}
            for eps in EPS_ORDER:
                heights = []

                for key in SETUP_ORDER:
                    row = df_s[
                        (df_s["setup_key"] == key) &
                        (df_s["epsilon"] == eps)
                    ]

                    height = (
                        np.nan
                        if row.empty
                        else float(row["mean_TRIR"].iloc[0])
                    )
                    heights.append(height)

                    if np.isfinite(height):
                        col_max[scenario] = max(
                            col_max[scenario],
                            height,
                        )

                bars_by_eps[eps] = heights

            baseline_rows = trir[
                (trir["scenario"] == scenario) &
                (trir["setup_type"] == "original") &
                (trir["model"] == model)
            ]

            baseline_value = (
                float(baseline_rows["mean_TRIR"].mean())
                if not baseline_rows.empty
                else None
            )

            if baseline_value is None:
                fallback_rows = trir[
                    (trir["scenario"] == scenario) &
                    (trir["setup_type"] == "original")
                ]

                if not fallback_rows.empty:
                    baseline_value = float(
                        fallback_rows["mean_TRIR"].mean()
                    )
                    print(
                        f"  NOTE: TRIR | {model} | {scenario} — "
                        f"using scenario-level original baseline "
                        f"({baseline_value:.2f}%)"
                    )
                else:
                    print(
                        f"  MISSING BASELINE: TRIR | {model} | "
                        f"{scenario}"
                    )

            if baseline_value is not None:
                col_max[scenario] = max(
                    col_max[scenario],
                    baseline_value,
                )

            panel_data[(model, scenario)] = (
                bars_by_eps,
                baseline_value,
            )

    # Second pass: draw the four panels.
    for c_idx, scenario in enumerate(scenarios):
        # Extra vertical space is reserved for labels above every bar.
        ylim_top = (
            col_max[scenario] * 1.28
            if col_max[scenario] > 0
            else 1.0
        )
        label_offset = ylim_top * 0.018

        for r_idx, model in enumerate(models):
            ax = axes[r_idx, c_idx]
            bars_by_eps, baseline_value = panel_data[
                (model, scenario)
            ]

            for e_idx, eps in enumerate(EPS_ORDER):
                offset = (
                    (e_idx - 1.5) *
                    bar_width *
                    group_gap
                )
                xs = x_positions + offset
                heights = bars_by_eps[eps]

                bars = ax.bar(
                    xs,
                    heights,
                    width=bar_width * 0.92,
                    color=trir_colors[eps],
                    edgecolor="#404040",
                    linewidth=0.45,
                    zorder=3,
                )

                # Display every available TRIR value above its bar.
                # Labels use normal weight and a small grey font.
                for bar, height in zip(bars, heights):
                    if not np.isfinite(height):
                        continue

                    # Stagger labels vertically so neighbouring values do not merge.
                    # The first and third epsilon labels use the lower row; the
                    # second and fourth use a slightly higher row.
                    stagger = label_offset * (0.0 if e_idx % 2 == 0 else 0.95)
                    label_y = max(height, 0.0) + label_offset + stagger

                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        label_y,
                        f"{height:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=6.4,
                        fontweight="normal",
                        color="#222222",
                        clip_on=False,
                        zorder=5,
                    )

            if baseline_value is not None:
                # Show the original non-privatized TRIR as a compact annotation
                # box instead of a horizontal reference line.
                ax.text(
                    0.98,
                    0.96,
                    f"Original: {baseline_value:.1f}%",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8.2,
                    color="#333333",
                    bbox={
                        "boxstyle": "round,pad=0.32",
                        "facecolor": "white",
                        "edgecolor": "#9A9A9A",
                        "linewidth": 0.8,
                        "alpha": 0.96,
                    },
                    zorder=6,
                )

            ax.set_ylim(0, ylim_top)
            ax.set_xlim(
                -0.62,
                x_positions[-1] + 0.62,
            )

            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [SHORT_METHOD_LABEL[key] for key in SETUP_ORDER]
            )

            # Only horizontal grid lines are useful for bar-height comparisons.
            ax.grid(
                axis="y",
                color="#D0D0D0",
                linewidth=0.45,
                alpha=0.58,
                zorder=0,
            )
            ax.grid(axis="x", visible=False)

            if r_idx == 0:
                ax.set_title(scenario)

            if c_idx == 0:
                ax.set_ylabel(
                    f"{MODEL_LABEL[model]}\nTRIR (%)"
                )

            if r_idx == 1:
                ax.set_xlabel("Method")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="None",
            markersize=7.5,
            markerfacecolor=trir_colors[eps],
            markeredgecolor="#404040",
            label=f"$\\varepsilon={eps}$",
        )
        for eps in EPS_ORDER
    ]

    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )

    fig.suptitle(
        f"Re-identification risk: {_task_name(task)}",
        fontsize=12.5,
        fontweight="semibold",
    )

    _print_caption("trir", task)
    save_all(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Privacy-utility landscape — kept for appendix/ad-hoc use, not called
#    by generate_all_figures by default.
# ---------------------------------------------------------------------------
def plot_privacy_utility_scatter(gamma: pd.DataFrame, task: str, out_path: Path,
                                  utility_col="ROC_private", baseline_prefix="ROC",
                                  utility_label="ROC-AUC"):
    scenarios = ["50E", "500R"]
    models = ["roberta_base", "roberta_clinical"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.6), sharex=True, sharey="col")

    for r_idx, model in enumerate(models):
        for c_idx, scenario in enumerate(scenarios):
            ax = axes[r_idx, c_idx]
            df_sm = gamma[(gamma["scenario"] == scenario) & (gamma["model"] == model)].copy()
            df_sm["setup_key"] = df_sm.apply(_setup_key, axis=1)

            for _, r in df_sm.iterrows():
                key = r["setup_key"]
                if key is None:
                    continue
                ax.scatter(
                    r[utility_col], r["TRIR_private"],
                    s=EPS_SIZES[int(r["epsilon"])],
                    color=METHOD_COLOR[key],
                    edgecolor="#333333", linewidth=0.6, alpha=0.85, zorder=3,
                )

            if not df_sm.empty:
                ax.scatter(
                    df_sm[f"{baseline_prefix}_baseline"].iloc[0], df_sm["TRIR_baseline"].iloc[0],
                    marker="*", s=180, color="#222222", edgecolor="white", linewidth=0.6, zorder=5,
                )

            if r_idx == 0:
                ax.set_title(scenario)
            if c_idx == 0:
                ax.set_ylabel(f"{MODEL_LABEL[model]}\nTRIR (%)")
            if r_idx == 1:
                ax.set_xlabel(f"{utility_label}")
            ax.margins(x=0.10, y=0.12)

    method_handles = [
        Line2D([0], [0], color=METHOD_COLOR[k], marker="o", linestyle="None",
               markersize=8, label=SHORT_METHOD_LABEL[k])
        for k in SETUP_ORDER
    ]
    size_handles = [
        Line2D([0], [0], color="#777777", marker="o", linestyle="None",
               markersize=np.sqrt(EPS_SIZES[e]) * 0.9, label=f"$\\varepsilon={e}$")
        for e in EPS_ORDER
    ]
    baseline_handle = Line2D([0], [0], color="#222222", marker="*", linestyle="None",
                              markersize=11, label="Original")
    fig.legend(
        handles=method_handles + size_handles + [baseline_handle],
        loc="outside lower center", ncol=5, frameon=False,
    )
    save_all(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Gamma diverging bar chart
# ---------------------------------------------------------------------------
def plot_gamma_diverging_bars(gamma: pd.DataFrame, task: str, out_path: Path):
    scenarios = ["50E", "500R"]
    models = ["roberta_base", "roberta_clinical"]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.6), sharex="col")
    y_positions = np.arange(len(SETUP_ORDER))
    bar_height = 0.19
    group_gap = 1.15

    panel_data = {}
    col_max_abs = {s: 0.0 for s in scenarios}

    for model in models:
        for scenario in scenarios:
            df_sm = gamma[(gamma["scenario"] == scenario) & (gamma["model"] == model)].copy()
            df_sm["setup_key"] = df_sm.apply(_setup_key, axis=1)
            pivot = df_sm.pivot_table(index="setup_key", columns="epsilon", values="gamma", aggfunc="mean")
            pivot = pivot.reindex(index=SETUP_ORDER, columns=EPS_ORDER)
            panel_data[(model, scenario)] = pivot
            if pivot.notna().any().any():
                col_max_abs[scenario] = max(col_max_abs[scenario], float(np.nanmax(np.abs(pivot.values))))

    for c_idx, scenario in enumerate(scenarios):
        xlim = col_max_abs[scenario] * 1.35 if col_max_abs[scenario] > 0 else 0.5
        for r_idx, model in enumerate(models):
            ax = axes[r_idx, c_idx]
            pivot = panel_data[(model, scenario)]

            all_vals = pivot.values[~np.isnan(pivot.values)]
            best_val = np.nanmax(all_vals) if all_vals.size else None

            for e_idx, eps in enumerate(EPS_ORDER):
                offset = (e_idx - 1.5) * bar_height * group_gap
                ys = y_positions + offset
                widths = pivot[eps].values if eps in pivot.columns else np.full(len(SETUP_ORDER), np.nan)
                ax.barh(
                    ys, widths, height=bar_height * 0.9,
                    color=EPS_COLORS[eps], edgecolor="#3a3a3a", linewidth=0.5, zorder=3,
                )
                for y, w in zip(ys, widths):
                    if np.isnan(w):
                        continue
                    is_best = best_val is not None and abs(w - best_val) < 1e-9
                    ha = "left" if w >= 0 else "right"
                    pad = xlim * 0.02 * (1 if w >= 0 else -1)
                    ax.text(w + pad, y, f"{w:+.2f}", va="center", ha=ha,
                            fontsize=8.3 if is_best else 7.3,
                            fontweight="bold" if is_best else "normal",
                            color=LABEL_COLOR_BEST if is_best else LABEL_COLOR,
                            zorder=4)

            ax.axvline(0, color="#3a3a3a", linewidth=0.9, zorder=2)
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-0.68, len(SETUP_ORDER) - 1 + 0.68)
            ax.set_yticks(y_positions)
            ax.set_yticklabels([SHORT_METHOD_LABEL[k] for k in SETUP_ORDER])
            ax.invert_yaxis()
            if r_idx == 0:
                ax.set_title(scenario)
            if c_idx == 0:
                ax.set_ylabel(MODEL_LABEL[model])
            if r_idx == 1:
                ax.set_xlabel(r"$\gamma$ = privacy gain $-$ average utility loss")

    eps_handles = [
        Line2D([0], [0], marker="s", linestyle="None", markersize=8,
               markerfacecolor=EPS_COLORS[e], markeredgecolor="#3a3a3a", label=f"$\\varepsilon={e}$")
        for e in EPS_ORDER
    ]
    fig.legend(handles=eps_handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"Privacy-utility trade-off γ: {_task_name(task)}", fontsize=12.5, fontweight="semibold")
    save_all(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Gamma heatmap
# ---------------------------------------------------------------------------
def _muted_diverging_cmap():
    # Higher-contrast diverging palette:
    # blue = negative gamma, light neutral = near zero, orange = positive gamma.
    stops = ["#2166AC", "#67A9CF", "#F7F7F7", "#FDAE61", "#B2182B"]
    return mcolors.LinearSegmentedColormap.from_list("blue_orange_gamma", stops)


def plot_gamma_heatmap(gamma: pd.DataFrame, task: str, out_path: Path):
    scenarios = ["50E", "500R"]
    models = ["roberta_base", "roberta_clinical"]

    # Same compact figure size and panel geometry as the earlier green/red version.
    # Match the more compact panel size of the earlier green/red version.
    # Disabling constrained layout here lets us control the gaps explicitly.
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.3, 5.6),
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=0.13,
        right=0.94,
        bottom=0.10,
        top=0.88,
        wspace=0.52,
        hspace=0.42,
    )

    col_abs_max = {}
    pivots = {}

    for model in models:
        for scenario in scenarios:
            df_sm = gamma[
                (gamma["scenario"] == scenario) &
                (gamma["model"] == model)
            ].copy()

            df_sm["setup_key"] = df_sm.apply(_setup_key, axis=1)

            pivot = df_sm.pivot_table(
                index="setup_key",
                columns="epsilon",
                values="gamma",
                aggfunc="mean",
            )
            pivot = pivot.reindex(index=SETUP_ORDER, columns=EPS_ORDER)
            pivots[(model, scenario)] = pivot

            if pivot.notna().any().any():
                max_abs = float(np.nanmax(np.abs(pivot.values)))
                col_abs_max[scenario] = max(
                    col_abs_max.get(scenario, 0.0),
                    max_abs,
                )

    cmap = _muted_diverging_cmap()

    for r_idx, model in enumerate(models):
        for c_idx, scenario in enumerate(scenarios):
            ax = axes[r_idx, c_idx]
            pivot = pivots[(model, scenario)]

            vmax = (col_abs_max.get(scenario, 0.5) or 0.5) * 1.1

            # aspect="auto" reproduces the wider rectangular cells
            # used in the original green/red heatmap.
            im = ax.imshow(
                pivot.values,
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
                aspect="auto",
            )

            # Identify exactly one maximum cell in this heatmap.
            # np.nanargmax returns a single flat index, even if values are close.
            if np.isfinite(pivot.values).any():
                max_flat_index = int(np.nanargmax(pivot.values))
                max_position = np.unravel_index(
                    max_flat_index,
                    pivot.values.shape,
                )
            else:
                max_position = None

            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    val = pivot.values[i, j]

                    if np.isnan(val):
                        ax.text(
                            j,
                            i,
                            "n/a",
                            ha="center",
                            va="center",
                            fontsize=8.0,
                            color="#777777",
                        )
                        continue

                    # Bold only the single cell selected by nanargmax.
                    is_highest = (
                        max_position is not None
                        and (i, j) == max_position
                    )

                    text_color = (
                        "white"
                        if abs(val) > vmax * 0.58
                        else "#222222"
                    )

                    ax.text(
                        j,
                        i,
                        f"{val:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=8.6,
                        fontweight="bold" if is_highest else "normal",
                        color=text_color,
                    )

            ax.set_xticks(range(len(EPS_ORDER)))
            ax.set_xticklabels(
                [f"$\\varepsilon={eps}$" for eps in EPS_ORDER]
            )

            ax.set_yticks(range(len(SETUP_ORDER)))
            ax.set_yticklabels(
                [SHORT_METHOD_LABEL[key] for key in SETUP_ORDER]
            )

            ax.set_xticks(
                np.arange(-0.5, len(EPS_ORDER), 1),
                minor=True,
            )
            ax.set_yticks(
                np.arange(-0.5, len(SETUP_ORDER), 1),
                minor=True,
            )

            ax.grid(
                which="minor",
                color="white",
                linewidth=1.2,
            )
            ax.tick_params(which="minor", length=0)
            ax.grid(which="major", visible=False)

            for spine in ax.spines.values():
                spine.set_visible(False)

            if r_idx == 0:
                ax.set_title(scenario)

            if c_idx == 0:
                ax.set_ylabel(MODEL_LABEL[model])

            # Four colorbars: one beside every heatmap panel.
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.03,
            )
            cbar.set_label(r"$\gamma$")
            cbar.outline.set_linewidth(0.5)

    fig.suptitle(
        f"Privacy-utility trade-off γ: {_task_name(task)}",
        fontsize=12.5,
        fontweight="semibold",
    )

    _print_caption("gamma_heatmap", task)
    save_all(fig, out_path)


# ---------------------------------------------------------------------------
def generate_all_figures(base: Path, task: str, out_dir: Path):
    util, gamma, roc, trir = load_task_data(base, task)
    plot_utility_grid(util, roc, task, out_dir / f"utility_grid_{task}.png")
    plot_trir_bars(trir, task, out_dir / f"trir_bars_{task}.png")
    plot_gamma_heatmap(gamma, task, out_dir / f"gamma_heatmap_{task}.png")
    plot_gamma_diverging_bars(gamma, task, out_dir / f"gamma_divbars_{task}.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mp", "proc", "all"])
    p.add_argument("--base", default=".")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    base = Path(args.base)
    tasks = ["mp", "proc"] if args.task == "all" else [args.task]

    for task in tasks:
        out_dir = Path(args.out_dir) if args.out_dir else Layout(base, task).figures_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*70}\nGenerating figures for task={task}\n{'='*70}")
        generate_all_figures(base, task, out_dir)
        print(f"All figures for {task} saved to: {out_dir}")


if __name__ == "__main__":
    main()