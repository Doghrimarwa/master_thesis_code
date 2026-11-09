"""
tri_step5_average.py  (thesis: compute_trir_average_v2.py)
Computes average TRIR over 5 seeds and plots results for both tasks (mp, proc).
 
IMPORTANT: tri.py already outputs TRIR as percentage internally
           (compute_metrics: accuracy = correct/total * 100)
           No further conversion needed — values like 14.0 already mean 14%
 
Change 2026-11: --base argument; writes results/trir_averaged_results.csv
(read by compute_gamma.py and generate_figures.py) and exploratory plots to
results/figures/. The thesis figures are made by evaluation/generate_figures.py.

Usage:
    python privacy/tri_step5_average.py --base .
"""
 
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import tri_run_dir, repo_root, trir_averaged_csv, TRI_SEEDS  # noqa: E402
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
 
# ── Config ────────────────────────────────────────────────────────────────────
_p = argparse.ArgumentParser()
_p.add_argument("--base", default=".")
_p.add_argument("--no_plots", action="store_true")
ARGS = _p.parse_args()
SEEDS     = TRI_SEEDS
TASKS     = ["mp", "proc"]
SCENARIOS = ["50E", "500R"]
N_PATIENTS = 50  # number of target patients evaluated (for reference only)
 
# ── Load all results ──────────────────────────────────────────────────────────
def load_all_results():
    all_dfs = []
    for seed in SEEDS:
        for task in TASKS:
            patterns = [
                tri_run_dir(ARGS.base, seed) / f"tri_results_{task}" / "tri_summary_complete.csv",
                tri_run_dir(ARGS.base, seed) / f"tri_results_{task}" / "tri_summary.csv",
            ]
            loaded = False
            for path in patterns:
                if path.exists():
                    df = pd.read_csv(path)
                    df["seed"] = seed
                    if "task" not in df.columns:
                        df["task"] = task
                    all_dfs.append(df)
                    print(f"  ✓ Loaded seed={seed}, task={task}: {path.name} ({len(df)} rows)")
                    loaded = True
                    break
            if not loaded:
                print(f"  ✗ MISSING: seed={seed}, task={task}")
 
    if not all_dfs:
        raise FileNotFoundError("No result files found!")
 
    combined = pd.concat(all_dfs, ignore_index=True)
 
    # tri.py already outputs TRIR as percentage (accuracy = correct/total * 100)
    # No conversion needed — values like 14.0 already mean 14%
    combined["TRIR"] = pd.to_numeric(combined["TRIR"], errors="coerce")
    combined["TRIR_pct"] = combined["TRIR"]  # already in %
 
    print(f"\nTotal rows: {len(combined)}")
    print(f"Seeds found: {sorted(combined['seed'].unique())}")
    print(f"Tasks found: {sorted(combined['task'].unique())}")
    print(f"\nSample TRIR values (already in %):")
    sample = combined[combined["status"] == "ok"][["task","scenario","setup_type","sigma","epsilon","model","TRIR"]].head(10)
    print(sample.to_string(index=False))
    return combined
 
# ── Compute averages ──────────────────────────────────────────────────────────
def compute_averages(df):
    group_cols = ["task", "scenario", "setup_type", "sigma", "epsilon", "model"]
    df_ok = df[df["status"] == "ok"].copy()
 
    # dropna=False ensures original rows (sigma=None, epsilon=None, model=None)
    # are NOT dropped — needed so TRIR_original is included in output CSV
    # and compute_gamma.py can use it for privacy_gain computation
    agg = df_ok.groupby(group_cols, dropna=False)["TRIR_pct"].agg(
        mean_TRIR="mean",
        std_TRIR="std",
        n_seeds="count"
    ).reset_index()
 
    print(f"\nAveraged configurations: {len(agg)}")
    print(f"Seeds per config: {agg['n_seeds'].value_counts().to_dict()}")
    return agg
 
# ── Bar Plot ──────────────────────────────────────────────────────────────────
def make_bar_plot(agg, task, output_dir=Path(".")):
    df_task = agg[agg["task"] == task].copy()
    if len(df_task) == 0:
        print(f"  No data for task={task}, skipping plot")
        return

    epsilons     = [25, 50, 75, 100]
    models       = ["roberta_base", "roberta_clinical"]
    model_labels = {"roberta_base": "RoBERTa-Base", "roberta_clinical": "RoBERTa-Clinical"}

    colors = {
        ("rare_sentence", "1.0"): "#2196F3",
        ("rare_sentence", "1.5"): "#FF9800",
        ("rare_sentence", "2.0"): "#4CAF50",
        ("full_rewriting", None): "#F44336",
        ("original",       None): "#9C27B0",
    }
    hatches = {"roberta_base": "", "roberta_clinical": "///"}

    setups = [
        ("original",       None,  "No\nPrivatization"),
        ("rare_sentence",  "1.0", "Selective\nk=1.0"),
        ("rare_sentence",  "1.5", "Selective\nk=1.5"),
        ("rare_sentence",  "2.0", "Selective\nk=2.0"),
        ("full_rewriting", None,  "Full\nDP-MLM"),
    ]

    task_label = "Mortality Prediction" if task == "mp" else "Procedures Prediction"

    # ── Save ONE figure per scenario (better for LaTeX) ───────────────────────
    for scenario in SCENARIOS:
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        fig.suptitle(
            f"Privacy: TRIR (%) — {task_label}\n"
            f"Scenario: {scenario}  |  Mean ± std over 5 seeds",
            fontsize=13, fontweight="bold"
        )

        df_s      = df_task[df_task["scenario"] == scenario]
        n_setups  = len(setups)
        bar_width = 0.10
        group_width = n_setups * len(models) * bar_width + 0.20
        x = np.arange(len(epsilons)) * group_width

        for s_idx, (stype, sigma, slabel) in enumerate(setups):
            color = colors.get((stype, sigma), "#888888")
            for m_idx, model in enumerate(models):
                # original has no model split — use roberta_base only
                if stype == "original" and model == "roberta_clinical":
                    continue

                hatch = hatches[model]
                vals, errs = [], []
                for eps in epsilons:
                    if stype == "original":
                        row = df_s[df_s["setup_type"] == "original"]
                    elif stype == "rare_sentence":
                        row = df_s[
                            (df_s["setup_type"] == stype) &
                            (df_s["sigma"].astype(str) == str(sigma)) &
                            (df_s["epsilon"] == eps) &
                            (df_s["model"] == model)
                        ]
                    else:
                        row = df_s[
                            (df_s["setup_type"] == stype) &
                            (df_s["epsilon"] == eps) &
                            (df_s["model"] == model)
                        ]
                    if len(row) > 0:
                        vals.append(float(row["mean_TRIR"].values[0]))
                        std_val = row["std_TRIR"].values[0]
                        errs.append(float(std_val) if not pd.isna(std_val) else 0)
                    else:
                        vals.append(0)
                        errs.append(0)

                offset = (s_idx * len(models) + m_idx) * bar_width
                # original uses center offset
                if stype == "original":
                    offset = s_idx * len(models) * bar_width + bar_width/2

                model_str = "" if stype == "original" else f" — {model_labels[model]}"
                label_str = (
                    f"No Privatization (original)"
                    if stype == "original"
                    else f"Selective DP-MLM k={sigma} (τ=μ+{sigma}σ){model_str}"
                    if stype == "rare_sentence"
                    else f"Full DP-MLM (all sentences){model_str}"
                )

                bars = ax.bar(
                    x + offset, vals,
                    bar_width * 0.88,
                    yerr=errs,
                    color=color,
                    hatch=hatch,
                    alpha=0.85,
                    capsize=2,
                    error_kw={"elinewidth": 1, "ecolor": "black"},
                    label=label_str
                )

                for bar, val in zip(bars, vals):
                    if val > 0.5:
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.3,
                            f"{val:.1f}",
                            ha="center", va="bottom",
                            fontsize=6, rotation=90
                        )

        ax.set_xlabel("Privacy Budget (ε)", fontsize=11)
        ax.set_ylabel("TRIR (%)", fontsize=11)
        tick_center = (n_setups * len(models) * bar_width) / 2
        ax.set_xticks(x + tick_center)
        ax.set_xticklabels([f"ε={e}" for e in epsilons], fontsize=10)
        max_val = df_s["mean_TRIR"].max() if len(df_s) > 0 else 25
        ax.set_ylim(0, max(30, max_val * 1.5))
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.legend(
            loc="upper right", fontsize=7.5, frameon=True,
            title="Setup (solid=RoBERTa-Base, hatched=RoBERTa-Clinical)",
            title_fontsize=7
        )

        plt.tight_layout()
        out = output_dir / f"trir_barplot_{task}_{scenario}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        plt.close()

    # ── Also save combined figure (both scenarios side by side) ───────────────
    fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=False)
    fig.suptitle(
        f"Privacy: TRIR (%) — {task_label}\n(Mean ± std over 5 seeds)",
        fontsize=13, fontweight="bold"
    )

    for ax_idx, scenario in enumerate(SCENARIOS):
        ax   = axes[ax_idx]
        df_s = df_task[df_task["scenario"] == scenario]

        n_setups    = len(setups)
        bar_width   = 0.09
        group_width = n_setups * len(models) * bar_width + 0.20
        x = np.arange(len(epsilons)) * group_width

        for s_idx, (stype, sigma, slabel) in enumerate(setups):
            color = colors.get((stype, sigma), "#888888")
            for m_idx, model in enumerate(models):
                if stype == "original" and model == "roberta_clinical":
                    continue
                hatch = hatches[model]
                vals, errs = [], []
                for eps in epsilons:
                    if stype == "original":
                        row = df_s[df_s["setup_type"] == "original"]
                    elif stype == "rare_sentence":
                        row = df_s[
                            (df_s["setup_type"] == stype) &
                            (df_s["sigma"].astype(str) == str(sigma)) &
                            (df_s["epsilon"] == eps) &
                            (df_s["model"] == model)
                        ]
                    else:
                        row = df_s[
                            (df_s["setup_type"] == stype) &
                            (df_s["epsilon"] == eps) &
                            (df_s["model"] == model)
                        ]
                    if len(row) > 0:
                        vals.append(float(row["mean_TRIR"].values[0]))
                        std_val = row["std_TRIR"].values[0]
                        errs.append(float(std_val) if not pd.isna(std_val) else 0)
                    else:
                        vals.append(0)
                        errs.append(0)

                offset = (s_idx * len(models) + m_idx) * bar_width
                if stype == "original":
                    offset = s_idx * len(models) * bar_width + bar_width/2

                ax.bar(
                    x + offset, vals,
                    bar_width * 0.88,
                    yerr=errs,
                    color=color,
                    hatch=hatch,
                    alpha=0.85,
                    capsize=2,
                    error_kw={"elinewidth": 1, "ecolor": "black"},
                )

        ax.set_title(f"Scenario: {scenario}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Privacy Budget (ε)", fontsize=11)
        ax.set_ylabel("TRIR (%)" if ax_idx == 0 else "", fontsize=11)
        tick_center = (n_setups * len(models) * bar_width) / 2
        ax.set_xticks(x + tick_center)
        ax.set_xticklabels([f"ε={e}" for e in epsilons], fontsize=9)
        max_val = df_s["mean_TRIR"].max() if len(df_s) > 0 else 25
        ax.set_ylim(0, max(30, max_val * 1.5))
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, fontsize=7.5,
        bbox_to_anchor=(0.5, -0.18), frameon=True,
        title="Setup (solid=RoBERTa-Base, hatched=RoBERTa-Clinical) — Selective: τ = μ + k·σ"
    )
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = output_dir / f"trir_barplot_{task}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()
 
# ── Line Plot ─────────────────────────────────────────────────────────────────
def make_line_plot(agg, task, output_dir=Path(".")):
    df_task = agg[agg["task"] == task].copy()
    if len(df_task) == 0:
        print(f"  No data for task={task}, skipping")
        return
 
    epsilons = [25, 50, 75, 100]
    sigmas   = ["1.0", "1.5", "2.0"]
    models   = ["roberta_base", "roberta_clinical"]
 
    colors    = {"1.0": "#2196F3", "1.5": "#FF9800", "2.0": "#4CAF50", "full": "#F44336"}
    markers   = {"roberta_base": "o", "roberta_clinical": "s"}
    linesty   = {"roberta_base": "-",  "roberta_clinical": "--"}
    mlabels   = {"roberta_base": "RoBERTa-Base", "roberta_clinical": "RoBERTa-Clinical"}
 
    task_label = "Mortality Prediction" if task == "mp" else "Procedure Prediction"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        f"TRI Re-Identification Risk (%) — {task_label}\n(Mean ± std over 5 seeds)",
        fontsize=13, fontweight="bold"
    )
 
    for ax_idx, scenario in enumerate(SCENARIOS):
        ax = axes[ax_idx]
        df_s = df_task[df_task["scenario"] == scenario]
 
        for model in models:
            # Rare sentence rewriting
            for sigma in sigmas:
                vals, errs = [], []
                for eps in epsilons:
                    row = df_s[
                        (df_s["setup_type"] == "rare_sentence") &
                        (df_s["sigma"].astype(str) == str(sigma)) &
                        (df_s["epsilon"] == eps) &
                        (df_s["model"] == model)
                    ]
                    vals.append(float(row["mean_TRIR"].values[0]) if len(row) > 0 else np.nan)
                    errs.append(float(row["std_TRIR"].values[0]) if len(row) > 0 and not pd.isna(row["std_TRIR"].values[0]) else 0)
 
                ax.errorbar(
                    epsilons, vals, yerr=errs,
                    label=f"Selective DP-MLM k={sigma} (τ=μ+{sigma}σ) — {mlabels[model]}",
                    color=colors[sigma],
                    linestyle=linesty[model],
                    marker=markers[model],
                    capsize=4, linewidth=1.5, markersize=5
                )
 
            # Full rewriting
            vals_f, errs_f = [], []
            for eps in epsilons:
                row = df_s[
                    (df_s["setup_type"] == "full_rewriting") &
                    (df_s["epsilon"] == eps) &
                    (df_s["model"] == model)
                ]
                vals_f.append(float(row["mean_TRIR"].values[0]) if len(row) > 0 else np.nan)
                errs_f.append(float(row["std_TRIR"].values[0]) if len(row) > 0 and not pd.isna(row["std_TRIR"].values[0]) else 0)
 
            ax.errorbar(
                epsilons, vals_f, yerr=errs_f,
                label=f"Full DP-MLM (all sentences) — {mlabels[model]}",
                color=colors["full"],
                linestyle=linesty[model],
                marker=markers[model],
                capsize=4, linewidth=2, markersize=5
            )
 
        ax.set_title(f"Scenario: {scenario}", fontsize=12, fontweight="bold")
        ax.set_xlabel("ε (Privacy Budget)", fontsize=11)
        ax.set_ylabel("TRIR (%)" if ax_idx == 0 else "", fontsize=11)
        ax.set_xticks(epsilons)
        ax.yaxis.grid(True, alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
 
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=4, fontsize=8,
        bbox_to_anchor=(0.5, -0.2), frameon=True
    )
 
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = output_dir / f"trir_lineplot_{task}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close()
 
# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    output_dir = trir_averaged_csv(ARGS.base).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
 
    print("="*60)
    print("Loading results...")
    print("="*60)
    df_all = load_all_results()
 
    print("\n" + "="*60)
    print("Computing averages...")
    print("="*60)
    agg = compute_averages(df_all)
 
    out_csv = trir_averaged_csv(ARGS.base)
    agg.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
 
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY (mean TRIR % over seeds)")
    print("="*60)
    for task in TASKS:
        sub = agg[agg["task"] == task].sort_values(
            ["scenario", "setup_type", "sigma", "epsilon", "model"]
        )
        if len(sub) > 0:
            print(f"\nTask: {task.upper()}")
            print(sub[["scenario","setup_type","sigma","epsilon","model",
                        "mean_TRIR","std_TRIR","n_seeds"]].to_string(index=False))

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    for task in TASKS:
        if ARGS.no_plots:
            break
        print(f"\nTask: {task.upper()}")
        make_bar_plot(agg,  task, output_dir)
        make_line_plot(agg, task, output_dir)

    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nOutput files:")
    for task in TASKS:
        print(f"  trir_barplot_{task}.png")
        print(f"  trir_lineplot_{task}.png")
    print(f"  trir_averaged_results.csv")