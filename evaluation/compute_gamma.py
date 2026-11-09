"""
compute_gamma.py
Computes privacy-utility trade-off (γ) as defined in the paper (Section 4.4):

    γ = (1/2) Σ_i (1 - P_i^p/P_i^b) - (1/3) Σ_j (1 - U_j^p/U_j^b)

Where:
    i ∈ {TRI, LLM}    → privacy metrics (lower is better)
    j ∈ {Util, S, C}  → utility metrics (higher is better)
    b = baseline (non-private), p = privatized

For MIMIC-III, we use:
    Privacy: TRI (TRIR)
    Utility: Util (ROC_AUC), S (semantic similarity), C (coherence)

Simplified formula:
    γ = (1 - TRIR_p/TRIR_b) - (1/3) × [(1 - ROC_p/ROC_b) + (1 - S_p/S_b) + (1 - C_p/C_b)]

Positive γ → privacy gains outweigh utility losses ✓
Negative γ → utility losses outweigh privacy gains ✗

Changes 2026-11: --base argument; inputs/outputs from common/layout.py
(results/trir_averaged_results.csv, results/utility_metrics_<task>.csv,
results/classification_<task>_averaged.csv -> results/gamma_tradeoff_<task>.csv).
The formula is unchanged.

Usage:
    python evaluation/compute_gamma.py --task mp   --base .
    python evaluation/compute_gamma.py --task proc --base .
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--task", required=True, choices=["mp", "proc"])
p.add_argument("--base", default=".")
args = p.parse_args()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, trir_averaged_csv  # noqa: E402

L = Layout(args.base, args.task)
OUT_DIR = L.figures_dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files
TRIR_CSV        = trir_averaged_csv(args.base)
UTILITY_CSV     = L.utility_csv
GAMMA_CSV       = L.gamma_csv

print("="*70)
print(f"Computing Privacy-Utility Trade-off (γ) — Task: {args.task.upper()}")
print("="*70)

# ── Load TRIR results ─────────────────────────────────────────────────────────
print("\n[1/4] Loading TRIR (privacy metric)...")
if not TRIR_CSV.exists():
    raise FileNotFoundError(f"TRIR file not found: {TRIR_CSV}")

df_trir = pd.read_csv(TRIR_CSV)
df_trir = df_trir[df_trir["task"] == args.task].copy()
print(f"  Loaded {len(df_trir)} TRIR configurations")

# ── TRIR baselines: one per attack scenario ──────────────────────────────────
original_trir = df_trir[df_trir["setup_type"] == "original"].copy()

if original_trir["scenario"].duplicated().any():
    raise ValueError("More than one original TRIR baseline found for a scenario.")

trir_baselines = original_trir.set_index("scenario")["mean_TRIR"].to_dict()
print(f"  TRIR baselines by scenario:")
for scenario, trir_val in trir_baselines.items():
    print(f"    {scenario}: {trir_val:.2f}%")

# ── Load Utility metrics (S, C) ───────────────────────────────────────────────
print("\n[2/4] Loading Utility metrics (S, C)...")
if not UTILITY_CSV.exists():
    raise FileNotFoundError(f"Utility file not found: {UTILITY_CSV}")

df_util = pd.read_csv(UTILITY_CSV)
print(f"  Loaded {len(df_util)} utility configurations")

# Original utility baselines: by definition, original vs original = 1.0
# S (semantic similarity): original text compared to itself = 1.0
# C (coherence): PPL_original / PPL_original = 1.0
S_b_global = 1.0
C_b_global = 1.0
print(f"  Baseline S (semantic similarity): {S_b_global:.4f} (original vs original)")
print(f"  Baseline C (coherence): {C_b_global:.4f} (original vs original)")

# ── Load ROC_AUC from averaged results ───────────────────────────────────────
print("\n[3/4] Loading ROC_AUC (downstream task performance)...")

# Load averaged results CSV
roc_csv = L.classification_avg
if not roc_csv.exists():
    raise FileNotFoundError(f"Averaged results file not found: {roc_csv}")

df_roc = pd.read_csv(roc_csv)
print(f"  Loaded {len(df_roc)} ROC_AUC configurations")

# Extract original baselines per model
# For MP: original rows have model=roberta_base or roberta_clinical
# Take the mean if epsilon varies (they should be the same)
roc_baselines = {}

for model_name in ["roberta_base", "roberta_clinical"]:
    baseline_rows = df_roc[
        (df_roc["setup"] == "original") &
        (df_roc["model"] == model_name)
    ]
    
    if len(baseline_rows) == 0:
        raise ValueError(f"Missing baseline ROC-AUC for {model_name} in {roc_csv}")
    
    # Take mean (should all be the same value)
    roc_baselines[model_name] = float(baseline_rows["roc_auc_mean"].mean())

print(f"  ROC_AUC baselines by model:")
for model_name, roc_val in roc_baselines.items():
    print(f"    {model_name}: {roc_val:.4f}")

# ── Compute Gamma for all configurations ─────────────────────────────────────
print("\n[4/4] Computing γ (privacy-utility trade-off)...")

results = []

for _, row in df_trir.iterrows():
    setup_type = row["setup_type"]
    
    # Skip original baseline (γ undefined for baseline)
    if setup_type == "original":
        continue
    
    scenario = row["scenario"]
    sigma = row["sigma"] if pd.notna(row["sigma"]) else None
    epsilon = row["epsilon"] if pd.notna(row["epsilon"]) else None
    model = row["model"] if pd.notna(row["model"]) else None
    
    # Validate scenario and model have baselines
    if scenario not in trir_baselines:
        raise ValueError(f"Missing TRIR baseline for scenario={scenario}")
    
    if model not in roc_baselines:
        raise ValueError(f"Missing ROC baseline for model={model}")
    
    # Get scenario-specific TRIR baseline
    TRIR_b = float(trir_baselines[scenario])
    TRIR_p = float(row["mean_TRIR"])
    
    # Get model-specific ROC baseline
    ROC_b = float(roc_baselines[model])
    S_b = S_b_global
    C_b = C_b_global
    
    # Get corresponding S, C from utility metrics
    if setup_type == "rare_sentence":
        util_row = df_util[
            (df_util["setup"] == f"rare_sigma_{sigma}") &
            (df_util["model"] == model) &
            (df_util["epsilon"] == epsilon)
        ]
    elif setup_type == "full_rewriting":
        util_row = df_util[
            (df_util["setup"] == "all_sentences") &
            (df_util["model"] == model) &
            (df_util["epsilon"] == epsilon)
        ]
    else:
        raise ValueError(f"Unexpected setup type: {setup_type}")
    
    if len(util_row) != 1:
        raise ValueError(
            f"Expected one utility row, found {len(util_row)} "
            f"for scenario={scenario}, setup={setup_type}, "
            f"sigma={sigma}, epsilon={epsilon}, model={model}"
        )
    
    S_p = float(util_row["S_avg"].iloc[0])
    C_p = float(util_row["C"].iloc[0])
    
    # Get ROC_AUC from averaged results
    # Map setup_type to setup column name in CSV
    if setup_type == "rare_sentence":
        setup_name = f"rare_sigma_{sigma}"
    elif setup_type == "full_rewriting":
        setup_name = "all_sentences"
    else:
        raise ValueError(f"Unexpected setup type: {setup_type}")
    
    roc_row = df_roc[
        (df_roc["setup"] == setup_name) &
        (df_roc["model"] == model) &
        (df_roc["epsilon"] == epsilon)
    ]
    
    if len(roc_row) != 1:
        raise ValueError(
            f"Expected one ROC_AUC row, found {len(roc_row)} "
            f"for setup={setup_name}, epsilon={epsilon}, model={model}"
        )
    
    ROC_p = float(roc_row["roc_auc_mean"].iloc[0])
    
    # Validate baselines are positive
    if min(TRIR_b, ROC_b, S_b, C_b) <= 0:
        raise ValueError(
            "Gamma cannot be computed with a zero or negative baseline."
        )
    
    # Compute privacy gain (positive = privacy improved)
    privacy_gain = 1.0 - TRIR_p / TRIR_b
    
    # Compute utility losses (positive = utility degraded)
    util_loss_roc = 1.0 - ROC_p / ROC_b
    util_loss_S = 1.0 - S_p / S_b
    util_loss_C = 1.0 - C_p / C_b
    
    # Average utility loss
    avg_util_loss = np.mean([util_loss_roc, util_loss_S, util_loss_C])
    
    # Compute gamma: γ = privacy_gain - avg_utility_loss
    # Positive γ → privacy gains outweigh utility losses
    gamma = privacy_gain - avg_util_loss
    
    results.append({
        "task": args.task,
        "scenario": scenario,
        "setup_type": setup_type,
        "sigma": sigma,
        "epsilon": epsilon,
        "model": model,
        # Privacy
        "TRIR_baseline": TRIR_b,
        "TRIR_private": TRIR_p,
        "privacy_gain": privacy_gain,
        # Utility
        "ROC_baseline": ROC_b,
        "ROC_private": ROC_p,
        "S_baseline": S_b,
        "S_private": S_p,
        "C_baseline": C_b,
        "C_private": C_p,
        # Losses
        "util_loss_ROC": util_loss_roc,
        "util_loss_S": util_loss_S,
        "util_loss_C": util_loss_C,
        "avg_util_loss": avg_util_loss,
        # Final
        "gamma": gamma,
    })

df_gamma = pd.DataFrame(results)

# Save results
df_gamma.to_csv(GAMMA_CSV, index=False)
print(f"\n✓ Saved results to: {GAMMA_CSV}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("GAMMA SUMMARY")
print("="*70)
print(f"\n{'Setup':<25} {'σ':<5} {'ε':<8} {'Model':<20} {'γ':>8}")
print("-"*70)

for _, row in df_gamma.sort_values(["setup_type", "sigma", "epsilon", "model"]).iterrows():
    setup_label = "Selective DP-MLM" if row["setup_type"] == "rare_sentence" else "Full DP-MLM"
    sigma_str = f"{row['sigma']}" if pd.notna(row['sigma']) else "all"
    gamma_val = row["gamma"]
    gamma_str = f"{gamma_val:+.4f}" if not np.isnan(gamma_val) else "N/A"
    
    print(f"{setup_label:<25} {sigma_str:<5} {row['epsilon']:<8.0f} {row['model']:<20} {gamma_str:>8}")

# Best configurations
print("\n" + "="*70)
print("TOP 5 CONFIGURATIONS (highest γ = best privacy-utility balance)")
print("="*70)
top5 = df_gamma.nlargest(5, "gamma")
for i, (_, row) in enumerate(top5.iterrows(), 1):
    setup_label = "Selective" if row["setup_type"] == "rare_sentence" else "Full"
    sigma_str = f"σ={row['sigma']}" if pd.notna(row['sigma']) else "all"
    print(f"{i}. {setup_label} {sigma_str}, ε={row['epsilon']:.0f}, {row['model']}: γ = {row['gamma']:+.4f}")

# ── Generate Plot ─────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("Generating plot...")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    f"Privacy-Utility Trade-off (γ) — {args.task.upper()}\n"
    f"γ = privacy_gain - avg_utility_loss (higher is better)",
    fontsize=13, fontweight="bold"
)

scenarios = ["50E", "500R"]
colors = {"1.0": "#2196F3", "1.5": "#FF9800", "2.0": "#4CAF50", "all": "#F44336"}
markers = {"roberta_base": "o", "roberta_clinical": "s"}
linestyles = {"roberta_base": "-", "roberta_clinical": "--"}

for ax_idx, scenario in enumerate(scenarios):
    ax = axes[ax_idx]
    df_s = df_gamma[df_gamma["scenario"] == scenario]
    
    epsilons = sorted(df_s["epsilon"].dropna().unique())
    
    for setup_type in ["rare_sentence", "full_rewriting"]:
        df_setup = df_s[df_s["setup_type"] == setup_type]
        
        if setup_type == "rare_sentence":
            sigmas = ["1.0", "1.5", "2.0"]
            for sigma in sigmas:
                for model in ["roberta_base", "roberta_clinical"]:
                    df_line = df_setup[
                        (df_setup["sigma"] == float(sigma)) &
                        (df_setup["model"] == model)
                    ].sort_values("epsilon")
                    
                    if len(df_line) > 0:
                        label = f"Selective k={sigma} — {model.replace('_', '-')}"
                        ax.plot(
                            df_line["epsilon"], df_line["gamma"],
                            label=label,
                            color=colors[sigma],
                            marker=markers[model],
                            linestyle=linestyles[model],
                            linewidth=1.5,
                            markersize=6
                        )
        else:  # full_rewriting
            for model in ["roberta_base", "roberta_clinical"]:
                df_line = df_setup[df_setup["model"] == model].sort_values("epsilon")
                
                if len(df_line) > 0:
                    label = f"Full DP-MLM — {model.replace('_', '-')}"
                    ax.plot(
                        df_line["epsilon"], df_line["gamma"],
                        label=label,
                        color=colors["all"],
                        marker=markers[model],
                        linestyle=linestyles[model],
                        linewidth=2,
                        markersize=6
                    )
    
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_title(f"Scenario: {scenario}", fontsize=12, fontweight="bold")
    ax.set_xlabel("ε (Privacy Budget)", fontsize=11)
    ax.set_ylabel("γ (Privacy-Utility Trade-off)" if ax_idx == 0 else "", fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="lower center", ncol=4, fontsize=8,
    bbox_to_anchor=(0.5, -0.12), frameon=True
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plot_path = OUT_DIR / f"gamma_tradeoff_{args.task}.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved plot to: {plot_path}")
plt.close()

print("\n" + "="*70)
print("DONE")
print("="*70)
print(f"Results: {GAMMA_CSV}")
print(f"Plot:    {plot_path}")
