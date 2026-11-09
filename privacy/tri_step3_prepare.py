"""
tri_step3_prepare.py  (thesis: step3_prepare_tri_data.py)
Prepares TRI input data for both 50E and 500R scenarios.
Creates ONE wide CSV per scenario (following the paper's structure).

CSV format:
  - Rows = patients
  - First column = id
  - Second column = original_background (Qwen from original notes - attacker's knowledge)
  - original_notes = the unprivatized notes (gives TRIR_original, the baseline)
  - 24 selective columns sigma<k>_eps<e>_<model> + 8 full columns full_eps<e>_<model>
    = privatized clinical notes (rebuilt step-7 notes, NOT Qwen backgrounds)
  - NO label column (tri.py uses id directly)

This measures: "Can attacker re-identify privatized clinical notes using public biographical info?"

Change 2026-11: --base argument; note files located via common/layout.py.

Usage:
  python privacy/tri_step3_prepare.py --seed 42 --task mp --base .
"""

import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, tri_run_dir, SIGMAS, EPSILONS, MODELS  # noqa: E402
import pandas as pd
import numpy as np
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--seed",     type=int, default=42)
p.add_argument("--base",     default=".")
p.add_argument("--task",     default="mp", choices=["mp", "proc"],
               help="Task: mp (mortality) or proc (procedures)")
args = p.parse_args()

L       = Layout(args.base, args.task)
RUN_DIR = tri_run_dir(args.base, args.seed)
TRI_DIR = RUN_DIR / f"tri_data_{args.task}"
TRI_DIR.mkdir(exist_ok=True, parents=True)

EPSILONS = [int(e) for e in EPSILONS]       # column names use 25, 50, ...
MODELS   = list(MODELS)
BASELINE_EPSILONS = EPSILONS                # Full DP-MLM at every epsilon

# ── Load fixed files ──────────────────────────────────────────────────────────
bg_orig = pd.read_csv(RUN_DIR / f"bg_{args.task}_original.csv")
bg_dist = pd.read_csv(RUN_DIR / f"bg_{args.task}_distractors_500R.csv")

sampled_ids = pd.read_csv(RUN_DIR / f"{args.task}_sampled_50_ids.csv")["id"].tolist()

bg_orig = bg_orig[bg_orig["id"].isin(sampled_ids)].reset_index(drop=True)

print(f"Seed {args.seed}, Task {args.task}")
print(f"Real patients   : {len(bg_orig)}")
print(f"Distractors     : {len(bg_dist)}")
print(f"Output dir      : {TRI_DIR}")
print(f"Task-specific dir to avoid mp/proc conflicts")

# ── Build 50E wide CSV ────────────────────────────────────────────────────────
print("\nBuilding 50E dataset...")
df_50E = bg_orig[["id", "background"]].copy()
df_50E.rename(columns={"background": "original_background"}, inplace=True)

# ── Add original_notes column for TRIR_baseline computation ──────────────────
# This column contains the ORIGINAL (non-privatized) clinical notes.
# When tri.py tests on this column:
#   train = Qwen background from original notes
#   test  = the actual original clinical notes
# → gives TRIR_original: upper bound re-identification risk without privatization
# → needed for γ = privacy_gain - utility_loss computation
print("  Adding original_notes column (baseline TRIR)...")
original_notes_csv = L.segmented
df_orig_notes = pd.read_csv(original_notes_csv)
df_orig_matched = df_orig_notes[df_orig_notes["id"].isin(sampled_ids)][["id", "text"]].copy()
df_orig_matched = df_orig_matched.rename(columns={"text": "original_notes"})
df_50E = df_50E.merge(df_orig_matched, on="id", how="left")
print(f"    ✓ original_notes: {df_50E['original_notes'].notna().sum()}/{len(df_50E)} patients")

# Add each rare-sentence anonymization as a column
missing_files = []
print("  Adding rare-sentence rewriting columns...")
for sigma in SIGMAS:
    for eps in EPSILONS:
        for model in MODELS:
            setup_name = f"sigma{sigma}_eps{eps}_{model}"
            col_name   = setup_name
            # Read PRIVATIZED NOTES from rare-sentence rewriting only
            priv_notes_csv = L.notes_file(f"selective_k{sigma}", model, eps)
            
            if not priv_notes_csv.exists():
                missing_files.append(setup_name)
                continue  # no column: tri.py fails on an all-empty column
            
            priv_df = pd.read_csv(priv_notes_csv)
            priv_matched = priv_df[priv_df["id"].isin(sampled_ids)].copy()
            
            # Use note_private column (the actual privatized clinical text)
            temp = priv_matched[["id", "note_private"]].rename(columns={"note_private": col_name})
            df_50E = df_50E.merge(temp, on="id", how="left")

# Add full rewriting baseline columns
print("  Adding full rewriting baseline columns...")
for eps in BASELINE_EPSILONS:
    for model in MODELS:
        setup_name = f"full_eps{eps}_{model}"
        col_name   = setup_name
        
        # Read from step7_baseline_notes
        baseline_csv = L.notes_file("full", model, eps)
        
        if not baseline_csv.exists():
            missing_files.append(setup_name)
            continue  # no column: tri.py fails on an all-empty column
        
        baseline_df = pd.read_csv(baseline_csv)
        baseline_matched = baseline_df[baseline_df["id"].isin(sampled_ids)].copy()
        
        # Use note_private column
        temp = baseline_matched[["id", "note_private"]].rename(columns={"note_private": col_name})
        df_50E = df_50E.merge(temp, on="id", how="left")

# Save 50E
df_50E.to_csv(TRI_DIR / "tri_50E.csv", index=False)
n_rare = len(SIGMAS) * len(EPSILONS) * len(MODELS)
n_full = len(BASELINE_EPSILONS) * len(MODELS)
has_original = "original_notes" in df_50E.columns
print(f"✓ 50E: {len(df_50E)} patients × {len(df_50E.columns)} columns")
print(f"  - 1 original_background column")
print(f"  - 1 original_notes column (baseline TRIR): {'YES' if has_original else 'NO'}")
print(f"  - {n_rare} rare-sentence rewriting columns")
print(f"  - {n_full} full rewriting baseline columns")
print(f"  Saved: {TRI_DIR / 'tri_50E.csv'}")

# ── Build 500R wide CSV ───────────────────────────────────────────────────────
print("\nBuilding 500R dataset...")

# Start with 50 real patients (same as 50E)
df_500R = df_50E.copy()

# Add 450 distractors (only have original_background, no anonymizations)
dist_df = bg_dist[["id", "background"]].copy()
dist_df.rename(columns={"background": "original_background"}, inplace=True)

# Add NaN columns for all anonymizations
for col in df_50E.columns:
    if col not in ["id", "original_background"]:
        dist_df[col] = None

# Concatenate and shuffle
df_500R = pd.concat([df_500R, dist_df], ignore_index=True)
df_500R = df_500R.sample(frac=1, random_state=args.seed).reset_index(drop=True)

# Save 500R
df_500R.to_csv(TRI_DIR / "tri_500R.csv", index=False)
print(f"✓ 500R: {len(df_500R)} patients × {len(df_500R.columns)} columns")
print(f"  - Real patients: 50 (with all anonymizations)")
print(f"  - Distractors: 450 (only original_background)")
print(f"  Saved: {TRI_DIR / 'tri_500R.csv'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("DONE!")
print(f"  50E:  {TRI_DIR / 'tri_50E.csv'}")
print(f"  500R: {TRI_DIR / 'tri_500R.csv'}")
if missing_files:
    print(f"\nWarning: {len(missing_files)} anonymization files missing:")
    for f in missing_files[:5]:
        print(f"  - {f}")
    if len(missing_files) > 5:
        print(f"  ... and {len(missing_files)-5} more")
print("="*60)