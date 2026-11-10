"""
tri_step4_attack.py  (thesis: step4_run_tri.py)
Runs the TRI (Text Re-Identification) classifier for all setups.
 
How tri.py works:
  - Reads ONE wide CSV with: id, background_column, and N protected_columns
  - Trains classifier on background_column (Qwen biographies from original notes)
  - Tests re-identification on EACH protected column (raw privatized clinical notes)
  - Returns N TRIR scores (one per protected column)
 
Our approach:
  1. For each scenario (50E, 500R):
     a. Create config.json pointing to tri_{scenario}.csv
     b. Run tri.py once
     c. Parse results (33 TRIR scores per scenario: original_notes + 24 rare-sentence + 8 full-rewriting)
  2. Save summary CSV with 66 results per task (2 scenarios × 33 configs)
 
CSV structure:
  - Columns: id, original_background (Qwen), original_notes, sigma*_eps*_roberta_base, full_eps*_...
  - 50E: 50 rows (all real patients)
  - 500R: 500 rows (50 real + 450 distractors with NaN in anonymization columns)
 
Measures: Can attacker re-identify privatized clinical notes using public biographical info?
 
TRI configuration used in the thesis (MIMIC_CONFIG below): DistilBERT
(distilbert-base-uncased), NO additional pre-training, 15 fine-tuning epochs,
batch size 16, learning rate 5e-5, block size 250, sliding window 100-25.

Change 2026-11: --base argument; tri.py is taken from external/tri_repo
(cloned by scripts/setup_external.sh).

Usage:
  python privacy/tri_step4_attack.py --seed 42 --task mp --base .
  python privacy/tri_step4_attack.py --seed 42 --task proc --base . --scenarios 50E
"""
 
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import tri_run_dir, repo_root, trir_averaged_csv, TRI_SEEDS  # noqa: E402
import csv
import json
import shutil
import subprocess
import pandas as pd
from pathlib import Path
 
# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--seed",       type=int, default=42,
               help="Random seed (42, 123, 456, 789, 1337)")
p.add_argument("--base",       default=".")
p.add_argument("--task",       default="mp", choices=["mp", "proc"],
               help="Task: mp (mortality) or proc (procedures)")
p.add_argument("--scenarios",  default="50E,500R",
               help="Scenarios to run: 50E,500R or just 50E or 500R")
p.add_argument("--base_model", default="distilbert-base-uncased",
               help="TRI attacker model (thesis: distilbert-base-uncased)")
p.add_argument("--finetuning_epochs", type=int, default=15, help="thesis: 15")
args = p.parse_args()
 
SCENARIOS = [s.strip() for s in args.scenarios.split(",")]
 
TRI_REPO = repo_root() / "external" / "tri_repo"
RUN_DIR  = tri_run_dir(args.base, args.seed)
TRI_DATA = RUN_DIR / f"tri_data_{args.task}"
TRI_RES  = RUN_DIR / f"tri_results_{args.task}"
TRI_RES.mkdir(exist_ok=True, parents=True)
 
print(f"{'='*60}")
print(f"TRI Classifier — seed={args.seed}, task={args.task}")
print(f"Scenarios      : {SCENARIOS}")
print(f"Data dir       : {TRI_DATA}")
print(f"Results dir    : {TRI_RES}")
print(f"Task-specific dirs to avoid mp/proc conflicts")
print(f"{'='*60}\n")
 
# ── Config template for MIMIC ─────────────────────────────────────────────────
MIMIC_CONFIG = {
    # ── Data columns ──────────────────────────────────────────────────────────
    "individual_name_column":               "id",
    "background_knowledge_column":          "original_background",
 
    # ── Anonymization settings ────────────────────────────────────────────────
    "anonymize_background_knowledge":       False,
    "only_use_anonymized_background_knowledge": False,
    "add_non_saved_anonymizations":         False,
 
    # ── Caching settings (all false = always train fresh) ─────────────────────
    "load_saved_pretreatment":              False,
    "save_pretreatment":                    False,
    "load_saved_finetuning":                False,
    "save_finetuning":                      False,
    "load_saved_pretraining":               False,
    "save_additional_pretraining":          False,
 
    # ── Document curation ─────────────────────────────────────────────────────
    "use_document_curation":                False,
 
    # ── Model ─────────────────────────────────────────────────────────────────
    "base_model_name":                      "distilbert-base-uncased",
    "tokenization_block_size":              250,
 
    # ── Pre-training (disabled for now) ───────────────────────────────────────
    "use_additional_pretraining":           False,
 
    # ── Fine-tuning (classification) ──────────────────────────────────────────
    "finetuning_epochs":                    150,
    "finetuning_batch_size":                16,
    "finetuning_learning_rate":             5e-5,
    "finetuning_sliding_window":            "100-25",
}


def parse_tri_results(result_dir: Path) -> dict:
    """
    Robustly parse TRI results from tri.py output directory.

    Expected Results.csv format:
      Row 0: timestamp or metadata
      Row 1: Time,Epoch,<protected_col_1>,...,Average
      Row 2+: timestamp,epoch,trir1,...,avg

    Why robust parsing is needed:
      After adding original_notes, the result width changes. If an old
      Results.csv is reused/appended, pandas.read_csv can fail with errors like
      "Expected 29 fields, saw 36". This parser finds the latest valid header
      and only reads rows with the same number of fields.
    """
    results_csv = result_dir / "Results.csv"
    if results_csv.exists():
        with open(results_csv, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        if not rows:
            return {}

        # Find all candidate header rows and use the LAST one with the widest schema.
        # This protects against stale Results.csv files containing old headers.
        header_candidates = [
            (idx, row) for idx, row in enumerate(rows)
            if len(row) >= 3 and row[0] == "Time" and row[1] == "Epoch"
        ]
        if not header_candidates:
            print(f"WARNING: Could not find a Time,Epoch header in {results_csv}")
            return {}

        max_width = max(len(row) for _, row in header_candidates)
        header_idx, header = [item for item in header_candidates if len(item[1]) == max_width][-1]
        expected_len = len(header)

        valid_rows = []
        skipped_rows = 0
        repeated_headers = 0

        for row in rows[header_idx + 1:]:
            if not row:
                continue
            if len(row) >= 2 and row[0] == "Time" and row[1] == "Epoch":
                repeated_headers += 1
                continue
            if len(row) != expected_len:
                skipped_rows += 1
                continue
            valid_rows.append(row)

        if skipped_rows:
            print(
                f"WARNING: skipped {skipped_rows} stale/malformed rows in {results_csv} "
                f"(expected {expected_len} fields)."
            )
        if repeated_headers:
            print(f"WARNING: skipped {repeated_headers} repeated header rows in {results_csv}.")

        if not valid_rows:
            print(f"WARNING: No valid result rows found in {results_csv}")
            return {}

        last_row = valid_rows[-1]
        results = {}
        for col, val in zip(header, last_row):
            if col in ["Time", "Epoch", "Average"]:
                continue
            try:
                results[col] = float(val)
            except (ValueError, TypeError):
                continue
        return results

    # Try results.json as fallback
    results_json = result_dir / "results.json"
    if results_json.exists():
        with open(results_json) as f:
            return json.load(f)

    return {}


def parse_setup_metadata(col_name: str):
    """Return (setup_type, sigma, eps, model) for a TRI protected-column name."""
    if col_name == "original_notes":
        return "original", None, None, None

    if col_name.startswith("sigma"):
        # Example: sigma1.0_eps25_roberta_base
        parts = col_name.replace("sigma", "", 1).split("_")
        sigma = parts[0]
        eps = int(parts[1].replace("eps", ""))
        model = "_".join(parts[2:])
        return "rare_sentence", sigma, eps, model

    if col_name.startswith("full_"):
        # Example: full_eps100_roberta_base
        parts = col_name.replace("full_", "", 1).split("_")
        eps = int(parts[0].replace("eps", ""))
        model = "_".join(parts[1:])
        return "full_rewriting", None, eps, model

    return None, None, None, None


# ── Main loop ─────────────────────────────────────────────────────────────────
all_results = []

for scenario in SCENARIOS:
    csv_file = TRI_DATA / f"tri_{scenario}.csv"
    
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario}")
    print(f"CSV: {csv_file}")
    print(f"{'='*60}")
    
    if not csv_file.exists():
        print(f"ERROR: {csv_file} not found — run step3_prepare_tri_data.py first")
        continue
    
    # Check CSV structure
    df = pd.read_csv(csv_file, nrows=5)
    print(f"Preview: {len(df)} rows sampled")
    print(f"Columns: {df.columns.tolist()}")
    
    # Validate expected protected-column structure.
    protected_cols = [c for c in df.columns if c not in ["id", "original_background"]]
    print(f"Protected columns to evaluate: {len(protected_cols)}")
    if "original_notes" not in protected_cols:
        print("WARNING: original_notes is missing; TRIR_original will not be computed.")

    # Output directory for this scenario
    out_dir = TRI_RES / f"{scenario}"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # Check if already done
    done_flag = out_dir / "done.flag"
    if done_flag.exists():
        print(f"SKIP — {scenario} already completed")
        tri_results = parse_tri_results(out_dir)
        for col_name, trir in tri_results.items():
            setup_type, sigma, eps, model = parse_setup_metadata(col_name)
            if setup_type is None:
                continue
            all_results.append({
                "task": args.task,
                "scenario": scenario,
                "setup_type": setup_type,
                "sigma": sigma,
                "epsilon": eps,
                "model": model,
                "column": col_name,
                "TRIR": trir,
                "status": "done"
            })
        continue

    # If this scenario was started before but not completed, remove stale files.
    # This prevents tri.py from appending new 36-field rows to an old Results.csv
    # with fewer columns, which causes pandas ParserError.
    if any(out_dir.iterdir()):
        print(f"Cleaning incomplete/stale output directory: {out_dir}")
        shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
    
    # Create config.json
    config = dict(MIMIC_CONFIG)
    config["data_file_path"]     = str(csv_file)
    config["output_folder_path"] = str(out_dir)
    config["base_model_name"]    = args.base_model
    config["finetuning_epochs"]  = args.finetuning_epochs
    
    cfg_path = out_dir / "config.json"
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfig: {cfg_path}")
    print(f"Output: {out_dir}")
    
    # Run tri.py
    print(f"\nRunning tri.py for {scenario}...")
    result = subprocess.run(
        [sys.executable, str(TRI_REPO / "tri.py"), str(cfg_path)],
        capture_output=True, text=True,
        cwd=str(TRI_REPO)
    )
    
    # Always print tri.py output for debugging
    if result.stdout:
        print(f"\n--- tri.py STDOUT ---")
        print(result.stdout[-2000:])  # Last 2000 chars
    if result.stderr:
        print(f"\n--- tri.py STDERR ---")
        print(result.stderr[-2000:])
    
    # Check result
    if result.returncode != 0:
        print(f"ERROR (returncode={result.returncode})")
        all_results.append({
            "task": args.task,
            "scenario": scenario,
            "setup_type": None,
            "sigma": None,
            "epsilon": None,
            "model": None,
            "column": None,
            "TRIR": None,
            "status": "error"
        })
        continue
    
    # Parse results
    tri_results = parse_tri_results(out_dir)
    if not tri_results:
        print(f"WARNING: No results found in {out_dir}")
        all_results.append({
            "task": args.task,
            "scenario": scenario,
            "setup_type": None,
            "sigma": None,
            "epsilon": None,
            "model": None,
            "column": None,
            "TRIR": None,
            "status": "no_results"
        })
        continue
    
    print(f"\n✓ Success! Found {len(tri_results)} TRIR scores:")
    for col_name, trir in tri_results.items():
        setup_type, sigma, eps, model = parse_setup_metadata(col_name)
        if setup_type is None:
            print(f"  WARNING: Unknown column format '{col_name}', skipping")
            continue

        print(f"  {col_name}: TRIR = {trir:.4f}")
        all_results.append({
            "task": args.task,
            "scenario": scenario,
            "setup_type": setup_type,
            "sigma": sigma,
            "epsilon": eps,
            "model": model,
            "column": col_name,
            "TRIR": trir,
            "status": "ok"
        })
    
    # Mark done
    done_flag.write_text("done")
    
    # Save intermediate results
    pd.DataFrame(all_results).to_csv(
        TRI_RES / "tri_summary.csv", index=False
    )

# ── Final summary ─────────────────────────────────────────────────────────────
df_summary = pd.DataFrame(all_results)
out_csv = TRI_RES / "tri_summary.csv"
df_summary.to_csv(out_csv, index=False)
 
print(f"\n{'='*60}")
print(f"FINAL SUMMARY — seed {args.seed}, task {args.task}")
print(f"{'='*60}")
 
for scenario in SCENARIOS:
    sub = df_summary[df_summary["scenario"] == scenario]
    print(f"\n{scenario} ({len(sub)} tests):")
    if len(sub) > 0:
        print(sub[["setup_type", "model", "sigma", "epsilon", "TRIR"]].to_string(index=False))
 
print(f"\nAll results saved to: {out_csv}")
print(f"Total configurations: {len(df_summary)}")
#print(f"Success rate: {(df_summary['status']=='ok').sum()}/{len(df_summary)}")