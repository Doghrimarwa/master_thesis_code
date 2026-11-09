"""
collect_classification.py — gathers the step-8c ROC-AUC results of all
classification seeds and averages them.

Writes
    results/classification_<task>_seed_<seed>.csv   (one row per experiment)
    results/classification_<task>_averaged.csv      (mean/std over seeds)

Setup labels in these tables (kept from the thesis CSVs so that
compute_gamma.py and generate_figures.py work unchanged):
    original | all_sentences (= full) | rare_sigma_1.0 / 1.5 / 2.0 (= selective_k*)
The original-data model does not depend on the rewriting model; its row is
written once for each rewriting model so every comparison has a baseline.

Replaces extract_results.py + average_results.py of the thesis runs.

Usage:
    python evaluation/collect_classification.py --task mp --base .
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import CLASSIFICATION_SEEDS, EPSILONS, MODELS, Layout, setup_names  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
from step8c_experiments import find_eval_results, parse_eval_results  # noqa: E402


def label(setup):
    if setup == "full":
        return "all_sentences", "all"
    if setup.startswith("selective_k"):
        k = setup.replace("selective_k", "")
        return f"rare_sigma_{k}", k
    return "original", None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mp", "proc"])
    p.add_argument("--base", default=".")
    p.add_argument("--seeds", default=",".join(map(str, CLASSIFICATION_SEEDS)))
    a = p.parse_args()
    L = Layout(a.base, a.task)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    all_rows = []
    for seed in seeds:
        rows = []
        m = parse_eval_results(find_eval_results(L.results_dir(seed, "original")))
        for model in MODELS:
            if m:
                rows.append({"setup": "original", "model": model, "sigma": None, "epsilon": None,
                             "roc_auc": m.get("roc_auc", m.get("eval_roc_auc")),
                             "f1_macro": m.get("f1_macro", m.get("eval_f1_macro"))})
        for setup in setup_names():
            lab, sig = label(setup)
            for model in MODELS:
                for eps in EPSILONS:
                    m = parse_eval_results(find_eval_results(L.results_dir(seed, setup, model, eps)))
                    if not m:
                        continue
                    rows.append({"setup": lab, "model": model, "sigma": sig, "epsilon": eps,
                                 "roc_auc": m.get("roc_auc", m.get("eval_roc_auc")),
                                 "f1_macro": m.get("f1_macro", m.get("eval_f1_macro"))})
        df = pd.DataFrame(rows)
        out = L.classification_seed_csv(seed)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"seed {seed}: {len(df)} results → {out}")
        df["seed"] = seed
        all_rows.append(df)

    full = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if full.empty:
        sys.exit("No classification results found — run step 8c first.")
    agg = (full.groupby(["setup", "model", "sigma", "epsilon"], dropna=False)["roc_auc"]
               .agg(roc_auc_mean="mean", roc_auc_std="std", roc_auc_min="min",
                    roc_auc_max="max", n_runs="count").reset_index())
    agg.to_csv(L.classification_avg, index=False)
    print(f"\n✓ averaged over seeds {seeds} → {L.classification_avg}")
    print(agg.to_string(index=False))
    if (agg["n_runs"] < len(seeds)).any():
        print("\nWARNING: some experiments have fewer runs than seeds (see n_runs).")


if __name__ == "__main__":
    main()
