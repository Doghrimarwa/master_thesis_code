"""
Step 8c - Train and evaluate Bio_ClinicalBERT (FARM doc_classification,
van Aken et al. 2021) on every split, for one classification seed.

Results go to   outputs_<task>/step8_results/seed_<seed>/<setup>/<model>/eps_<eps>/
(original data: outputs_<task>/step8_results/seed_<seed>/original/)
Experiments that already have a valid eval_results.txt are skipped, so a job
that hit the time limit can simply be resubmitted.

Runs in the FARM environment (python 3.8, requirements-farm.txt).

Based on the thesis script; changes 2026-11:
  * setups are called original / selective_k* / full (the old 'baseline' name
    meant different things in different scripts)
  * all experiments are listed, including roberta_clinical full eps=25
    (missing from the old list, which made compute_gamma fail)
  * seeds live in one folder tree (seed_11, seed_22, seed_33) instead of
    separate outputs_<task>_run2 / _run3 roots
  * the original-data model is trained once per seed (it does not depend on
    the rewriting model)

Usage:
    python pipeline/step8c_experiments.py --task mp --seed 11
    python pipeline/step8c_experiments.py --task proc --seed 22 --setups full --models roberta_clinical
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, EPSILONS, MODELS, repo_root, setup_names  # noqa: E402

REPO = repo_root()
EXPERIMENTS_DIR = REPO / "external/clinical-outcome-prediction/experiments"
# DPO_DOC_CLASSIFICATION lets the smoke test replace FARM by a stand-in
DOC_CLASSIFICATION = Path(os.environ.get("DPO_DOC_CLASSIFICATION", EXPERIMENTS_DIR / "doc_classification.py"))
DEFAULT_CLASSIFIER = REPO / "cache/bio_clinical_bert"


def find_eval_results(search_dir: Path):
    if not search_dir.exists():
        return None
    matches = list(search_dir.rglob("eval_results.txt"))
    return sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0] if matches else None


def parse_eval_results(path):
    metrics = {}
    if path is None or not Path(path).exists():
        return metrics
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ": " in line:
                k, v = line.split(": ", 1)
                try:
                    metrics[k.strip()] = float(v.strip())
                except ValueError:
                    metrics[k.strip()] = v.strip()
    return metrics


def roc(m):
    return m.get("roc_auc", m.get("eval_roc_auc"))


def run_one(cfg_path, results_dir, seed, classifier, tag):
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["output_dir"] = str(results_dir)
    cfg["log_dir"] = str(results_dir / "logs")
    patched = Path(f"/tmp/config_{tag}_seed{seed}.yaml")
    with open(patched, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    cache = Path(f"/tmp/cache_{tag}_seed{seed}")
    shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True)

    cmd = [sys.executable, str(DOC_CLASSIFICATION),
           "--task_config", str(patched),
           "--model_name_or_path", str(classifier),
           "--cache_dir", str(cache),
           "--model_class", "Bert",
           "--tokenizer_class", "BertTokenizer",
           "--do_train", "True", "--do_eval", "True",
           "--seed", str(seed)]
    print("  " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(DOC_CLASSIFICATION.parent)).returncode
    return rc, (time.time() - t0) / 60


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mp", "proc"])
    p.add_argument("--base", default=".")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--setups", default=",".join(["original"] + setup_names()))
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--eps", default=",".join(str(e) for e in EPSILONS))
    p.add_argument("--classifier", default=str(DEFAULT_CLASSIFIER),
                   help="Path of the Bio_ClinicalBERT weights (scripts/setup_external.sh)")
    p.add_argument("--force", action="store_true", help="Re-run even if results exist")
    a = p.parse_args()

    L = Layout(a.base, a.task)
    if not DOC_CLASSIFICATION.exists():
        sys.exit(f"{DOC_CLASSIFICATION} not found — run scripts/setup_external.sh")

    jobs = []
    for setup in [s.strip() for s in a.setups.split(",") if s.strip()]:
        if setup == "original":
            jobs.append((setup, None, None))
            continue
        for model in [m.strip() for m in a.models.split(",") if m.strip()]:
            for eps in [float(e) for e in a.eps.split(",") if e.strip()]:
                jobs.append((setup, model, eps))

    summary = []
    for setup, model, eps in jobs:
        cfg = L.config_file(setup, model, eps)
        out = L.results_dir(a.seed, setup, model, eps)
        tag = f"{a.task}_{setup}_{model}_{eps}"
        row = {"task": a.task, "setup": setup, "model": model, "epsilon": eps, "seed": a.seed,
               "results_dir": str(out), "start": datetime.now().isoformat(timespec="seconds")}
        if not cfg.exists():
            print(f"SKIP {tag}: no config ({cfg})")
            continue
        done = parse_eval_results(find_eval_results(out))
        if not a.force and roc(done):
            print(f"SKIP {tag}: already done (ROC-AUC {roc(done):.4f})")
            row.update(metrics=done, returncode=0, skipped=True)
        else:
            print(f"\n=== {tag} seed={a.seed}")
            rc, minutes = run_one(cfg, out, a.seed, a.classifier, tag)
            row.update(metrics=parse_eval_results(find_eval_results(out)),
                       returncode=rc, minutes=round(minutes, 1), skipped=False)
        summary.append(row)
        js = L.root / "step8_results" / f"seed_{a.seed}" / "all_results.json"
        js.parent.mkdir(parents=True, exist_ok=True)
        old = json.loads(js.read_text()) if js.exists() else []
        key = lambda r: (r["setup"], r["model"], r["epsilon"])  # noqa: E731
        merged = {key(r): r for r in old}
        merged.update({key(r): r for r in summary})
        js.write_text(json.dumps(list(merged.values()), indent=2))

    print("\nSUMMARY")
    for r in summary:
        print(f"  {r['setup']:<16} {str(r['model']):<18} {str(r['epsilon']):<6} "
              f"ROC-AUC={roc(r['metrics'])}  rc={r['returncode']}")


if __name__ == "__main__":
    main()
