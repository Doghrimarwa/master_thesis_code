"""
Step 8b - Write one FARM / doc_classification YAML config per split folder.

For every split folder created by step 8a, writes
    outputs_<task>/step8_configs/<same sub-path>/config.yaml
The output_dir in the config is a placeholder: step 8c sets the real,
seed-specific results folder when it runs the experiment.

Based on the thesis script; changes 2026-11:
  * configs are generated for ALL setups found (original, selective_k*, full),
    with the exact paths step 8c reads (the old 'baseline' configs did not match
    what step 8c expected and had to be renamed by hand)
  * for procedures, the label file is referenced by name; step 8a copies
    ALL_3_DIGIT_PRO_CODES.txt into every split folder

Usage:
    python pipeline/step8b_generate_configs.py --task mp --base .
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout  # noqa: E402


def make_config(L, data_dir, exp_name):
    cfg = L.cfg
    label_list = cfg["label_file_name"] if cfg["multilabel"] else ["0", "1"]
    return {
        "experiment_name": exp_name,
        "task_type": "text_classification",
        "output_type": "per_sequence",
        "multilabel": cfg["multilabel"],
        "metric": "binary_classification_metrics",
        "output_dir": "SET_BY_STEP8C",
        "log_dir": "SET_BY_STEP8C",
        "data": {
            "data_dir": str(data_dir),
            "train_filename": f"{cfg['task_name']}_adm_train.csv",
            "dev_filename": f"{cfg['task_name']}_adm_val.csv",
            "test_filename": f"{cfg['task_name']}_adm_test.csv",
            "label_list": label_list,
            "parsing": {"label_column": cfg["label_col"], "delimiter": ",", "quote_char": '"'},
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mp", "proc"])
    p.add_argument("--base", default=".")
    a = p.parse_args()
    L = Layout(a.base, a.task)
    L.cfg = dict(L.cfg, label_file_name=L.label_file.name)

    splits_root = L.root / "step8_splits"
    configs_root = L.root / "step8_configs"
    train_name = f"{L.cfg['task_name']}_adm_train.csv"
    n = 0
    for train_csv in sorted(splits_root.rglob(train_name)):
        data_dir = train_csv.parent
        rel = data_dir.relative_to(splits_root)
        out = configs_root / rel / "config.yaml"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(make_config(L, data_dir, str(rel).replace(os.sep, "__")), f,
                      default_flow_style=False, sort_keys=False)
        print(f"  ✓ {out.relative_to(L.root)}")
        n += 1
    if n == 0:
        sys.exit(f"No split folders found under {splits_root}; run step 8a first.")
    print(f"\n✓ {n} configs written to {configs_root}")


if __name__ == "__main__":
    main()
