"""
Step 8a - Create train/val/test splits for every setup.

One 70/10/20 split (seed 123) is computed ONCE on the original dataset and
re-used for every condition, so all conditions are evaluated on the same
patients. For each privatized setup the note text is replaced by the rebuilt
private note from step 7 (`note_private`).

Writes   outputs_<task>/step8_splits/<setup>/<model>/eps_<eps>/<TASK>_adm_{train,val,test}.csv
and      outputs_<task>/step8_splits/original/<TASK>_adm_{train,val,test}.csv

Based on the thesis script; changes 2026-11:
  * one call creates all splits (original + selective_k* + full), paths from common/layout.py
  * the original-data splits are created here (previously made by hand)
  * a check that EVERY note was replaced (a mismatch between --original_csv and
    the step-1 input used to leave notes silently un-privatized)
  * empty private notes are no longer turned into the string "nan"

Usage:
    python pipeline/step8a_create_splits.py --task mp   --base .
    python pipeline/step8a_create_splits.py --task proc --base . --setups full --models roberta_base
"""
import argparse
import csv
import os
import shutil
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, EPSILONS, MODELS, SPLIT_SEED, setup_names  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=["mp", "proc"])
    p.add_argument("--base", default=".")
    p.add_argument("--original_csv", default=None,
                   help="Default: the step-1 input (data/<task>_50pct.csv). Must be the SAME file.")
    p.add_argument("--setups", default=",".join(["original"] + setup_names()))
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--eps", default=",".join(str(e) for e in EPSILONS))
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--train_size", type=float, default=0.7)
    p.add_argument("--val_size", type=float, default=0.1)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--id_col", default="id")
    p.add_argument("--text_col", default="text")
    p.add_argument("--priv_text_col", default="note_private")
    p.add_argument("--min_replaced", type=float, default=0.99,
                   help="Fraction of notes that must be replaced (notes with empty text have no private version)")
    return p.parse_args()


def save_split(df, folder, split_name, task_name, text_col, label_col):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{task_name}_adm_{split_name}.csv")
    df[[text_col, label_col]].to_csv(path, index=False, quoting=csv.QUOTE_ALL)
    print(f"    ✓ {split_name:5s} ({len(df):,} rows) → {path}")


def apply_privatization(df, id_col, text_col, id_to_private):
    df_out = df.copy()
    mask = df_out[id_col].astype(str).isin(id_to_private)
    df_out.loc[mask, text_col] = df_out.loc[mask, id_col].astype(str).map(id_to_private)
    return df_out, int(mask.sum())


def load_step7_file(path, id_col, priv_text_col):
    if not os.path.exists(path):
        return None
    df_priv = pd.read_csv(path)
    df_priv.columns = df_priv.columns.str.lower()
    ptc = priv_text_col.lower()
    if ptc not in df_priv.columns:
        raise ValueError(f"'{ptc}' not in columns of {path}: {list(df_priv.columns)}")
    df_priv = df_priv.dropna(subset=[ptc])            # no "nan" strings
    return dict(zip(df_priv[id_col].astype(str), df_priv[ptc].astype(str)))


def main():
    a = parse_args()
    L = Layout(a.base, a.task)
    label_col = L.cfg["label_col"].lower()
    task_name = L.cfg["task_name"]
    setups = [s.strip() for s in a.setups.split(",") if s.strip()]
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    eps_values = [float(e) for e in a.eps.split(",") if e.strip()]

    original_csv = a.original_csv or L.input_csv
    df = pd.read_csv(original_csv)
    df.columns = df.columns.str.lower()
    id_col, text_col = a.id_col.lower(), a.text_col.lower()
    for col in [id_col, text_col, label_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    print("=" * 70)
    print(f"STEP 8a: CREATE SPLITS [task={a.task}]  {len(df):,} notes from {original_csv}")
    print("=" * 70)

    # ── one split for everything ───────────────────────────────────────────
    val_test = a.val_size + a.test_size
    val_ratio = a.val_size / val_test
    try:
        tr, tmp = train_test_split(df, test_size=val_test, random_state=a.seed, stratify=df[label_col])
        va, te = train_test_split(tmp, test_size=1 - val_ratio, random_state=a.seed, stratify=tmp[label_col])
    except ValueError:
        print("  NOTE: stratified split not possible (multilabel). Using random split.")
        tr, tmp = train_test_split(df, test_size=val_test, random_state=a.seed)
        va, te = train_test_split(tmp, test_size=1 - val_ratio, random_state=a.seed)
    print(f"  train:{len(tr):,}  val:{len(va):,}  test:{len(te):,}")
    parts = [(tr, "train"), (va, "val"), (te, "test")]

    def copy_label_file(folder):
        # procedures: doc_classification needs the label list next to the data
        if L.cfg["multilabel"]:
            if not L.label_file.exists():
                raise FileNotFoundError(f"Missing {L.label_file} (see README, procedures task)")
            shutil.copy(L.label_file, folder)

    if "original" in setups:
        print("\n── original (no privatization)")
        folder = L.splits_dir("original")
        for part, name in parts:
            save_split(part, folder, name, task_name, text_col, label_col)
        copy_label_file(folder)

    problems = []
    for setup in [s for s in setups if s != "original"]:
        for model in models:
            for eps in eps_values:
                path = L.notes_file(setup, model, eps)
                id_to_private = load_step7_file(path, id_col, a.priv_text_col)
                if id_to_private is None:
                    print(f"  skip {setup}/{model}/eps={eps}: no step-7 file ({path.name})")
                    continue
                print(f"\n── {setup} / {model} / eps={eps}")
                folder = L.splits_dir(setup, model, eps)
                n_rep = 0
                for part, name in parts:
                    priv, n = apply_privatization(part, id_col, text_col, id_to_private)
                    n_rep += n
                    save_split(priv, folder, name, task_name, text_col, label_col)
                copy_label_file(folder)
                frac = n_rep / len(df)
                print(f"    replaced {n_rep:,}/{len(df):,} notes ({100 * frac:.1f}%)")
                if frac < a.min_replaced:
                    problems.append(f"{setup}/{model}/eps={eps}: only {100 * frac:.1f}% replaced")

    if problems:
        print("\nERROR: some notes were not replaced by their private version:")
        for p_ in problems:
            print("  -", p_)
        print("Check that --original_csv is the same file that went into step 1.")
        sys.exit(1)
    print("\n✓ ALL SPLITS CREATED")


if __name__ == "__main__":
    main()
