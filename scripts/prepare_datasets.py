"""
prepare_datasets.py
Builds the two thesis input files from the clinical-outcome-prediction task
files (van Aken et al., 2021 — https://github.com/bvanaken/clinical-outcome-prediction):

    data/mp_50pct.csv    id, text, hospital_expire_flag      (≈24,342 notes)
    data/proc_50pct.csv  id, text, short_codes                (3-digit ICD-9 procedure groups)

Prerequisite: run that repository's `tasks/` scripts on MIMIC-III
(NOTEEVENTS, ADMISSIONS, PROCEDURES_ICD) to obtain
    MP_IN_adm_{train,val,test}.csv and PRO_GROUPS_3_DIGITS_adm_{train,val,test}.csv
Two fixes were needed in mimic_utils.py with recent pandas (see docs/HISTORY.md):
    .str.strip() on CATEGORY/DESCRIPTION before filtering, and
    regex=True in notes_df['TEXT'].str.replace(r"\\n", r"\\\\n", regex=True)

RECONSTRUCTED from chat history (2026-03): all splits are concatenated, duplicates
on (id, text) dropped, then a 50 % random sample is drawn with random_state=42.
The thesis then re-splits 70/10/20 (seed 123) in step 8a.

Usage:
    python scripts/prepare_datasets.py --mp_dir data/mimic_mp --proc_dir data/mimic_procedure --out_dir data
"""
import argparse
from pathlib import Path

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--mp_dir", required=True)
p.add_argument("--proc_dir", required=True)
p.add_argument("--out_dir", default="data")
p.add_argument("--frac", type=float, default=0.5)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)


def build(folder, prefix, label_col, out_name):
    parts = []
    for split in ["train", "val", "test"]:
        f = Path(folder) / f"{prefix}_adm_{split}.csv"
        if f.exists():
            parts.append(pd.read_csv(f))
    if not parts:
        f = Path(folder) / f"{prefix}_adm_all.csv"
        parts = [pd.read_csv(f)]
    df = pd.concat(parts, ignore_index=True)
    df.columns = df.columns.str.lower()
    df = df[["id", "text", label_col]].drop_duplicates(subset=["id", "text"])
    df50 = df.sample(frac=args.frac, random_state=args.seed).reset_index(drop=True)
    df50.to_csv(out / out_name, index=False)
    print(f"{prefix}: {len(df):,} unique notes → {len(df50):,} in {out/out_name}")


build(args.mp_dir, "MP_IN", "hospital_expire_flag", "mp_50pct.csv")
build(args.proc_dir, "PRO_GROUPS_3_DIGITS", "short_codes", "proc_50pct.csv")
