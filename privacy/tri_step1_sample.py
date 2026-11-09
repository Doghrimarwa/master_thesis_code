"""
tri_step1_sample.py  (thesis: step1_sample_run.py)
Samples 50 real patients + 450 distractors for one TRI run.
Saves everything to privacy_eval/seed_{seed}/
 
Change 2026-11: --base argument instead of the hard-coded cluster path.

Usage (thesis seeds 42 123 456 789 1337, both tasks):
    python privacy/tri_step1_sample.py --seed 42 --task mp --base .
"""
 
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, tri_run_dir, repo_root, trir_averaged_csv, SIGMAS, EPSILONS, MODELS, TRI_SEEDS  # noqa: E402
import pandas as pd
from pathlib import Path
 
p = argparse.ArgumentParser()
p.add_argument("--seed",          type=int, default=42)
p.add_argument("--base",          default=".")
p.add_argument("--task",          default="mp", choices=["mp", "proc"],
               help="Task: mp (mortality) or proc (procedures)")
p.add_argument("--n_real",        type=int, default=50)
p.add_argument("--n_distractors", type=int, default=450)
args = p.parse_args()
 
INPUT_CSV = Layout(args.base, args.task).segmented
RUN_DIR   = tri_run_dir(args.base, args.seed)
RUN_DIR.mkdir(parents=True, exist_ok=True)
 
real_out = RUN_DIR / f"{args.task}_sampled_50_ids.csv"
dist_out = RUN_DIR / f"{args.task}_distractor_450_ids.csv"
 
df          = pd.read_csv(INPUT_CSV)
real        = df.sample(n=args.n_real, random_state=args.seed)
remaining   = df[~df["id"].isin(real["id"])]
distractors = remaining.sample(n=args.n_distractors, random_state=args.seed)
 
real[["id"]].to_csv(real_out,  index=False)
distractors[["id"]].to_csv(dist_out, index=False)
 
print(f"Task    : {args.task}")
print(f"Seed    : {args.seed}")
print(f"Run dir : {RUN_DIR}")
print(f"Input   : {INPUT_CSV}")
print(f"Real    : {len(real)} patients → {real_out}")
print(f"Distract: {len(distractors)} patients → {dist_out}")