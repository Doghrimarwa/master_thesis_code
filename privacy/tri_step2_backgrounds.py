"""
tri_step2_backgrounds.py  (thesis: step2_generate_backgrounds_run.py)
Generates Qwen2.5 backgrounds for TRI classifier training.
Reads sampled IDs from privacy_eval/seed_{seed}/
Saves backgrounds to privacy_eval/seed_{seed}/

Only generates backgrounds from ORIGINAL clinical notes:
  - bg_mp_original.csv (50 real patients)
  - bg_mp_distractors_500R.csv (450 distractors)

Privatized notes will be read directly from DP-MLM outputs in step3.
 
Must run AFTER step1_sample_run.py for the same seed.
 
Changes 2026-11: --base argument; the random seed is now fixed before
generation (Qwen samples with temperature 0.2, so biographies differed
slightly between runs before).

Usage:
    python privacy/tri_step2_backgrounds.py --seed 42 --task mp --base .
"""
 
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import Layout, tri_run_dir, repo_root, trir_averaged_csv, SIGMAS, EPSILONS, MODELS, TRI_SEEDS  # noqa: E402
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
 
p = argparse.ArgumentParser()
p.add_argument("--seed",       type=int, default=42)
p.add_argument("--base",       default=".")
p.add_argument("--task",       default="mp", choices=["mp", "proc"],
               help="Task: mp (mortality) or proc (procedures)")
p.add_argument("--qwen_model", default="Qwen/Qwen2.5-7B-Instruct")
args = p.parse_args()

RUN_DIR = tri_run_dir(args.base, args.seed)
SEGMENTED = Layout(args.base, args.task).segmented
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if not RUN_DIR.exists():
    raise FileNotFoundError(f"{RUN_DIR} not found — run step1_sample_run.py first")

SAMPLED_IDS_CSV    = RUN_DIR / f"{args.task}_sampled_50_ids.csv"
DISTRACTOR_IDS_CSV = RUN_DIR / f"{args.task}_distractor_450_ids.csv"

if not SAMPLED_IDS_CSV.exists():
    raise FileNotFoundError(f"{SAMPLED_IDS_CSV} not found — run step1_sample_run.py first")
if not DISTRACTOR_IDS_CSV.exists():
    raise FileNotFoundError(f"{DISTRACTOR_IDS_CSV} not found — run step1_sample_run.py first")

sampled_ids = pd.read_csv(SAMPLED_IDS_CSV)["id"].tolist()
distractor_ids = pd.read_csv(DISTRACTOR_IDS_CSV)["id"].tolist()
print(f"Seed {args.seed}, Task {args.task}: {len(sampled_ids)} real + {len(distractor_ids)} distractors")

# ── Prompt (same as before) ───────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert in data security and privacy. "
    "Your task is to extract social and personal details from clinical text "
    "to create a Background Knowledge biography for a specific patient. "
    "This biography will be used to train an adversary model to assess "
    "the re-identification risk after anonymization."
)
 
USER_TEMPLATE = """### TASK:
Write a short (50-100 words) biographical summary about the following patient.
Extract Indirect Personal Identifiers such as:
1. Occupation/Employment
2. Family situation
3. Habits (alcohol, smoking, drugs)
4. Appearance (height, weight, scars, tattoos)
5. Socio-economic or criminal history
6. Specific events before or during admission
 
### RULES:
1. Focus on the narrative (job, family, hobbies, the event).
2. Do NOT include medical vitals (BP, HR) or diagnoses.
3. Focus on identifying information only.
4. Names and dates are de-identified — do not mention them.
 
### CLINICAL NOTE:
{note}
 
### OUTPUT FORMAT:
[BIOGRAPHY_START]
The patient is a... (continue biography)
[BIOGRAPHY_END]"""
 
# ── Load Qwen2.5 ──────────────────────────────────────────────────────────────
print(f"Loading {args.qwen_model}...")
tokenizer = AutoTokenizer.from_pretrained(args.qwen_model)
qwen = AutoModelForCausalLM.from_pretrained(
    args.qwen_model, torch_dtype=torch.float16, device_map="auto"
)
print("Model ready.\n")
 
 
def get_biography(note_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": USER_TEMPLATE.format(
            note=str(note_text)[:3000])}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to(qwen.device)
    with torch.no_grad():
        out = qwen.generate(
            **inputs, max_new_tokens=200, temperature=0.2,
            do_sample=True, pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return (response
            .replace("[BIOGRAPHY_START]", "")
            .replace("[BIOGRAPHY_END]", "")
            .strip())
 
 
def generate_and_save(ids: list, csv_path: Path,
                      text_col: str, out_csv: Path, desc: str):
    if out_csv.exists():
        print(f"  SKIP — {out_csv.name} already exists")
        return
    if not csv_path.exists():
        print(f"  SKIP — {csv_path.name} missing")
        return
 
    df      = pd.read_csv(csv_path)
    df      = df[df["id"].isin(ids)].reset_index(drop=True)
    results = []
 
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        bio = get_biography(row[text_col])
        results.append({"id": row["id"], "background": bio})
 
        if len(results) % 50 == 0:
            pd.DataFrame(results).to_csv(out_csv, index=False)
 
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"  Saved {len(results)} → {out_csv}")
 
 
# ── 1. Original backgrounds (50 real patients) ────────────────────────────────
print("="*60)
print(f"ORIGINAL backgrounds for task={args.task} (50 real patients)")
generate_and_save(
    ids      = sampled_ids,
    csv_path = SEGMENTED,
    text_col = "text",
    out_csv  = RUN_DIR / f"bg_{args.task}_original.csv",
    desc     = "original"
)

# ── 2. Distractor backgrounds (450 patients for 500R) ─────────────────────────
print(f"\n{'='*60}")
print(f"DISTRACTOR backgrounds for task={args.task} (450 patients)")
dist_out_csv = RUN_DIR / f"bg_{args.task}_distractors_500R.csv"
generate_and_save(
    ids      = distractor_ids,
    csv_path = SEGMENTED,
    text_col = "text",
    out_csv  = dist_out_csv,
    desc     = "distractors"
)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"DONE — seed {args.seed}, task {args.task}")
print(f"Generated 2 background files (original + distractors):")
print(f"  - {RUN_DIR / f'bg_{args.task}_original.csv'}")
print(f"  - {RUN_DIR / f'bg_{args.task}_distractors_500R.csv'}")
print(f"\nNote: Privatized texts will be read directly from DP-MLM outputs in step3")