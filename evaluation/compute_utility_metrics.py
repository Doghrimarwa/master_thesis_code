"""
compute_utility_metrics.py
Computes semantic similarity (S) and text coherence (C) for all setups.
Follows the DP-MLM / DP-IPI evaluation protocol:
  - S: cosine similarity (original note vs privatized note) averaged over
       3 embedding models (all-mpnet-base-v2, gte-large, jina-embeddings-v3)
  - C: perplexity ratio PPL_original / PPL_private using GPT-2 (first 128 tokens)

Up to 500 notes per setup are sampled with random_state=42.

Recovered from chat history (2026-07); changes 2026-11: paths come from
common/layout.py, output goes to results/utility_metrics_<task>.csv.

Usage:
    python evaluation/compute_utility_metrics.py --task mp --base .
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.layout import EPSILONS, MODELS, SIGMAS, Layout  # noqa: E402

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--task",        required=True, choices=["mp", "proc"])
p.add_argument("--base",        default=".")
p.add_argument("--max_samples", type=int, default=500, help="Notes per setup (thesis: 500)")
p.add_argument("--max_tokens",  type=int, default=128, help="Tokens for perplexity (thesis: 128)")
p.add_argument("--embedding_models", default="mpnet,gte,jina",
               help="Subset of mpnet,gte,jina (thesis: all three)")
p.add_argument("--gpt2_model", default="gpt2", help="Perplexity model (thesis: gpt2)")
p.add_argument("--embedding_path", default=None,
               help="TESTING ONLY: use this local model for every embedding name")
args = p.parse_args()

L = Layout(args.base, args.task)
RESULTS_CSV = L.utility_csv
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

# ── Define all setups ─────────────────────────────────────────────────────────
SETUPS = []
for setup, label, sigma in [(f"selective_k{k}", f"rare_sigma_{k}", k) for k in SIGMAS] + [("full", "all_sentences", "all")]:
    for eps in EPSILONS:
        for model in MODELS:
            SETUPS.append({
                "name": f"{setup}_eps{int(eps)}_{model}", "setup": label, "sigma": sigma,
                "epsilon": eps, "model": model, "csv": L.notes_file(setup, model, eps),
            })

# ── Load GPT-2 for perplexity ─────────────────────────────────────────────────
print("Loading GPT-2 for perplexity...")
gpt2_tokenizer = GPT2TokenizerFast.from_pretrained(args.gpt2_model)
gpt2_model     = GPT2LMHeadModel.from_pretrained(args.gpt2_model)
gpt2_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
gpt2_model.to(device)
print(f"GPT-2 loaded on {device}")

# ── Load embedding models for semantic similarity ────────────────────────────
print("\nLoading embedding models...")
EMB_IDS = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "gte":   "thenlper/gte-large",
    "jina":  "jinaai/jina-embeddings-v3",
}
EMB_MODELS = {
    name: SentenceTransformer(args.embedding_path or EMB_IDS[name],
                              trust_remote_code=(name == "jina" and not args.embedding_path))
    for name in [m.strip() for m in args.embedding_models.split(",") if m.strip()]
}
print("Embedding models loaded.")


def compute_perplexity(texts: list, max_tokens: int = 128) -> float:
    """Compute mean perplexity over a list of texts using GPT-2."""
    perplexities = []
    for text in texts:
        encodings = gpt2_tokenizer(
            text[:1024], return_tensors="pt", truncation=True,
            max_length=max_tokens
        )
        input_ids = encodings.input_ids.to(device)
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = gpt2_model(input_ids, labels=input_ids)
            loss    = outputs.loss
        perplexities.append(math.exp(loss.item()))
    return float(np.mean(perplexities)) if perplexities else float("nan")


def compute_semantic_similarity(orig_texts: list,
                                 priv_texts: list) -> dict:
    """
    Compute cosine similarity between original and private texts.
    Returns per-model scores and their average.
    """
    scores = {}
    for name, model in EMB_MODELS.items():
        print(f"    Computing {name} embeddings...")
        orig_emb = model.encode(orig_texts, batch_size=32,
                                show_progress_bar=False)
        priv_emb = model.encode(priv_texts, batch_size=32,
                                show_progress_bar=False)
        sims = [
            cosine_similarity([orig_emb[i]], [priv_emb[i]])[0][0]
            for i in range(len(orig_texts))
        ]
        scores[name] = float(np.mean(sims))
        print(f"      {name}: {scores[name]:.4f}")
    scores["average"] = float(np.mean(list(scores.values())))
    return scores


# ── Compute baseline perplexity (original notes, once) ───────────────────────
print("\n=== Computing baseline perplexity (original notes) ===")

first_csv = next((s["csv"] for s in SETUPS if s["csv"].exists()), None)
if first_csv is None:
    raise FileNotFoundError("No step7 files found. Run step7 first.")

df_sample = pd.read_csv(first_csv).dropna(subset=["text", "note_private"])
if args.max_samples:
    df_sample = df_sample.sample(
        min(args.max_samples, len(df_sample)), random_state=42
    )

orig_texts = df_sample["text"].tolist()
ppl_original = compute_perplexity(orig_texts, args.max_tokens)
print(f"Baseline perplexity (original): {ppl_original:.2f}")

# ── Process each setup ────────────────────────────────────────────────────────
results = []

for setup_info in SETUPS:
    name = setup_info["name"]
    csv  = setup_info["csv"]

    print(f"\n{'='*60}")
    print(f"Setup: {name}")

    if not csv.exists():
        print(f"  SKIP — file not found: {csv.name}")
        continue

    df = pd.read_csv(csv).dropna(subset=["text", "note_private"])
    if args.max_samples:
        df = df.sample(min(args.max_samples, len(df)), random_state=42)

    orig  = df["text"].tolist()
    priv  = df["note_private"].tolist()
    n     = len(df)
    print(f"  Notes: {n}")

    print("  Computing semantic similarity...")
    sim_scores = compute_semantic_similarity(orig, priv)

    print("  Computing perplexity (privatized notes)...")
    ppl_private = compute_perplexity(priv, args.max_tokens)
    # C = perplexity_original / perplexity_private (max=1, lower=worse)
    coherence   = ppl_original / ppl_private if ppl_private > 0 else float("nan")

    print(f"  S (avg)    = {sim_scores['average']:.4f}")
    print(f"  PPL_orig   = {ppl_original:.2f}")
    print(f"  PPL_priv   = {ppl_private:.2f}")
    print(f"  C (ratio)  = {coherence:.4f}")

    results.append({
        "setup":       setup_info["setup"],
        "model":       setup_info["model"],
        "sigma":       setup_info["sigma"],
        "epsilon":     setup_info["epsilon"],
        "n_notes":     n,
        "S_mpnet":     sim_scores.get("mpnet"),
        "S_gte":       sim_scores.get("gte"),
        "S_jina":      sim_scores.get("jina"),
        "S_avg":       sim_scores["average"],
        "PPL_original": ppl_original,
        "PPL_private":  ppl_private,
        "C":           coherence,
    })

    # Save after each setup (safe to interrupt)
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"  → Saved to {RESULTS_CSV}")

# ── Final summary ──────────────────────────────────────────────────────────────
df_out = pd.DataFrame(results)
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(df_out[["setup", "model", "epsilon", "S_avg", "C"]].to_string())
print(f"\nSaved to: {RESULTS_CSV}")
