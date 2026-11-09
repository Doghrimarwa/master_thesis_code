"""
Step 5: Calculate DPMLM Clipping Bounds
Calculate C_min and C_max for DPMLM from sample of rare sentences
"""

import random
import numpy as np
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


def calculate_clipping_bounds(df_rare, 
                               text_column='dpmlm_input_sentence',
                               n_samples=1000,
                               model_name="FacebookAI/roberta-base",
                               seed=42,
                               sigma_factor=4.0):
    """
    Calculate clipping bounds (C_min, C_max) from sample of rare sentences.
    
    Method:
    - Sample random sentences
    - Mask one token per sentence
    - Collect MLM logits
    - C_min = μ (mean)
    - C_max = μ + 4σ (mean + 4 std)
    
    Args:
        df_rare: DataFrame with rare sentences
        text_column: Column containing sentence text
        n_samples: Number of sentences to sample
        model_name: Hugging Face model name
        seed: Random seed
        
    Returns:
        dict: {'C_min': float, 'C_max': float, 'mu': float, 'sigma': float}
    """
    print("="*70)
    print("CALCULATING CLIPPING BOUNDS")
    print("="*70)
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Sample sentences (reproducible, without replacement)
    n_samples = min(n_samples, len(df_rare))
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(df_rare), size=n_samples, replace=False)
    sampled_sentences = df_rare.iloc[sample_indices][text_column].tolist()
    
    print(f"Sampled {n_samples} from {len(df_rare)} rare sentences")
    
    # Load model
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Model loaded on {device}")
    
    # Collect logits
    all_logits = []
    skipped = 0
    
    for text in tqdm(sampled_sentences, desc="Collecting logits"):
        if not isinstance(text, str) or len(text.split()) < 3:
            skipped += 1
            continue

        words = text.split()
        mask_pos = random.randint(0, len(words) - 1)
        masked_words = words.copy()
        masked_words[mask_pos] = tokenizer.mask_token
        masked_text = " ".join(masked_words)

        try:
            inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits  # [1, seq_len, vocab]

            mask_positions = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_positions) > 0:
                all_logits.extend(logits[0, mask_positions[0], :].detach().cpu().numpy().tolist())
        except Exception as e:
            skipped += 1
            continue

    if skipped > 0:
        print(f"Skipped {skipped} sentences due to errors or invalid format")
    
    all_logits = np.array(all_logits, dtype=np.float32)
    
    if len(all_logits) == 0:
        raise ValueError("No logits collected! Check your input data and text column.")
    
    # Calculate bounds
    mu = float(np.mean(all_logits))
    sigma = float(np.std(all_logits))
    C_min = mu
    C_max = mu + float(sigma_factor) * sigma
    
    results = {
        'mu': mu,
        'sigma': sigma,
        'C_min': C_min,
        'C_max': C_max,
        'sensitivity': abs(C_max - C_min),
        'n_sampled': int(n_samples),
        'n_logits': int(len(all_logits)),
        'model_name': model_name,
        'sigma_factor': float(sigma_factor),
        'seed': int(seed),
    }
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"μ (mean):       {mu:.6f}")
    print(f"σ (std dev):    {sigma:.6f}")
    print(f"C_min = μ:      {C_min:.6f}")
    print(f"C_max = μ+{sigma_factor}σ:   {C_max:.6f}")
    print(f"Sensitivity:    {results['sensitivity']:.6f}")
    print("="*70)
    
    return results


if __name__ == "__main__":
    import sys
    import json
    import os
    
    if len(sys.argv) < 3:
        print("Usage: python step5_clipping_bounds.py <input_rare_csv> <output_json> --model <hf_name> [--samples 1000] [--sigma_factor 4.0]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Fix 2026-11: --model is REQUIRED. The bounds must be computed with the
    # same model that step 6 uses for rewriting (thesis: FacebookAI/roberta-base
    # and pminervini/RoBERTa-base-PM-M3-Voc-hf). The old silent default
    # (distilroberta-base) gave bounds for the wrong model.
    if "--model" not in sys.argv:
        print("ERROR: --model is required (e.g. --model FacebookAI/roberta-base)")
        sys.exit(1)
    model_name = sys.argv[sys.argv.index("--model") + 1]
    n_samples = 1000
    if "--samples" in sys.argv:
        n_samples = int(sys.argv[sys.argv.index("--samples") + 1])
    sigma_factor = 4.0
    if "--sigma_factor" in sys.argv:
        sigma_factor = float(sys.argv[sys.argv.index("--sigma_factor") + 1])

    df_rare = pd.read_csv(input_path)
    bounds = calculate_clipping_bounds(
        df_rare,
        model_name=model_name,
        n_samples=n_samples,
        sigma_factor=sigma_factor,
    )
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create if there's a directory component
        os.makedirs(output_dir, exist_ok=True)
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(bounds, f, indent=2)
    
    print(f"\n✓ Saved to {output_path}")
