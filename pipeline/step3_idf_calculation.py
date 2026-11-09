"""
Step 3: IDF Calculation
Calculate token-level and sentence-level IDF scores
"""

import ast
import math
import numpy as np
import pandas as pd
from collections import Counter
from tqdm.auto import tqdm


def calculate_token_idf(df, tokenized_column='sentences_tokenized'):
    """
    Calculate IDF score for each token across all documents.
    
    IDF = log10((N + 1) / (DF + 1)) + 1.0
    where N = number of documents, DF = document frequency
    
    Args:
        df: DataFrame with tokenized sentences
        tokenized_column: Column containing list of tokenized sentences
        
    Returns:
        dict: Token to IDF score mapping
    """
    print("Calculating token-level IDF scores...")
    
    N = len(df)
    df_counter = Counter()
    
    for doc in tqdm(df[tokenized_column], total=N, desc="Counting document frequencies"):
        # Parse string representation of list
        if isinstance(doc, str):
            try:
                doc = ast.literal_eval(doc)
            except (ValueError, SyntaxError):
                continue
        
        seen = set()
        if not isinstance(doc, list):
            continue
        
        for sent_tokens in doc:
            if not isinstance(sent_tokens, list):
                continue
            for tok in sent_tokens:
                if isinstance(tok, str) and tok:
                    seen.add(tok)
        
        df_counter.update(seen)
    
    # Calculate IDF
    token_idf = {
        tok: math.log10((N + 1) / (df + 1)) + 1.0 
        for tok, df in df_counter.items()
    }
    
    print(f"✓ Token IDF calculated")
    print(f"  Vocab size: {len(token_idf):,}")
    if token_idf:
        idf_values = list(token_idf.values())
        print(f"  Mean IDF: {np.mean(idf_values):.4f}")
        print(f"  Std IDF: {np.std(idf_values):.4f}")
    
    return token_idf


_DEFAULT_CACHE = {}


def sentence_mean_idf(sent_tokens, idf_dict):
    """
    Calculate mean IDF score for a sentence.
    
    Args:
        sent_tokens: List of tokens in sentence
        idf_dict: Token to IDF mapping
        
    Returns:
        float: Mean IDF score (None if no tokens)
    """
    if not sent_tokens or len(sent_tokens) == 0:
        return None
    
    # Fallback IDF for unseen tokens = smallest IDF in the vocabulary.
    # (Fix 2026-11: computed once per dictionary instead of once per sentence;
    #  identical result, much faster on the full corpus.)
    key = id(idf_dict)
    if key not in _DEFAULT_CACHE:
        _DEFAULT_CACHE[key] = min(idf_dict.values()) if idf_dict else 1.0
    default = _DEFAULT_CACHE[key]
    vals = [idf_dict.get(t, default) for t in sent_tokens if t]
    
    # Handle case where all tokens were filtered out
    if not vals or len(vals) == 0:
        return None
    
    return sum(vals) / len(vals)


def calculate_sentence_idf(df, token_idf, tokenized_column='sentences_tokenized'):
    """
    Calculate sentence-level IDF scores.
    
    Args:
        df: DataFrame with tokenized sentences
        token_idf: Token to IDF mapping
        tokenized_column: Column with tokenized sentences
        
    Returns:
        DataFrame with added 'sentence_idf_mean' column
    """
    print("Calculating sentence-level IDF scores...")
    
    def calculate_sentence_scores(doc):
        # Parse string representation of list (common when loading from CSV)
        if isinstance(doc, str):
            try:
                doc = ast.literal_eval(doc)
            except (ValueError, SyntaxError) as e:
                print(f"⚠ Warning: Failed to parse tokenized sentences: {e}")
                return []
        
        if not isinstance(doc, list):
            print(f"⚠ Warning: Expected list but got {type(doc)}")
            return []
        
        # Calculate IDF for each sentence in the document
        scores = []
        for sent_tokens in doc:
            if isinstance(sent_tokens, str):
                # Sometimes individual sentences are also strings - parse them
                try:
                    sent_tokens = ast.literal_eval(sent_tokens)
                except (ValueError, SyntaxError):
                    scores.append(None)
                    continue
            
            if isinstance(sent_tokens, list):
                scores.append(sentence_mean_idf(sent_tokens, token_idf))
            else:
                scores.append(None)
        
        return scores
    
    print(f"  Processing {len(df)} documents...")
    df["sentence_idf_mean"] = df[tokenized_column].apply(calculate_sentence_scores)
    
    # Debug: Check how many documents have empty sentence_idf_mean
    empty_count = df["sentence_idf_mean"].apply(lambda x: len(x) == 0).sum()
    if empty_count > 0:
        print(f"⚠ Warning: {empty_count} documents have empty sentence_idf_mean")
    
    # Collect all scores (skip None)
    all_scores = []
    for scores in df["sentence_idf_mean"]:
        if isinstance(scores, list):
            for s in scores:
                if s is not None and np.isfinite(s):
                    all_scores.append(s)
    
    all_scores = np.array(all_scores)
    
    # Avoid warnings if no valid scores
    if len(all_scores) == 0:
        print("⚠ Warning: No valid sentence IDF scores found")
        return df, all_scores
    
    print(f"✓ Sentence IDF calculated")
    print(f"  Total sentences: {len(all_scores):,}")
    print(f"  Mean: {np.mean(all_scores):.4f}")
    print(f"  Std: {np.std(all_scores):.4f}")
    print(f"  Min: {np.min(all_scores):.4f}")
    print(f"  Max: {np.max(all_scores):.4f}")
    
    return df, all_scores


def identify_rare_sentences(df, threshold=None, method='mean_plus_2std'):
    """
    Identify rare sentences above IDF threshold.
    
    Args:
        df: DataFrame with sentence_idf_mean column
        threshold: Fixed threshold (if None, calculated from data)
        method: 'mean_plus_1std', 'mean_plus_2std', etc.
        
    Returns:
        DataFrame with added 'rare_idx' and 'n_rare' columnsls
    """
    print("Identifying rare sentences...")
    
    # Collect all scores
    all_scores = []
    for scores in df["sentence_idf_mean"]:
        if isinstance(scores, list):
            for s in scores:
                if s is not None and np.isfinite(s):
                    all_scores.append(s)
    
    if len(all_scores) == 0:
        print("⚠ Warning: No valid sentence IDF scores found for rare sentence identification")
        df["rare_idx"] = df["sentence_idf_mean"].apply(lambda x: [])
        df["n_rare"] = 0
        return df, 0.0
    
    all_scores = np.array(all_scores)
    mean_idf = np.mean(all_scores)
    std_idf = np.std(all_scores)
    
    # Calculate threshold
    if threshold is None:
        if method == 'mean_plus_1std':
            threshold = mean_idf + 1.0 * std_idf
        elif method == 'mean_plus_1.5std':
            threshold = mean_idf + 1.5 * std_idf
        elif method == 'mean_plus_2std':
            threshold = mean_idf + 2.0 * std_idf
        else:
            raise ValueError(f"Unknown method: {method}")
    
    print(f"  Threshold ({method}): {threshold:.4f}")
    print(f"  Mean: {mean_idf:.4f}, Std: {std_idf:.4f}")
    
    # Mark rare sentences
    df["rare_idx"] = df["sentence_idf_mean"].apply(
        lambda scores: [i for i, s in enumerate(scores) if s is not None and s >= threshold]
    )
    df["n_rare"] = df["rare_idx"].apply(len)
    
    total_rare = df["n_rare"].sum()
    total_sentences = df["sentence_idf_mean"].apply(len).sum()
    
    print(f"✓ Rare sentences identified")
    print(f"  Total rare: {total_rare:,} ({100*total_rare/total_sentences:.2f}%)")
    print(f"  Avg rare/document: {df['n_rare'].mean():.2f}")
    
    return df, threshold


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python step3_idf_calculation.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    df = pd.read_csv(input_path)
    
    # Calculate IDF
    token_idf = calculate_token_idf(df)
    df, all_scores = calculate_sentence_idf(df, token_idf)
    df, threshold = identify_rare_sentences(df)
    
    # Save
    df.to_csv(output_path, index=False)
    
    # Validate output before finishing
    print(f"\n=== VALIDATION ===")
    print(f"DataFrame shape: {df.shape}")
    print(f"sentence_idf_mean column exists: {'sentence_idf_mean' in df.columns}")
    
    if 'sentence_idf_mean' in df.columns:
        # Check first few sentence_idf_mean values
        sample_values = df['sentence_idf_mean'].head(3)
        for i, val in enumerate(sample_values):
            print(f"Row {i} sentence_idf_mean: {type(val)}, length: {len(val) if isinstance(val, list) else 'N/A'}")
            if isinstance(val, list) and len(val) > 0:
                print(f"  Sample scores: {val[:3]}...")
    
    # Save token IDF
    token_idf_df = pd.DataFrame([
        {'token': tok, 'idf': idf}
        for tok, idf in sorted(token_idf.items(), key=lambda x: -x[1])
    ])
    token_idf_path = output_path.replace('.csv', '_token_idf.csv')
    token_idf_df.to_csv(token_idf_path, index=False)
    
    print(f"✓ Saved to {output_path}")
    print(f"✓ Token IDF saved to {token_idf_path}")
