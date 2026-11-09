"""
Step 4: Extract Rare Sentences
Create dataset of rare sentences for DPMLM processing
"""

import pandas as pd
import re
import numpy as np

PHI_UNWRAP_RE = re.compile(r"\[\*\*(.*?)\*\*\]")

def remove_phi_brackets(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return PHI_UNWRAP_RE.sub(r"\1", text)


def extract_rare_sentences(df,
                           id_column='id',
                           binary_column='hospital_expire_flag',
                           multiclass_column='short_codes',
                           sents_column='sentences_mimic',
                           scores_column='sentence_idf_mean',
                           rare_idx_column='rare_idx',
                           threshold=None,
                           sigma_multiplier=None,
                           sigma_multipliers=None):
    """
    Extract rare sentences into flat DataFrame for DPMLM.
    
    Args:
        df: DataFrame with rare_idx, sentences_mimic, sentence_idf_mean columns
        id_column: Name of note ID column
        label_column: Name of label column
        
    Returns:
        DataFrame with one row per rare sentence
    """
    print("Extracting rare sentences...")

    # Handle multiple sigma multipliers (like notebook SIGMA_MULTIPLIERS)
    if sigma_multipliers is not None:
        if not isinstance(sigma_multipliers, list):
            sigma_multipliers = [sigma_multipliers]
        
        # Compute all scores once (efficiency improvement)
        all_scores = []
        for scores in df[scores_column]:
            # Parse string representation of list (common when loading from CSV)
            if isinstance(scores, str):
                try:
                    import ast
                    scores = ast.literal_eval(scores)
                except (ValueError, SyntaxError):
                    continue
            
            if isinstance(scores, list):
                for s in scores:
                    if s is not None and np.isfinite(s):
                        all_scores.append(s)
        if len(all_scores) == 0:
            raise ValueError("No valid sentence IDF scores found to compute threshold.")
        
        mean_idf = float(np.mean(all_scores))
        std_idf = float(np.std(all_scores))
        
        print(f"  IDF Statistics: mean = {mean_idf:.4f}, std = {std_idf:.4f}")
        print(f"  Total sentences: {len(all_scores):,}")
        
        # Show diagnostics like your notebook plot
        for test_k in [1.0, 1.5, 2.0]:
            test_thr = mean_idf + test_k * std_idf
            count_above = sum(1 for s in all_scores if s >= test_thr)
            pct = 100 * count_above / len(all_scores)
            print(f"  Mean+{test_k}σ: {test_thr:.4f} → {count_above:,} sentences ({pct:.2f}%)")
        
        # Process each threshold
        results = {}
        for sigma_mult in sigma_multipliers:
            threshold_val = mean_idf + float(sigma_mult) * std_idf
            print(f"\n  Extracting for threshold = mean + {sigma_mult}σ = {threshold_val:.4f}")
            
            # Extract directly without recursion
            rare_rows = []
            
            for row in df.itertuples(index=False):
                note_id = getattr(row, id_column)
                label = int(getattr(row, binary_column, 0))
                multiclass_label = getattr(row, multiclass_column, None)
                sents = getattr(row, sents_column)
                scores = getattr(row, scores_column)
                
                # Parse string representations (from CSV loading)
                if isinstance(sents, str):
                    try:
                        import ast
                        sents = ast.literal_eval(sents)
                    except (ValueError, SyntaxError):
                        continue
                
                if isinstance(scores, str):
                    try:
                        import ast
                        scores = ast.literal_eval(scores)
                    except (ValueError, SyntaxError):
                        continue

                if not (isinstance(sents, list) and isinstance(scores, list)):
                    continue

                # Find sentences exceeding threshold
                idx_iter = [i for i, s in enumerate(scores) if s is not None and np.isfinite(s) and s >= threshold_val]

                for i in idx_iter:
                    if i < len(sents) and i < len(scores):
                        rare_rows.append({
                            "note_id": note_id,
                            binary_column: label,
                            multiclass_column: multiclass_label,
                            "sent_idx": i,
                            "original_sentence": sents[i],
                            "dpmlm_input_sentence": remove_phi_brackets(sents[i]),
                            "idf_score": float(scores[i])
                        })

            df_rare = pd.DataFrame(rare_rows)
            df_rare["threshold_value"] = float(threshold_val)
            df_rare["threshold_k"] = float(sigma_mult)
            
            print(f"    ✓ Extracted {len(df_rare):,} rare sentences")
            if len(df_rare) > 0:
                print(f"    From {df_rare['note_id'].nunique():,} unique documents")
                print(f"    IDF range: {df_rare['idf_score'].min():.4f} - {df_rare['idf_score'].max():.4f}")
            else:
                print(f"    No rare sentences found for this threshold")
            
            results[f"sigma_{sigma_mult}"] = df_rare
        
        return results

    # Original single threshold logic
    if threshold is None and sigma_multiplier is not None:
        all_scores = []
        for scores in df[scores_column]:
            # Parse string representation (from CSV loading)
            if isinstance(scores, str):
                try:
                    import ast
                    scores = ast.literal_eval(scores)
                except (ValueError, SyntaxError):
                    continue
            
            if isinstance(scores, list):
                for s in scores:
                    if s is not None and np.isfinite(s):
                        all_scores.append(s)
        if len(all_scores) == 0:
            raise ValueError("No valid sentence IDF scores found to compute threshold.")
        mean_idf = float(np.mean(all_scores))
        std_idf = float(np.std(all_scores))
        threshold = mean_idf + float(sigma_multiplier) * std_idf
        print(f"  Using threshold = mean + {sigma_multiplier}σ = {threshold:.4f}")
    
    rare_rows = []
    
    for row in df.itertuples(index=False):
        note_id = getattr(row, id_column)
        label = int(getattr(row, binary_column, 0))
        multiclass_label = getattr(row, multiclass_column, None)
        sents = getattr(row, sents_column)
        scores = getattr(row, scores_column)
        rare_i = getattr(row, rare_idx_column, None)
        
        # Parse string representations (from CSV loading)
        if isinstance(sents, str):
            try:
                import ast
                sents = ast.literal_eval(sents)
            except (ValueError, SyntaxError):
                continue
        
        if isinstance(scores, str):
            try:
                import ast
                scores = ast.literal_eval(scores)
            except (ValueError, SyntaxError):
                continue

        if not (isinstance(sents, list) and isinstance(scores, list)):
            continue

        # If rare_idx is present and threshold not provided, use it directly
        if isinstance(rare_i, list) and threshold is None:
            idx_iter = rare_i
        else:
            if threshold is None:
                raise ValueError("Provide threshold or sigma_multiplier if rare_idx is not available.")
            idx_iter = [i for i, s in enumerate(scores) if s is not None and np.isfinite(s) and s >= threshold]

        for i in idx_iter:
            if i < len(sents) and i < len(scores):
                rare_rows.append({
                    "note_id": note_id,
                    binary_column: label,
                    multiclass_column: multiclass_label,
                    "sent_idx": i,
                    "original_sentence": sents[i],
                    "dpmlm_input_sentence": remove_phi_brackets(sents[i]),
                    "idf_score": float(scores[i])
                })

    df_rare = pd.DataFrame(rare_rows)
    if threshold is not None:
        df_rare["threshold_value"] = float(threshold)
    else:
        df_rare["threshold_value"] = None
    if sigma_multiplier is not None:
        df_rare["threshold_k"] = float(sigma_multiplier)
    else:
        df_rare["threshold_k"] = None
    
    print(f"✓ Extracted {len(df_rare):,} rare sentences")
    if len(df_rare) > 0:
        print(f"  From {df_rare['note_id'].nunique():,} unique documents")
        print(f"  IDF range: {df_rare['idf_score'].min():.4f} - {df_rare['idf_score'].max():.4f}")
    else:
        print("  No rare sentences found - check threshold or input data")
    
    return df_rare


if __name__ == "__main__":
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python step4_extract_rare.py <input_csv> [output_csv] [--threshold <float> | --sigma <float> | --multi-sigma]")
        print("  --multi-sigma: Extract for sigma multipliers 1.0, 1.5, 2.0 (like notebook)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    # Check for multi-sigma mode (like notebook)
    use_multi_sigma = "--multi-sigma" in sys.argv
    
    if use_multi_sigma:
        # Multi-threshold mode (like notebook)
        output_dir = "outputs"
        if "--output-dir" in sys.argv:
            output_dir = sys.argv[sys.argv.index("--output-dir") + 1]
        
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(input_path)
        
        # Process all three thresholds like notebook
        sigma_multipliers = [1.0, 1.5, 2.0]
        print(f"Processing multiple thresholds: {sigma_multipliers}")
        
        results = extract_rare_sentences(df, sigma_multipliers=sigma_multipliers)
        
        for key, df_rare in results.items():
            output_file = os.path.join(output_dir, f"step4_rare_{key}.csv")
            df_rare.to_csv(output_file, index=False)
            print(f"✓ Saved {key} to {output_file}")
    else:
        # Original single threshold mode
        if len(sys.argv) < 3:
            print("Usage: python step4_extract_rare.py <input_csv> <output_csv> [--threshold <float> | --sigma <float>]")
            sys.exit(1)
        
        output_path = sys.argv[2]
        
        threshold = None
        sigma_multiplier = None
        if "--threshold" in sys.argv:
            threshold = float(sys.argv[sys.argv.index("--threshold") + 1])
        if "--sigma" in sys.argv:
            sigma_multiplier = float(sys.argv[sys.argv.index("--sigma") + 1])
        
        df = pd.read_csv(input_path)
        df_rare = extract_rare_sentences(df, threshold=threshold, sigma_multiplier=sigma_multiplier)
        df_rare.to_csv(output_path, index=False)
        
        print(f"✓ Saved to {output_path}")
