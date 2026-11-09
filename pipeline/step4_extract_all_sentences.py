"""
Step 4b: Extract ALL Sentences (Baseline)
Extract all sentences from notes without any IDF threshold filtering.
Used to create the baseline dataset where DPMLM is applied to ALL sentences.
 
Output format is identical to step4_extract_rare.py so it can be used
directly with step6_apply_dpmlm.py and step7_rebuild_notes.py.
 
Usage (mortality dataset):
    python master_thesis/step4_extract_all_sentences.py \
        outputs_mp/step1_segmented.csv \
        outputs_mp/step4_all_sentences.csv
 
Usage (procedures dataset):
    python master_thesis/step4_extract_all_sentences.py \
        outputs_proc/step1_segmented.csv \
        outputs_proc/step4_all_sentences.csv \
        --label_col short_codes
"""
 
import pandas as pd
import ast
import re
import sys
import os
import argparse
 
PHI_UNWRAP_RE = re.compile(r"\[\*\*(.*?)\*\*\]")
 
 
def remove_phi_brackets(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return PHI_UNWRAP_RE.sub(r"\1", text)
 
 
def parse_list(x):
    """Parse string-encoded list from CSV."""
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            return list(parsed) if isinstance(parsed, list) else [x]
        except Exception:
            return [x]
    return []
 
 
def extract_all_sentences(df,
                          id_col="id",
                          label_col="hospital_expire_flag",
                          sents_col="sentences_mimic"):
    """
    Extract ALL sentences from all notes into a flat DataFrame.
    No IDF threshold — every sentence is included.
 
    Output columns match step4_extract_rare.py exactly:
        note_id, <label_col>, sent_idx,
        original_sentence, dpmlm_input_sentence
    """
    print("=" * 70)
    print("EXTRACTING ALL SENTENCES (NO THRESHOLD)")
    print("=" * 70)
    print(f"  Input shape  : {df.shape}")
    print(f"  Input columns: {list(df.columns)}")
    print(f"  id_col       : {id_col}")
    print(f"  label_col    : {label_col}")
    print(f"  sents_col    : {sents_col}")
 
    rows = []
 
    for row in df.itertuples(index=False):
        note_id = getattr(row, id_col)
        label   = getattr(row, label_col, 0)
        sents   = parse_list(getattr(row, sents_col, []))
 
        for i, sent in enumerate(sents):
            if not isinstance(sent, str) or not sent.strip():
                continue
            rows.append({
                "note_id":              note_id,
                label_col:              label,
                "sent_idx":             i,
                "original_sentence":    sent,
                "dpmlm_input_sentence": remove_phi_brackets(sent),
            })
 
    df_all = pd.DataFrame(rows)
 
    print(f"\n✓ Extracted {len(df_all):,} sentences")
    print(f"  From {df_all['note_id'].nunique():,} unique notes")
    print(f"  Avg sentences per note: {len(df_all) / df_all['note_id'].nunique():.1f}")
    print(f"  Columns: {list(df_all.columns)}")
 
    return df_all
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input_csv",  help="Path to step1_segmented.csv")
    p.add_argument("output_csv", help="Path to save step4_all_sentences.csv")
    p.add_argument("--label_col", default="hospital_expire_flag",
                   help="Label column name. "
                        "Use 'hospital_expire_flag' for mortality (default), "
                        "'short_codes' for procedures dataset.")
    p.add_argument("--id_col",    default="id",
                   help="ID column name (default: id)")
    p.add_argument("--sents_col", default="sentences_mimic",
                   help="Sentences column name (default: sentences_mimic)")
    args = p.parse_args()
 
    print(f"Loading: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
 
    df_all = extract_all_sentences(
        df,
        id_col    = args.id_col,
        label_col = args.label_col,
        sents_col = args.sents_col,
    )
 
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_all.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved → {args.output_csv}")
    print(f"  Shape : {df_all.shape}")