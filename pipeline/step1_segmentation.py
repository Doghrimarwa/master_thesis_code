"""
Step 1: Text Segmentation
Segment clinical notes into sentences using rule-based approach
"""

import re
import pandas as pd
from tqdm.auto import tqdm

# Abbreviations and titles that should NOT trigger sentence breaks
ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "vs", "etc", "e.g", "i.e", "pt", "mg", "mcg",
    "ml", "hr", "hrs", "min", "bid", "tid", "qid", "qhs", "qod", "prn", "po", "iv"
}

TITLES = {"mr", "mrs", "ms", "dr"}

# Simple PHI matcher (used only to detect non-content segments)
PHI_RE_SIMPLE = re.compile(r"\[\*\*.*?\*\*\]")


def _is_noncontent(seg: str) -> bool:
    """True if segment is basically only PHI/numbers/punct (low lexical content)."""
    if not seg or not seg.strip():
        return True
    s = seg.strip()
    s_no_phi = PHI_RE_SIMPLE.sub(" ", s)
    # count alphabetic words
    alpha_words = re.findall(r"[A-Za-z]{2,}", s_no_phi)
    if len(alpha_words) >= 2:
        return False
    # if mostly digits/punct/phi, treat as non-content
    has_phi = bool(PHI_RE_SIMPLE.search(s))
    has_digit = bool(re.search(r"\d", s_no_phi))
    return has_phi or has_digit or len(alpha_words) == 0


def segment_mimic_simple(text, min_words=6):
    """
    Segment MIMIC clinical note into sentences.
    
    Args:
        text: Raw clinical note text
        min_words: Minimum words per sentence
        
    Returns:
        List of sentence strings
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # Clean and prepare text
    t = text.strip()
    t = re.sub(r'(?m)^\s*([A-Z][A-Z0-9\s/\-]{2,40}):\s*', r'\1: ', t)
    t = re.sub(r'(?m)([:;])\s*-\s*', r'\1\n- ', t)
    t = re.sub(r'(\S)\s+(\d+[\.\)])\s+', r'\1\n\2 ', t)

    chunks = [line.strip() for line in t.split('\n') if line.strip()]
    segments = []
    buffer = ""

    def clean_last_word(s):
        word = s.split()[-1].lower()
        return re.sub(r'[^a-z0-9/]+$', '', word)

    for chunk in chunks:
        parts = re.split(r'(?<=[.!?])\s+', chunk)
        for part in parts:
            part = part.strip()
            if not part:
                continue

            candidate = (buffer + " " + part).strip() if buffer else part
            words = candidate.split()

            # Skip splitting if too short, is a header, ends with title/abbr, or is non-content
            last_clean = clean_last_word(candidate)
            if (len(words) < min_words or
                (candidate.endswith(":") and len(words) <= 4) or
                last_clean in TITLES or
                last_clean in ABBREVIATIONS or
                _is_noncontent(candidate)):
                buffer = candidate
                continue

            # Safe to split here
            segments.append(candidate)
            buffer = ""

    # Handle leftover
    if buffer:
        if segments and (len(buffer.split()) < min_words or _is_noncontent(buffer)):
            segments[-1] += " " + buffer
        else:
            segments.append(buffer)

    # Clean up spaces
    result = [' '.join(s.split()) for s in segments if s.strip()]

    # Fallback: if everything is too short, return whole note
    if result and all(len(s.split()) < min_words for s in result):
        return [' '.join(text.split())]

    return result


def segment_dataset(df, text_column='text', min_words=6):
    """
    Segment all documents in a dataset.
    
    Args:
        df: DataFrame with text column
        text_column: Name of text column
        min_words: Minimum words per sentence
        
    Returns:
        DataFrame with added 'sentences_mimic' and 'sentence_count_mimic' columns
    """
    print(f"Segmenting {len(df):,} documents...")
    
    tqdm.pandas(desc="Segmenting sentences")
    df['sentences_mimic'] = df[text_column].progress_apply(
        lambda x: segment_mimic_simple(x, min_words=min_words)
    )
    df['sentence_count_mimic'] = df['sentences_mimic'].apply(len)
    
    print(f"✓ Segmentation complete")
    print(f"  Total sentences: {df['sentence_count_mimic'].sum():,}")
    print(f"  Avg sentences/document: {df['sentence_count_mimic'].mean():.1f}")
    
    return df


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python step1_segmentation.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    df = pd.read_csv(input_path)
    df = segment_dataset(df, text_column='text')
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved to {output_path}")
