"""
Step 2: Tokenization for IDF
Tokenize sentences for IDF calculation
"""

import re
import string
import pandas as pd
from tqdm.auto import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ast

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

# PHI placeholder pattern
PHI_RE = re.compile(r"\[\*\*.*?\*\*\]")

# Stopwords (keep negations for clinical context)
STOP = set(stopwords.words("english"))
KEEP_NEG = {"no", "not", "nor", "without", "never", "denies", "denied"}
STOP = STOP - KEEP_NEG


def sentence_to_idf_tokens(sentence):
    """
    Convert ONE sentence into tokens for IDF scoring.
    
    Steps:
    - Remove PHI placeholders [**...**]
    - Lowercase
    - Tokenize
    - Remove punctuation-only, digits, non-alpha, short tokens
    - Remove stopwords (keep negations)
    
    Args:
        sentence: Original sentence string
        
    Returns:
        List of tokens (empty list if no valid tokens)
    """
    if not isinstance(sentence, str) or not sentence.strip():
        return []

    s = sentence.lower()
    s = PHI_RE.sub(" ", s)  # Remove PHI markers

    toks = word_tokenize(s)
    out = []
    
    for tok in toks:
        tok = tok.strip()
        if not tok:
            continue
        
        # Skip punctuation-only tokens
        if all(ch in string.punctuation for ch in tok):
            continue
        
        # Skip tokens with digits
        if any(ch.isdigit() for ch in tok):
            continue
        
        # Check if alphabetic (after removing hyphens/apostrophes)
        alpha = tok.replace("-", "").replace("'", "")
        if not alpha.isalpha():
            continue
        
        # Skip very short tokens
        if len(alpha) < 2:
            continue
        
        # Skip stopwords (except negations)
        if tok in STOP:
            continue
        
        out.append(tok)

    return out


def tokenize_dataset(df, sentences_column='sentences_mimic'):
    """
    Tokenize all sentences in dataset.
    
    Args:
        df: DataFrame with sentences column
        sentences_column: Name of column containing list of sentences
        
    Returns:
        DataFrame with added 'sentences_tokenized' column
    """
    print(f"Tokenizing sentences from {len(df):,} documents...")
    
    tqdm.pandas(desc="Tokenizing")
    def _tokenize_row(sent_list):
        # If list was saved as string in CSV, parse it
        if isinstance(sent_list, str):
            try:
                sent_list = ast.literal_eval(sent_list)
            except (ValueError, SyntaxError):
                return []
        if not isinstance(sent_list, list):
            return []
        return [sentence_to_idf_tokens(s) for s in sent_list]

    df["sentences_tokenized"] = df[sentences_column].progress_apply(_tokenize_row)
    
    # Count empty tokenized sentences
    empty_count = df["sentences_tokenized"].apply(
        lambda doc: sum(len(s) == 0 for s in doc)
    ).sum()
    total_count = df["sentences_tokenized"].apply(len).sum()
    
    print(f"✓ Tokenization complete")
    print(f"  Total sentences: {total_count:,}")
    print(f"  Empty tokenized sentences: {empty_count:,} ({100*empty_count/max(total_count,1):.2f}%)")
    
    return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python step2_tokenization.py <input_csv> <output_csv>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    df = pd.read_csv(input_path)
    df = tokenize_dataset(df)
    df.to_csv(output_path, index=False)
    
    print(f"✓ Saved to {output_path}")
