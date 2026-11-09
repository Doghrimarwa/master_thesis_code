# Thesis text vs. code

**All results reported in the thesis were produced by the code in this repository.** In a few places the submitted text describes the method differently, or leaves out an implementation detail. This page lists them, explains which choice the code made, and why.

## Overview

| Step | Thesis says | Code does | Verdict |
|---|---|---|---|
| Starting data | MIMIC-III is de-identified; placeholders are a starting point, not a substitute | same | same |
| Segmentation | notes are split into an ordered list of sentences | rule-based segments of ≥ 6 words, order kept | wording: detail not described |
| Stopwords | removed for scoring, negations kept; the full sentence goes to DP-MLM | same | same |
| **Placeholders in rarity** | tokenized and scored like other tokens | **removed** before IDF | **gap** (code more reasonable) |
| **Placeholders in rewriting** | part of the sentence given to DP-MLM, brackets included | brackets removed, contents rewritten | **gap** (small) |
| IDF and score | IDF per note, mean token IDF | same | same |
| Threshold | τ = μ + kσ, k ∈ {1.0, 1.5, 2.0} | same | same |
| **Clipping bounds** | 1000 samples, μ / μ + 4σ, per model; Appendix A: Full uses the whole corpus | same formula, computed on the k = 2.0 rare pool and used for Selective **and** Full | **gap** (small; same Δ everywhere) |
| DP-MLM mechanism | exponential mechanism, ε per token | same | same |
| **Long segments** | "no segmentation or chunking step" | segments > 200 tokens rewritten in chunks | **gap** (technical safeguard) |
| Failed rewrites | omitted; the original sentence stays | the original sentence is kept | same outcome |
| Rebuilding notes | sentences put back at their position | same, joined with spaces | same |
| Utility, privacy, γ | Bio_ClinicalBERT, S, C, TRI, γ | same | same |

## 1. Placeholders and rarity

MIMIC-III is **de-identified, not anonymised**: direct identifiers were replaced by placeholders such as `[**2158-1-3**]` (dates are shifted surrogates), but indirect clues remain in the text.

- **Thesis (§5.2):** placeholders are tokenized and scored like any other token, so they can make a sentence rare.
- **Code:** `step2_tokenization.py` deletes them before IDF, so only the remaining words decide rarity.
- **Why the code is more reasonable:**
  - Surrogate dates and IDs are almost unique, so counting them would make nearly every sentence with a date look rare.
  - Rarity would then point at material that de-identification has already handled, instead of at what it leaves behind (ethnicity, occupation, unusual history).
  - That remaining risk is exactly what the thesis addresses.

## 2. Placeholders and rewriting

Selection works **per sentence**; DP-MLM then rewrites **every word** of a selected sentence (`STOP=False`). A placeholder therefore gets rewritten whenever the sentence around it was selected because of its other words. Placeholders in non-selected sentences stay unchanged; the thesis case study shows both cases (sentences 55 and 58).

- **Thesis:** placeholders, with brackets, are part of the text DP-MLM rewrites.
- **Code:** brackets are removed and the contents are rewritten (`Name (NI) 1403` → `Number (Ni) 1400`).
- **In both versions, placeholders are rewritten.** That adds little protection, because the value already stands in for a removed identifier. It also spends ε (per token) and produces nonsense.
- **Improvement for future work:** keep placeholders unchanged and rewrite only the surrounding words.

## 3. Long segments (chunking)

The segmenter can produce long segments: lists, or text without full stops. RoBERTa reads at most 512 tokens, and CONCAT mode puts the sentence in twice. Before the fix, the first over-long sentence crashed the GPU and all later sentences failed.

Step 6 therefore splits segments over 200 sub-word tokens into word-boundary chunks, rewrites each chunk and joins them back. Mechanism and per-token ε are unchanged; this mainly affects Full DP-MLM. Wording for the text: *"Segments longer than the model's input limit are rewritten in 200-token chunks and joined back; this is a technical safeguard, not part of the method."*

## 4. Clipping bounds

Bounds depend on the model's logit scale and on the text domain, so the DP-MLM paper's values cannot be reused for clinical text or a clinical model. They were recalculated with the same procedure, once per rewriting model.

- **Thesis, Appendix A:** Full used bounds from the whole corpus.
- **Code:** the k = 2.0 bounds were used for all conditions, which gives the same Δ, and hence a fair comparison.
- **Limitation:** the bounds are computed on the private corpus itself. Computing them on a separate public clinical corpus (or with a DP estimate) would make the guarantee fully independent of the private data.

## 5. Classification details

- The original-data condition was trained once per rewriting model in the thesis runs (identical data and seed). This repository trains it once per seed and copies the row for both models.
- In the thesis runs the three seeds were stored in separate output folders (`outputs_<task>`, `_run2`, `_run3`). Here they are `step8_results/seed_11|22|33`.

Neither change affects how the numbers are computed.
