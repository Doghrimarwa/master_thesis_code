# Pipeline in detail

What each step does, why, and what it writes. Paths are relative to `<BASE>/outputs_<task>/` unless stated otherwise. All paths are defined in `common/layout.py`.

## Phase A: find the rare sentences (steps 1–4)

### Step 1: segmentation (`pipeline/step1_segmentation.py`)
Rule-based splitter adapted to clinical notes:
- breaks at line breaks and at `. ! ?`;
- does not break after abbreviations and titles (`Dr.`, `mg`, `b.i.d.`, ...);
- glues pieces shorter than **6 words** onto the next one;
- merges pieces that are only placeholders, numbers or punctuation;
- keeps the order of segments (needed to put rewritten sentences back in step 7).

A "sentence" is therefore a **segment of at least 6 words**. Some segments are long (lists, text without full stops); step 6 handles those by chunking.
Output: `step1_segmented.csv` with `sentences_mimic` (list of segments) and `sentence_count_mimic`.

### Step 2: tokenization for IDF (`pipeline/step2_tokenization.py`)
Builds a filtered copy of every segment, used **only** for scoring:
- lowercases and **removes MIMIC placeholders** `[**…**]`;
- drops punctuation, tokens containing digits, and 1-letter tokens;
- removes English stopwords but **keeps negations** (no, not, nor, without, never, denies, denied).

Output: `step2_tokenized.csv` with `sentences_tokenized`. The original segment text is kept untouched for rewriting.

### Step 3: IDF (`pipeline/step3_idf_calculation.py`)
- Token IDF = log10((N + 1) / (DF + 1)) + 1, where N = number of notes and DF = number of notes containing the token.
- Sentence rarity r(s) = mean IDF of its tokens. A sentence with no tokens left has no score and is never selected.

Output: `step3_idf.csv` (`sentence_idf_mean`) and `step3_idf_token_idf.csv` (vocabulary sorted by IDF).

### Step 4: selection (`pipeline/step4_extract_rare.py`, `pipeline/step4_extract_all_sentences.py`)
- μ and σ of r(s) are computed **once over all scored sentences of the corpus**.
- A sentence is rare if r(s) ≥ τ_k = μ + kσ, for k ∈ {1.0, 1.5, 2.0}.
- `dpmlm_input_sentence` is the segment with the placeholder brackets removed and the contents kept (`[**Name (NI) 1403**]` → `Name (NI) 1403`).

Output: one row per selected sentence (`note_id, sent_idx, original_sentence, dpmlm_input_sentence, idf_score, threshold_value, threshold_k`) in `step4_rare_sigma_<k>.csv`. For Full DP-MLM, `step4_all_sentences.csv` contains every sentence.

## Phase B: rewrite (steps 5–7)

### Step 5: clipping bounds (`pipeline/step5_clipping_bounds.py`)
Follows Meisenbacher et al. (2024):
1. sample 1000 sentences (seed 42);
2. mask one random word in each;
3. collect all vocabulary logits at the masked position;
4. set C_min = μ and C_max = μ + 4σ.

The sensitivity is Δ = C_max − C_min. Bounds depend on the model and on the text domain, so they are recomputed **per rewriting model**. In the thesis runs they were computed on the k = 2.0 rare pool and used for **all** conditions (Selective and Full), so every condition has the same Δ for a given model.

Reference value (MP, RoBERTa-clinical): C_min = −2.110, C_max = 14.495, Δ = 16.605, n_logits = 50,008,000.

### Step 6: DP-MLM (`pipeline/step6_apply_dpmlm.py`, uses `external/DPMLM/DPMLM.py`)
For every selected sentence and every ε, DP-MLM goes through the sentence **word by word**:
1. masks the word, using CONCAT mode (the original sentence is given as context);
2. clips the logits to [C_min, C_max];
3. samples a replacement with probability ∝ exp(ε · logit / (2Δ)).

Settings: `TEMP=True` (DP sampling) and `STOP=False` (stopwords are rewritten too). ε is **per token**, so a sentence with n words uses n·ε.

Practical safeguards (not part of the method):
- segments over 200 sub-word tokens are split at word boundaries into chunks, rewritten and joined again (2 × 200 + 4 < 512 tokens in CONCAT mode);
- sentences with fewer than 3 words, and chunks that raise an error, keep their original text;
- a crashed CUDA context is rebuilt automatically;
- a checkpoint is written every `--checkpoint_every` sentences, and re-running the command resumes from it.

Output: `step6_dpmlm/<setup>/<model>/df_rare_dpmlm_sigma_<k|all>_eps_<ε>.csv` with `dpmlm_sentence`, `perturbed`, `total`, `perturbation_rate`.

### Step 7: rebuild notes (`pipeline/step7_rebuild_notes.py`)
Replaces each segment at its (note_id, sent_idx) position with its rewrite and joins all segments with single spaces into `note_private`. Non-selected segments are unchanged, placeholders included.
Output: `step7_notes/<setup>/<model>/priv_notes_<step-6 file name>.csv`.

## Phase C: evaluate

### Step 8: downstream utility (`pipeline/step8a…8c`, `evaluation/collect_classification.py`)
- **8a:** one stratified 70/10/20 split (seed 123; random split for the multilabel PROC task) for every condition, so all conditions are tested on the same admissions. It also writes the original-data splits.
- **8b:** one FARM YAML config per split folder.
- **8c:** fine-tunes and evaluates Bio_ClinicalBERT with `doc_classification.py` (van Aken et al.) for one seed. Run it for seeds 11, 22 and 33, in the FARM environment.
- **collect_classification.py:** reads all `eval_results.txt` files and writes per-seed and averaged ROC-AUC tables.

### Linguistic utility (`evaluation/compute_utility_metrics.py`)
On 500 sampled notes per setup:
- **S** = mean cosine similarity between original and private notes, averaged over all-mpnet-base-v2, gte-large and jina-embeddings-v3;
- **C** = PPL_orig / PPL_priv, with GPT-2 perplexity on the first 128 tokens.

### Privacy: TRI attack (`privacy/tri_step1…5`)
1. Sample 50 target patients and 450 distractors per seed.
2. Qwen2.5-7B-Instruct writes a short biography (occupation, family, habits, events) from the first 3000 characters of each **original** note. This is the attacker's background knowledge.
3. Build one wide table per scenario with columns:
   - `original_notes` (gives the baseline TRIR);
   - 24 selective columns `sigma<k>_eps<ε>_<model>`;
   - 8 full columns `full_eps<ε>_<model>`.
4. `tri.py` (Manzanares-Salor et al.) trains DistilBERT on the biographies and tries to re-identify every column. **50E**: 50 candidates; **500R**: 50 + 450 distractors.
5. Average TRIR over seeds 42, 123, 456, 789, 1337 → `results/trir_averaged_results.csv`.

### Trade-off and figures
- `evaluation/compute_gamma.py`: γ = (1 − TRIR_p/TRIR_o) − ⅓[(1 − ROC_p/ROC_o) + (1 − S) + (1 − C)], per task, scenario, setup, model and ε.
- `evaluation/generate_figures.py`: all Chapter 6 figures (utility grid, TRIR bars, γ heatmap and bars).

## Output layout

```
<BASE>/
├── data/{mp,proc}_50pct.csv
├── outputs_<task>/
│   ├── step1_segmented.csv  step2_tokenized.csv  step3_idf.csv  step3_idf_token_idf.csv
│   ├── step4_rare_sigma_{1.0,1.5,2.0}.csv  step4_all_sentences.csv
│   ├── step5_clipping/<model>/step5_clipping_bounds_sigma_2.0.json
│   ├── step6_dpmlm/<setup>/<model>/df_rare_dpmlm_sigma_<k|all>_eps_<ε>.csv
│   ├── step7_notes/<setup>/<model>/priv_notes_df_rare_dpmlm_sigma_<k|all>_eps_<ε>.csv
│   ├── step8_splits/{original | <setup>/<model>/eps_<ε>}/<TASK>_adm_{train,val,test}.csv
│   ├── step8_configs/{original | <setup>/<model>/eps_<ε>}/config.yaml
│   ├── step8_results/seed_<s>/{original | <setup>/<model>/eps_<ε>}/…/eval_results.txt
│   └── ALL_3_DIGIT_PRO_CODES.txt            (procedures only)
├── privacy_eval/seed_<s>/
│   ├── <task>_sampled_50_ids.csv  <task>_distractor_450_ids.csv
│   ├── bg_<task>_original.csv  bg_<task>_distractors_500R.csv
│   ├── tri_data_<task>/tri_{50E,500R}.csv
│   └── tri_results_<task>/{50E,500R}/Results.csv, tri_summary.csv
└── results/
    ├── classification_<task>_seed_<s>.csv  classification_<task>_averaged.csv
    ├── utility_metrics_<task>.csv  trir_averaged_results.csv  gamma_tradeoff_<task>.csv
    └── figures/*.png
```

`<setup>` ∈ {`selective_k1.0`, `selective_k1.5`, `selective_k2.0`, `full`}; `<model>` ∈ {`roberta_base`, `roberta_clinical`}.

Inside the CSV tables the thesis labels are kept so the evaluation scripts work unchanged: `rare_sigma_<k>` = `selective_k<k>`, `all_sentences` = `full`, `rare_sentence` / `full_rewriting` in the TRIR table.
