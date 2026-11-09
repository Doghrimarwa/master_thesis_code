# Development history

This history was rebuilt from the chat sessions in which the code was written. It records what changed and why, so the design decisions in the thesis can be traced back to concrete problems.

## Phase 1: Getting DP-MLM to run on MIMIC-III (Feb 2026)

- **Start.** A Google Colab notebook applied DP-MLM to rare MIMIC-III sentences. It failed with `'DPMLM' object has no attribute 'MODEL'`. Cause: a `wget` of the upstream `DPMLM.py` overwrote the locally fixed copy.
- **Patched `DPMLM.py`** (`external/DPMLM_patch/`):
  - store `self.MODEL`;
  - accept `clip_min`/`clip_max` in the constructor, so the sensitivity Δu = C_max − C_min is computed from MIMIC-specific bounds instead of the SST-2 defaults (−3.21 / 16.30);
  - add `_encode_safe()`, because long clinical sentences pushed `<mask>` past the 512-token limit (`ValueError: '50264 is not in list'`).
- **Moved to the DFKI SLURM cluster.** The code was split into `step1`–`step7` and `run_pipeline.py`; outputs go to `outputs_mp/`.
- **Three thresholds, one set of bounds.** Rare sentences are extracted for k ∈ {1.0, 1.5, 2.0}. Clipping bounds are computed once on the k = 2.0 set and reused (`step5_clipping_bounds_sigma_2.0.json`). Output files carry a `sigma_<k>` prefix.
- **26 Feb.** Line-by-line comparison of the patched and upstream DP-MLM, to confirm the exponential mechanism is untouched: clip → divide by 2Δu/ε → softmax → sample.

## Phase 2: Data and downstream tasks (Mar 2026)

- **Datasets.** Built with the van Aken et al. `tasks/` scripts. Two pandas incompatibilities were fixed in `mimic_utils.py`: `.str.strip()` before the CATEGORY/DESCRIPTION filters (otherwise 0 rows) and `regex=True` in the newline replacement (otherwise every section came back empty). Splits were concatenated, deduplicated on (id, text) and sampled at 50 % with seed 42, giving `mp_50pct.csv` (24,342 notes). The procedure dataset uses `short_codes` (3-digit ICD-9 groups).
- **Method 2 (explored, not in the thesis).** Token-level IDF, IPI detection based on Baroud et al. (2025), and KMeans clusters of averaged sub-word embeddings that constrain DP-MLM replacements. The token-IDF distribution spikes at hapax legomena (IDF ≈ 5.03), so percentile and absolute thresholds were studied. 98.5 % of rare tokens are multi-subword, which motivated embedding averaging. The thesis lists this direction as future work.
- **25 Mar.** Step 6 now passes `TEMP=True` explicitly (the DP sampling path), uses `torch.no_grad()` instead of `inference_mode()` (the latter broke hidden-state extraction), and skips sentences with fewer than 3 words. Rewriting is token-level, so a sentence with n tokens consumes n·ε under sequential composition.

## Phase 3: Models, containers, classification (Apr 2026)

- **Rewriting models.** `distilroberta-base` (Wikipedia/books) was replaced by `FacebookAI/roberta-base` and the clinical `pminervini/RoBERTa-base-PM-M3-Voc-hf`. Clipping bounds are recomputed per model (`outputs_mp/clipping/<model>/`).
- **Container issues.** NVIDIA 24.01 ships PyTorch 2.2, but newer `transformers` need ≥ 2.4. 24.05 ships `2.4.0a0`, which fails the version check, so `transformers==4.40.2` was pinned.
- **Classifier.** A self-written HuggingFace trainer was compared with the original FARM `doc_classification.py`. FARM was kept, as the faithful reproduction of van Aken et al. It needs Python 3.8 (container 21.06), `dotmap==1.3.0`, `mlflow==1.20.2`, `protobuf==3.20.1`, and ray removed with three `sed` commands. Bio_ClinicalBERT is always the classifier; the rewriting model only determines the data.
- **Step 8** was split into `step8a_create_splits` (70/10/20, seed 123, same admissions for every configuration), `step8b_generate_configs` (absolute paths) and `step8c_experiments` (a separate `/tmp` cache per run, because FARM otherwise reused data cached from another ε).
- **Storage.** `language_model.bin` and `prediction_head_0.bin` were deleted after evaluation, freeing about 5 GB.
- **23 Apr. Full-rewriting baseline crash.** On all 567k sentences, the first over-long sentence fired a CUDA device-side assert. That killed the GPU context, and 567,137 sentences then "failed" at 147 it/s. Root cause: `privatize()` encodes without truncation. The fix in `step6_apply_dpmlm.py` splits sentences into word-boundary chunks of ≤ 200 tokens (CONCAT input 2·200 + 4 ≤ 512), rewrites each chunk and joins them, and rebuilds the model if the CUDA context dies anyway.

## Phase 4: Baselines and repeated runs (May–Jun 2026)

- **Baseline semantics corrected.** "Baseline" first meant classifying the original notes. The thesis now has three modes:
  - `baseline`: original notes;
  - `all`: Full DP-MLM, every sentence rewritten (`step4_extract_all_sentences.py`, `--sigma all`);
  - `sigma`: selective rewriting.
- **Linguistic utility** following DP-MLM / DP-IPI: S is the cosine similarity averaged over all-mpnet-base-v2, gte-large and jina-embeddings-v3; C = PPL_orig / PPL_priv with GPT-2 on the first 128 tokens, on 500 sampled notes (seed 42).
- **Repeated runs.** `--output_root` (`outputs_mp`, `outputs_mp_run2`, `outputs_mp_run3`) holds the three classification runs.
- **500R scenario.** 450 distractor patients are added. They only have background knowledge, with NaN in every evaluated column, so tri.py trains a 500-class model but evaluates the 50 real patients only.

## Phase 5: Privacy evaluation and trade-off (Jul 2026)

- **8 Jul.** `step8c_experiments.py` v2: `--task mp|proc`, `--mode sigma|all|baseline`, `--seed`, and skipping of finished experiments, so resubmitted 72 h jobs resume where they stopped. Procedure prediction is multi-label: the label list comes from `ALL_3_DIGIT_PRO_CODES.txt`, and the stratified split falls back to a random split.
- **TRI pipeline** (`tri/`), per seed in {42, 123, 456, 789, 1337}:
  1. sample 50 real patients and 450 distractors;
  2. generate Qwen2.5-7B-Instruct biographies (DP-IPI Figure 6 prompt, 50–100 words of indirect identifiers) from the original notes;
  3. build one wide CSV per scenario: `original_background` for training, and one test column per privatized setup;
  4. run tri.py (DistilBERT, additional MLM pre-training, 15 fine-tuning epochs).
  
  `step3` v2 skips missing setups instead of adding all-NaN columns, which had crashed tri.py.
- **10 Jul. Baseline TRIR.** An `original_notes` column (the un-privatized note) gives TRIR_original, which γ needs.
- **15 Jul.** `compute_trir_average_v2.py` (needs `dropna=False`, or the original rows are dropped) and `compute_gamma.py`:
  γ = (1 − TRIR_p/TRIR_b) − mean(1 − ROC_p/ROC_b, 1 − S, 1 − C).
- **16 Jul. Pipeline review.** Issues found (and partly reported as limitations in the thesis):
  - clipping bounds are calibrated on the k = 2.0 rare sentences, which are also rewritten (circular, and a confound between selective and full rewriting);
  - the OOV fallback in `sentence_mean_idf` uses the minimum IDF;
  - ε is per token, so composition must be stated explicitly.

## Phase 6: Writing (Jul–Aug 2026)

- Chapters 1–7 were written and revised.
- The procedure code count was corrected from 731 to 711, matching van Aken et al. Table 2.
- The thesis was submitted on 5 Aug 2026.
