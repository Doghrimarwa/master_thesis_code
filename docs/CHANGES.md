# Changes compared with the code used for the thesis

The scripts in this repository are the final versions run on the DFKI cluster, plus the fixes below. None of the fixes changes how a **successful** run computes its numbers. They stop crashes and silent errors, remove hard-coded paths, and make the naming consistent.

## Fixes

| # | Where | Problem in the thesis code | Fix |
|---|---|---|---|
| 1 | 8a / 8b / 8c | Configs written by 8b (`step8_configs_baseline/…`) did not match the names 8c read (`step8_configs_all/…`, `step8_configs_original/…`). No script created the original-data splits; they were made by hand. | One layout for all steps (`common/layout.py`); 8a creates the original splits too; 8b writes a config for every split folder. |
| 2 | everywhere | "baseline" meant **Full DP-MLM** in 8a/8b/TRI but the **original data** in 8c. | Folders are named `original`, `selective_k<k>`, `full`. |
| 3 | 8c | `roberta_clinical` Full ε = 25 was missing from the experiment list, so `compute_gamma.py` failed on a fresh run. | All experiments are listed. |
| 4 | step 6 | If CUDA recovery failed, a truncated final file was written and the checkpoint deleted. Step 7 then left the remaining sentences un-privatized, without a warning. | The checkpoint is kept and no final file is written; re-running resumes. |
| 5 | step 6 | The output folder was only created at the end, so early checkpoints silently failed. | Folder created before the loop. |
| 6 | step 6 | The DP-MLM import path pointed one folder above the repository. | `<repo>/external/DPMLM`. |
| 7 | step 5 | Default model `distilroberta-base`: forgetting `--model` silently gave bounds for the wrong model. | `--model` is required. |
| 8 | TRI, γ, figures | Hard-coded `/netscratch/mdoghri/...`; TRIR table read from the current folder. | `--base` argument; all tables in `results/`. |
| 9 | TRI step 2 | Qwen sampled without a fixed seed. | `torch.manual_seed(seed)` before generation. |
| 10 | 8a | No check that every note was replaced by its private version; empty private notes became the text `"nan"`. | 8a stops if fewer than 99 % of notes are replaced; empty notes are skipped. |
| 11 | TRI step 3 | Missing setups were added as all-empty columns, which makes `tri.py` crash. | Missing setups are skipped with a warning. |
| 12 | step 3 | The fallback IDF (`min` over the whole vocabulary) was recomputed for every sentence. | Computed once; identical result, much faster. |
| 13 | external | Upstream DP-MLM was restructured into a package in April 2026, so the patched `DPMLM.py` no longer fits the newest version. | `setup_external.sh` pins DP-MLM to commit `42b78f2` (and the other repositories to the versions checked in 2026-11). |

## Re-organisation

- `extract_results.py` + `average_results.py` → `evaluation/collect_classification.py`. Classification seeds are stored in `step8_results/seed_<s>/` instead of separate `outputs_<task>_run2/_run3` roots.
- The original-data classifier is trained once per seed instead of once per rewriting model, because the data are identical. Its row is copied for both models in the tables.
- `step11_generate_figures.py` → `evaluation/generate_figures.py`; `step10_compute_gamma.py` → `evaluation/compute_gamma.py`.
- TRI scripts renamed `privacy/tri_step1_sample.py` … `tri_step5_average.py`.
- New: `scripts/run_all.sh` (all commands), `scripts/make_toy_data.py` and `tests/smoke_test.sh` (end-to-end test on synthetic data).

## Reconstructed parts

Steps 1–8, the TRI scripts, γ and the figure script are the final cluster versions. `evaluation/compute_utility_metrics.py`, `scripts/prepare_datasets.py` and the DP-MLM patch were recovered from the development chats (see [HISTORY.md](HISTORY.md)), not from the cluster, so check them first if a re-run differs.
