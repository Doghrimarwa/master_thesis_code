#!/usr/bin/env bash
# Runs the complete thesis pipeline for ONE task, phase by phase.
#
#   bash scripts/run_all.sh <task> <phase>
#     task : mp | proc
#     phase: prepare | rewrite | classify | utility | privacy | report | all
#
# Environment variables:
#   BASE       project/data root (default: repository root). Must contain data/<task>_50pct.csv
#   RUN        command prefix for heavy GPU steps, e.g. a SLURM call:
#              RUN="srun -p RTXA6000 --gpus=1 --mem=64G --time=24:00:00 --container-image=... --container-workdir=$PWD"
#   FARM_PY    python of the FARM environment for step 8c (default: python)
#   MODELS     rewriting models (default: "roberta_base roberta_clinical")
#   EPS        privacy budgets   (default: "25 50 75 100")
#   SIGMAS     rarity thresholds (default: "1.0 1.5 2.0")
#   HF_roberta_base / HF_roberta_clinical   override the model id/path (testing)
#   UTILITY_ARGS, TRI1_ARGS, TRI2_ARGS, TRI4_ARGS   extra arguments (used by tests/smoke_test.sh)
#
# Phases can be run separately (they take days on the full MIMIC subset).
# Every heavy step can be resubmitted: step 6 resumes from checkpoints,
# step 8c and TRI skip finished experiments.
set -euo pipefail
cd "$(dirname "$0")/.."
TASK=${1:?task: mp|proc}
PHASE=${2:-all}
BASE=${BASE:-$PWD}
RUN=${RUN:-}
FARM_PY=${FARM_PY:-python}
MODELS=${MODELS:-"roberta_base roberta_clinical"}
EPS=${EPS:-"25 50 75 100"}
SIGMAS=${SIGMAS:-"1.0 1.5 2.0"}
OUT="$BASE/outputs_$TASK"
declare -A HF=([roberta_base]="${HF_roberta_base:-FacebookAI/roberta-base}"
               [roberta_clinical]="${HF_roberta_clinical:-pminervini/RoBERTa-base-PM-M3-Voc-hf}")
UTILITY_ARGS=${UTILITY_ARGS:-}; TRI1_ARGS=${TRI1_ARGS:-}; TRI2_ARGS=${TRI2_ARGS:-}; TRI4_ARGS=${TRI4_ARGS:-}
[ "$TASK" = "mp" ] && LABEL=hospital_expire_flag || LABEL=short_codes
EPS_LIST=$(echo $EPS | sed 's/ /,/g')
mkdir -p "$OUT"

prepare() {   # steps 1-5: segmentation, IDF, rare sentences, clipping bounds
  python pipeline/step1_segmentation.py "$BASE/data/${TASK}_50pct.csv" "$OUT/step1_segmented.csv"
  python pipeline/step2_tokenization.py "$OUT/step1_segmented.csv" "$OUT/step2_tokenized.csv"
  python pipeline/step3_idf_calculation.py "$OUT/step2_tokenized.csv" "$OUT/step3_idf.csv"
  python pipeline/step4_extract_rare.py "$OUT/step3_idf.csv" --multi-sigma --output-dir "$OUT"
  python pipeline/step4_extract_all_sentences.py "$OUT/step1_segmented.csv" "$OUT/step4_all_sentences.csv" --label_col "$LABEL"
  for M in $MODELS; do   # bounds on the k=2.0 rare pool, used for ALL conditions (as in the thesis)
    $RUN python pipeline/step5_clipping_bounds.py "$OUT/step4_rare_sigma_2.0.csv" \
      "$OUT/step5_clipping/$M/step5_clipping_bounds_sigma_2.0.json" --model "${HF[$M]}"
  done
}

rewrite() {   # steps 6-7: DP-MLM rewriting and note reconstruction
  for M in $MODELS; do
    B="$OUT/step5_clipping/$M/step5_clipping_bounds_sigma_2.0.json"
    for K in $SIGMAS; do
      $RUN python pipeline/step6_apply_dpmlm.py "$OUT/step4_rare_sigma_$K.csv" "$B" \
        "$OUT/step6_dpmlm/selective_k$K/$M" --eps "$EPS_LIST" --sigma "$K" \
        --model "${HF[$M]}" --checkpoint_every 1000
      python pipeline/step7_rebuild_notes.py "$OUT/step6_dpmlm/selective_k$K/$M" \
        "$OUT/step1_segmented.csv" "$OUT/step7_notes/selective_k$K/$M"
    done
    # Full DP-MLM: tip from the thesis runs, submit one job per epsilon in parallel (--eps 25 ...)
    $RUN python pipeline/step6_apply_dpmlm.py "$OUT/step4_all_sentences.csv" "$B" \
      "$OUT/step6_dpmlm/full/$M" --eps "$EPS_LIST" --sigma all \
      --model "${HF[$M]}" --checkpoint_every 1000
    python pipeline/step7_rebuild_notes.py "$OUT/step6_dpmlm/full/$M" \
      "$OUT/step1_segmented.csv" "$OUT/step7_notes/full/$M"
  done
}

classify() {  # step 8: splits, configs, Bio_ClinicalBERT x 3 seeds
  python pipeline/step8a_create_splits.py --task "$TASK" --base "$BASE"
  python pipeline/step8b_generate_configs.py --task "$TASK" --base "$BASE"
  for S in 11 22 33; do
    $RUN $FARM_PY pipeline/step8c_experiments.py --task "$TASK" --base "$BASE" --seed $S
  done
  python evaluation/collect_classification.py --task "$TASK" --base "$BASE"
}

utility() {   # semantic similarity S and coherence C
  $RUN python evaluation/compute_utility_metrics.py --task "$TASK" --base "$BASE" $UTILITY_ARGS
}

privacy() {   # TRI attack, 5 seeds
  for S in 42 123 456 789 1337; do
    python privacy/tri_step1_sample.py --seed $S --task "$TASK" --base "$BASE" $TRI1_ARGS
    $RUN python privacy/tri_step2_backgrounds.py --seed $S --task "$TASK" --base "$BASE" $TRI2_ARGS
    python privacy/tri_step3_prepare.py --seed $S --task "$TASK" --base "$BASE"
    $RUN python privacy/tri_step4_attack.py --seed $S --task "$TASK" --base "$BASE" $TRI4_ARGS
  done
}

report() {    # TRIR average (both tasks found), gamma, figures
  python privacy/tri_step5_average.py --base "$BASE"
  python evaluation/compute_gamma.py --task "$TASK" --base "$BASE"
  python evaluation/generate_figures.py --task "$TASK" --base "$BASE"
}

case "$PHASE" in
  prepare|rewrite|classify|utility|privacy|report) $PHASE ;;
  all) prepare; rewrite; classify; utility; privacy; report ;;
  *) echo "unknown phase $PHASE"; exit 1 ;;
esac
echo "✓ $TASK / $PHASE done"
