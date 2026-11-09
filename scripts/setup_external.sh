#!/usr/bin/env bash
# Clones the external code the pipeline depends on and applies the thesis changes.
# Run once from the repository root:   bash scripts/setup_external.sh
# Set SKIP_CLASSIFIER=1 to skip downloading Bio_ClinicalBERT.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p external cache

# 1) DP-MLM (Meisenbacher et al., 2024) + the thesis patch of DPMLM.py
# Pinned to commit 42b78f2 (2025-12-09), the last version with DPMLM.py at the
# repository root, which the thesis used. Later upstream versions (from April
# 2026) are a restructured package (src/DPMLM/core.py) and are NOT compatible.
if [ ! -d external/DPMLM ]; then
  git clone https://github.com/sjmeis/DPMLM external/DPMLM
fi
git -C external/DPMLM checkout -q 42b78f2
cp external/DPMLM_patch/DPMLM.py external/DPMLM/DPMLM.py
echo "✓ external/DPMLM @42b78f2 (patched DPMLM.py copied in)"

# 2) Clinical outcome prediction (van Aken et al., 2021): doc_classification.py (FARM)
if [ ! -d external/clinical-outcome-prediction ]; then
  git clone https://github.com/bvanaken/clinical-outcome-prediction external/clinical-outcome-prediction
fi
git -C external/clinical-outcome-prediction checkout -q b376f4b   # version checked in 2026-11
DOC=external/clinical-outcome-prediction/experiments/doc_classification.py
# remove the ray[tune] dependency (no hyper-parameter search is used)
sed -i 's/^from ray import tune//' "$DOC"
sed -i 's/tune.session.get_trial_name() if do_hpo else None/None/' "$DOC"
sed -i 's/tune.report(roc_auc_dev=eval_score, train_loss=train_loss)/pass/' "$DOC"
echo "✓ external/clinical-outcome-prediction (ray removed)"

# 3) Text Re-Identification attack (Manzanares-Salor et al., 2024)
# tri.py needs spaCy's en_core_web_lg model:  python -m spacy download en_core_web_lg
if [ ! -d external/tri_repo ]; then
  git clone https://github.com/BenetManzanaresSalor/TextRe-Identification external/tri_repo
fi
git -C external/tri_repo checkout -q a62b3fe   # version checked in 2026-11
echo "✓ external/tri_repo @a62b3fe"

# 4) Bio_ClinicalBERT classifier, downloaded once into cache/
if [ "${SKIP_CLASSIFIER:-0}" != "1" ]; then
python - <<'PY'
from transformers import AutoModel, AutoTokenizer
p = "cache/bio_clinical_bert"
AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").save_pretrained(p)
AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").save_pretrained(p)
print("✓ cache/bio_clinical_bert")
PY
fi
