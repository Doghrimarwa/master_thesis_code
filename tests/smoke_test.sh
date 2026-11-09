#!/usr/bin/env bash
# End-to-end test of the whole pipeline on SYNTHETIC notes, on CPU, in minutes.
#
# Prerequisite (once):  pip install -r requirements.txt
#                       SKIP_CLASSIFIER=1 bash scripts/setup_external.sh
#                       python -c "import wn; wn.download('oewn:2022')"
#
# Real code everywhere except three stand-ins (results are meaningless):
#   * tiny random models instead of RoBERTa / DistilBERT / GPT-2 / Qwen (tests/make_tiny_models.py)
#   * a TF-IDF classifier instead of FARM/Bio_ClinicalBERT (tests/stubs/doc_classification_stub.py)
#   * an en_core_web_lg stub for tri.py (not used with the thesis TRI configuration)
set -euo pipefail
cd "$(dirname "$0")/.."
W=$PWD/tests/_work
rm -rf "$W"
python scripts/make_toy_data.py --out "$W" --n 60
python tests/make_tiny_models.py --texts "$W/data/mp_50pct.csv" --out "$W/models"

export BASE="$W"
export HF_roberta_base="$W/models/tiny-roberta" HF_roberta_clinical="$W/models/tiny-roberta"
export EPS="25 100"
export DPO_DOC_CLASSIFICATION="$PWD/tests/stubs/doc_classification_stub.py"
export PYTHONPATH="$PWD/tests/stubs${PYTHONPATH:+:$PYTHONPATH}"
export UTILITY_ARGS="--gpt2_model $W/models/tiny-gpt2 --embedding_path $W/models/tiny-roberta --max_samples 20"
export TRI1_ARGS="--n_real 10 --n_distractors 30"
export TRI2_ARGS="--qwen_model $W/models/tiny-gpt2"
export TRI4_ARGS="--base_model $W/models/tiny-distilbert --finetuning_epochs 1"

for TASK in mp proc; do
  for PHASE in prepare rewrite classify utility privacy; do
    echo; echo "################ $TASK / $PHASE ################"
    bash scripts/run_all.sh $TASK $PHASE
  done
done
for TASK in mp proc; do bash scripts/run_all.sh $TASK report; done

python - <<'PY'
import os, pandas as pd
W = os.environ["BASE"]
need = ["results/classification_mp_averaged.csv", "results/classification_proc_averaged.csv",
        "results/utility_metrics_mp.csv", "results/utility_metrics_proc.csv",
        "results/trir_averaged_results.csv",
        "results/gamma_tradeoff_mp.csv", "results/gamma_tradeoff_proc.csv"]
for f in need:
    df = pd.read_csv(os.path.join(W, f))
    assert len(df) > 0, f
    print(f"  ✓ {f}: {len(df)} rows")
figs = os.listdir(os.path.join(W, "results/figures"))
assert any(f.startswith("gamma_heatmap") for f in figs), figs
print(f"  ✓ {len(figs)} figures in results/figures")
PY
echo; echo "PIPELINE SMOKE TEST PASSED"

echo; echo "################ step 6: chunking + checkpoint/resume ################"
python - <<'PY'
import os, subprocess, sys, pandas as pd
W = os.environ["BASE"]; d = os.path.join(W, "step6_extra"); os.makedirs(d, exist_ok=True)
long_sent = " ".join(["patient reports chest pain and shortness of breath"] * 60)   # > 200 tokens
rows = [{"note_id": i, "sent_idx": 0, "dpmlm_input_sentence": s}
        for i, s in enumerate([long_sent] + ["she works as a lighthouse keeper on an island"] * 5)]
pd.DataFrame(rows).to_csv(f"{d}/in.csv", index=False)
bounds = os.path.join(W, "outputs_mp/step5_clipping/roberta_base/step5_clipping_bounds_sigma_2.0.json")
out = f"{d}/out"
# simulate an interrupted run: a checkpoint holding the first 3 rows
pd.DataFrame(rows[:3]).assign(dpmlm_sentence="DONE BEFORE", perturbed=0, total=0, epsilon=25.0,
    C_min=0, C_max=1, perturbation_rate=0).to_csv(f"{d}/ckpt.csv", index=False)
os.makedirs(out, exist_ok=True)
os.replace(f"{d}/ckpt.csv", f"{out}/checkpoint_sigma_test_eps_25.0.csv")
cmd = [sys.executable, "pipeline/step6_apply_dpmlm.py", f"{d}/in.csv", bounds, out, "--eps", "25",
       "--sigma", "test", "--model", os.environ["HF_roberta_base"], "--checkpoint_every", "2"]
log = subprocess.run(cmd, capture_output=True, text=True).stdout
res = pd.read_csv(f"{out}/df_rare_dpmlm_sigma_test_eps_25.0.csv")
assert len(res) == 6, len(res)
assert (res["dpmlm_sentence"].iloc[:3] == "DONE BEFORE").all(), "resume did not keep checkpoint rows"
assert "Already done     : 3" in log, log[-2000:]
assert not os.path.exists(f"{out}/checkpoint_sigma_test_eps_25.0.csv"), "checkpoint not removed"
chunk_run = subprocess.run(cmd[:4] + [f"{d}/out2"] + cmd[5:], capture_output=True, text=True).stdout
assert "Sentences chunked    : 1" in chunk_run, chunk_run[-2000:]
print("  ✓ resume from checkpoint, ✓ long sentence chunked")
PY
echo "STEP 6 CHECKS PASSED"
