#!/usr/bin/env bash
set -x
export PMLDL_QUICK=1
cd /home/claude/proj
rm -f results/*.json
run () {
  d=$(dirname "$1"); f=$(basename "$1")
  (cd "$d" && python3 -m nbconvert --to notebook --execute --inplace "$f" --ExecutePreprocessor.timeout=3600)
  echo "DONE $1 rc=$?"
}
run baselines/01_simple_baselines.ipynb
run baselines/02_strong_baseline_61st.ipynb
run baselines/03_strong_baseline_100th.ipynb
run proposed_model/proposed_hybrid_model.ipynb
run model_improvements/improved_ensemble_optuna.ipynb
echo "ALL FINISHED"
