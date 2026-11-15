#!/usr/bin/env bash
# End-to-end smoke test of EVERY experiment script with tiny settings.
# Writes only to *_smoke result folders; never touches real results. ~10-20 min on CPU.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=.venv/bin/python
CFG=experiments/configs/simon32_smoke.json
run() { echo "=== $*"; "$PY" "$@" > /dev/null 2> results/logs/smoke_last.err || { echo "FAILED: $*"; tail -20 results/logs/smoke_last.err; exit 1; }; }
mkdir -p results/logs
rm -f results/phase5/simon32_smoke/neural/final_test_done.json   # smoke only: allow re-running the final stage
run experiments/phase1_verify.py --config $CFG
run experiments/phase3_score_variability.py --config $CFG
run experiments/phase4_validate_scorer.py --config $CFG
run experiments/phase5_scoring_compare.py --config $CFG
if "$PY" -c "import tensorflow" 2>/dev/null; then
  run experiments/phase2_baseline.py --config $CFG --n-eval 2000
  run experiments/phase3_neural_correlation.py --config $CFG
  run experiments/phase5_neural_eval.py --config $CFG --stage select
  run experiments/phase5_neural_eval.py --config $CFG --stage final
else
  echo "=== TensorFlow not importable: skipping phase2 / phase3b / phase5b (run on a TF 2.15 machine)"
fi
run experiments/novel_e2_detection_depth.py --t 2000 --reps 1 --max-rounds 4 --out e2_smoke
run experiments/novel_e3_search.py --seeds 1 --generations 1 --t 300 --out e3_smoke
run experiments/novel_e1_score_candidates.py --cipher speck3264 --t 2000 --max-round 7 --n-select 4 --generations 1 --out e1_smoke
run experiments/novel_e1_neural.py --cipher speck3264 train --smoke --out e1_smoke
run experiments/novel_e1_neural.py --cipher speck3264 analyse --out e1_smoke
run experiments/novel_e5_compare.py --seeds 1 --t-ref 2000 --t-fixed 500 --pool-dir e1_smoke --out e5_smoke
echo "ALL SMOKE TESTS PASSED"
