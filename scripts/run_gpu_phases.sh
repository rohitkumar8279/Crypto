#!/usr/bin/env bash
# Neural phases (2, 3b, 5b). Run after scripts/setup_gpu_linux.sh, on a GPU.
# Usage: bash scripts/run_gpu_phases.sh experiments/configs/simon32_demo.json
# The CPU phases (1, 3a, 4, 5a) must have produced results/ for the same config first
# (copy the results/ folder over, or re-run them here -- they are seeded and reproducible).
set -euo pipefail
CFG="${1:-experiments/configs/simon32_demo.json}"
PY=.venv-gpu/bin/python
$PY experiments/phase2_baseline.py --config "$CFG"
$PY experiments/phase3_neural_correlation.py --config "$CFG"
$PY experiments/phase5_neural_eval.py --config "$CFG" --stage select
echo "Selection done. Review results/phase5/<name>/neural/selection.json, then run:"
echo "  $PY experiments/phase5_neural_eval.py --config $CFG --stage final   # evaluates the frozen test set ONCE"
