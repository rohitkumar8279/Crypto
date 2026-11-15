#!/usr/bin/env bash
# CPU-only phases: 1 (verify + pool), 3a (score variability + reference), 4 (scorer validation), 5a (scoring comparison).
set -euo pipefail
CFG="${1:-experiments/configs/simon32_demo.json}"
PY="${PY:-.venv/Scripts/python.exe}"; [ -x "$PY" ] || PY=.venv/bin/python
$PY -m pytest -q
$PY experiments/phase1_verify.py --config "$CFG"
$PY experiments/phase3_score_variability.py --config "$CFG"
$PY experiments/phase4_validate_scorer.py --config "$CFG"
$PY experiments/phase5_scoring_compare.py --config "$CFG"
