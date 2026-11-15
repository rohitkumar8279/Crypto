#!/usr/bin/env bash
# Linux GPU machine or Google Colab: Python 3.11 venv with TensorFlow 2.15 + CUDA wheels.
# (Colab's default Python is newer than 3.11, which TF 2.15 does not support, hence the venv.)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
pip install --quiet uv
uv venv --python 3.11 .venv-gpu
uv pip install --python .venv-gpu/bin/python "tensorflow[and-cuda]==2.15.1" "numpy<2" scipy pandas matplotlib pytest
bash scripts/setup_autond.sh
.venv-gpu/bin/python -c "import tensorflow as tf; print('TF', tf.__version__, tf.config.list_physical_devices('GPU'))"
.venv-gpu/bin/python -m pytest -q
