# Phase 1: setup notes and audit record

*Recorded 14 Nov 2026. Every experiment output also writes its own `manifest.json`, which records the package versions, the AutoND commit and whether its checkout is unmodified, a SHA-256 of this project's source, the hardware, and the full configuration.*

## Baseline code

| Item | Value |
|---|---|
| Repository | https://github.com/Crypto-TII/AutoND |
| Pinned commit | `8d66e0a43a9e6fbbb3624d68296c1b77aa2da779` (13 Mar 2025, the latest at audit time) |
| Location | `third_party/AutoND` (unmodified; `scripts/setup_autond.sh` reproduces it) |
| Licence | **No LICENSE file in the repository.** Without one, default copyright applies. We therefore keep AutoND as an unmodified external checkout, import it at runtime, and do not redistribute modified copies. Before publishing any code that includes AutoND files, ask the authors about licensing. |
| Paper | Bellini, Gérault, Hambitzer, Rossi, IACR ToSC 2023(3):184–212, DOI 10.46586/tosc.v2023.i3.184-212 (ePrint 2022/1467) |

## Environment (this laptop)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26300 |
| CPU / RAM | Intel Core Ultra 9 185H (16 cores / 22 threads), 15.4 GB |
| GPU | Intel Arc (integrated). **No NVIDIA GPU.** |
| Python | 3.11.9 (official, signed CPython from the Python install manager) in `.venv` |
| Packages | numpy 1.26.4, scipy 1.17.1, pandas 3.0.6, matplotlib 3.11.2, pytest 9.2.0, tensorflow 2.15.1 (installed but cannot load) |

### Setup blockers found

1. **Python version.** TensorFlow 2.15 has wheels only for Python ≤ 3.11, and the machine default was 3.12. AutoND warns that its code is incompatible with TF 2.16. **Fix:** a 3.11 virtual environment.
2. **Windows Smart App Control / Application Control** blocks unsigned native DLLs:
   - The uv-downloaded CPython 3.11 failed on `_ctypes`. **Fix:** use the PSF-signed CPython from `py install 3.11`.
   - TensorFlow's own DLLs (`_pywrap_*`, `_dtypes`) are blocked too. **TensorFlow cannot run on this machine** while that policy is enabled. We did not try to get around the policy. Note that turning Smart App Control off on Windows 11 generally cannot be undone without resetting Windows, so that is the owner's decision.
   - **Consequence:** the neural phases (2, 3b, 5b) run on a Linux GPU machine or Colab via `scripts/setup_gpu_linux.sh`. All NumPy phases (1, 3a, 4, 5a) run locally.
3. **Colab caveat (not verified here).** Colab's current default Python is newer than 3.11, so `setup_gpu_linux.sh` creates a separate 3.11 virtual environment with `tensorflow[and-cuda]==2.15.1`. Record the actual GPU type in each run's manifest. Session limits and hardware tiers change.

### Update 14–15 Nov 2026

- **TensorFlow 2.15.1 now loads on this machine (CPU only).** From the second Phase 1 run onward the subprocess probe reports `available: True`. No security setting was changed: Application Control evidently re-evaluated the files. The original Keras pipeline (Phases 2, 3b and 5b) can therefore be smoke-tested on CPU here (`scripts/smoke_all.sh`), but substantial runs still need a GPU.
- **PyTorch 2.15 (CPU), h5py and scikit-learn also load.** The novel neural experiment E1 uses a **PyTorch port of DBitNet** ([cgsearch/torch_dbitnet.py](../cgsearch/torch_dbitnet.py)), validated against AutoND's released Keras weights: 0.5181 ± 0.0004 versus the documented 0.5182 ± 0.0004, enforced by `tests/test_torch_dbitnet.py`. The port picks up CUDA automatically when a GPU is present.
- **Sessions get interrupted.** Background jobs, including processes started detached with `Start-Process`, ended with exit code 4 whenever the session was torn down or restarted. Every long experiment is therefore **resumable**:
  - E1 keeps one JSON per finished (candidate, seed) and uses lock files.
  - E3 writes `runs_partial.jsonl`.
  - The adaptive scorer supports checkpoints.

  If a run dies, delete any `*.lock` file older than the run and relaunch the same command. Finished work is skipped.
- **AutoND checkout hygiene.** Importing AutoND used to write `__pycache__` into `third_party/AutoND`. The bridge now sets `sys.dont_write_bytecode` so `git status` stays clean. No AutoND source file has ever been modified; `autond_worktree_clean` in every manifest records this.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `DLL load failed … Application Control policy has blocked this file` | Windows Smart App Control / WDAC blocking unsigned DLLs | Use the PSF-signed CPython (`py install 3.11`). If TensorFlow is blocked, use the PyTorch port for E1, or run on Linux/Colab. **Do not disable the policy** (on Windows 11 it generally cannot be turned back on). |
| `No matching distribution found for tensorflow==2.15.1` | Python ≥ 3.12 | Create a Python 3.11 venv (`py -3.11 -m venv .venv`, or `uv venv --python 3.11` on Linux). |
| `UnicodeDecodeError: 'charmap' codec …` when scripting edits | Windows default cp1252 | Set `PYTHONUTF8=1`. |
| Eval accuracy ≈ 0.5 with falling training loss (PyTorch port, small data) | BatchNorm running statistics not converged | Already handled by `recalibrate_bn`; keep it when changing the recipe. |
| Experiment stops with exit code 4 and an empty log | Session or machine teardown | Relaunch; runs resume. Remove stale `*.lock` files first. |
| `Final test already evaluated` | The test-set lock in phase5b works as intended | Only use `--force-rerun-test` with a logged reason. |

## Implementation audit findings

| # | Finding | Impact / decision |
|---|---|---|
| 1 | **SIMON32/64 is correct.** AutoND asserts the published vector (key 1918 1110 0908 0100, plaintext 6565 6877 → c69b e9bb) on import. Our independent reference ([cgsearch/simon_ref.py](../cgsearch/simon_ref.py), written from the specification) agrees with AutoND on 300 random inputs for every round count 1–32, with 0 mismatches. | Verified. |
| 2 | `ciphers/simon3264.py: dec_one_round` uses `tmp` before assigning it and would raise `NameError`. | Never called by the pipeline; no impact. Noted only. |
| 3 | **The negative class is uniformly random bits.** In `main.make_train_data`, label-0 samples are 64 uniformly random bits, not encryptions of random plaintext pairs as the paper's wording suggests. | Kept as is for comparability. Our seeded generator reproduces the same format and distribution; see tests. |
| 4 | **Nothing can be replayed from a seed.** All randomness comes from `os.urandom`. | Added seeded generation with named streams, so train, selection and test data never overlap ([cgsearch/data.py](../cgsearch/data.py)). |
| 5 | **The demo threshold differs from the paper.** `optimizer.optimize` uses `T = 0.05`, while the paper says `T_b = 0.01`. With the demo's 10³ samples the random-data floor is ≈ 0.0126, so 0.01 would never stop; the paper's T = 0.01 pairs with t = 10⁴ (floor ≈ 0.004). | The demo config uses 0.05 with 10³ samples. The main config uses the paper's 0.01 with 10⁴. |
| 6 | **The weighted score is `Σ_r r · b̃_r`**, rounds 1..R. The paper's typeset formula lost the weight. | Implemented exactly as the code does. |
| 7 | **Rounds share samples, rounds don't.** `evaluate_multiple_differences` shares (P, K, C0) across all candidates within a round (common random numbers), and uses fresh samples for each round. The zero difference is forced to score 0. | Same cost model and sampling scheme for every policy. |
| 8 | **The released optimizer differs from the paper on seeding.** It seeds each round's population with the previous round's survivors; the paper used a random start every round, as a code comment notes. | We use the released behaviour. |
| 9a | SPECK32/64 is also verified. AutoND asserts the published vector (6574 694c → a868 42f2). Our independent reference ([cgsearch/speck_ref.py](../cgsearch/speck_ref.py)) matches AutoND on 64 random inputs at each of 1, 2, 3, 5, 7, 8, 10 and 22 rounds, and decrypt inverts encrypt. | Verified (tests/test_cipher.py). |
| 9 | **The cipher list differs from the README.** `ciphers/` also contains `simeck3264`, `simeck4896`, `simeck64128` and `chacha`, which the README doesn't list. There is no GIFT. | SIMECK has code but has not been validated by us; do not claim support without test vectors. GIFT is not present. |
| 10 | **Neural training keeps going until accuracy drops.** `train_nets` aborts when validation accuracy ≤ 0.505; validation data double as checkpoint selection; there is no separate test set. | Our wrapper trains to a fixed target round (equal budget per candidate) and keeps a frozen test stream. |
| 11 | **The released weights** in `model_weights/` include `simon3264_single-key_0x400_round11_polished.h5`, documented at 0.5182 ± 0.0004. | Used as the Phase 2 reproduction target. |

## Commands used

```bash
git clone https://github.com/Crypto-TII/AutoND third_party/AutoND   # then checkout the pinned commit
py install 3.11 && py -3.11 -m venv .venv
python -m uv pip install --python .venv/Scripts/python.exe "tensorflow==2.15.1" "numpy<2" pandas scipy matplotlib pytest
.venv/Scripts/python.exe -m pytest -q                                    # 37 passed
.venv/Scripts/python.exe experiments/phase1_verify.py --config experiments/configs/simon32_demo.json
```
