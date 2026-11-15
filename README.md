# Confidence-guided candidate search for AutoND (SIMON32/64, reduced rounds)

This is the research code for the plan *"Improving AutoND's Candidate Search"*. It reproduces the scoring stage of the official [AutoND](https://github.com/Crypto-TII/AutoND) pipeline and compares two ways of spending encryption samples on candidate input differences:

- **AutoND's way:** the same fixed number of samples for every candidate.
- **The proposed way (confidence-guided):** spend samples only where more evidence could change the top-K shortlist.

The comparison uses the same encryption budget and the same neural-training budget for both. It is a reduced-round cryptanalysis study; see [docs/claims_not_made.md](docs/claims_not_made.md).

## Layout

```
third_party/AutoND/   official AutoND at pinned commit 8d66e0a (unmodified; scripts/setup_autond.sh)
cgsearch/             this project: seeded data, scoring policies, intervals, DBitNet wrapper, metrics
experiments/          one script per phase + configs/ (demo = small; main = paper settings, pre-registered)
tests/                37 tests: cipher vectors, data labels, equivalence with AutoND, scorer behaviour, neural plumbing
docs/                 setup notes, implementation map, pre-registration, claims not made, related work, results
results/              outputs; every folder has a manifest.json (versions, commit, seeds, hardware, config)
scripts/              setup and run helpers (CPU phases locally; GPU phases on Linux/Colab)
```

## Quick start (CPU: phases 1, 3a, 4 and 5a)

```bash
bash scripts/setup_autond.sh                      # clone AutoND at the pinned commit
py -3.11 -m venv .venv                            # Python <= 3.11 (TensorFlow 2.15 constraint)
.venv/Scripts/python -m pip install -r requirements-cpu-scoring.txt
bash scripts/run_cpu_phases.sh experiments/configs/simon32_demo.json
```

## Neural phases (2, 3b and 5b): NVIDIA GPU, Linux or Colab

```bash
bash scripts/setup_gpu_linux.sh                   # Python 3.11 venv + tensorflow[and-cuda]==2.15.1, runs the tests
bash scripts/run_gpu_phases.sh experiments/configs/simon32_main.json
.venv-gpu/bin/python experiments/phase5_neural_eval.py --config experiments/configs/simon32_main.json --stage final
```

The final test stream (`test_seed`) is used only in `--stage final`. A lock file blocks re-evaluation unless `--force-rerun-test` is passed, and that flag is recorded in the output.

## Phase status

| Phase | Script | Where it runs | Status |
|---|---|---|---|
| 0 Literature and scope | [docs/related_work_matrix.md](docs/related_work_matrix.md), [docs/claims_not_made.md](docs/claims_not_made.md) | — | done; novelty re-check due in Phase 6 |
| 1 Environment and audit | `phase1_verify.py`, [docs/setup_notes.md](docs/setup_notes.md) | CPU | **done**: all checks pass |
| 2 Baseline reproduction | `phase2_baseline.py` | GPU | written; **not yet run** (TensorFlow is blocked on the dev laptop) |
| 3a Score variability | `phase3_score_variability.py` | CPU | **done** (demo and main) |
| 3b Score vs DBitNet (H1) | `phase3_neural_correlation.py` | GPU | written; not yet run |
| 4 Adaptive scorer and validation | `cgsearch/scoring.py`, `phase4_validate_scorer.py` | CPU | **done** |
| 5a Scoring comparison | `phase5_scoring_compare.py` | CPU | **done** (demo and main) |
| 5b Neural comparison | `phase5_neural_eval.py` | GPU | written; not yet run |
| 6 Interpretation and write-up | [docs/results.md](docs/results.md) | — | interim results recorded |

The protocol is in [docs/preregistration.md](docs/preregistration.md).
