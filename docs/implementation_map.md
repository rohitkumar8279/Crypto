# Implementation map

## AutoND (unmodified, `third_party/AutoND`)

| File | Role | Used by this project |
|---|---|---|
| `ciphers/simon3264.py` | SIMON32/64 on bit matrices: `encrypt(p, k, r)`, `plain_bits=32`, `key_bits=64`, `word_size=16` | yes: every encryption |
| `optimizer.py` | `evaluate_multiple_differences` (bias score), `evo` (evolutionary search), `optimize` (round loop, re-scoring, weighted score), `empirical_threshold_estimation` | `evo` and `evaluate_multiple_differences` are called directly; `optimize`'s loop is mirrored with seeding |
| `main.py` | `make_train_data` (pair data), `findGoodInputDifferences`, `trainNeuralDistinguishers` | format replicated in `cgsearch/data.py` (main.py imports TensorFlow at module level) |
| `dbitnet.py` | `make_model(input_size)`: the DBitNet architecture | yes, unchanged |
| `train_nets.py` | simple training pipeline: Adam with AMSGrad, MSE, batch 5000, best-val-loss checkpoint, abort at ≤ 0.505 | recipe replicated in `cgsearch/neural.py` |
| `eval_nets.py` | 5 fresh evaluation sets of 10⁶ samples each | replicated by `evaluate_saved_weights` |
| `model_weights/*.h5` | released polished models | Phase 2 reproduction |

## Data flow

```
Phase 1  candidates.build_candidate_pool ──► results/phase1/<cfg>/candidate_pool.json  (frozen pool, rounds 1..R)
            uses AutoND evo + evaluate_multiple_differences, seeded
Phase 3a phase3_score_variability ──► reference_scores.json (t_reference, independent stream) + variability stats
Phase 4  phase4_validate_scorer   ──► interval coverage, allocation checks
Phase 5a phase5_scoring_compare   ──► runs.csv, summary.csv, shortlists.json (A, B, C, C2 × seeds × budgets)
Phase 3b phase3_neural_correlation (GPU) ──► score vs DBitNet selection accuracy (H1)
Phase 5b phase5_neural_eval --stage select (GPU) ──► selection.json (finalist per policy; selection data only)
         phase5_neural_eval --stage final  (GPU) ──► final_test.csv  (frozen test stream, evaluated once, locked)
```

## This project (`cgsearch/`)

| Module | Contents |
|---|---|
| `autond_bridge.py` | path setup, cipher/optimizer/DBitNet loaders, pinned-commit check |
| `simon_ref.py` | independent integer reference of SIMON32/64 (spec), with decrypt |
| `data.py` | bit conversions; seeded `make_pairs_dataset` (AutoND format); `seed_tree` named RNG streams |
| `scoring.py` | `Ledger`; `CipherSampler` (common random numbers) and `SyntheticSampler` (known ground truth); bias score; Poisson bootstrap; intervals (`bias_bounded`, `mcdiarmid`, `basic`, `percentile`); policies `score_fixed` (A/C), `AdaptiveScorer` (B, checkpoint/resume), `score_successive_halving` (C2) |
| `candidates.py` | seeded pool generation with AutoND's `evo` |
| `neural.py` | seeded DBitNet training to a fixed target round, separate selection and test streams, resumable runs, evaluation of released weights |
| `metrics.py` | advantage, Wilson CI, t-interval over seeds, Spearman/Kendall, top-K overlap, shortlist regret |
| `manifest.py` | reproducibility manifest |
| `speck_ref.py` | independent integer reference of SPECK32/64 (spec), with decrypt |
| `spectral.py` | **novel:** low-degree Walsh correlations (Gram matrix), unbiased capacities C₁ / C₂ / C_{≤2}, χ² statistics, certified accuracy floor, delta-method SE (`cap12_and_se`) |
| `spectral_search.py` | **novel:** drop-in spectral fitness for AutoND's `evo`, χ² fresh-sample stopping (`degree` 1 or 2), deepest-round re-scoring, rotation orbits |
| `torch_dbitnet.py` | PyTorch port of DBitNet + Keras `.h5` weight loader + AutoND training recipe (AMSGrad, MSE, best val-loss) + BatchNorm recalibration; CUDA if available |

## Novel experiments (`experiments/novel_*.py`)

| Script | Experiment | Hardware |
|---|---|---|
| `novel_e2_detection_depth.py` | E2: detection depth, bias vs degree-1 χ² vs degree-≤2 χ² | CPU |
| `novel_e3_search.py` | E3: AutoND search vs spectral search vs degree-1 ablation (resumable) | CPU |
| `novel_e1_score_candidates.py` | E1 part 1: pool + proxy scores + pre-declared candidate selection | CPU |
| `novel_e1_neural.py` | E1 part 2: DBitNet (PyTorch) training, frozen test, pre-registered analysis (resumable, lock files) | CPU (hours) or GPU |
| `novel_e5_compare.py` | E5: allocation policies under both statistics | CPU |
| `make_figures.py` | all figures, from CSV/JSON only | CPU |

## Where the research contribution lives

Only in the **candidate scoring and selection layer**:
- `AdaptiveScorer` and its ablations (the original plan);
- the spectral score, χ² test, certified floor and spectral search (the novel extension).

The cipher implementations, data format, bias-score formula, evolutionary optimizer and DBitNet architecture are AutoND's, used unmodified. The PyTorch port reimplements the architecture and is validated against AutoND's weights.
