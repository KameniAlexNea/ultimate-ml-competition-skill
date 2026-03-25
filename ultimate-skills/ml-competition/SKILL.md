---
name: ml-competition
description: "Build, debug, review and improve ML competition pipelines for tabular competitions (Kaggle, Zindi, etc.) covering binary classification, multiclass, regression, multi-label, and ranking tasks. Use when: identifying competition type and setting up the metric; structuring a competition codebase from scratch; reviewing training code for correctness bugs; implementing CatBoost/LightGBM/XGBoost/NN base models; debugging metric leakage or train-LB gaps; adding pseudo-labeling; setting up Optuna hyperparameter tuning with YAML config; building ensemble (weighted blend / LogReg stacking / dynamic gating); managing OOF files; aligning early-stopping metrics with competition objective; adding auxiliary/prior data safely; engineering and selecting features; post-processing and calibrating predictions. NOT for NLP/CV competitions."
argument-hint: "Describe your task: e.g. 'review training code for bugs', 'implement pseudo-labeling for regression', 'set up Optuna for LGB multiclass', 'debug LB gap'"
---

# ML Competition Pipeline Skill

## When to Use
- Starting or structuring a new competition codebase
- Reviewing any training/tuning code before wasting GPU hours
- Adding a new model family or modifying an existing one
- Debugging train OOF vs LB submission gap
- Setting up or fixing Optuna tuning (metric alignment, best.json format)
- Implementing pseudo-labeling or ensemble meta-learners
- Safely incorporating auxiliary/prior data

---

## Task Type Decision Guide

**Identify your task type first** — it determines which parameters, objectives, and techniques apply.

| Task Type | Framework objectives | Binary-only params to remove |
|-----------|---------------------|------------------------------|
| **Binary classification** | XGB: `binary:logistic` · LGB: `binary` · CB: `CatBoostClassifier` | — (all apply) |
| **Regression** | XGB: `reg:squarederror` · LGB: `regression` · CB: `CatBoostRegressor` | `scale_pos_weight`, `is_unbalance`, `auto_class_weights: Balanced`, threshold=0.5 |
| **Multiclass** | XGB: `multi:softprob` · LGB: `multiclass` · CB: `MultiClass` loss | `scale_pos_weight`, `is_unbalance` |
| **Multi-label** | N independent binary models OR single NN with N sigmoid heads | — (each target is binary) |
| **Ranking** | XGB: `rank:pairwise` · LGB: `lambdarank` · CB: `YetiRank` | all imbalance params |

**TARGETS pattern:**
- Single target: `TARGETS = ["target"]` — dict-of-arrays still works, simplest to keep uniform
- Multi-target / multi-label: `TARGETS = ["target_a", "target_b", "target_c"]` — dict-of-arrays everywhere, one model per target per fold
- Multiclass: `TARGETS = ["label"]` — predictions are 2D `(n, n_classes)`; OOF shape changes

**CatBoost API difference by task:**
- Binary/multiclass: `approxes[0]` = raw **logits** → must apply sigmoid (binary) or softmax (multiclass)
- Regression: `approxes[0]` = raw **prediction values** → no activation needed

---

## Separation of Concerns — Non-Negotiable

Every layer owns exactly one thing. Never blur boundaries.

| Layer | Owns | Key files |
|-------|------|-----------|
| **Config** | All user-tunable knobs | `base/config.py` (RunConfig dataclass + singleton) |
| **Features** | Raw → engineered DataFrames, versioned pkl cache | `base/features.py`, `cache/features_vN.pkl` |
| **Matrices** | NumPy/DataFrame arrays per model family | `base/features.py::build_model_matrices()` |
| **Metrics** | Competition metric, framework-specific wrappers | `base/metrics.py` |
| **Trainers** | Stateless fold-loop engines, injectable params | `base/lgb_trainer.py`, `xgb_trainer.py`, `nn_trainer.py` |
| **Entrypoints** | Load → call trainer → score → save OOF | `train/cat.py`, `lgb.py`, `xgb.py`, `nn.py` |
| **Tuning** | Optuna objectives, run_study, save JSON | `tune/tune_*.py`, `tune.py` |
| **Pseudo** | Retrain on train + pseudo-labeled test | `train/pseudo.py` |
| **Meta** | Weighted blend / stacking / gating on OOF | `train/meta.py`, `train/meta_gating.py` |
| **Orchestrator** | YAML-driven sequential runner + resume logic | `train.py` |

---

## Coding Standards — Non-Negotiable

Apply to every `src/*.py` and `scripts/*.py` file. These are hard quality gates, not suggestions.

| Rule | What it means |
|------|---------------|
| **No dead code** | No unused imports, variables, parameters, or uncalled private helpers (`_name`) |
| **Clear contracts** | Every public function has explicit input/output docs; optional params state their default behavior |
| **Single responsibility** | Functions do one job — if a function loads data AND computes metrics AND logs, split it |
| **Explicit types and names** | Type hints on all signatures; use `train_df`, `class_weights` not `tmp`, `d`, `x1` |
| **Predictable data handling** | Validate required columns up front and `raise` with a specific message; never silently mutate |
| **Structured logging** | Use `logger.*`; no `print`, no commented-out debug blocks, no stale TODOs |

**Quick review gate** — before committing any Python change, verify:
1. No unused imports, variables, or parameters
2. No uncalled private helper functions
3. Function signatures match actual behavior (not aspirational behavior)
4. Error messages are specific and actionable
5. Logs are concise and useful for iteration debugging

See [coding-rules.md](./references/coding-rules.md) for anti-patterns and extended examples.

---

## Process Management — Non-Negotiable

Training scripts can run for hours. The most expensive mistake is launching a duplicate process.

### Before every training launch — pre-flight check

```bash
# 1. Is training already running?
RUNNING_PIDS=$(pgrep -f "python scripts/train.py" 2>/dev/null)
[ -n "$RUNNING_PIDS" ] && echo "⚠️  Already running — PIDs: $RUNNING_PIDS" && exit 0

# 2. Are artifacts already fresh? (< 5 min old)
[ -f "artifacts/oof.npy" ] && \
  ARTIFACT_AGE=$(( $(date +%s) - $(stat -c %Y artifacts/oof.npy) )) && \
  [ $ARTIFACT_AGE -lt 300 ] && echo "✅ Artifacts are fresh — skip retraining"
```

### Correct launch pattern

```bash
# TRAIN_PID=$! MUST be on its own line immediately after & — not on the same line
nohup uv run python scripts/train.py > train.log 2>&1 &
TRAIN_PID=$!
echo "Training started — PID: $TRAIN_PID"

# Wait loop
while kill -0 $TRAIN_PID 2>/dev/null; do sleep 60; tail -5 train.log; done
echo "✅ Done"; tail -50 train.log
```

### Hard rules

- **Never launch without the pre-flight check** — two processes writing the same `oof.npy` corrupt results silently
- **`TRAIN_PID=$!` must be on its own line** — assigning on the same compound line as `&` captures an empty string
- **Never pipe `train.py` to `head`** — `head` closes the pipe and kills the process; always redirect to a log file
- **Fast exit ≠ failure** — check artifact timestamps and CPU load before concluding anything went wrong
- **Kill before relaunch** — if code changed and an old process is running, kill it first, then relaunch

See [process-management.md](./references/process-management.md) for all four workflow patterns (pre-flight, launch/wait, kill/relaunch, diagnose fast exit).

---

## Competition Metrics — Key Rules

Two completely independent concepts — never confuse them:

| Concept | Who uses it | What it is |
|---------|------------|------------|
| **Training loss** | Optimizer | Differentiable objective: BCE, Focal, SmoothBCE, MSELoss, CrossEntropyLoss |
| **Eval metric** | Early stopping | **Exact competition formula** — must match the leaderboard precisely |

**Getting the eval metric wrong wastes all training** — early stopping fires at the wrong iteration.

### Framework injection summary

| Framework | Metric injection | Input type in callback |
|-----------|-----------------|------------------------|
| **CatBoost binary** | `eval_metric=CatBoostCompMetric()` | Raw logits → apply `sigmoid` |
| **CatBoost regression** | `eval_metric=CatBoostCompMetric()` | Raw predictions → use directly |
| **CatBoost multiclass** | `eval_metric=CatBoostCompMetric()` | K logit arrays → apply `softmax` |
| **LightGBM** | `"metric": "None"` + `feval=make_lgb_feval()` | Already-transformed probs/values |
| **XGBoost** | `"disable_default_eval_metric": 1` + `custom_metric=make_xgb_eval()` + `maximize=True` | Already-transformed probs/values |
| **NN** | Compute in epoch loop; `model.load_state_dict(best_state)` | You control activation in `forward()` |

**Single source of truth**: define `competition_score(y_true, y_pred) -> float` once in `base/metrics.py`; never inline the formula elsewhere. `competition_score` always maximizes — negate RMSE/MAE/logloss when the leaderboard is "lower is better".

See [competition-metrics.md](./references/competition-metrics.md) for full implementations of every wrapper.

---

## First-Principles Checklist (new competition)

Complete in this exact order — each depends on the previous:

1. [ ] Identify **task type** (binary/regression/multiclass/multi-label/ranking) and read the leaderboard formula carefully
2. [ ] Implement **competition metric** exactly — single function, unit-tested against sample submission scores
3. [ ] Wrap metric for every framework (CatBoost class / LGB feval / XGB custom_metric / NN monitor)
4. [ ] Identify **group column**: use `GroupKFold` when a natural integrity unit (entity ID, session ID) exists; use `StratifiedKFold` for i.i.d. data with class imbalance; use `KFold` only when rows are truly independent and balanced; use `TimeSeriesSplit` for temporal competitions.
5. [ ] Build **feature pipeline** with versioned cache; bump version on any change
6. [ ] Build **model matrices** — single `build_model_matrices()`, process-level cache
7. [ ] Create **RunConfig dataclass** singleton; `init_config(yaml_dict)` at startup
8. [ ] Write **base model entrypoints** with `tune_dir=None` param
9. [ ] Set up **load_tuned_params** returning full JSON dict (see [hyperparameter-tuning.md](./references/hyperparameter-tuning.md))
10. [ ] Set up **Optuna** with identical folds/metrics to training
11. [ ] Write **OOF save/load** with versioned tags
12. [ ] Add **pseudo-labeling** only after base models converge
13. [ ] Add **meta ensemble** after pseudo converges
14. [ ] Wire everything in **YAML orchestrator** with resume-by-existence logic
15. [ ] Add **calibration step** as an optional post-meta pass — but verify on OOF first. For ensembles of 4+ models the ensemble already softens extremes and calibration often hurts (e.g. −0.006 OOF observed on a 4-model binary blend). Keep `enabled: false` by default; enable only when OOF gain is confirmed. See [submission-postprocessing.md](./references/submission-postprocessing.md).

---

## Critical Rules (hard constraints)

### Training loss vs eval metric — two separate things
- **Training loss** (what the optimizer minimizes): tree models use their built-in objective; NNs use `FocalLoss`, `SmoothBCE`, `MSELoss`, or `BCELoss` depending on task type
- **Eval metric** (what drives early stopping): must be the **exact competition formula** for every model
- These are completely independent — getting the eval metric wrong wastes all your training

See [competition-metrics.md](./references/competition-metrics.md) for the generic `competition_score` pattern and correct per-framework implementation.

**Critical API difference (task-type dependent):**
- CatBoost **classification**: `approxes[0]` = raw logits → must apply sigmoid (binary) or softmax (multiclass)
- CatBoost **regression**: `approxes[0]` = raw prediction → no activation, use directly
- LGB and XGB: always pass already-transformed outputs (probabilities for classification, raw values for regression) → no activation needed

### Validation hygiene
- `GroupKFold` when a natural integrity unit exists (user ID, entity ID, session ID)
- `StratifiedKFold` for i.i.d. rows with class imbalance (no group column)
- `KFold` only when rows are truly independent and the target is balanced
- For time-series: use `TimeSeriesSplit` or a rolling-window split — never GroupKFold when temporal ordering matters
- OOF arrays are **train-rows only** — even if combined data drives fold splits
- Tuner folds can be 3-fold for speed; final training uses 5-fold × 5 seeds
- Target encoding and any statistics must be computed fold-by-fold inside cross-validation

### load_tuned_params contract
```python
# common.py — always return the FULL json dict
def load_tuned_params(model_name, tune_dir):
    data = json.load(open(path))
    return data          # {"value": 0.959, "params": {...}, "trial": 48, ...}

# caller — extract params separately; log value
tuned = load_tuned_params("cat", tune_dir)
hp = tuned.get("params", {})
params[t].update(hp)
logger.info(f"loaded (value={tuned.get('value', '?')})")
# BUG if you do: params[t].update(tuned)  ← merges "value", "trial" etc. into params
```

### Prior / auxiliary data
- CatBoost handles heterogeneous cardinality and distribution shift better than other frameworks → auxiliary data often helps
- LGB / XGB / NN **may overfit severely** when auxiliary data comes from a different distribution than test; always verify OOF vs LB gap before committing
- When using combined splits: fold on combined, but score OOF on `va_idx[va_idx < n_train]` only
- **Never add two changes at once** — add auxiliary data alone, submit, then decide

---

## Reference Files

| Topic | File |
|-------|------|
| Project structure, config singleton, YAML orchestrator | [project-structure.md](./references/project-structure.md) |
| Base model setup — CB/LGB/XGB/NN params, training objective vs eval metric | [model-training.md](./references/model-training.md) |
| Training losses (FocalLoss/SmoothBCE/MSE) + `competition_score` wrappers for all frameworks | [competition-metrics.md](./references/competition-metrics.md) |
| Optuna tuning — setup, saving/loading best.json | [hyperparameter-tuning.md](./references/hyperparameter-tuning.md) |
| GroupKFold / TimeSeriesSplit, OOF management, leakage prevention | [validation-strategy.md](./references/validation-strategy.md) |
| Feature engineering patterns — encoding, aggregations, selection, cache discipline | [feature-engineering.md](./references/feature-engineering.md) |
| Ensemble: blending, LogReg stacking, dynamic gating | [ensemble-meta.md](./references/ensemble-meta.md) |
| Pseudo-labeling: when, how, weight, pitfalls | [pseudo-labeling.md](./references/pseudo-labeling.md) |
| Prediction post-processing: calibration, clipping, domain constraints | [submission-postprocessing.md](./references/submission-postprocessing.md) |
| Experiment tracking: logging scores, deciding what to submit | [experiment-tracking.md](./references/experiment-tracking.md) |
| Common bugs from production (never repeat these) | [common-pitfalls.md](./references/common-pitfalls.md) |
| Python coding standards — dead code, contracts, naming, logging | [coding-rules.md](./references/coding-rules.md) |
| Process management — pre-flight checks, PID tracking, launch/kill patterns | [process-management.md](./references/process-management.md) |
