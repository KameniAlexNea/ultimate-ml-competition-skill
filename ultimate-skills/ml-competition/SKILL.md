---
name: ml-competition
description: "Router and overview for tabular ML competition pipelines (Kaggle, Zindi, etc.). Use when: identifying competition task type (binary/regression/multiclass/multi-label/ranking); getting the first-principles checklist for a new competition; deciding which sub-skill to load next; understanding the full pipeline at a glance. For deeper work load the focused sub-skill: ml-competition-setup (project structure, process management), ml-competition-features (feature engineering, validation), ml-competition-training (models, metrics, output format), ml-competition-tuning (Optuna), ml-competition-advanced (pseudo-labeling, ensemble, post-processing, tracking), ml-competition-quality (coding rules, pitfalls). NOT for NLP/CV competitions."
argument-hint: "Describe your task or competition type to be routed to the right sub-skill, e.g. 'starting a new binary classification competition', 'not sure where to start'"
license: MIT
metadata:
    skill-author: eak
---

# ML Competition Pipeline — Router

## Overview

This skill is the entry point and router for tabular ML competition pipelines (Kaggle, Zindi, and equivalents). Load it first to identify your task type and find the right sub-skill. For deeper work, load the focused sub-skill directly — each is self-contained with its own reference files.

**Critical principle: every layer of the pipeline owns exactly one thing.** Config drives knobs. Trainers are stateless fold engines. Metrics are defined once and reused everywhere. Violating these boundaries is the single most common source of silent bugs.

This skill is **not** for NLP or CV competitions.

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

## First-Principles Checklist (new competition)

Complete in this exact order — each step links to the sub-skill that covers it:

1. [ ] Identify **task type** (binary/regression/multiclass/multi-label/ranking) and read the leaderboard formula carefully — *(this file)*
2. [ ] Implement **competition metric** exactly — single function, unit-tested → [ml-competition-training](../ml-competition-training/SKILL.md)
3. [ ] Wrap metric for every framework (CB / LGB / XGB / NN) → [ml-competition-training](../ml-competition-training/SKILL.md)
4. [ ] Choose **CV split** (GroupKFold / StratifiedKFold / TimeSeriesSplit / KFold) → [ml-competition-features](../ml-competition-features/SKILL.md)
5. [ ] Build **feature pipeline** with versioned cache; bump version on any change → [ml-competition-features](../ml-competition-features/SKILL.md)
6. [ ] Build **model matrices** — single `build_model_matrices()`, process-level cache → [ml-competition-features](../ml-competition-features/SKILL.md)
7. [ ] Create **RunConfig dataclass** singleton; scaffold package layout → [ml-competition-setup](../ml-competition-setup/SKILL.md)
8. [ ] Write **base model entrypoints** with `tune_dir=None` param → [ml-competition-training](../ml-competition-training/SKILL.md)
9. [ ] Set up **`load_tuned_params`** returning full JSON dict → [ml-competition-tuning](../ml-competition-tuning/SKILL.md)
10. [ ] Set up **Optuna** with identical folds/metrics to training → [ml-competition-tuning](../ml-competition-tuning/SKILL.md)
11. [ ] Write **OOF save/load** with versioned tags → [ml-competition-advanced](../ml-competition-advanced/SKILL.md)
12. [ ] Add **pseudo-labeling** only after base models converge → [ml-competition-advanced](../ml-competition-advanced/SKILL.md)
13. [ ] Add **meta ensemble** after pseudo converges → [ml-competition-advanced](../ml-competition-advanced/SKILL.md)
14. [ ] Wire everything in **YAML orchestrator** with resume-by-existence logic → [ml-competition-setup](../ml-competition-setup/SKILL.md)
15. [ ] Add **calibration** as optional post-meta pass — verify OOF gain, keep disabled by default → [ml-competition-advanced](../ml-competition-advanced/SKILL.md)

---

## Sub-Skills

| Sub-Skill | When to load it |
|-----------|----------------|
| [ml-competition-setup](../ml-competition-setup/SKILL.md) | Scaffolding project, RunConfig, YAML orchestrator, process management (pre-flight, PID tracking) |
| [ml-competition-features](../ml-competition-features/SKILL.md) | Feature engineering, validation strategy, OOF leakage, cache discipline |
| [ml-competition-training](../ml-competition-training/SKILL.md) | Model training (CB/LGB/XGB/NN), competition metric wrappers, submission output format |
| [ml-competition-tuning](../ml-competition-tuning/SKILL.md) | Optuna hyperparameter tuning, load_tuned_params contract, search spaces |
| [ml-competition-advanced](../ml-competition-advanced/SKILL.md) | Pseudo-labeling, ensemble meta-learning, post-processing/calibration, experiment tracking |
| [ml-competition-quality](../ml-competition-quality/SKILL.md) | Code review, 6-rule quality gate, 16 production bug patterns |
