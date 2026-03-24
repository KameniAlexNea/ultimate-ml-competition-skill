# ultimate-ml-competition-skill

> **Add this skill to your favorite AI agent and start finishing in the top 10 of tabular ML competitions.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Skills](https://img.shields.io/badge/Skills-5-brightgreen.svg)](#whats-included)
[![References](https://img.shields.io/badge/References-12-orange.svg)](#whats-included)
[![Agent Skills](https://img.shields.io/badge/Standard-Agent_Skills-blueviolet.svg)](https://agentskills.io/)
[![Works with](https://img.shields.io/badge/Works_with-Cursor_|_Claude_Code_|_Codex_|_Gemini_CLI-blue.svg)](#getting-started)

A focused collection of **5 production-grade skills** covering every phase of a tabular ML competition — from project structure and feature engineering to hyperparameter tuning, ensemble meta-learning, and final submission validation. Built for AI agents that follow the open [Agent Skills](https://agentskills.io/) standard.

---

## 📦 What's Included

### Skills (5)

| Skill | Purpose |
|-------|---------|
| [`ml-competition`](ultimate-skills/ml-competition/SKILL.md) | End-to-end competition pipeline — structure, training, tuning, pseudo-labeling, ensembling |
| [`coding-rules`](ultimate-skills/coding-rules/SKILL.md) | Hard quality gate: no dead code, clear contracts, single responsibility, explicit types |
| [`metrics`](ultimate-skills/metrics/SKILL.md) | Correct prediction type per metric — prevents submitting labels when probabilities are expected |
| [`process-management`](ultimate-skills/process-management/SKILL.md) | Safe background training — no duplicate processes, correct PID capture, artifact freshness checks |
| [`validation`](ultimate-skills/validation/SKILL.md) | Pre-submission gate — leakage check, submission file validation, adversarial validation |

Each skill includes a `SKILL.md` with decision tables, code templates, anti-patterns, and step-by-step workflows your agent follows automatically.

### Reference Docs (12)

Curated references loaded by the `ml-competition` and `validation` skills:

**`ml-competition/references/`**
- `common-pitfalls.md` — recurring mistakes and how to avoid them
- `competition-metrics.md` — metric catalogue with sklearn call reference
- `ensemble-meta.md` — stacking, blending, and rank-average recipes
- `experiment-tracking.md` — OOF score logging, run naming conventions
- `feature-engineering.md` — tabular feature recipes per data type
- `hyperparameter-tuning.md` — Optuna integration patterns
- `model-training.md` — XGBoost / LightGBM / CatBoost trainer templates
- `project-structure.md` — canonical folder layout and module responsibilities
- `pseudo-labeling.md` — thresholding, confidence filtering, retraining loop
- `submission-postprocessing.md` — clipping, rank transformation, calibration
- `validation-strategy.md` — KFold / StratifiedKFold / GroupKFold / TimeSeries split selection

**`validation/references/`**
- `checklist.md` — programmatic pre-submission checks

### Scripts (1)

- `validation/scripts/adversarial_validation.py` — detects train/test distribution shift; AUC interpretation guide built in

---

## 🚀 Why Use This?

### ⚡ Eliminate Silent Score Destruction
The most common causes of a bad leaderboard score — wrong prediction type, leaky OOF, stale tuned params, duplicate training processes, or submission format errors — are all caught and fixed automatically by these skills before you submit.

### 🎯 Opinionated, Competition-Tested Defaults
- **Task-type routing** — binary, regression, multiclass, multi-label, ranking each map to the correct framework objective and output format
- **Separation of concerns** enforced by layer: Config → Features → Matrices → Metrics → Trainers → Entrypoints → Tuning → Pseudo → Meta → Orchestrator
- **First-principles checklist** — 15 ordered steps from raw data to final ensemble, every time

### 🔧 Fits Any Tabular Competition
Works with XGBoost, LightGBM, CatBoost, scikit-learn, and PyTorch tabular models. Covers Kaggle, Zindi, DrivenData, and any platform that produces `train.csv` / `test.csv` / `sample_submission.csv`.

---

## 🎯 Getting Started

These skills follow the open [Agent Skills](https://agentskills.io/) standard. Copy the skill folders to your agent's skills directory and your agent will automatically discover and apply them.

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ultimate-ml-competition-skill.git
```

### Step 2: Copy Skills to Your Skills Directory

**Global installation** (skills available across all projects):

| Tool | Directory |
|------|-----------|
| Cursor | `~/.cursor/skills/` |
| Claude Code | `~/.claude/skills/` |
| Codex | `~/.codex/skills/` |
| Gemini CLI | `~/.gemini/skills/` |

**Project-level installation** (skills scoped to one competition repo):

| Tool | Directory |
|------|-----------|
| Cursor | `.cursor/skills/` (in your project root) |
| Claude Code | `.claude/skills/` (in your project root) |
| Codex | `.codex/skills/` (in your project root) |
| Gemini CLI | `.gemini/skills/` (in your project root) |

**Example — global install for Claude Code:**
```bash
cp -r ultimate-ml-competition-skill/ultimate-skills/* ~/.claude/skills/
```

**Example — project-level install for Cursor:**
```bash
mkdir -p .cursor/skills
cp -r ultimate-ml-competition-skill/ultimate-skills/* .cursor/skills/
```

**That's it.** Your agent will auto-discover the skills and invoke them when relevant. You can also invoke a skill manually by mentioning its name in your prompt (e.g., *"use the ml-competition skill to scaffold this pipeline"*).

---

## 📋 Table of Contents

- [What's Included](#whats-included)
- [Why Use This?](#-why-use-this)
- [Getting Started](#-getting-started)
- [Skills Overview](#-skills-overview)
- [Prerequisites](#-prerequisites)
- [License](#-license)

---

## 🔍 Skills Overview

### `ml-competition`
The core skill. Covers the full lifecycle of a tabular competition:

- **Task-type decision table** — maps task type to framework objective for XGBoost, LightGBM, and CatBoost
- **15-step first-principles checklist** — from identifying the task to calibrating the final ensemble
- **Auxiliary data rules** — when and how to incorporate external datasets safely
- **OOF save/load pattern** — reproducible out-of-fold predictions across runs

### `coding-rules`
A non-negotiable quality gate applied to all `src/*.py` and `scripts/*.py` files:

- No dead code (unused imports, variables, uncalled helpers)
- Every public function has an explicit input/output contract and docstring
- Type hints and descriptive names throughout
- Structured logs, actionable errors — no debug `print` statements

### `metrics`
Prevents the #1 source of silent score destruction: **submitting labels when the metric expects probabilities**.

- Metric → prediction type mapping table
- sklearn call reference for every major competition metric
- Submission format templates per task type
- 6-point submission format scout checklist

### `process-management`
Prevents ghost training processes and wasted GPU hours:

- Pre-flight check before every training run
- Correct PID capture pattern (common shell bug avoided)
- Kill-and-relaunch workflow after code changes
- Artifact freshness check before diagnosing failure

### `validation`
Three mandatory checks before any result is reported:

1. **Code review** — data leakage, metric correctness, scaler fit discipline
2. **Submission file validation** — column names, row count, NaN check, value range against `sample_submission.csv`
3. **Adversarial validation** — AUC-based train/test shift detection with built-in severity interpretation

---

## ⚙️ Prerequisites

- **Python**: 3.9+ (3.12+ recommended)
- **Client**: Any agent supporting the [Agent Skills](https://agentskills.io/) standard (Cursor, Claude Code, Gemini CLI, Codex, etc.)
- **Packages**: `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `optuna`, `pandas`, `numpy` — installed per competition as needed

---

## 📄 License

[MIT](LICENSE)
