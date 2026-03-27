# ultimate-ml-competition-skill

> Add this skill to your AI agent and start finishing in the top 10 of tabular ML competitions.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Agent Skills](https://img.shields.io/badge/Standard-Agent_Skills-blueviolet.svg)](https://agentskills.io/)
[![Works with](https://img.shields.io/badge/Works_with-Cursor_|_Claude_Code_|_Codex_|_Gemini_CLI-blue.svg)](#install)

---

A focused collection of **8 production-grade skills** covering every phase of a tabular ML competition — from project structure and feature engineering to hyperparameter tuning, ensemble meta-learning, and final submission validation. Built for AI agents that follow the open [Agent Skills](https://agentskills.io/) standard.

## 🚀 Why Use This?

**Eliminate silent score destruction** — the most common causes of a bad leaderboard score (wrong prediction type, leaky OOF, stale tuned params, duplicate training processes, submission format errors) are caught and fixed automatically before you submit.

**Opinionated, competition-tested defaults:**

- Task-type routing — binary, regression, multiclass, multi-label, ranking each map to the correct framework objective and output format
- Separation of concerns enforced by layer: Config → Features → Matrices → Metrics → Trainers → Entrypoints → Tuning → Pseudo → Meta → Orchestrator
- First-principles checklist — 15 ordered steps from raw data to final ensemble, every time

**Fits any tabular competition** — XGBoost, LightGBM, CatBoost, scikit-learn, PyTorch tabular. Covers Kaggle, Zindi, DrivenData, and any platform producing `train.csv` / `test.csv` / `sample_submission.csv`.

---

## What's Included

| Skill | Purpose |
| ----- | ------- |
| [`ml-competition`](ultimate-skills/ml-competition/SKILL.md) | Router and overview — task-type guide, first-principles checklist, sub-skill routing |
| [`ml-competition-setup`](ultimate-skills/ml-competition-setup/SKILL.md) | Project structure, `RunConfig` singleton, YAML orchestrator, process management |
| [`ml-competition-features`](ultimate-skills/ml-competition-features/SKILL.md) | Feature engineering, validation strategy (GroupKFold / TimeSeriesSplit), leakage prevention |
| [`ml-competition-training`](ultimate-skills/ml-competition-training/SKILL.md) | CB/LGB/XGB/NN training, competition metric wrappers, submission output format |
| [`ml-competition-tuning`](ultimate-skills/ml-competition-tuning/SKILL.md) | Optuna hyperparameter tuning, `load_tuned_params` contract, per-model search spaces |
| [`ml-competition-advanced`](ultimate-skills/ml-competition-advanced/SKILL.md) | Pseudo-labeling, ensemble meta-learning, post-processing/calibration, experiment tracking |
| [`ml-competition-quality`](ultimate-skills/ml-competition-quality/SKILL.md) | Code quality gate — no dead code, clear contracts; 16 production bug patterns |
| [`ml-competition-pre-submit`](ultimate-skills/ml-competition-pre-submit/SKILL.md) | Pre-submission gate — leakage check, submission file validation, adversarial validation |

---

## Install

```bash
git clone https://github.com/your-username/ultimate-ml-competition-skill.git

# Global (Claude Code)
cp -r ultimate-ml-competition-skill/ultimate-skills/* ~/.claude/skills/

# Project-level
cp -r ultimate-ml-competition-skill/ultimate-skills/* .claude/skills/
```

Other agents: Cursor → `.cursor/skills/`, Codex → `.codex/skills/`, Gemini CLI → `.gemini/skills/`

---

## License

[MIT](LICENSE)
