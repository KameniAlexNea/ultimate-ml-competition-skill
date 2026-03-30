# ultimate-ml-competition-skill

> A curated collection of skills and specialized agents for building top-tier ML competition pipelines.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Agent Skills](https://img.shields.io/badge/Standard-Agent_Skills-blueviolet.svg)](https://agentskills.io/)
[![Works with](https://img.shields.io/badge/Works_with-Cursor_|_Claude_Code_|_Codex_|_Gemini_CLI-blue.svg)](#install)

---

A production-grade collection of **43 skills** and **11 specialized agents** covering every phase of a tabular ML competition pipeline — from research and data profiling to training, tuning, ensemble, and final submission. Built for AI agents following the open [Agent Skills](https://agentskills.io/) standard.

The pipeline is organized in two layers:

- **Skills** — on-demand domain knowledge that agents load when they need it
- **Agents** — specialized workers with clear scope boundaries, each owning one phase of the competition

---

## Why Use This?

**Eliminate silent score destruction.** The most common causes of a bad leaderboard score — wrong prediction type, leaky OOF, stale tuned params, temporal leakage in time series, running training twice in parallel — are caught and blocked before they reach the leaderboard.

**Research before code.** A dedicated `research-analyst` agent mines arXiv and literature for winning approaches and generates a ranked hypothesis bank before a single training line is written.

**Covers the full competition spectrum.** Standard tabular (XGB/LGB/CB), neural (PyTorch Lightning), text (Transformers), time series (TimesFM + aeon), graph (NetworkX + PyG), reinforcement learning (SB3 + PufferLib), survival analysis, and symbolic feature engineering.

**Every agent owns exactly one thing.** Hard scope boundaries prevent agents from stepping on each other's work. The `team-lead` orchestrator gates each phase on well-defined criteria before the next begins.

---

## Repository Structure

```
ultimate-skills/
├── skills/          # 43 on-demand skill files (Claude Code / Codex / VS Code)
└── agents/          # 11 specialized agent definitions (Claude Code / Codex format)

scripts/
└── generate_vscode_agents.sh   # converts agents to VS Code frontmatter format
```

> **VS Code users**: the agent files in `ultimate-skills/agents/` use Claude Code / Codex frontmatter. Run [`scripts/generate_vscode_agents.sh`](scripts/generate_vscode_agents.sh) to generate VS Code-compatible versions before installing.

---

## Skills

Skills are organized into 10 groups. Each is a self-contained `SKILL.md` that agents load on demand.

### Core ML Pipeline (8 skills)

These skills define the competition pipeline architecture and are pre-loaded by most agents.

| Skill | Purpose |
| ----- | ------- |
| [`ml-competition`](ultimate-skills/skills/ml-competition/SKILL.md) | Router and overview — task-type guide, first-principles checklist, sub-skill routing |
| [`ml-competition-setup`](ultimate-skills/skills/ml-competition-setup/SKILL.md) | Project structure, `RunConfig` singleton, YAML orchestrator, process management |
| [`ml-competition-features`](ultimate-skills/skills/ml-competition-features/SKILL.md) | Feature engineering, validation strategy (GroupKFold / TimeSeriesSplit), leakage prevention |
| [`ml-competition-training`](ultimate-skills/skills/ml-competition-training/SKILL.md) | CB/LGB/XGB/NN training, competition metric wrappers, submission output format |
| [`ml-competition-tuning`](ultimate-skills/skills/ml-competition-tuning/SKILL.md) | Optuna hyperparameter tuning, `load_tuned_params` contract, per-model search spaces |
| [`ml-competition-advanced`](ultimate-skills/skills/ml-competition-advanced/SKILL.md) | Pseudo-labeling, ensemble meta-learning, post-processing/calibration, experiment tracking |
| [`ml-competition-quality`](ultimate-skills/skills/ml-competition-quality/SKILL.md) | Code quality gate — no dead code, clear contracts; 16 production bug patterns |
| [`ml-competition-pre-submit`](ultimate-skills/skills/ml-competition-pre-submit/SKILL.md) | Pre-submission gate — leakage check, submission file validation, adversarial validation |

### Core Data Processing (4 skills)

| Skill | Purpose |
| ----- | ------- |
| [`exploratory-data-analysis`](ultimate-skills/skills/exploratory-data-analysis/SKILL.md) | Comprehensive EDA across 200+ file formats — quality reports, format detection |
| [`polars`](ultimate-skills/skills/polars/SKILL.md) | Fast in-memory DataFrames for datasets up to ~2 GB |
| [`dask`](ultimate-skills/skills/dask/SKILL.md) | Larger-than-RAM distributed DataFrame processing |
| [`vaex`](ultimate-skills/skills/vaex/SKILL.md) | Billions-of-rows out-of-core analytics |

### Core ML & Statistics (5 skills)

| Skill | Purpose |
| ----- | ------- |
| [`scikit-learn`](ultimate-skills/skills/scikit-learn/SKILL.md) | Classification, regression, clustering, pipelines, preprocessing |
| [`shap`](ultimate-skills/skills/shap/SKILL.md) | Model explainability — SHAP values, feature importance, interaction effects |
| [`statistical-analysis`](ultimate-skills/skills/statistical-analysis/SKILL.md) | Test selection, assumption checking, distribution comparison, reporting |
| [`statsmodels`](ultimate-skills/skills/statsmodels/SKILL.md) | OLS/GLM/ARIMA with full diagnostics |
| [`pymc`](ultimate-skills/skills/pymc/SKILL.md) | Bayesian modeling, hierarchical models, MCMC |

### Deep Learning (2 skills)

| Skill | Purpose |
| ----- | ------- |
| [`pytorch-lightning`](ultimate-skills/skills/pytorch-lightning/SKILL.md) | Structured PyTorch — LightningModule, Trainer, multi-GPU/TPU, callbacks |
| [`transformers`](ultimate-skills/skills/transformers/SKILL.md) | Pre-trained transformer models for NLP and mixed-modal tabular inputs |

### Visualization (5 skills)

| Skill | Purpose |
| ----- | ------- |
| [`matplotlib`](ultimate-skills/skills/matplotlib/SKILL.md) | Low-level plotting with full customization control |
| [`seaborn`](ultimate-skills/skills/seaborn/SKILL.md) | Statistical visualization — distributions, heatmaps, pairplots |
| [`plotly`](ultimate-skills/skills/plotly/SKILL.md) | Interactive charts for team dashboards and exploration |
| [`scientific-visualization`](ultimate-skills/skills/scientific-visualization/SKILL.md) | Publication-ready multi-panel figures with significance annotations |
| [`umap-learn`](ultimate-skills/skills/umap-learn/SKILL.md) | 2D/3D manifold embeddings for feature space and clustering visualization |

### Time Series (2 skills)

| Skill | Purpose |
| ----- | ------- |
| [`timesfm-forecasting`](ultimate-skills/skills/timesfm-forecasting/SKILL.md) | Zero-shot univariate forecasting with Google's TimesFM foundation model |
| [`aeon`](ultimate-skills/skills/aeon/SKILL.md) | Time series classification, regression, clustering, anomaly detection |

### Graph / Network (2 skills)

| Skill | Purpose |
| ----- | ------- |
| [`networkx`](ultimate-skills/skills/networkx/SKILL.md) | Graph construction, centrality, community detection, per-node feature extraction |
| [`torch-geometric`](ultimate-skills/skills/torch-geometric/SKILL.md) | GNNs — GCN, GAT, GraphSAGE, molecular graphs, link prediction |

### Reinforcement Learning (2 skills)

| Skill | Purpose |
| ----- | ------- |
| [`stable-baselines3`](ultimate-skills/skills/stable-baselines3/SKILL.md) | Production-ready RL — PPO, SAC, DQN, TD3, A2C |
| [`pufferlib`](ultimate-skills/skills/pufferlib/SKILL.md) | High-performance vectorized environments and multi-agent RL |

### Specialized ML (3 skills)

| Skill | Purpose |
| ----- | ------- |
| [`scikit-survival`](ultimate-skills/skills/scikit-survival/SKILL.md) | Survival analysis — Cox PH, Random Survival Forest, C-index, Brier score |
| [`pymoo`](ultimate-skills/skills/pymoo/SKILL.md) | Multi-objective optimization — NSGA-II/III, Pareto threshold tuning |
| [`sympy`](ultimate-skills/skills/sympy/SKILL.md) | Symbolic math for deriving domain-equation features |

### Critical Thinking & Research (7 skills)

| Skill | Purpose |
| ----- | ------- |
| [`scientific-critical-thinking`](ultimate-skills/skills/scientific-critical-thinking/SKILL.md) | Evaluate experimental design, identify biases and confounders |
| [`scientific-brainstorming`](ultimate-skills/skills/scientific-brainstorming/SKILL.md) | Open-ended interdisciplinary ideation |
| [`hypothesis-generation`](ultimate-skills/skills/hypothesis-generation/SKILL.md) | Structured testable hypotheses from data observations |
| [`hypogenic`](ultimate-skills/skills/hypogenic/SKILL.md) | LLM-driven automated hypothesis exploration on tabular datasets |
| [`what-if-oracle`](ultimate-skills/skills/what-if-oracle/SKILL.md) | Structured what-if scenario analysis and failure mode exploration |
| [`literature-review`](ultimate-skills/skills/literature-review/SKILL.md) | Systematic reviews across PubMed, Semantic Scholar, Google Scholar |
| [`arxiv-database`](ultimate-skills/skills/arxiv-database/SKILL.md) | Search and retrieve ML/CS preprints from arXiv |

### Compute / Infrastructure (3 skills)

| Skill | Purpose |
| ----- | ------- |
| [`modal`](ultimate-skills/skills/modal/SKILL.md) | Cloud GPU training and serverless inference |
| [`get-available-resources`](ultimate-skills/skills/get-available-resources/SKILL.md) | Detect CPU/GPU/RAM/disk before launching heavy jobs |
| [`zarr-python`](ultimate-skills/skills/zarr-python/SKILL.md) | Chunked compressed arrays for memory-efficient large feature matrices |

---

## Agents

Agents are specialized workers with strict scope boundaries. Each owns one phase of the pipeline and writes progress to a shared `EXPERIMENT_STATE.json`. The `team-lead` orchestrates the full flow.

### Orchestrator

| Agent | Role |
| ----- | ---- |
| [`team-lead`](ultimate-skills/agents/team-lead.agent.md) | Routes agents in dependency order, gates on each agent's output, owns the final submission decision |

### Execution order and worker agents

```
1. research-analyst          always — research and hypothesis bank before any code
2. infrastructure-expert     always — hardware profiling before any heavy computation
3. data-processing-expert    always — verified data contract
4. visualization-expert      always — full diagnostic figure suite
5. ml-statistics-expert      always — statistical baselines and SHAP audit
6. ─ conditional ────────────────────────────────────────────────────────────
   time-series-expert         if TIMESTAMP_FEATURES are present
   graph-ml-expert            if entity relationship columns are present
   deep-learning-expert       if text/embedding columns present or NN requested
   rl-expert                  if the competition is simulation-based
   specialized-ml-expert      if survival targets, Pareto metric, or symbolic features
7. ml-competition sub-skills  training → tuning → advanced
8. ml-competition-pre-submit  always — mandatory gate before final submission
```

| Agent | Skills used | Scope |
| ----- | ----------- | ----- |
| [`research-analyst`](ultimate-skills/agents/research-analyst.agent.md) | `scientific-critical-thinking`, `hypothesis-generation`, `scientific-brainstorming`, `literature-review`, `arxiv-database`, `hypogenic`, `what-if-oracle` | Literature mining, hypothesis bank, experiment roadmap |
| [`infrastructure-expert`](ultimate-skills/agents/infrastructure-expert.agent.md) | `get-available-resources`, `modal`, `zarr-python`, `dask`, `vaex` | Hardware profiling, OOM guard, Zarr storage, Modal cloud offload |
| [`data-processing-expert`](ultimate-skills/agents/data-processing-expert.agent.md) | `exploratory-data-analysis`, `statistical-analysis`, `polars`, `dask`, `vaex` | Data contract, EDA, leakage/drift/imbalance/missing profiling |
| [`visualization-expert`](ultimate-skills/agents/visualization-expert.agent.md) | `matplotlib`, `seaborn`, `plotly`, `scientific-visualization`, `umap-learn` | Diagnostic figures, UMAP embeddings, feature distributions |
| [`ml-statistics-expert`](ultimate-skills/agents/ml-statistics-expert.agent.md) | `scikit-learn`, `shap`, `statistical-analysis`, `statsmodels`, `pymc` | Statistical baselines, assumption tests, SHAP audit, Bayesian models |
| [`time-series-expert`](ultimate-skills/agents/time-series-expert.agent.md) | `timesfm-forecasting`, `aeon`, `statistical-analysis`, `statsmodels` | Temporal CV, lag/rolling features, TimesFM baseline, stationarity checks |
| [`graph-ml-expert`](ultimate-skills/agents/graph-ml-expert.agent.md) | `networkx`, `torch-geometric` | Graph construction, node features, GNN training |
| [`deep-learning-expert`](ultimate-skills/agents/deep-learning-expert.agent.md) | `pytorch-lightning`, `transformers` | Tabular MLP, transformer text encoder, OOF fold predictions |
| [`rl-expert`](ultimate-skills/agents/rl-expert.agent.md) | `stable-baselines3`, `pufferlib`, `get-available-resources` | Gym environment wrapper, PPO/SAC training, policy evaluation |
| [`specialized-ml-expert`](ultimate-skills/agents/specialized-ml-expert.agent.md) | `scikit-survival`, `pymoo`, `sympy` | Survival analysis, Pareto threshold tuning, symbolic features |

---

## Install

```bash
git clone https://github.com/KameniAlexNea/ultimate-ml-competition-skill.git
cd ultimate-ml-competition-skill
```

### Claude Code / Codex (default format)

```bash
# Global install
cp -r ultimate-skills/skills/* ~/.claude/skills/
cp -r ultimate-skills/agents/* ~/.claude/agents/

# Project-level
cp -r ultimate-skills/skills/* .claude/skills/
cp -r ultimate-skills/agents/* .claude/agents/
```

### VS Code Copilot

The agent files in `ultimate-skills/agents/` use Claude Code / Codex frontmatter. Run the generator script to produce VS Code-compatible versions before copying:

```bash
# Generate VS Code agents into .github/agents/ (default, auto-detected by VS Code)
bash scripts/generate_vscode_agents.sh

# Or generate into a custom location (e.g. user profile)
bash scripts/generate_vscode_agents.sh ~/.copilot/agents
```

Then copy the skills (no conversion needed — `SKILL.md` format is the same):

```bash
# Project-level
cp -r ultimate-skills/skills/* .github/skills/

# User profile (available in all workspaces)
cp -r ultimate-skills/skills/* ~/.copilot/skills/
```

Other agents: Cursor → `.cursor/skills/` and `.cursor/agents/`, Gemini CLI → `.gemini/skills/`

---

## Use in VS Code

### How skills and agents are resolved

VS Code reads skills from `.github/skills/` or `~/.copilot/skills/` and agents from `.github/agents/` or `~/.copilot/agents/`. After the setup above, skills appear as `/` slash commands in the Chat view and the `team-lead` agent appears in the agent picker dropdown.

### Start a competition run

1. Open a new chat and select **`team-lead`** from the agents dropdown.
2. Describe the competition — paste the competition summary, a link to the brief, or point to the data folder.
3. `team-lead` orchestrates the full pipeline. It invokes all specialist agents as subagents in dependency order: `research-analyst` → `infrastructure-expert` → `data-processing-expert` → ... → `ml-competition-pre-submit`.

### Enable the subagent pattern

`team-lead` delegates work using VS Code's subagent feature. The `agent` tool must be enabled in your chat session (it is on by default in agent mode). Each subagent runs in isolated context and reports its result back via `EXPERIMENT_STATE.json`.

To allow specialist agents to invoke further subagents if needed, enable:

```
chat.subagents.allowInvocationsFromSubagents = true
```

### Specialist agents are subagent-only

All 10 specialist agents are set to `user-invocable: false` — they do not appear in the dropdown and can only be invoked by the `team-lead`. To work with a specialist directly, start from `team-lead` and ask it to delegate to the specific specialist.

### Load a skill on demand

Skills can be invoked as slash commands at any point during a session:

```
/ml-competition-features    Derive lag and rolling features for the timestamp columns.
/shap                       Explain the top-20 features driving predictions.
/exploratory-data-analysis  Profile the test set for distribution shift.
```

VS Code loads only the relevant `SKILL.md` into context — no token cost for skills you don't use.

### Resume an interrupted run

After each phase the active agent writes its result to `EXPERIMENT_STATE.json`. `team-lead` reads that file at startup and skips any step already marked `"success"`, so you can resume without restarting from scratch.

---

## License

[MIT](LICENSE)

