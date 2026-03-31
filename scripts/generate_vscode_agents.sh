#!/usr/bin/env bash
# generate_vscode_agents.sh
#
# Generates VS Code-compatible .agent.md files from the canonical Claude Code
# agent definitions in ultimate-skills/agents/, and copies skills to the
# matching skills/ directory beside the agents output.
#
# The source files use Claude Code / Codex frontmatter format:
#   tools: Read, Write, Edit, MultiEdit, Bash, Glob, Grep, Skill
#   model: inherit
#   maxTurns: N
#   skills: [...]
#
# The generated files use VS Code frontmatter format:
#   tools: ['read', 'write', 'edit', 'terminal', 'search']    (VS Code tool names)
#   agents: [...]                                              (team-lead only)
#   user-invocable: false                                      (specialists only)
#
# Usage:
#   ./scripts/generate_vscode_agents.sh [output_dir]
#
# Defaults:
#   output_dir = .github/agents      (workspace-level, picked up by VS Code automatically)
#   skills are copied to   .github/skills/
#
# Other common destinations:
#   .github/agents     workspace — available to all workspace users (default)
#   ~/.copilot/agents  user profile — available in all workspaces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_AGENTS="$REPO_ROOT/ultimate-skills/agents"
SOURCE_SKILLS="$REPO_ROOT/ultimate-skills/skills"
OUTPUT_DIR="${1:-$REPO_ROOT/.github/agents}"
SKILLS_DIR="$(dirname "$OUTPUT_DIR")/skills"

mkdir -p "$OUTPUT_DIR" "$SKILLS_DIR"

python3 - "$SOURCE_AGENTS" "$OUTPUT_DIR" << 'PYEOF'
import sys, re, pathlib, textwrap

source_dir = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])

# ---------------------------------------------------------------------------
# VS Code tool lists per agent (exact tool names from VS Code)
# team-lead: orchestration only — no execute/edit
# specialists: full tool access including Python, Jupyter, and MCP tools
# ---------------------------------------------------------------------------
_SPECIALIST_TOOLS = (
    "[vscode, execute, read, agent, edit, search, web, 'pylance-mcp-server/*', browser, "
    "vscode.mermaid-chat-features/renderMermaidDiagram, "
    "ms-azuretools.vscode-containers/containerToolsConfig, "
    "ms-mssql.mssql/mssql_schema_designer, ms-mssql.mssql/mssql_dab, "
    "ms-mssql.mssql/mssql_connect, ms-mssql.mssql/mssql_disconnect, "
    "ms-mssql.mssql/mssql_list_servers, ms-mssql.mssql/mssql_list_databases, "
    "ms-mssql.mssql/mssql_get_connection_details, ms-mssql.mssql/mssql_change_database, "
    "ms-mssql.mssql/mssql_list_tables, ms-mssql.mssql/mssql_list_schemas, "
    "ms-mssql.mssql/mssql_list_views, ms-mssql.mssql/mssql_list_functions, "
    "ms-mssql.mssql/mssql_run_query, "
    "ms-python.python/getPythonEnvironmentInfo, ms-python.python/getPythonExecutableCommand, "
    "ms-python.python/installPythonPackage, ms-python.python/configurePythonEnvironment, "
    "ms-toolsai.jupyter/configureNotebook, ms-toolsai.jupyter/listNotebookPackages, "
    "ms-toolsai.jupyter/installNotebookPackages, todo]"
)

_ORCHESTRATOR_TOOLS = "[vscode, read, agent, search, web, browser, todo]"

VSCODE_TOOLS = {
    # Orchestrators — coordinate agents, do not write model/data code
    "team-lead":                  _ORCHESTRATOR_TOOLS,
    "data-pipeline-expert":       _ORCHESTRATOR_TOOLS,
    "mle-expert":                 _ORCHESTRATOR_TOOLS,
    # Specialists — full tool access
    "research-analyst":           _SPECIALIST_TOOLS,
    "setup-expert":               _SPECIALIST_TOOLS,
    "data-processing-expert":     _SPECIALIST_TOOLS,
    "visualization-expert":       _SPECIALIST_TOOLS,
    "feature-engineering-expert": _SPECIALIST_TOOLS,
    "baseline-expert":            _SPECIALIST_TOOLS,
    "gradient-boosting-expert":   _SPECIALIST_TOOLS,
    "deep-learning-expert":       _SPECIALIST_TOOLS,
    "time-series-expert":         _SPECIALIST_TOOLS,
    "graph-ml-expert":            _SPECIALIST_TOOLS,
    "rl-expert":                  _SPECIALIST_TOOLS,
    "specialized-ml-expert":      _SPECIALIST_TOOLS,
    "ensemble-expert":            _SPECIALIST_TOOLS,
}

SPECIALIST_NAMES = [k for k in VSCODE_TOOLS if k != "team-lead"]

TEAM_LEAD_AGENTS = (
    "agents: ['research-analyst', 'setup-expert', 'data-pipeline-expert', "
    "'data-processing-expert', 'visualization-expert', 'feature-engineering-expert', "
    "'baseline-expert', 'mle-expert', 'gradient-boosting-expert', "
    "'deep-learning-expert', 'time-series-expert', 'graph-ml-expert', "
    "'rl-expert', 'specialized-ml-expert', 'ensemble-expert']"
)

# ---------------------------------------------------------------------------
# Process each source agent file
# ---------------------------------------------------------------------------
for src_path in sorted(source_dir.glob("*.agent.md")):
    stem = src_path.stem.replace(".agent", "")          # e.g. "team-lead"
    text = src_path.read_text()

    # Separate frontmatter and body
    m = re.match(r'^---\n(.*?)\n---\n(.*)', text, re.DOTALL)
    if not m:
        print(f"  SKIP (no frontmatter): {src_path.name}", file=sys.stderr)
        continue

    fm_raw, body = m.group(1), m.group(2)

    # Extract fields we keep
    name  = re.search(r'^name:\s*(.+)$',        fm_raw, re.MULTILINE)
    desc  = re.search(r'^description:\s*(.+)$', fm_raw, re.MULTILINE)
    name_val = name.group(1).strip() if name else stem
    desc_val = desc.group(1).strip() if desc else ""

    # Build VS Code frontmatter
    fm_lines = [
        f"name: {name_val}",
        f"description: {desc_val}",
        f"tools: {VSCODE_TOOLS.get(stem, _SPECIALIST_TOOLS)}",
    ]
    if stem == "team-lead":
        fm_lines.append(TEAM_LEAD_AGENTS)
    else:
        fm_lines.append("user-invocable: false")

    new_fm = "\n".join(fm_lines)
    out_path = output_dir / src_path.name
    out_path.write_text(f"---\n{new_fm}\n---\n{body}")
    print(f"  generated: {out_path}")

print(f"\nDone — {len(list(output_dir.glob('*.agent.md')))} agent files written to {output_dir}")
PYEOF

# ---------------------------------------------------------------------------
# Copy skills (no conversion needed — SKILL.md format is the same for VS Code)
# ---------------------------------------------------------------------------
echo "Copying skills → $SKILLS_DIR"
find "$SOURCE_SKILLS" -name "SKILL.md" | while read -r skill_file; do
    skill_name="$(basename "$(dirname "$skill_file")")"
    dest_dir="$SKILLS_DIR/$skill_name"
    mkdir -p "$dest_dir"
    cp "$skill_file" "$dest_dir/SKILL.md"
    for subdir in references assets scripts; do
        src_sub="$(dirname "$skill_file")/$subdir"
        if [ -d "$src_sub" ]; then
            cp -r "$src_sub" "$dest_dir/"
        fi
    done
done
SKILL_COUNT=$(find "$SOURCE_SKILLS" -name "SKILL.md" | wc -l | tr -d ' ')
echo "  $SKILL_COUNT skill directories copied"
