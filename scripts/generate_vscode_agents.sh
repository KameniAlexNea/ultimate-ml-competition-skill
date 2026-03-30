#!/usr/bin/env bash
# generate_vscode_agents.sh
#
# Generates VS Code-compatible .agent.md files from the canonical Claude Code
# agent definitions in ultimate-skills/agents/.
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
#
# Other common destinations:
#   .github/agents     workspace — available to all workspace users (default)
#   ~/.copilot/agents  user profile — available in all workspaces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_DIR="$REPO_ROOT/ultimate-skills/agents"
OUTPUT_DIR="${1:-$REPO_ROOT/.github/agents}"

mkdir -p "$OUTPUT_DIR"

python3 - "$SOURCE_DIR" "$OUTPUT_DIR" << 'PYEOF'
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

VSCODE_TOOLS = {
    "team-lead":             "[vscode, read, agent, search, web, browser, todo]",
    "research-analyst":      _SPECIALIST_TOOLS,
    "infrastructure-expert": _SPECIALIST_TOOLS,
    "data-processing-expert":_SPECIALIST_TOOLS,
    "visualization-expert":  _SPECIALIST_TOOLS,
    "ml-statistics-expert":  _SPECIALIST_TOOLS,
    "time-series-expert":    _SPECIALIST_TOOLS,
    "graph-ml-expert":       _SPECIALIST_TOOLS,
    "deep-learning-expert":  _SPECIALIST_TOOLS,
    "rl-expert":             _SPECIALIST_TOOLS,
    "specialized-ml-expert": _SPECIALIST_TOOLS,
}

SPECIALIST_NAMES = [k for k in VSCODE_TOOLS if k != "team-lead"]

TEAM_LEAD_AGENTS = (
    "agents: ['research-analyst', 'infrastructure-expert', 'data-processing-expert', "
    "'visualization-expert', 'ml-statistics-expert', 'time-series-expert', "
    "'graph-ml-expert', 'deep-learning-expert', 'rl-expert', 'specialized-ml-expert']"
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
        f"tools: {VSCODE_TOOLS.get(stem, VSCODE_TOOLS['data-processing-expert'])}",
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
