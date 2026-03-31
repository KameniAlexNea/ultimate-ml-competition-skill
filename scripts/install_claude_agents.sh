#!/usr/bin/env bash
# install_claude_agents.sh
#
# Installs canonical Claude Code agent definitions and skills from
# ultimate-skills/ into the correct Claude Code directories.
#
# The source files in ultimate-skills/agents/ are already in Claude Code
# frontmatter format — this script copies them and renames AGENTS.md to
# CLAUDE.md (the filename Claude Code reads automatically).
#
# Usage:
#   ./scripts/install_claude_agents.sh [scope]
#
# Scopes:
#   project  (default) — installs into .claude/ in the current working directory
#   global             — installs into ~/.claude/ (available in all projects)
#
# Examples:
#   ./scripts/install_claude_agents.sh           # project-level
#   ./scripts/install_claude_agents.sh project   # same as above
#   ./scripts/install_claude_agents.sh global    # user-level

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_AGENTS="$REPO_ROOT/ultimate-skills/agents"
SOURCE_SKILLS="$REPO_ROOT/ultimate-skills/skills"
AGENTS_MD="$REPO_ROOT/ultimate-skills/AGENTS.md"

SCOPE="${1:-project}"

case "$SCOPE" in
  project)
    TARGET_ROOT="${PWD}/.claude"
    ;;
  global)
    TARGET_ROOT="${HOME}/.claude"
    ;;
  *)
    echo "ERROR: unknown scope '$SCOPE'. Use 'project' or 'global'." >&2
    exit 1
    ;;
esac

TARGET_AGENTS="$TARGET_ROOT/agents"
TARGET_SKILLS="$TARGET_ROOT/skills"

mkdir -p "$TARGET_AGENTS" "$TARGET_SKILLS"

# ---------------------------------------------------------------------------
# 1. Copy agent files (already in Claude Code format — no conversion needed)
# ---------------------------------------------------------------------------
echo "Installing agents → $TARGET_AGENTS"
cp "$SOURCE_AGENTS"/*.agent.md "$TARGET_AGENTS/"
echo "  $(ls "$SOURCE_AGENTS"/*.agent.md | wc -l | tr -d ' ') agent files copied"

# ---------------------------------------------------------------------------
# 2. Copy skills
# ---------------------------------------------------------------------------
echo "Installing skills → $TARGET_SKILLS"
# Each skill lives in its own subdirectory with a SKILL.md
find "$SOURCE_SKILLS" -name "SKILL.md" | while read -r skill_file; do
    skill_name="$(basename "$(dirname "$skill_file")")"
    dest_dir="$TARGET_SKILLS/$skill_name"
    mkdir -p "$dest_dir"
    # Copy SKILL.md and any references/ or assets/ subdirectories
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

# ---------------------------------------------------------------------------
# 3. Install AGENTS.md as CLAUDE.md (Claude Code reads this automatically)
# ---------------------------------------------------------------------------
if [ -f "$AGENTS_MD" ]; then
    cp "$AGENTS_MD" "$TARGET_ROOT/CLAUDE.md"
    echo "Installed AGENTS.md → $TARGET_ROOT/CLAUDE.md"
else
    echo "WARNING: $AGENTS_MD not found — skipping CLAUDE.md install" >&2
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "Done — Claude Code installation complete"
echo "  Scope  : $SCOPE"
echo "  Target : $TARGET_ROOT"
echo ""
if [ "$SCOPE" = "project" ]; then
    echo "Agents and skills are now available in Claude Code for this project."
    echo "Add .claude/ to .gitignore if you do not want to commit it:"
    echo "  echo '.claude/' >> .gitignore"
else
    echo "Agents and skills are now available globally in all Claude Code projects."
fi
