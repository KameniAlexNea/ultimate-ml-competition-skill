#!/usr/bin/env node
// install_agents.mjs
//
// Installs agents and skills from ultimate-skills/ into the correct
// directories for Claude Code or Cursor, with auto-detection.
//
// Usage:
//   node scripts/install_agents.mjs                         # auto-detect, project scope
//   node scripts/install_agents.mjs --agent claude-code     # force Claude Code
//   node scripts/install_agents.mjs --agent cursor          # force Cursor
//   node scripts/install_agents.mjs --scope global          # install globally
//   node scripts/install_agents.mjs --print                 # print AGENTS.md and exit
//
// Supported agents:
//   claude-code  →  .claude/  (agents/, skills/, CLAUDE.md)
//   cursor       →  .cursor/  (rules/ml-competition-agents.mdc, skills/)

import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const home = process.env.HOME || process.env.USERPROFILE
const repoRoot = path.join(__dirname, '..')
const sourceAgents = path.join(repoRoot, 'ultimate-skills', 'agents')
const sourceSkills = path.join(repoRoot, 'ultimate-skills', 'skills')
const agentsMd = path.join(repoRoot, 'ultimate-skills', 'AGENTS.md')

// ---------------------------------------------------------------------------
// Agent registry
// ---------------------------------------------------------------------------
const AGENTS = {
  'claude-code': {
    detect: path.join(home, '.claude'),
    label: 'Claude Code',
    usage: 'Agents and skills are available in Claude Code.',
  },
  cursor: {
    detect: path.join(home, '.cursor'),
    label: 'Cursor',
    usage: 'Use @Skills in Cursor chat to reference installed skills.',
  },
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
const agentFlag = process.argv.indexOf('--agent')
const forcedAgent = agentFlag !== -1 ? process.argv[agentFlag + 1] : null

const scopeFlag = process.argv.indexOf('--scope')
const scope = scopeFlag !== -1 ? process.argv[scopeFlag + 1] : 'project'

if (process.argv.includes('--print')) {
  console.log(fs.readFileSync(agentsMd, 'utf8'))
  process.exit(0)
}

if (!['project', 'global'].includes(scope)) {
  console.error(`ERROR: unknown scope "${scope}". Use "project" or "global".`)
  process.exit(1)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Recursively copy src directory into dest. */
function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true })
  for (const entry of fs.readdirSync(src)) {
    const srcPath = path.join(src, entry)
    const destPath = path.join(dest, entry)
    if (fs.statSync(srcPath).isDirectory()) {
      copyDir(srcPath, destPath)
    } else {
      fs.copyFileSync(srcPath, destPath)
    }
  }
}

/** Install all skill directories (each must contain a SKILL.md). */
function installSkills(targetSkills) {
  fs.mkdirSync(targetSkills, { recursive: true })
  const skillDirs = fs.readdirSync(sourceSkills).filter(d =>
    fs.statSync(path.join(sourceSkills, d)).isDirectory()
    && fs.existsSync(path.join(sourceSkills, d, 'SKILL.md'))
  )
  for (const skillDir of skillDirs) {
    copyDir(path.join(sourceSkills, skillDir), path.join(targetSkills, skillDir))
  }
  return skillDirs.length
}

// ---------------------------------------------------------------------------
// Agent installers
// ---------------------------------------------------------------------------

function installClaudeCode() {
  const targetRoot = scope === 'global'
    ? path.join(home, '.claude')
    : path.join(process.cwd(), '.claude')
  const targetAgents = path.join(targetRoot, 'agents')
  const targetSkills = path.join(targetRoot, 'skills')

  // Agents
  fs.mkdirSync(targetAgents, { recursive: true })
  const agentFiles = fs.readdirSync(sourceAgents).filter(f => f.endsWith('.agent.md'))
  for (const f of agentFiles) {
    fs.copyFileSync(path.join(sourceAgents, f), path.join(targetAgents, f))
  }

  // Skills
  const skillCount = installSkills(targetSkills)

  // AGENTS.md → CLAUDE.md (Claude Code reads this automatically)
  if (fs.existsSync(agentsMd)) {
    fs.copyFileSync(agentsMd, path.join(targetRoot, 'CLAUDE.md'))
  }

  console.log(`\n✓ Installed for Claude Code (scope: ${scope})`)
  console.log(`  Target  : ${targetRoot}`)
  console.log(`  Agents  : ${agentFiles.length}`)
  console.log(`  Skills  : ${skillCount}`)
  console.log(`  CLAUDE.md installed`)
  if (scope === 'project') {
    console.log(`\n  Tip: add .claude/ to .gitignore if you don't want to commit it`)
  }
  console.log()
}

function installCursor() {
  const targetRoot = scope === 'global'
    ? path.join(home, '.cursor')
    : path.join(process.cwd(), '.cursor')
  const targetRules = path.join(targetRoot, 'rules')
  const targetSkills = path.join(targetRoot, 'skills')

  // AGENTS.md → .cursor/rules/ml-competition-agents.mdc
  fs.mkdirSync(targetRules, { recursive: true })
  if (fs.existsSync(agentsMd)) {
    const body = fs.readFileSync(agentsMd, 'utf8')
    const mdc = [
      '---',
      'description: ML Competition Pipeline — agents and skills reference',
      'alwaysApply: false',
      '---',
      '',
      body,
    ].join('\n')
    fs.writeFileSync(path.join(targetRules, 'ml-competition-agents.mdc'), mdc)
  }

  // Skills
  const skillCount = installSkills(targetSkills)

  console.log(`\n✓ Installed for Cursor (scope: ${scope})`)
  console.log(`  Target  : ${targetRoot}`)
  console.log(`  Skills  : ${skillCount}`)
  console.log(`  Rule    : rules/ml-competition-agents.mdc`)
  if (scope === 'project') {
    console.log(`\n  Tip: add .cursor/ to .gitignore if you don't want to commit it`)
  }
  console.log()
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

function install(agentKey) {
  if (!AGENTS[agentKey]) {
    console.error(`ERROR: unknown agent "${agentKey}". Supported: ${Object.keys(AGENTS).join(', ')}`)
    process.exit(1)
  }
  if (agentKey === 'claude-code') installClaudeCode()
  else if (agentKey === 'cursor') installCursor()
}

if (forcedAgent) {
  install(forcedAgent)
} else {
  // Auto-detect the first installed agent
  const detected = Object.keys(AGENTS).find(key => fs.existsSync(AGENTS[key].detect))

  if (detected) {
    console.log(`Detected ${AGENTS[detected].label} — installing...`)
    install(detected)
  } else {
    console.log('\nNo supported agent detected.')
    console.log('Supported agents: claude-code, cursor')
    console.log('\nOptions:')
    console.log('  node scripts/install_agents.mjs --agent claude-code')
    console.log('  node scripts/install_agents.mjs --agent cursor')
    console.log('  node scripts/install_agents.mjs --scope global   (global install)')
    console.log('  node scripts/install_agents.mjs --print          (print AGENTS.md)\n')
    process.exit(1)
  }
}
