# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

The vaEvas Agent framework — orchestrates the full closed-loop pipeline for Verilog-A generation:
LLM generation → EVAS evaluation → diagnosis → targeted repair → repeat until PASS.

Depends on three sibling repos:
- `EVAS/` — event-driven Verilog-A simulator
- `behavioral-veriloga-eval/` — benchmark tasks, runners, checkers
- `veriloga-skills/` — Verilog-A coding knowledge base (category references)

## Architecture

```
vaevas_agent/
  agent.py          Agent class — top-level orchestrator
  cli.py            CLI: init, run, list, config, doctor
  config.py         AgentConfig dataclass + YAML loader + .env auto-load
  display.py        Terminal output (spinners, boxes, colors, ASCII-safe)
  doctor.py         Environment checker with 10 checks + auto-fix (5 fixable)
  llm/client.py     Anthropic + OpenAI (native + compatible providers)
  skills/           SkillManager (keyword match -> category injection, 54 keywords)
  loop/             LoopController state machine + LoopState + Terminator
  prompts/pipeline.py  Prompt assembly (system, task, repair) — central injection logic
```

## Commands

```bash
# Install
pip install evas-sim anthropic numpy matplotlib pyyaml

# First-time setup
python -m vaevas_agent init

# Check environment
python -m vaevas_agent doctor
python -m vaevas_agent doctor --fix

# List tasks (scans tasks/ + benchmark-v2/tasks/ recursively)
python -m vaevas_agent list
python -m vaevas_agent list --family end-to-end

# Run a task
python -m vaevas_agent run digital_basics_smoke --max-rounds 3

# Configuration
python -m vaevas_agent config --show
python -m vaevas_agent config --set-provider anthropic-compatible

# Tests
pytest tests/
python tests/test_integration.py
```

## LLM Provider Modes

| Provider | API | base_url |
|----------|-----|----------|
| `anthropic` | native Anthropic Messages | default |
| `anthropic-compatible` | Anthropic Messages at custom endpoint | required |
| `openai` | native OpenAI Chat Completions | default |
| `openai-compatible` | OpenAI Chat Completions at custom endpoint | required |

## .env File

Auto-loaded on import. Set keys:
```
ANTHROPIC_API_KEY=sk-xxx
ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
ANTHROPIC_MODEL=deepseek-v4-flash
OPENAI_API_KEY=sk-xxx
```

## Prompt Injection Pipeline

See `prompts/pipeline.py` for the complete injection chain:
- `build_system_prompt()` → system message (rules + skill context)
- `build_task_prompt()` → user message for Round 0 (10 injection layers)
- `build_repair_prompt()` → user message for Round 1+ (EVAS feedback + targeted repair rules)
