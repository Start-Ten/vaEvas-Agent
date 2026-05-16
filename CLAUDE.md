# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

The vaEvas Agent framework — orchestrates the full closed-loop pipeline for Verilog-A generation:
LLM generation → EVAS evaluation → diagnosis → targeted repair → repeat until PASS.

Depends on three sibling repos:
- `EVAS/` — event-driven Verilog-A simulator
- `behavioral-veriloga-eval/` — benchmark tasks, runners, checkers
- `veriloga-skills/` — Verilog-A coding knowledge base (SKILL.md + category references)

## Architecture

```
vaevas_agent/
  agent.py          Agent class — top-level orchestrator
  cli.py            CLI entry: vaevas-agent run/list/config/doctor
  config.py         AgentConfig dataclass + YAML loader
  display.py        Terminal output (spinners, boxes, colors)
  doctor.py         Environment checker with auto-fix
  llm/client.py     Anthropic + OpenAI (native + compatible providers)
  skills/           SkillManager: keyword match → category reference injection
  loop/             LoopController state machine + LoopState types + Terminator
  prompts/pipeline.py  Prompt assembly (system, task, repair) — central injection logic
```

## Commands

```bash
# Install in dev mode
pip install -e ".[all]"

# Check environment
python -m vaevas_agent doctor
python -m vaevas_agent doctor --fix

# List available tasks
python -m vaevas_agent list

# Run a single task
python -m vaevas_agent run digital_basics_smoke --max-rounds 3

# Configure
python -m vaevas_agent config --set-model claude-sonnet-4-6
python -m vaevas_agent config --set-provider anthropic-compatible --set-base-url https://my-endpoint/v1
python -m vaevas_agent config --show

# Run tests
pytest tests/
```

## LLM Provider Modes

| Provider | API | base_url |
|----------|-----|----------|
| `anthropic` | native Anthropic Messages | default |
| `anthropic-compatible` | Anthropic Messages at custom endpoint | required |
| `openai` | native OpenAI Chat Completions | default |
| `openai-compatible` | OpenAI Chat Completions at custom endpoint | required |

API keys: set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in environment.

## Prompt Injection Pipeline

See `prompts/pipeline.py` for the complete injection chain:
- `build_system_prompt()` → system message (rules + skill context)
- `build_task_prompt()` → user message for Round 0 (10 injection layers)
- `build_repair_prompt()` → user message for Round 1+ (EVAS feedback + targeted repair rules)
