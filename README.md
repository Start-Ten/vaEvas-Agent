# vaEvas Agent

Agent framework for the **vaEvas closed-loop Verilog-A generation pipeline**:

LLM generation → EVAS simulation → scoring → diagnosis → targeted repair → repeat until PASS.

## Quick Start

```bash
# Create conda environment
conda create -n vaevas python=3.12 -y
conda activate vaevas

# Clone and enter
git clone https://github.com/Start-Ten/vaEvas-Agent.git
cd vaEvas-Agent

# Install dependencies
pip install evas-sim anthropic openai numpy matplotlib pyyaml

# Interactive setup (recommended)
python -m vaevas_agent init

# Or manual setup: copy .env.example to .env and fill in keys
```

## Commands

```bash
python -m vaevas_agent init                     # Interactive first-time setup
python -m vaevas_agent doctor                   # Check environment
python -m vaevas_agent doctor --fix             # Auto-fix issues
python -m vaevas_agent list                     # List all benchmarks
python -m vaevas_agent list --family end-to-end # Filter by family
python -m vaevas_agent config --show            # Show configuration
python -m vaevas_agent run <task_id>            # Run closed loop
```

## LLM Provider Modes

| Provider | API | Configuration |
|----------|-----|---------------|
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `anthropic-compatible` | Anthropic Messages API (custom endpoint) | `ANTHROPIC_API_KEY` + `ANTHROPIC_BASE_URL` |
| `openai` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `openai-compatible` | OpenAI Chat Completions (custom endpoint) | `OPENAI_API_KEY` + custom base URL |

## Project Structure

```
vaevas_agent/
  agent.py          Agent class — top-level orchestrator
  cli.py            CLI: init, run, list, config, doctor
  config.py         AgentConfig dataclass + YAML + .env loader
  display.py        Terminal output (spinners, boxes, ASCII-safe)
  doctor.py         Environment checker (10 checks + auto-fix)
  llm/client.py     LLM client (Anthropic + OpenAI, native + compatible)
  skills/           SkillManager: keyword match -> category reference injection
  loop/             LoopController state machine + Terminator
  prompts/          Prompt injection pipeline (system, task, repair)
```

## Requirements

- Python >= 3.11
- [EVAS](https://github.com/Arcadia-1/EVAS) simulator
- [veriloga-skills](https://github.com/Arcadia-1/veriloga-skills) (for skill injection)
- [behavioral-veriloga-eval](https://github.com/Arcadia-1/behavioral-veriloga-eval) (for benchmark tasks)
