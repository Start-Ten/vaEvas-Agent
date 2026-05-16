# vaEvas Agent

Agent framework for the **vaEvas closed-loop Verilog-A generation pipeline**.

LLM generation → EVAS simulation → scoring → diagnosis → targeted repair → repeat until PASS.

## Quick Start

```bash
# Create conda environment
conda create -n vaevas python=3.12 -y
conda activate vaevas

# Install
pip install evas-sim anthropic pyyaml numpy matplotlib
git clone https://github.com/Start-Ten/vaEvas-Agent.git
cd vaEvas-Agent

# Check environment
python -m vaevas_agent doctor
python -m vaevas_agent doctor --fix

# List available tasks
python -m vaevas_agent list
python -m vaevas_agent list --family end-to-end

# Run the closed loop
python -m vaevas_agent run digital_basics_smoke --max-rounds 3
```

## Configuration

```bash
# Show current config
python -m vaevas_agent config --show

# Set model and provider
python -m vaevas_agent config --set-model deepseek-v4-flash
python -m vaevas_agent config --set-provider anthropic-compatible
python -m vaevas_agent config --set-base-url https://api.deepseek.com/anthropic
```

### LLM Provider Modes

| Provider | API | Requires |
|----------|-----|----------|
| `anthropic` | Anthropic Messages API | `ANTHROPIC_API_KEY` |
| `anthropic-compatible` | Anthropic Messages API (custom endpoint) | `ANTHROPIC_API_KEY` + `--set-base-url` |
| `openai` | OpenAI Chat Completions | `OPENAI_API_KEY` |
| `openai-compatible` | OpenAI Chat Completions (custom endpoint) | `OPENAI_API_KEY` + `--set-base-url` |

## Project Structure

```
vaevas_agent/
  agent.py          Agent class — top-level orchestrator
  cli.py            CLI: run, list, config, doctor
  config.py         AgentConfig dataclass + YAML
  display.py        Terminal output (spinners, boxes, colors)
  doctor.py         Environment checker with auto-fix
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

