"""Agent configuration — dataclass + YAML loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class LLMConfig:
    provider: Literal["anthropic", "openai", "anthropic-compatible", "openai-compatible"] = "anthropic"
    model: str = "claude-sonnet-4-6"
    temperature: float = 0.0
    repair_temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 1.0
    timeout: int = 120
    base_url: str = ""                  # custom endpoint for compatible providers
    api_key_env: str = ""               # override env var name for API key


@dataclass
class PathsConfig:
    veriloga_skills: str = "../veriloga-skills"
    behavioral_eval: str = "../behavioral-veriloga-eval"


@dataclass
class LoopConfig:
    max_rounds: int = 3
    stall_limit: int = 2
    regress_limit: int = 2


@dataclass
class SkillsConfig:
    enabled: bool = True
    max_chars: int = 3000
    categories_dir: str = "../veriloga-skills/veriloga/references/categories"


@dataclass
class OutputConfig:
    dir: str = "./output"
    save_artifacts: bool = True
    save_prompts: bool = True


@dataclass
class AgentConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Resolved absolute paths (set after load)
    project_root: Path = field(default_factory=Path.cwd)

    def resolve_path(self, raw: str) -> Path:
        """Resolve a path relative to project_root, expanding ~ and env vars."""
        import os
        expanded = os.path.expandvars(os.path.expanduser(raw))
        p = Path(expanded)
        if p.is_absolute():
            return p
        return (self.project_root / p).resolve()


# ─── Known mapping: config key → dataclass type ──────────────

_NESTED_TYPES: dict[str, type] = {
    "llm": LLMConfig,
    "paths": PathsConfig,
    "loop": LoopConfig,
    "skills": SkillsConfig,
    "output": OutputConfig,
}


def _dict_to_dataclass(data: dict, klass):
    """Recursively convert dict to dataclass, ignoring unknown keys."""
    import dataclasses
    known = {f.name for f in dataclasses.fields(klass)}
    kwargs = {}
    for k, v in data.items():
        if k not in known:
            continue
        nested_type = _NESTED_TYPES.get(k)
        if isinstance(v, dict) and nested_type is not None:
            kwargs[k] = _dict_to_dataclass(v, nested_type)
        else:
            kwargs[k] = v
    return klass(**kwargs)


def load_config(config_path: Path | str | None = None) -> AgentConfig:
    """Load configuration from YAML file, with sensible defaults."""
    if config_path is None:
        # Search order: env var > ./config/default.yaml > package default
        import os
        env_path = os.environ.get("VAEVAS_AGENT_CONFIG")
        if env_path:
            config_path = Path(env_path)
        else:
            package_default = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
            if package_default.exists():
                config_path = package_default
            else:
                return AgentConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        return AgentConfig()

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    config = _dict_to_dataclass(raw, AgentConfig)
    config.project_root = config_path.parent.parent.resolve()  # project root is parent of config/
    return config


def save_config(config: AgentConfig, config_path: Path | str) -> None:
    """Save configuration to YAML file (preserves comments by overwriting)."""
    import dataclasses
    raw = {}
    for f in dataclasses.fields(AgentConfig):
        val = getattr(config, f.name)
        if dataclasses.is_dataclass(val):
            raw[f.name] = dataclasses.asdict(val)
        else:
            raw[f.name] = val
    raw.pop("project_root", None)  # don't persist resolved paths
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.dump(raw, fh, default_flow_style=False, allow_unicode=True, sort_keys=False)
