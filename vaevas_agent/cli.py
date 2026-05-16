"""CLI entry point for vaEvas Agent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent
from .config import AgentConfig, load_config, save_config
from .display import dim, green, red, yellow
from .doctor import Doctor, DoctorConfig
from .skills.manager import SkillManager


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="vaevas-agent",
        description="vaEvas Agent — closed-loop Verilog-A generation pipeline",
    )
    sub = parser.add_subparsers(dest="command")

    # ── run ──
    p_run = sub.add_parser("run", help="Run the closed loop for a task")
    p_run.add_argument("task_id", help="Task ID (e.g., digital_basics_smoke)")
    p_run.add_argument("--task-root", default=None,
                       help="Root directory of tasks (default: auto-detect)")
    p_run.add_argument("--model", default=None, help="LLM model override")
    p_run.add_argument("--provider", choices=["anthropic", "openai",
                         "anthropic-compatible", "openai-compatible"], default=None)
    p_run.add_argument("--base-url", default=None, help="Custom LLM endpoint URL")
    p_run.add_argument("--max-rounds", type=int, default=None)
    p_run.add_argument("--no-skill", action="store_true", help="Disable skill injection")
    p_run.add_argument("--output-dir", default=None)
    p_run.add_argument("--skip-doctor", action="store_true",
                       help="Skip environment check")
    p_run.set_defaults(func=cmd_run)

    # ── list ──
    p_list = sub.add_parser("list", help="List available tasks")
    p_list.add_argument("--family", default=None, help="Filter by family")
    p_list.add_argument("--task-root", default=None)
    p_list.set_defaults(func=cmd_list)

    # ── config ──
    p_config = sub.add_parser("config", help="Show or update configuration")
    p_config.add_argument("--set-model", default=None, help="Set default model")
    p_config.add_argument("--set-provider", choices=["anthropic", "openai",
                          "anthropic-compatible", "openai-compatible"], default=None)
    p_config.add_argument("--set-base-url", default=None, help="Set custom LLM endpoint")
    p_config.add_argument("--show", action="store_true", help="Show current config")
    p_config.set_defaults(func=cmd_config)

    # ── doctor ──
    p_doctor = sub.add_parser("doctor", help="Check environment readiness")
    p_doctor.add_argument("--fix", action="store_true", help="Auto-fix issues")
    p_doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


# ─── Command implementations ─────────────────────────────────

def cmd_run(args) -> None:
    config = load_config()

    # Apply CLI overrides
    if args.model:
        config.llm.model = args.model
    if args.provider:
        config.llm.provider = args.provider
    if args.base_url:
        config.llm.base_url = args.base_url
    if args.max_rounds:
        config.loop.max_rounds = args.max_rounds
    if args.no_skill:
        config.skills.enabled = False
    if args.output_dir:
        config.output.dir = args.output_dir

    # Doctor check (unless skipped)
    if not args.skip_doctor:
        dr = Doctor(_make_doctor_config(config))
        if dr.run(auto_fix=False) != 0:
            print(yellow("\nRun 'vaevas-agent doctor --fix' to auto-repair, "
                        "or 'vaevas-agent run --skip-doctor ...' to skip.\n"))
            sys.exit(1)

    # Find task directory (scans all task locations recursively)
    task_dir = _find_task_dir(args.task_id, args.task_root, config)
    if task_dir is None:
        print(red(f"Task '{args.task_id}' not found. Use 'vaevas-agent list' to see available tasks."))
        sys.exit(1)

    agent = Agent(config)
    history = agent.run(args.task_id, task_dir)

    last = history[-1] if history else None
    if last and last.status == "PASS":
        sys.exit(0)
    else:
        sys.exit(1)


def cmd_list(args) -> None:
    config = load_config()
    root = _resolve_eval_root(args.task_root, config)
    if root is None or not root.exists():
        print(red("Task root not found. Set behavioral-veriloga-eval path in config."))
        sys.exit(1)

    tasks = _scan_all_tasks(root)
    family_filter = args.family

    # Group by prefix path, then by family
    by_prefix: dict[str, list[dict]] = {}
    for t in tasks:
        if family_filter and t["family"] != family_filter:
            continue
        prefix = t["prefix"] or "(root)"
        by_prefix.setdefault(prefix, []).append(t)

    for prefix in sorted(by_prefix):
        group = by_prefix[prefix]
        print(f"\n{bold(prefix)}/  ({len(group)} tasks)")
        for t in sorted(group, key=lambda x: x["task_id"]):
            icon = "gold" if t["has_gold"] else ("chk" if t["has_checks"] else "inc")
            line = f"  {t['task_id']:<40s} {t['family']:<16s} {t['category']:<20s} {t['difficulty']:<8s} {icon}"
            print(dim(line) if sys.stdout.isatty() else line)
    total_tasks = len(tasks)
    total_dirs = len(by_prefix)
    print(f"\nTotal: {total_tasks} tasks across {total_dirs} directories")


def cmd_config(args) -> None:
    config_path = Path.cwd() / "config" / "default.yaml"

    if args.show or (not args.set_model and not args.set_provider and not args.set_base_url):
        _print_config(load_config(config_path))
        return

    config = load_config(config_path)
    if args.set_model:
        config.llm.model = args.set_model
    if args.set_provider:
        config.llm.provider = args.set_provider
    if args.set_base_url:
        config.llm.base_url = args.set_base_url

    save_config(config, config_path)
    print(green(f"Config saved to {config_path}"))
    _print_config(config)


def cmd_doctor(args) -> None:
    config = load_config()
    dr = Doctor(_make_doctor_config(config))
    sys.exit(dr.run(auto_fix=args.fix))


# ─── Helpers ─────────────────────────────────────────────────

def _find_task_dir(task_id: str, task_root_override: str | None, config: AgentConfig) -> Path | None:
    root = _resolve_eval_root(task_root_override, config)
    if root is None:
        return None
    for meta_path in sorted(root.rglob("meta.json")):
        # Skip meta.json not belonging to a benchmark task (must have gold/ or checks.yaml)
        task_dir = meta_path.parent
        if not ((task_dir / "gold").is_dir() or (task_dir / "checks.yaml").exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tid = meta.get("task_id") or meta.get("id") or task_dir.name
        except Exception:
            tid = task_dir.name
        if tid == task_id:
            return task_dir
    return None


def _resolve_eval_root(override: str | None, config: AgentConfig) -> Path | None:
    """Resolve the behavioral-veriloga-eval repo root directory."""
    if override:
        p = Path(override).resolve()
        return p if p.exists() else None
    eval_path = config.resolve_path(config.paths.behavioral_eval)
    if eval_path.exists():
        return eval_path
    candidates = [
        Path.cwd() / ".." / "behavioral-veriloga-eval",
        Path.cwd() / "behavioral-veriloga-eval",
    ]
    for c in candidates:
        if c.resolve().exists():
            return c.resolve()
    return None


def _scan_all_tasks(root: Path) -> list[dict]:
    """Scan eval repo root for all benchmark tasks with complete files.

    Returns list of {task_id, family, category, difficulty, path, has_gold}.
    A task is considered "complete" if it has meta.json AND (gold/ directory or checks.yaml).
    """
    tasks = []
    for meta_path in sorted(root.rglob("meta.json")):
        task_dir = meta_path.parent
        has_gold = (task_dir / "gold").is_dir()
        has_checks = (task_dir / "checks.yaml").exists()
        has_prompt = (task_dir / "prompt.md").exists()

        if not has_gold and not has_checks:
            continue  # incomplete task, skip

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        tid = meta.get("task_id") or meta.get("id") or task_dir.name
        family = meta.get("family", "unknown")
        category = meta.get("category", "unknown")
        difficulty = meta.get("difficulty", "?")

        # Determine a display prefix from the relative path
        rel = task_dir.relative_to(root)
        prefix = str(rel.parent) if rel.parent != Path(".") else ""

        tasks.append({
            "task_id": tid,
            "family": family,
            "category": category,
            "difficulty": difficulty,
            "path": task_dir,
            "prefix": prefix,
            "has_gold": has_gold,
            "has_checks": has_checks,
            "has_prompt": has_prompt,
        })
    return tasks


def _make_doctor_config(config: AgentConfig) -> DoctorConfig:
    return DoctorConfig(
        veriloga_skills_path=config.resolve_path(config.paths.veriloga_skills),
        behavioral_eval_path=config.resolve_path(config.paths.behavioral_eval),
        config_path=Path.cwd() / "config" / "default.yaml",
    )


def _print_config(config: AgentConfig) -> None:
    print(f"\n{bold('LLM Configuration')}")
    print(f"  provider:   {config.llm.provider}")
    print(f"  model:      {config.llm.model}")
    print(f"  temperature: {config.llm.temperature} (repair: {config.llm.repair_temperature})")
    print(f"  base_url:   {config.llm.base_url or '(default)'}")
    print(f"\n{bold('Loop Configuration')}")
    print(f"  max_rounds:  {config.loop.max_rounds}")
    print(f"  stall_limit: {config.loop.stall_limit}")
    print(f"\n{bold('Skills')}")
    print(f"  enabled:     {config.skills.enabled}")
    print(f"  categories:  {config.resolve_path(config.skills.categories_dir)}")
    mgr = SkillManager(skills_root=config.resolve_path(config.paths.veriloga_skills))
    print(f"  available:   {mgr.category_count} categories" if mgr.available else "  available:   no")
    print(f"\n{bold('Paths')}")
    print(f"  skills:  {config.resolve_path(config.paths.veriloga_skills)}")
    print(f"  eval:    {config.resolve_path(config.paths.behavioral_eval)}")
    print(f"  output:  {config.resolve_path(config.output.dir)}")
    print()


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m" if sys.stdout.isatty() else text
