"""Environment doctor — checks and auto-fixes the vaEvas Agent runtime environment."""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .display import (
    doctor_check, doctor_header, doctor_summary, dim, green, red, yellow,
)


@dataclass
class CheckResult:
    name: str
    status: Literal["pass", "fail", "warn"]
    message: str
    fixable: bool = False
    fix_func: Callable[[], bool] | None = None
    required: bool = True


@dataclass
class DoctorConfig:
    """Paths checked by the doctor. Resolved from AgentConfig or defaults."""
    veriloga_skills_path: Path | None = None
    behavioral_eval_path: Path | None = None
    config_path: Path | None = None


class Doctor:
    def __init__(self, config: DoctorConfig | None = None):
        self.config = config or DoctorConfig()

    def run(self, auto_fix: bool = False) -> int:
        """Run all checks. Returns exit code: 0 = ready, 1 = issues found."""
        results = self.check_all()
        fixed = 0

        print(doctor_header())

        for r in results:
            if r.status == "fail" and auto_fix and r.fixable and r.fix_func:
                print(doctor_check(r.name, "fail", r.message, fixing=True))
                try:
                    success = r.fix_func()
                    if success:
                        r.status = "pass"
                        r.message = "fixed"
                        fixed += 1
                        print(green(f"     ✓ done"))
                    else:
                        print(red(f"     ✗ fix failed"))
                except Exception as e:
                    print(red(f"     ✗ error: {e}"))
            else:
                print(doctor_check(r.name, r.status, r.message))

        passed = sum(1 for r in results if r.status == "pass")
        total = len(results)
        all_required_pass = all(
            r.status == "pass" for r in results if r.required
        )

        print(doctor_summary(passed, total, fixed, has_failures=not all_required_pass))

        if not all_required_pass:
            print("\n  Some required checks failed. Run with --fix to auto-repair.")
            return 1
        return 0

    def check_all(self) -> list[CheckResult]:
        checkers = [
            self._check_python_version,
            self._check_pip_deps,
            self._check_evas_cli,
            self._check_veriloga_skills,
            self._check_behavioral_eval,
            self._check_api_key,
            self._check_api_connectivity,
            self._check_tasks_integrity,
            self._check_git_repos,
            self._check_config_file,
        ]
        results = []
        for checker in checkers:
            try:
                result = checker()
            except Exception as e:
                result = CheckResult(
                    name=checker.__name__.replace("_check_", "").replace("_", " "),
                    status="fail",
                    message=str(e),
                    required=False,
                )
            results.append(result)
        return results

    # ─── Individual checks ────────────────────────────────────

    def _check_python_version(self) -> CheckResult:
        vi = sys.version_info
        ok = vi >= (3, 11)
        return CheckResult(
            name="Python",
            status="pass" if ok else "fail",
            message=f"{vi.major}.{vi.minor}.{vi.micro}" if ok else f"{vi.major}.{vi.minor}.{vi.micro} (need >= 3.11)",
            fixable=False,
        )

    def _check_pip_deps(self) -> CheckResult:
        required = ["evas", "numpy", "matplotlib", "yaml"]
        missing = []
        for pkg in required:
            try:
                __import__(pkg.replace("-", "_"))
            except ImportError:
                missing.append(pkg)
        if not missing:
            return CheckResult(name="pip deps", status="pass",
                              message="evas-sim, numpy, matplotlib, pyyaml")

        def _fix():
            for pkg in missing:
                name = {"evas": "evas-sim", "yaml": "pyyaml"}.get(pkg, pkg)
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            return True

        return CheckResult(
            name="pip deps", status="fail",
            message=f"missing: {', '.join(missing)}",
            fixable=True, fix_func=_fix,
        )

    def _check_evas_cli(self) -> CheckResult:
        # Check multiple locations: PATH, conda env bin, pip show
        candidates = ["evas"]
        # Also check next to python interpreter (conda env)
        py_dir = Path(sys.executable).parent
        evas_in_py_dir = py_dir / "evas.exe"
        if evas_in_py_dir.exists():
            candidates.append(str(evas_in_py_dir))

        for cmd in candidates:
            try:
                result = subprocess.run(
                    [cmd, "--version"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    version = result.stdout.strip() or "installed"
                    return CheckResult(name="EVAS CLI", status="pass", message=version)
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                continue

        # Also check via import
        try:
            import evas
            ver = getattr(evas, "__version__", "installed")
            return CheckResult(name="EVAS CLI", status="pass",
                              message=f"importable (v{ver})")
        except ImportError:
            pass

        def _fix():
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "evas-sim"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            return True

        return CheckResult(
            name="EVAS CLI", status="fail", message="not found",
            fixable=True, fix_func=_fix,
        )

    def _check_veriloga_skills(self) -> CheckResult:
        path = self._resolve_skills_path()
        if path and path.exists():
            categories = path / "veriloga" / "references" / "categories"
            if categories.exists():
                count = len(list(categories.glob("*.md")))
                return CheckResult(
                    name="veriloga-skills", status="pass",
                    message=f"{path} ({count} categories)",
                )
            return CheckResult(
                name="veriloga-skills", status="fail",
                message=f"found {path} but missing categories/",
            )
        return CheckResult(
            name="veriloga-skills", status="fail",
            message=f"not found at {path or 'unknown'}",
        )

    def _check_behavioral_eval(self) -> CheckResult:
        path = self._resolve_eval_path()
        if path and path.exists():
            tasks = path / "tasks"
            if tasks.exists():
                families = [d.name for d in tasks.iterdir() if d.is_dir() and not d.name.startswith(".")]
                return CheckResult(
                    name="behavioral-va-eval", status="pass",
                    message=f"{path} (families: {', '.join(families[:4])})",
                )
            return CheckResult(
                name="behavioral-va-eval", status="fail",
                message=f"found {path} but missing tasks/",
            )
        return CheckResult(
            name="behavioral-va-eval", status="fail",
            message=f"not found",
        )

    def _check_api_key(self) -> CheckResult:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if anthropic_key:
            masked = anthropic_key[:10] + "***" if len(anthropic_key) > 10 else "***"
            return CheckResult(name="API Key", status="pass",
                              message=f"ANTHROPIC_API_KEY={masked}")
        if openai_key:
            masked = openai_key[:10] + "***" if len(openai_key) > 10 else "***"
            return CheckResult(name="API Key", status="pass",
                              message=f"OPENAI_API_KEY={masked}")
        return CheckResult(
            name="API Key", status="fail",
            message="neither ANTHROPIC_API_KEY nor OPENAI_API_KEY set. Set in .env or environment.",
            fixable=False,
        )

    def _check_api_connectivity(self) -> CheckResult:
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        if not anthropic_key and not openai_key:
            return CheckResult(
                name="LLM connectivity", status="warn",
                message="skipped (no API key configured)",
                required=False,
            )

        if anthropic_key:
            try:
                import anthropic
                client = anthropic.Anthropic(api_key=anthropic_key)
                # minimal API call to verify key works
                client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1,
                    messages=[{"role": "user", "content": "hi"}],
                )
                return CheckResult(
                    name="LLM connectivity", status="pass",
                    message="anthropic: OK",
                )
            except Exception as e:
                return CheckResult(
                    name="LLM connectivity", status="fail",
                    message=f"anthropic: {_short_error(e)}",
                    required=False,
                )

        if openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)
                client.models.list()
                return CheckResult(
                    name="LLM connectivity", status="pass",
                    message="openai: OK",
                )
            except Exception as e:
                return CheckResult(
                    name="LLM connectivity", status="fail",
                    message=f"openai: {_short_error(e)}",
                    required=False,
                )

        return CheckResult(
            name="LLM connectivity", status="warn",
            message="skipped",
            required=False,
        )

    def _check_tasks_integrity(self) -> CheckResult:
        path = self._resolve_eval_path()
        if not path or not path.exists():
            return CheckResult(name="Tasks", status="fail",
                              message="behavioral-veriloga-eval path not configured")

        # Scan ALL meta.json files recursively (covers tasks/ + benchmark-v2/tasks/)
        gold_count = 0
        task_count = 0
        complete_count = 0
        for meta_path in sorted(path.rglob("meta.json")):
            task_dir = meta_path.parent
            task_count += 1
            has_gold = (task_dir / "gold").is_dir()
            has_checks = (task_dir / "checks.yaml").exists()
            if has_gold:
                gold_count += 1
            if has_gold or has_checks:
                complete_count += 1

        if task_count == 0:
            return CheckResult(name="Tasks", status="fail",
                              message="no meta.json found in eval repo")
        return CheckResult(
            name="Tasks", status="pass" if gold_count > 0 else "warn",
            message=f"{complete_count} complete tasks ({gold_count} with gold/) out of {task_count} total",
            required=complete_count == 0,
        )

    def _check_git_repos(self) -> CheckResult:
        repos = []
        for label, path in [
            ("veriloga-skills", self._resolve_skills_path()),
            ("behavioral-va-eval", self._resolve_eval_path()),
        ]:
            if path and path.exists():
                git_dir = path / ".git"
                if git_dir.exists():
                    repos.append(label)
        if repos:
            return CheckResult(
                name="Git repos", status="pass",
                message=f"{len(repos)} repos OK",
                required=False,
            )
        return CheckResult(
            name="Git repos", status="warn",
            message="no git repos detected",
            required=False,
        )

    def _check_config_file(self) -> CheckResult:
        config_path = self.config.config_path
        if config_path and config_path.exists():
            return CheckResult(
                name="Config", status="pass",
                message=str(config_path),
            )

        # Try default location
        default = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
        if default.exists():
            return CheckResult(
                name="Config", status="pass",
                message=f"config/default.yaml",
            )

        def _fix():
            default.parent.mkdir(parents=True, exist_ok=True)
            default.write_text(DEFAULT_CONFIG_CONTENT, encoding="utf-8")
            return True

        return CheckResult(
            name="Config", status="fail",
            message="no config file found",
            fixable=True, fix_func=_fix,
        )

    # ─── Path resolution ──────────────────────────────────────

    def _resolve_skills_path(self) -> Path | None:
        if self.config.veriloga_skills_path:
            return self.config.veriloga_skills_path
        # Search relative to project root
        candidates = [
            Path.cwd() / ".." / "veriloga-skills",
            Path.cwd() / "veriloga-skills",
            Path.home() / "WorkSpace" / "VerilogA" / "veriloga-skills",
        ]
        for c in candidates:
            if c.resolve().exists():
                return c.resolve()
        return None

    def _resolve_eval_path(self) -> Path | None:
        if self.config.behavioral_eval_path:
            return self.config.behavioral_eval_path
        candidates = [
            Path.cwd() / ".." / "behavioral-veriloga-eval",
            Path.cwd() / "behavioral-veriloga-eval",
            Path.home() / "WorkSpace" / "VerilogA" / "behavioral-veriloga-eval",
        ]
        for c in candidates:
            if c.resolve().exists():
                return c.resolve()
        return None


def _short_error(e: Exception) -> str:
    msg = str(e)
    return msg[:80] + "..." if len(msg) > 80 else msg


DEFAULT_CONFIG_CONTENT = """\
# vaEvas Agent default configuration
llm:
  provider: anthropic
  model: claude-sonnet-4-6
  temperature: 0.0
  repair_temperature: 0.3
  max_tokens: 4096
  top_p: 1.0
  timeout: 120

paths:
  veriloga_skills: ../veriloga-skills
  behavioral_eval: ../behavioral-veriloga-eval

loop:
  max_rounds: 3
  stall_limit: 2
  regress_limit: 2

skills:
  enabled: true
  max_chars: 3000
  categories_dir: ../veriloga-skills/veriloga/references/categories

output:
  dir: ./output
  save_artifacts: true
  save_prompts: true
"""
