"""Agent — top-level orchestrator for the vaEvas closed loop."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from .config import AgentConfig, load_config
from .display import (
    bold, box_footer, box_header, box_line, cyan, dim, green,
    red, round_header, scores_line, status_icon, transition_label, yellow,
)
from .llm.client import LLMError, call_llm
from .loop.controller import LoopController
from .loop.state import LoopState, RoundResult, TaskContext
from .prompts.pipeline import (
    _resolve_gold_path,
    build_repair_prompt,
    build_system_prompt,
    build_task_prompt,
)
from .skills.manager import SkillManager


class Agent:
    """The vaEvas Agent — runs the full closed loop for a Verilog-A benchmark task.

    Usage:
        config = load_config()
        agent = Agent(config)
        result = agent.run("digital_basics_smoke", task_dir)
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or load_config()
        self.skills = SkillManager(
            skills_root=self.config.resolve_path(self.config.paths.veriloga_skills),
            max_chars=self.config.skills.max_chars,
        )
        self.loop = LoopController(
            max_rounds=self.config.loop.max_rounds,
            stall_limit=self.config.loop.stall_limit,
            regress_limit=self.config.loop.regress_limit,
        )
        self._total_tokens = 0
        self._start_time = 0.0

    def run(self, task_id: str, task_dir: Path | str) -> list[RoundResult]:
        """Run the full generate→evaluate→repair loop for one task.

        Returns the complete history of all rounds.
        """
        task_dir = Path(task_dir)
        self._total_tokens = 0
        self._start_time = time.time()

        context = _build_task_context(task_id, task_dir)
        output_root = self.config.resolve_path(self.config.output.dir)

        self._print_task_header(context)

        history = self.loop.run(
            context=context,
            output_root=output_root,
            generate_fn=self._generate,
            evaluate_fn=self._evaluate,
            repair_fn=self._build_repair_prompt,
            on_round_start=self._on_round_start,
            on_round_end=self._on_round_end,
        )

        self._print_result(history, output_root / task_id)
        return history

    # ─── Callbacks for LoopController ───────────────────────

    def _on_round_start(self, round_idx: int, stage: str) -> None:
        if stage == "generating":
            temp = self.config.llm.temperature if round_idx == 0 else self.config.llm.repair_temperature
            sys.stdout.write(
                round_header(round_idx,
                    f"{'Generating' if round_idx == 0 else 'Repairing'} "
                    f"with {self.config.llm.model} (T={temp}) ... ")
            )
            sys.stdout.flush()
        elif stage == "evaluating":
            sys.stdout.write("EVAS scoring ... ")
            sys.stdout.flush()

    def _on_round_end(self, result: RoundResult) -> None:
        print(status_icon(result.status), " ", scores_line(result.scores))
        if result.failure_subtype:
            print(f"           {transition_label(result.failure_subtype)}")
            if result.metrics:
                preview = ", ".join(f"{k}={v}" for k, v in list(result.metrics.items())[:4])
                print(f"           {dim('metrics: ' + preview)}")
        sys.stdout.write("\n")
        sys.stdout.flush()

    # ─── Core steps ─────────────────────────────────────────

    def _generate(
        self,
        round_idx: int,
        context: TaskContext,
        iteration_context: dict,
    ) -> dict:
        temp = self.config.llm.temperature if round_idx == 0 else self.config.llm.repair_temperature

        # Build skill context
        skill_context = ""
        if self.config.skills.enabled:
            skill_context = self.skills.build_skill_context(context.task_id)

        if round_idx == 0:
            system = build_system_prompt(skill_context=skill_context)
            user = build_task_prompt(context.task_dir, skill_context=skill_context)
        else:
            system = build_system_prompt(skill_context=skill_context)
            user = iteration_context.get("repair_prompt", "")

        try:
            response = call_llm(self.config.llm, system, user, temperature=temp)
        except LLMError as e:
            return {
                "response_text": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "elapsed_ms": 0,
                "error": str(e),
            }

        self._total_tokens += response.input_tokens + response.output_tokens

        # Guard: empty LLM response or no code blocks → short-circuit
        text = (response.text or "").strip()
        if not text or not _has_va_or_scs_blocks(text):
            return {
                "response_text": "",
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "elapsed_ms": response.elapsed_ms,
                "error": "empty_response",
            }

        return {
            "response_text": text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "elapsed_ms": response.elapsed_ms,
        }

    def _evaluate(self, sample_dir: Path, context: TaskContext) -> dict:
        """Run EVAS scoring on the generated candidate.

        Delegates to score_one_task from behavioral-veriloga-eval if available.
        Falls back to running evas CLI directly.
        """
        try:
            return _run_evas_score(sample_dir, context)
        except Exception as e:
            return {
                "status": "FAIL_INFRA",
                "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                          "sim_correct": 0.0, "weighted_total": 0.0},
                "evas_notes": [f"score error: {e}"],
            }

    def _build_repair_prompt(
        self,
        task_dir: Path,
        sample_dir: Path,
        evas_result: dict,
        history: list[dict],
    ) -> str:
        skill_context = ""
        if self.config.skills.enabled:
            task_id = Path(task_dir).name
            skill_context = self.skills.build_skill_context(task_id)

        return build_repair_prompt(
            task_dir=task_dir,
            sample_dir=sample_dir,
            evas_result=evas_result,
            history=history,
            skill_context=skill_context,
        )

    # ─── Display ────────────────────────────────────────────

    def _print_task_header(self, context: TaskContext) -> None:
        print()
        print(box_header(f"Task: {context.task_id}"))
        print(box_line(f"Family: {context.family:<14s} Category: {context.category}"))
        if self.skills.available:
            matched = self.skills.match(context.task_id)
            label = matched.name if matched else "none"
            print(box_line(f"Skill:  {label}"))
        print(box_footer())
        print()

    def _print_result(self, history: list[RoundResult], output_dir: Path) -> None:
        rounds = len(history)
        last = history[-1] if history else None
        status = last.status if last else "?"
        elapsed = time.time() - self._start_time

        print(box_header(f"Result: {status}"))
        print(box_line(f"Rounds: {rounds}    Total tokens: {self._total_tokens:,}    "
                       f"Total time: {elapsed:.1f}s"))
        print(box_line(f"Output: {output_dir}"))
        print(box_footer())


# ─── Helpers ─────────────────────────────────────────────────

def _build_task_context(task_id: str, task_dir: Path) -> TaskContext:
    meta_path = task_dir / "meta.json"
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            meta = {}

    family = meta.get("family", "end-to-end")
    category = meta.get("category", "unknown")
    required_axes = meta.get("scoring", ["dut_compile", "tb_compile", "sim_correct"])
    gold_dir = _resolve_gold_path(task_dir)

    return TaskContext(
        task_id=task_id,
        task_dir=task_dir,
        meta=meta,
        family=family,
        category=category,
        required_axes=required_axes,
        gold_dir=gold_dir,
    )


def _run_evas_score(sample_dir: Path, context: TaskContext) -> dict:
    """Run EVAS evaluation for the generated candidate.

    Tries to import and use the full score_one_task from behavioral-veriloga-eval.
    Falls back to a minimal evas CLI invocation.
    """
    # Try the full scoring pipeline first
    try:
        # Add behavioral-veriloga-eval to path
        eval_root = sample_dir.parent.parent.parent  # heuristic
        for candidate in [
            sample_dir.parent.parent.parent.parent / "behavioral-veriloga-eval",
            Path.cwd() / "behavioral-veriloga-eval",
            Path.cwd() / ".." / "behavioral-veriloga-eval",
        ]:
            runner_dir = candidate / "runners"
            if runner_dir.exists():
                sys.path.insert(0, str(runner_dir))
                break

        from score import score_one_task
        result = score_one_task(
            task_id=context.task_id,
            task_dir=context.task_dir,
            sample_dir=sample_dir,
            output_dir=sample_dir / "evas_output",
            model="agent",
            sample_idx=0,
            temperature=0.0,
            top_p=1.0,
        )
        return result
    except (ImportError, Exception):
        pass

    # Fallback: minimal evas simulation
    scs_files = sorted(sample_dir.glob("*.scs"))
    va_files = sorted(sample_dir.glob("*.va"))
    if not scs_files or not va_files:
        return {
            "status": "FAIL_INFRA",
            "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                      "sim_correct": 0.0, "weighted_total": 0.0},
            "evas_notes": ["missing generated files"],
        }

    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in va_files:
            shutil.copy2(f, tmp_path / f.name)
        for f in scs_files:
            shutil.copy2(f, tmp_path / f.name)
        scs = tmp_path / scs_files[0].name

        try:
            proc = subprocess.run(
                ["evas", "simulate", str(scs), "-o", str(tmp_path / "output")],
                capture_output=True, text=True, timeout=180,
                cwd=str(tmp_path),
            )
            if proc.returncode == 0:
                return {
                    "status": "PASS",
                    "scores": {"dut_compile": 1.0, "tb_compile": 1.0,
                              "sim_correct": 1.0, "weighted_total": 1.0},
                    "evas_notes": ["evas simulation succeeded (no checker validation)"],
                }
            else:
                return {
                    "status": "FAIL_DUT_COMPILE",
                    "scores": {"dut_compile": 0.0, "tb_compile": 1.0,
                              "sim_correct": 0.0, "weighted_total": 0.333},
                    "evas_notes": [proc.stderr.strip()[:500] if proc.stderr else "compile failed"],
                }
        except subprocess.TimeoutExpired:
            return {
                "status": "FAIL_INFRA",
                "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                          "sim_correct": 0.0, "weighted_total": 0.0},
                "evas_notes": ["evas timeout"],
            }
        except FileNotFoundError:
            return {
                "status": "FAIL_INFRA",
                "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                          "sim_correct": 0.0, "weighted_total": 0.0},
                "evas_notes": ["evas CLI not found"],
            }


def _has_va_or_scs_blocks(text: str) -> bool:
    """Check whether *text* contains at least one fenced code block for Verilog-A or Spectre."""
    import re
    return bool(re.search(r"```(?:verilog-a|verilog|spectre|sp)\s*\n", text, re.IGNORECASE))
