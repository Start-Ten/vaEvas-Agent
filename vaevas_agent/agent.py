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

import shutil
import subprocess
import tempfile
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

    # v3 task: read task.toml for metadata
    if not meta_path.exists():
        toml_path = task_dir / "task.toml"
        if toml_path.exists():
            v3_meta = _read_v3_toml_meta(toml_path)
            family = v3_meta.get("family", family)
            category = v3_meta.get("category", category)
            required_axes = v3_meta.get("scoring", required_axes)
            # v3 uses solution/ as gold
            if gold_dir is None:
                sol_dir = task_dir / "solution"
                if sol_dir.is_dir():
                    gold_dir = sol_dir

    return TaskContext(
        task_id=task_id,
        task_dir=task_dir,
        meta=meta,
        family=family,
        category=category,
        required_axes=required_axes,
        gold_dir=gold_dir,
    )


def _read_v3_toml_meta(toml_path: Path) -> dict:
    """Parse basic metadata from a v3 task.toml file."""
    meta: dict = {}
    form_map = {"dut": "spec-to-va", "bugfix": "bugfix",
                "tb": "tb-generation", "e2e": "end-to-end"}
    try:
        for line in toml_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("["):
                continue
            if "=" not in s:
                continue
            k, _, v = s.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "form":
                meta["family"] = form_map.get(v, "spec-to-va")
            elif k == "category":
                meta["category"] = v
            elif k == "difficulty":
                meta["difficulty"] = v
            elif k == "level":
                meta["level"] = v
    except (OSError, UnicodeDecodeError):
        pass
    return meta


def _run_v3_evas_score(sample_dir: Path, context: TaskContext) -> dict:
    """Score a v3 task by staging solution + hidden testbench and running EVAS.

    For v3 tasks the scoring pipeline is:
      1. Copy generated ``.va`` files from *sample_dir* into a staging area.
      2. Copy the hidden testbench from ``test_hidden/hidden.scs`` (if it
         exists) or fall back to ``test_visible/visible.scs``.
      3. Copy any helper ``.va`` files referenced by the testbench from the
         task ``solution/`` or ``starter/`` directories.
      4. Run ``evas_simulate`` via the EVAS Python API.
    """
    task_dir = context.task_dir
    evas_output = sample_dir / "evas_output"
    evas_output.mkdir(parents=True, exist_ok=True)

    # Locate the right testbench: hidden > visible > tests subdirs
    tb_candidates = [
        task_dir / "test_hidden" / "hidden.scs",
        task_dir / "test_visible" / "visible.scs",
    ]
    tb_path = None
    for c in tb_candidates:
        if c.exists():
            tb_path = c
            break
    # Fallback: search tests/ subdirectories for any .scs file
    if tb_path is None:
        for subdir in ["test_hidden", "test_visible"]:
            scs_files = sorted((task_dir / subdir).rglob("*.scs"))
            if scs_files:
                tb_path = scs_files[0]
                break
    if tb_path is None:
        return {
            "status": "FAIL_INFRA",
            "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                      "sim_correct": 0.0, "weighted_total": 0.0},
            "evas_notes": ["no testbench found (test_hidden/hidden.scs or test_visible/visible.scs)"],
        }

    # Stage all files in a temp directory
    with tempfile.TemporaryDirectory(prefix=f"v3score_{context.task_id}_") as tmp:
        tmp_path = Path(tmp)

        # 1. Copy generated .va files
        va_files = sorted(sample_dir.glob("*.va"))
        for f in va_files:
            shutil.copy2(f, tmp_path / f.name)

        # 2. Copy the testbench
        shutil.copy2(tb_path, tmp_path / tb_path.name)

        # 3. Copy any helper .va files referenced by the testbench
        #    (search task solution/ and starter/ as fallback)
        tb_text = tb_path.read_text(encoding="utf-8", errors="ignore")
        import re
        for match in re.finditer(r'ahdl_include\s+"([^"]+)"', tb_text):
            inc_name = match.group(1)
            inc_path = tmp_path / inc_name
            if inc_path.exists():
                continue
            # Search solution/, starter/, and test dirs
            for search_dir in [task_dir / "solution", task_dir / "starter",
                               task_dir / "test_visible", task_dir / "test_hidden"]:
                candidate = search_dir / inc_name
                if candidate.exists():
                    shutil.copy2(candidate, inc_path)
                    break
                # Also search subdirectories
                for found in sorted(search_dir.rglob(inc_name)):
                    shutil.copy2(found, inc_path)
                    break
                if inc_path.exists():
                    break

        # 4. Run EVAS simulation
        scs_path = tmp_path / tb_path.name
        try:
            from evas.netlist.runner import evas_simulate
            sim_ok = evas_simulate(str(scs_path), output_dir=str(evas_output))
        except ImportError:
            # Fallback: evas CLI
            try:
                proc = subprocess.run(
                    ["evas", "simulate", str(scs_path), "-o", str(evas_output)],
                    capture_output=True, text=True, timeout=180, cwd=str(tmp_path),
                )
                sim_ok = proc.returncode == 0
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

        if sim_ok:
            return {
                "status": "PASS",
                "scores": {"dut_compile": 1.0, "tb_compile": 1.0,
                          "sim_correct": 1.0, "weighted_total": 1.0},
                "evas_notes": ["v3 EVAS simulation succeeded (basic pass)"],
            }
        else:
            return {
                "status": "FAIL_DUT_COMPILE",
                "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                          "sim_correct": 0.0, "weighted_total": 0.0},
                "evas_notes": ["v3 EVAS simulation failed"],
            }


def _run_evas_score(sample_dir: Path, context: TaskContext) -> dict:
    """Run EVAS evaluation for the generated candidate.

    Tries v3 scoring first (``task.toml``-based), then the full
    ``score_one_task`` pipeline from behavioral-veriloga-eval, then
    a minimal ``evas simulate`` CLI invocation as last resort.
    """
    # ── v3 scoring path ────────────────────────────────────
    if (context.task_dir / "task.toml").exists():
        try:
            return _run_v3_evas_score(sample_dir, context)
        except Exception as e:
            return {
                "status": "FAIL_INFRA",
                "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                          "sim_correct": 0.0, "weighted_total": 0.0},
                "evas_notes": [f"v3 score error: {e}"],
            }

    # ── v1 scoring: try score_one_task ─────────────────────
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
