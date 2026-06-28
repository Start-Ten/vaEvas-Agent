"""Loop controller — state machine driving generate → evaluate → diagnose → repair."""
from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from .state import LoopState, RoundResult, TaskContext
from .terminator import Terminator


class LoopController:
    """Orchestrates the agent's generate→evaluate→repair cycle.

    Does NOT own LLM, skills, or config — those are injected by Agent.
    """

    def __init__(self, max_rounds: int = 3, stall_limit: int = 2, regress_limit: int = 2):
        self.terminator = Terminator(max_rounds, stall_limit, regress_limit)
        self._output_root: Path | None = None

    def run(
        self,
        context: TaskContext,
        *,
        output_root: Path,
        generate_fn,      # callable(round_idx, task_context, iteration_context) -> RoundResult
        evaluate_fn,      # callable(sample_dir, task_context) -> dict (evas_result)
        repair_fn,        # callable(sample_dir, evas_result, history) -> str (repair_prompt)
        on_round_start=None,   # optional callback(round_idx, stage)
        on_round_end=None,     # optional callback(result: RoundResult)
    ) -> list[RoundResult]:
        """Run the full loop. Returns complete history of all rounds."""
        self._output_root = output_root
        state = LoopState(task_context=context)
        iteration_context: dict = {}

        while True:
            round_idx = state.current_round
            stopped, reason = self.terminator.should_stop(state)
            if stopped:
                break

            if on_round_start:
                on_round_start(round_idx, "generating")

            # Step 1: Generate/Repair
            sample_dir = self._make_sample_dir(context.task_id, round_idx)

            try:
                if round_idx == 0:
                    # Initial generation
                    result = generate_fn(round_idx, context, iteration_context)
                else:
                    # Repair: build prompt from previous failure
                    last = state.last_result()
                    if last is None:
                        break
                    evas_result = {
                        "status": last.status,
                        "scores": last.scores,
                        "evas_notes": last.evas_notes,
                    }
                    history_dicts = [
                        {"round": r.round_idx, "status": r.status,
                         "scores": r.scores, "transition": r.transition}
                        for r in state.history
                    ]
                    repair_prompt = repair_fn(context.task_dir, last.sample_dir,
                                             evas_result, history_dicts)
                    iteration_context["repair_prompt"] = repair_prompt
                    result = generate_fn(round_idx, context, iteration_context)
            except Exception as e:
                result = {
                    "response_text": "",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "elapsed_ms": 0,
                    "error": f"generate callback failed: {e}",
                }

            # Write generated files to sample_dir
            _save_generated_files(result, sample_dir)

            # Step 2: Evaluate
            if on_round_start:
                on_round_start(round_idx, "evaluating")

            try:
                evas_result = evaluate_fn(sample_dir, context)
            except Exception as e:
                evas_result = {
                    "status": "FAIL_INFRA",
                    "scores": {"dut_compile": 0.0, "tb_compile": 0.0,
                              "sim_correct": 0.0, "weighted_total": 0.0},
                    "evas_notes": [f"evaluate callback failed: {e}"],
                }

            # Step 3: Diagnose + update state
            round_result = _build_round_result(round_idx, sample_dir, evas_result, result)
            state.add_result(round_result)

            if on_round_end:
                on_round_end(round_result)

            # Update iteration context for next round
            iteration_context["history"] = state.history
            iteration_context["last_result"] = round_result

        return state.history

    def _make_sample_dir(self, task_id: str, round_idx: int) -> Path:
        d = self._output_root / task_id / f"round_{round_idx}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _save_generated_files(result, sample_dir: Path) -> None:
    """Extract and save generated .va and .scs files from LLM response."""
    text = result.get("response_text", "") if isinstance(result, dict) else getattr(result, "response_text", "")

    if not text:
        return

    # Extract code blocks
    va_blocks = _extract_code_blocks(text, "verilog-a")
    scs_blocks = _extract_code_blocks(text, "spectre")

    for i, code in enumerate(va_blocks):
        module_name = _infer_module_name(code) or f"module_{i}"
        try:
            (sample_dir / f"{module_name}.va").write_text(code, encoding="utf-8")
        except OSError:
            pass  # best-effort write

    for i, code in enumerate(scs_blocks):
        tb_name = _infer_tb_name(code) or f"tb_generated_{i}"
        try:
            (sample_dir / f"{tb_name}.scs").write_text(code, encoding="utf-8")
        except OSError:
            pass  # best-effort write


def _extract_code_blocks(text: str, lang: str) -> list[str]:
    pattern = rf"```(?:{lang}|{'verilog' if lang == 'verilog-a' else lang})\s*\n(.*?)```"
    return [m.group(1).strip() for m in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE)]


def _infer_module_name(va_code: str) -> str | None:
    m = re.search(r"\bmodule\s+(\w+)", va_code)
    return m.group(1) if m else None


def _infer_tb_name(scs_code: str) -> str | None:
    m = re.search(r"Cell name:\s*(\S+)", scs_code)
    if m:
        return m.group(1)
    m = re.search(r"(tb_\w+)", scs_code)
    return m.group(1) if m else None


def _build_round_result(
    round_idx: int,
    sample_dir: Path,
    evas_result: dict,
    generation_meta: dict,
) -> RoundResult:
    notes = evas_result.get("evas_notes") or evas_result.get("notes") or []
    scores = evas_result.get("scores", {})
    status = evas_result.get("status", "FAIL_INFRA")

    # Classify failure subtype
    note_text = " ".join(str(n) for n in notes).lower()
    if any(m in note_text for m in ("missing ", "tran.csv missing", "too_few_", "insufficient_")):
        subtype = "observability_contract"
    elif any(m in note_text for m in ("timeout", "evas_timeout", "tb_not_executed")):
        subtype = "simulation_artifact"
    else:
        subtype = "behavior_semantic"

    # Extract metrics from notes
    metrics = {}
    for note in notes:
        for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s,;]+)", str(note)):
            key, val = match.group(1), match.group(2)
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = val

    return RoundResult(
        round_idx=round_idx,
        sample_dir=sample_dir,
        status=status,
        scores=scores,
        evas_notes=[str(n)[:200] for n in notes],
        metrics=metrics,
        failure_subtype=subtype,
        generation_meta=generation_meta,
        evas_timing=evas_result.get("timing", {}),
    )
