"""Loop state types — TaskContext, RoundResult, LoopState."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TaskContext:
    """Immutable context for a single benchmark task."""
    task_id: str
    task_dir: Path
    meta: dict
    family: str                     # end-to-end | spec-to-va | bugfix | tb-generation
    category: str
    required_axes: list[str]        # e.g., ["dut_compile", "tb_compile", "sim_correct"]
    gold_dir: Path | None


@dataclass
class RoundResult:
    """Result of a single agent round (generate + evaluate)."""
    round_idx: int
    sample_dir: Path                # where generated files live
    status: str                     # PASS / FAIL_DUT_COMPILE / FAIL_TB_COMPILE / FAIL_SIM_CORRECTNESS / FAIL_INFRA
    scores: dict                    # {dut_compile, tb_compile, sim_correct, weighted_total}
    evas_notes: list[str]           # raw EVAS diagnostic notes
    metrics: dict = field(default_factory=dict)  # extracted key=value metrics
    failure_subtype: str = ""       # observability_contract / simulation_artifact / behavior_semantic
    transition: str = ""            # improved / regressed / stalled / lateral (relative to best)
    comparison: str = ""            # human-readable comparison with best result
    generation_meta: dict = field(default_factory=dict)  # tokens, timing, etc.
    evas_timing: dict = field(default_factory=dict)


@dataclass
class LoopState:
    """Full state of a repair loop for one task."""
    task_context: TaskContext
    history: list[RoundResult] = field(default_factory=list)
    best_result: RoundResult | None = None
    current_round: int = 0

    def last_result(self) -> RoundResult | None:
        return self.history[-1] if self.history else None

    def is_pass(self) -> bool:
        if self.best_result is None:
            return False
        return self.best_result.status == "PASS"

    def add_result(self, result: RoundResult) -> None:
        self.history.append(result)
        self.current_round = len(self.history)  # next round index

        if self.best_result is None:
            self.best_result = result
        elif _result_rank(result) > _result_rank(self.best_result):
            result.transition = "improved"
            result.comparison = _compare_results(result, self.best_result)
            self.best_result = result
        elif _result_rank(result) == _result_rank(self.best_result):
            result.transition = "lateral"
            result.comparison = "same weighted score, different failure surface"
        else:
            result.transition = "regressed"
            result.comparison = _compare_results(self.best_result, result)


def _result_rank(r: RoundResult) -> tuple:
    return (
        1 if r.status == "PASS" else 0,
        float(r.scores.get("weighted_total", 0.0)),
        float(r.scores.get("sim_correct", 0.0)),
        float(r.scores.get("tb_compile", 0.0)),
        float(r.scores.get("dut_compile", 0.0)),
    )


def _compare_results(better: RoundResult, worse: RoundResult) -> str:
    b_wt = better.scores.get("weighted_total", 0.0)
    w_wt = worse.scores.get("weighted_total", 0.0)
    return f"weighted_total {w_wt:.3f} → {b_wt:.3f}"
