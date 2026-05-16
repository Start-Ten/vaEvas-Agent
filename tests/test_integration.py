"""Integration test: full pipeline walkthrough without real LLM calls.
Verifies:
  1. Task discovery (CLI list)
  2. Config loading
  3. Skill matching
  4. Prompt construction (system + task + repair)
  5. EVAS scoring (real, on gold answer)
  6. Loop controller state machine
  7. Agent.run() with mock LLM
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure we can import from vaevas_agent
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vaevas_agent.config import AgentConfig, load_config
from vaevas_agent.skills.manager import SkillManager
from vaevas_agent.skills.loader import resolve_skills_root
from vaevas_agent.loop.state import LoopState, RoundResult, TaskContext
from vaevas_agent.loop.terminator import Terminator
from vaevas_agent.loop.controller import LoopController
from vaevas_agent.prompts.pipeline import (
    build_system_prompt,
    build_task_prompt,
    build_repair_prompt,
)
from vaevas_agent.prompts.system import SYSTEM_PROMPT


def find_first_gold_task() -> tuple[str, Path] | None:
    """Find the first available task with a gold/ directory."""
    candidates = [
        Path.cwd() / ".." / "behavioral-veriloga-eval" / "tasks",
        Path.cwd() / "behavioral-veriloga-eval" / "tasks",
    ]
    for base in candidates:
        if not base.resolve().exists():
            continue
        for meta_path in sorted(base.rglob("meta.json")):
            task_dir = meta_path.parent
            if (task_dir / "gold").is_dir():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                tid = meta.get("task_id") or meta.get("id") or task_dir.name
                return tid, task_dir
    return None


def test_task_discovery():
    """Test 1: Verify we can find tasks."""
    result = find_first_gold_task()
    if result is None:
        print("  SKIP: no tasks found (need behavioral-veriloga-eval repo)")
        return None
    tid, task_dir = result
    print(f"  OK: found task '{tid}' at {task_dir}")
    return tid, task_dir


def test_config():
    """Test 2: Config loads correctly with defaults."""
    config = load_config()
    assert config.llm.provider == "anthropic", f"bad provider: {config.llm.provider}"
    assert config.loop.max_rounds == 3, f"bad max_rounds: {config.loop.max_rounds}"
    print(f"  OK: provider={config.llm.provider}, model={config.llm.model}, max_rounds={config.loop.max_rounds}")


def test_skill_matching(task_id: str):
    """Test 3: Skill matching by keyword works."""
    mgr = SkillManager(skills_root=resolve_skills_root())
    if not mgr.available:
        print("  SKIP: veriloga-skills not found")
        return
    match = mgr.match(task_id)
    print(f"  OK: {mgr.category_count} categories available, match='{match.name if match else 'none'}'")
    context = mgr.build_skill_context(task_id)
    if context:
        print(f"  OK: skill context: {len(context)} chars")
    else:
        print(f"  OK: no skill context for this task (expected if no keyword match)")


def test_prompt_construction(task_dir: Path):
    """Test 4: All prompt builders produce valid output."""
    # System prompt
    sys_prompt = build_system_prompt()
    assert len(sys_prompt) > 500, f"system prompt too short: {len(sys_prompt)}"
    assert "Verilog-A" in sys_prompt
    print(f"  OK: system prompt: {len(sys_prompt)} chars")

    # Task prompt
    task_prompt = build_task_prompt(task_dir)
    assert len(task_prompt) > 200, f"task prompt too short: {len(task_prompt)}"
    print(f"  OK: task prompt: {len(task_prompt)} chars")

    # Repair prompt (mock EVAS result)
    sample_dir = task_dir / "gold"  # use gold as fake sample for prompt test
    evas_result = {
        "status": "FAIL_SIM_CORRECTNESS",
        "scores": {"dut_compile": 1.0, "tb_compile": 1.0, "sim_correct": 0.0, "weighted_total": 0.667},
        "evas_notes": ["unique_codes=3 (expected=16)", "too_few_edges"],
    }
    repair_prompt = build_repair_prompt(task_dir, sample_dir, evas_result, history=[])
    assert len(repair_prompt) > 300, f"repair prompt too short: {len(repair_prompt)}"
    assert "EVAS Result" in repair_prompt
    assert "FAIL_SIM_CORRECTNESS" in repair_prompt
    print(f"  OK: repair prompt: {len(repair_prompt)} chars")


def test_evas_scoring(task_dir: Path):
    """Test 5: EVAS can actually score a gold answer."""
    gold_dir = task_dir / "gold"
    if not gold_dir.exists():
        print("  SKIP: no gold directory")
        return

    # Find gold files
    va_files = sorted(gold_dir.glob("*.va"))
    scs_files = sorted(gold_dir.glob("*.scs"))
    if not va_files or not scs_files:
        print(f"  SKIP: gold has {len(va_files)} .va, {len(scs_files)} .scs")
        return

    print(f"  gold: {va_files[0].name} + {scs_files[0].name}")

    # Run EVAS via subprocess
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for f in va_files:
            shutil.copy2(f, tmp_path / f.name)
        for f in scs_files:
            shutil.copy2(f, tmp_path / f.name)

        scs = tmp_path / scs_files[0].name
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        try:
            proc = subprocess.run(
                ["evas", "simulate", str(scs), "-o", str(output_dir)],
                capture_output=True, text=True, timeout=120,
                cwd=str(tmp_path),
            )
            if proc.returncode == 0:
                csv_file = output_dir / "tran.csv"
                csv_size = csv_file.stat().st_size if csv_file.exists() else 0
                print(f"  OK: EVAS PASS — tran.csv: {csv_size} bytes")
                # Check CSV has expected structure
                if csv_file.exists():
                    import csv as csv_mod
                    with open(csv_file) as f:
                        reader = csv_mod.reader(f)
                        header = next(reader, [])
                        row_count = sum(1 for _ in reader)
                    print(f"  OK: CSV header={header[:5]}... rows={row_count}")
            else:
                stderr_tail = proc.stderr.strip()[-300:] if proc.stderr else ""
                print(f"  FAIL: EVAS returncode={proc.returncode}")
                print(f"  stderr: {stderr_tail}")
        except FileNotFoundError:
            print("  SKIP: evas CLI not installed in current env")
        except subprocess.TimeoutExpired:
            print("  FAIL: EVAS timeout")


def test_loop_state_machine():
    """Test 6: Loop state transitions and terminator logic."""
    ctx = TaskContext(
        task_id="test",
        task_dir=Path("/fake"),
        meta={},
        family="end-to-end",
        category="test",
        required_axes=["dut_compile", "tb_compile", "sim_correct"],
        gold_dir=None,
    )

    # Test PASS terminates
    state = LoopState(task_context=ctx)
    state.add_result(RoundResult(
        round_idx=0, sample_dir=Path("/fake/r0"),
        status="PASS",
        scores={"dut_compile": 1.0, "tb_compile": 1.0, "sim_correct": 1.0, "weighted_total": 1.0},
        evas_notes=[],
    ))
    t = Terminator(max_rounds=3)
    stopped, reason = t.should_stop(state)
    assert stopped and reason == "PASS", f"expected PASS stop, got: {stopped}, {reason}"
    print("  OK: PASS terminates loop")

    # Test max_rounds
    state2 = LoopState(task_context=ctx)
    for i in range(3):
        state2.add_result(RoundResult(
            round_idx=i, sample_dir=Path(f"/fake/r{i}"),
            status="FAIL_SIM_CORRECTNESS",
            scores={"dut_compile": 1.0, "tb_compile": 1.0, "sim_correct": 0.0, "weighted_total": 0.667},
            evas_notes=["test"],
        ))
    stopped2, reason2 = t.should_stop(state2)
    assert stopped2 and "max_rounds" in reason2, f"expected max_rounds stop, got: {reason2}"
    print("  OK: max_rounds terminates loop")

    # Test best-so-far selection
    state3 = LoopState(task_context=ctx)
    r0 = RoundResult(round_idx=0, sample_dir=Path("/fake/r0"),
        status="FAIL_DUT_COMPILE",
        scores={"dut_compile": 0.0, "tb_compile": 1.0, "sim_correct": 0.0, "weighted_total": 0.333},
        evas_notes=[])
    state3.add_result(r0)
    assert state3.best_result == r0
    r1 = RoundResult(round_idx=1, sample_dir=Path("/fake/r1"),
        status="FAIL_SIM_CORRECTNESS",
        scores={"dut_compile": 1.0, "tb_compile": 1.0, "sim_correct": 0.0, "weighted_total": 0.667},
        evas_notes=[])
    state3.add_result(r1)
    assert state3.best_result == r1, "R1 should be better than R0"
    assert r1.transition == "improved"
    print("  OK: best-so-far selection works (R0 fail_compile → R1 fail_behavior = improved)")


def test_agent_mock_run(task_dir: Path, task_id: str):
    """Test 7: Agent.run() with mocked LLM (no API call)."""
    from vaevas_agent.agent import Agent, _build_task_context

    os.environ["ANTHROPIC_API_KEY"] = "test-key"  # fake key for mock

    config = AgentConfig()
    config.loop.max_rounds = 1  # single round

    # Use a gold .va file as the mock LLM "generated" output
    gold_dir = task_dir / "gold"
    va_files = sorted(gold_dir.glob("*.va"))
    scs_files = sorted(gold_dir.glob("*.scs"))

    if not va_files or not scs_files:
        print("  SKIP: need gold .va and .scs for mock test")
        return

    gold_va = va_files[0].read_text(encoding="utf-8")
    gold_scs = scs_files[0].read_text(encoding="utf-8") if scs_files else ""

    # Mock LLM response to return gold code
    mock_llm_text = f"```verilog-a\n{gold_va}\n```\n\n```spectre\n{gold_scs}\n```"

    agent = Agent(config)

    # Replace _generate to return mock
    original_generate = agent._generate

    def mock_generate(round_idx, context, iter_ctx):
        return {
            "response_text": mock_llm_text,
            "input_tokens": 100,
            "output_tokens": 200,
            "elapsed_ms": 100,
        }

    agent._generate = mock_generate

    try:
        print(f"  Running Agent with mock LLM (max_rounds=1)...")
        history = agent.run(task_id, task_dir)
        print(f"  OK: Agent.run() completed with {len(history)} rounds")
        for r in history:
            print(f"    R{r.round_idx}: status={r.status} scores={r.scores}")
    except Exception as e:
        print(f"  WARN: Agent.run() error: {e}")
        # This is expected if EVAS is not installed or gold answer doesn't match
        import traceback
        traceback.print_exc()
    finally:
        agent._generate = original_generate


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("vaEvas Agent Integration Test")
    print("=" * 60)

    result = test_task_discovery()
    if result is None:
        print("\nCannot continue without tasks. Ensure behavioral-veriloga-eval repo is present.")
        sys.exit(0)

    task_id, task_dir = result
    print(f"\n--- Testing with: {task_id} ---\n")

    print("[test 2] Config loading")
    test_config()

    print("\n[test 3] Skill matching")
    test_skill_matching(task_id)

    print("\n[test 4] Prompt construction")
    test_prompt_construction(task_dir)

    print("\n[test 5] EVAS scoring (gold answer)")
    test_evas_scoring(task_dir)

    print("\n[test 6] Loop state machine")
    test_loop_state_machine()

    print("\n[test 7] Agent mock run")
    test_agent_mock_run(task_dir, task_id)

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)