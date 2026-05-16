"""Prompt injection pipeline — assembles system + task + repair prompts for LLM calls.

This is the central file that controls WHAT the LLM sees at each stage of the loop.

Pipeline entry points:
  build_system_prompt()     → system message (rules + skill context)
  build_task_prompt()        → user message for Round 0 (generation)
  build_repair_prompt()      → user message for Round 1+ (repair with EVAS feedback)
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

from .system import SYSTEM_PROMPT


# ─── System prompt ───────────────────────────────────────────

def build_system_prompt(
    skill_context: str = "",
    extra_rules: list[str] | None = None,
) -> str:
    """Build the system prompt with optional skill context appended.

    Args:
        skill_context: Circuit-specific knowledge from SkillManager.
        extra_rules: Additional rules to append (e.g., for specific experiment conditions).
    """
    prompt = SYSTEM_PROMPT

    if skill_context:
        prompt += skill_context

    if extra_rules:
        prompt += "\n\n" + "\n".join(extra_rules)

    return prompt


# ─── Task prompt (Round 0: generation) ───────────────────────

def build_task_prompt(
    task_dir: Path,
    *,
    include_public_contract: bool = True,
    skill_context: str = "",
) -> str:
    """Build the full user prompt for initial generation (Round 0).

    Injection order:
      1. prompt.md (human-written spec)
      2. Buggy DUT code (bugfix only)
      3. End-to-End output contract (end-to-end only)
      4. Strict EVAS validation contract (tran statement from gold TB)
      5. Module name contract (expected module name + file name)
      6. Verilog-A mandatory syntax rules
      7. DUT contract (tb-generation only)
      8. Public behavioral contract (checker expectations, non-gold)
      9. Observable CSV contract (checker-required columns)
      10. Skill context (if --include-skill)
    """
    meta = _read_meta(task_dir)
    task_id = meta.get("task_id") or meta.get("id") or task_dir.name
    family = meta.get("family", "end-to-end")
    prompt_md = _read_prompt_md(task_dir)

    # --- Layer 1: Buggy DUT (bugfix only) ---
    if family == "bugfix":
        buggy_code = _read_buggy_dut(task_dir)
        if buggy_code:
            prompt_md += f"\n\n## Buggy DUT\n\n```verilog-a\n{buggy_code}\n```\n"

    # --- Layer 2: End-to-End contract ---
    if family == "end-to-end":
        prompt_md += """
## End-To-End Output Contract (MANDATORY)

You MUST return both deliverables:
1. DUT Verilog-A code block: ```verilog-a ... ```
2. Spectre testbench code block: ```spectre ... ```

Do not return DUT-only output for this task.

## Hierarchical Modeling Policy

Choose the module boundary from the public task contract:
- If the task asks for a single behavioral component, keep it as one coherent module.
- If the task asks for multiple named blocks, implement those blocks separately.
- Do not add extra hidden submodules unless necessary.
"""

    # --- Layer 3: Gold-derived contracts ---
    gold_dir = task_dir / "gold"
    gold_tb_text = ""
    if gold_dir.exists() and family in ("spec-to-va", "bugfix", "end-to-end"):
        gold_tb = _find_gold_tb(gold_dir)
        if gold_tb:
            gold_tb_text = gold_tb.read_text(encoding="utf-8", errors="ignore")

    if gold_tb_text and family in ("spec-to-va", "bugfix", "end-to-end"):
        prompt_md += _inject_evas_validation_contract(gold_tb_text)
        prompt_md += _inject_module_name_contract(family, task_dir, gold_tb_text, task_id)

    # --- Layer 4: Verilog-A syntax rules ---
    if family in ("spec-to-va", "bugfix", "end-to-end"):
        prompt_md += _VA_SYNTAX_RULES

    # --- Layer 5: DUT contract (tb-generation only) ---
    if family == "tb-generation":
        prompt_md += _inject_dut_contract(task_dir)

    # --- Layer 6: Public contracts (non-gold) ---
    if include_public_contract:
        prompt_md += _inject_public_behavior_contract(task_dir)

    # --- Layer 7: Skill context ---
    if skill_context:
        prompt_md += skill_context

    return prompt_md


# ─── Repair prompt (Round 1+: repair with EVAS feedback) ─────

def build_repair_prompt(
    task_dir: Path,
    sample_dir: Path,
    evas_result: dict,
    *,
    history: list[dict] | None = None,
    skill_context: str = "",
) -> str:
    """Build the repair prompt for Round 1+ with full EVAS feedback.

    Injection order:
      1. Repair header (customized by failure status)
      2. Artifact contract
      3. Targeted repair rules (failure-type-gated)
      4. EVAS result (status + scores + notes)
      5. Loop state (multi-round progress)
      6. Original task prompt
      7. Current candidate files
      8. Skill context (if available)
    """
    meta = _read_meta(task_dir)
    family = meta.get("family", "end-to-end")
    task_id = meta.get("task_id") or meta.get("id") or task_dir.name
    status = evas_result.get("status", "FAIL_OTHER")
    scores = evas_result.get("scores", {})
    notes = evas_result.get("evas_notes") or evas_result.get("notes") or []

    parts = []

    # Header
    parts.append(f"# EVAS-Guided Repair — Round {len(history or []) + 1}\n")
    parts.append(f"Status: {status}  |  Scores: {_format_scores(scores)}\n")

    # Focus instruction
    parts.append(_repair_focus(status))

    # Artifact contract
    parts.append(_artifact_contract(family))

    # Output rules
    parts.append(textwrap.dedent("""
        Output rules:
        - Output complete replacement code blocks only. No explanations outside code blocks.
        - Preserve required module names, port order, and observable names from the task.
        - Use EVAS/Spectre-compatible voltage-domain Verilog-A only.
        - Prefer the smallest semantic change that addresses the failure.
        - If the candidate already compiles, do not change working syntax or structure.
    """))

    # Targeted repair rules (failure-type-gated)
    parts.append(_targeted_repair_rules(status, family, notes, task_id))

    # EVAS result detail
    parts.append("# EVAS Result\n")
    parts.append(f"Status: {status}")
    parts.append(f"Scores: dut_compile={scores.get('dut_compile', '?')}  "
                 f"tb_compile={scores.get('tb_compile', '?')}  "
                 f"sim_correct={scores.get('sim_correct', '?')}  "
                 f"weighted_total={scores.get('weighted_total', '?')}")
    if notes:
        parts.append("\nEVAS notes (first 15):")
        for n in notes[:15]:
            parts.append(f"  - {str(n)[:200]}")

    # Loop state
    if history and len(history) > 0:
        parts.append(_loop_progress(history, status))

    # Original task prompt
    original_prompt = _read_prompt_md(task_dir)
    parts.append("\n# Original Task\n")
    parts.append(original_prompt)

    # Current candidate files
    parts.append("\n# Current Candidate Files\n")
    parts.append(_read_candidate_files(sample_dir))

    # Skill context
    if skill_context:
        parts.append(skill_context)

    return "\n\n".join(parts)


# ─── Inline injection helpers ────────────────────────────────

_VA_SYNTAX_RULES = """
## Verilog-A Syntax Rules (MANDATORY)

Your code must be pure Verilog-A, not digital Verilog. Spectre VACOMP will reject:
1. `reg`, `wire`, `logic` — use `electrical` for signals, `integer` for state.
2. Packed bit-select like `sig[3] = ...` on scalar integers.
3. `always @(...)` — use `analog begin` with `@(cross(...))`.
4. `initial begin` — use `@(initial_step)` inside `analog`.
5. Bit literals like `7'b0000001` — use integer constants.
6. Multiple `<+` to the same node adds contributions, not overwrites.

Correct template:
```verilog-a
module NAME (ports);
    electrical ports;
    integer state;
    analog begin
        @(initial_step) state = 0;
        @(cross(V(clk) - vth, 1))
            state = state + 1;
        V(out) <+ transition(state * vstep, 0, 1n);
    end
endmodule
```
"""


def _inject_evas_validation_contract(gold_tb_text: str) -> str:
    """Extract tran statement from gold TB and present as validation contract."""
    tran_match = re.search(r'^\s*tran\s+\w+.*$', gold_tb_text, re.MULTILINE | re.IGNORECASE)
    if not tran_match:
        return ""
    tran_line = re.sub(r'\s+', ' ', tran_match.group(0).strip())
    return f"""
## Strict EVAS Validation Contract (MANDATORY)

The final EVAS validation uses this transient setting:
```spectre
{tran_line}
```
A fixed reference testbench will validate your DUT using this timing window.
Do not shorten the stop time or use a coarser maxstep.
"""


def _inject_module_name_contract(family: str, task_dir: Path, gold_tb_text: str, task_id: str) -> str:
    """Extract expected module name from gold TB's ahdl_include and XDUT instantiation."""
    include_match = re.search(r'ahdl_include\s+"([^"]+\.va)"', gold_tb_text)
    if not include_match:
        return ""

    include_file = include_match.group(1)
    include_stem = Path(include_file).stem

    # For bugfix: module name comes from XDUT instantiation
    expected_mod = include_stem
    if family == "bugfix":
        xdut_match = re.search(r'\bXDUT\s+\([^)]+\)\s+(\w+)', gold_tb_text)
        if xdut_match:
            expected_mod = xdut_match.group(1)

    lines = [
        "",
        "## Module Name Contract",
        f"Your module **MUST** be named exactly **`{expected_mod}`**.",
        f"- Your file will be included as `ahdl_include \"{include_file}\"`",
        f"- Your module declaration MUST be: `module {expected_mod}(...);`",
    ]
    if expected_mod != task_id:
        lines.append(f"- Do **not** use `{task_id}` — the correct name is `{expected_mod}`.")
    return "\n".join(lines)


def _inject_dut_contract(task_dir: Path) -> str:
    """For tb-generation: inject gold DUT interface contract."""
    gold_dir = task_dir / "gold"
    if not gold_dir.exists():
        return ""

    parts = ["\n## DUT Contract — MUST follow exactly\n"]
    for va_file in sorted(gold_dir.glob("*.va")):
        text = va_file.read_text(encoding="utf-8", errors="ignore")
        mod_match = re.search(r'\bmodule\s+(\w+)\s*\(([^)]*)\)\s*;', text, re.DOTALL)
        if not mod_match:
            continue
        mod_name = mod_match.group(1)
        ports_raw = mod_match.group(2)
        ports = [p.strip().split()[-1] if p.strip().split() else p.strip()
                 for p in ports_raw.split(",")]
        parts.append(
            f"**DUT: `{mod_name}`** (`{va_file.name}`)\n"
            f"- Port order: `({', '.join(ports)})`\n"
            f"- Include line: `ahdl_include \"{va_file.name}\"`\n"
        )
    parts.append("\n## Testbench Structure Rules\n"
                 "- Use only `vsource` elements (type=pulse or pwl).\n"
                 "- Single `tran` analysis only.\n"
                 "- `save` plain signal names (no colon syntax).\n"
                 "- `ahdl_include` as the LAST line.\n"
                 "- Add `global 0` after `simulator lang=spectre`.\n")
    return "\n".join(parts)


def _inject_public_behavior_contract(task_dir: Path) -> str:
    """Inject evaluator-aligned public behavioral indicators.

    This reads the checker source to extract what signals/metrics the checker
    validates, WITHOUT exposing the gold implementation.
    """
    meta = _read_meta(task_dir)
    task_id = meta.get("task_id") or meta.get("id") or task_dir.name
    scoring = meta.get("scoring", [])
    if "sim_correct" not in scoring:
        return ""
    # For end-to-end tasks, inject the observable CSV contract
    family = meta.get("family", "end-to-end")
    if family in ("spec-to-va", "bugfix", "end-to-end"):
        return _inject_csv_observable_contract(task_dir)
    return ""


def _inject_csv_observable_contract(task_dir: Path) -> str:
    """Extract required CSV columns from the gold TB's save statement."""
    gold_dir = task_dir / "gold"
    if not gold_dir.exists():
        return ""
    gold_tb = _find_gold_tb(gold_dir)
    if not gold_tb:
        return ""
    text = gold_tb.read_text(encoding="utf-8", errors="ignore")
    save_signals = _extract_save_signals(text)
    if not save_signals:
        return ""
    return f"""
## Observable CSV Contract (MANDATORY)

The EVAS checker reads `tran.csv` by exact column names:
`{"`, `".join(save_signals[:12])}`

- Use plain scalar save names exactly as listed.
- Do not rely on hierarchical names or instance-qualified names.
"""


# ─── Repair rule helpers ─────────────────────────────────────

def _repair_focus(status: str) -> str:
    mapping = {
        "FAIL_DUT_COMPILE":
            "Focus: Fix Verilog-A DUT syntax, module/interface mismatch, banned operators.",
        "FAIL_TB_COMPILE":
            "Focus: Fix Spectre testbench syntax, ahdl_include paths, save directives.",
        "FAIL_SIM_CORRECTNESS":
            "Focus: Fix behavioral semantics. Check thresholds, edge direction, reset, output logic.",
        "FAIL_INFRA":
            "Focus: Ensure all required files are present. Check code block extraction.",
    }
    return mapping.get(status, "Focus: Identify and fix the root cause of the failure.")


def _artifact_contract(family: str) -> str:
    contracts = {
        "spec-to-va": "Return exactly one Verilog-A DUT file in a `verilog-a` code block.",
        "bugfix": "Return exactly one corrected Verilog-A DUT file in a `verilog-a` code block.",
        "tb-generation": "Return exactly one Spectre testbench in a `spectre` code block.",
        "end-to-end": "Return the complete DUT set (one or more `verilog-a` blocks) + exactly one `spectre` testbench block.",
    }
    return contracts.get(family, contracts["end-to-end"])


def _targeted_repair_rules(status: str, family: str, notes: list, task_id: str) -> str:
    """Failure-type-gated repair rules — most critical injection logic."""
    rules = ["\n# Targeted Repair Rules\n"]

    if status == "FAIL_DUT_COMPILE":
        rules.extend([
            "- Preserve exact module name expected by the task.",
            "- Port order: VDD/VSS first, then signal ports.",
            "- One ANSI-inline port per line: `input electrical NAME,`",
            "- Do NOT re-declare ports after ANSI header.",
            "- All declarations at module scope (before `analog begin`).",
            "- No `reg`/`wire`/`logic` — use `electrical` + `integer`.",
            "- No packed bit-select on `integer` variables.",
            "- Edge detection: `@(cross(V(clk)-vth, +1))` not `always @(posedge clk)`.",
            "- Outputs use `transition()` with discrete target variable.",
            "- `@(initial_step)` for initialization.",
            "- `@(cross(...))` must be top-level in `analog begin`, not inside `if`/`else`/`case`.",
            "- `genvar` at module scope, not inside `analog begin`.",
            "- Do NOT use runtime analog bus indexing: `V(bus[i])` with `integer i`.",
        ])
    elif status == "FAIL_TB_COMPILE":
        rules.extend([
            "- Use `simulator lang=spectre` header.",
            "- `global 0` on the second line.",
            "- Plain signal names in `save` — no colon-instance syntax.",
            "- Single `tran` statement. No `dc`/`ac` sweep.",
            "- `ahdl_include` as the LAST line.",
            "- Match DUT instance module name and port order exactly.",
        ])
    elif status == "FAIL_SIM_CORRECTNESS":
        subtype = _classify_failure_subtype(notes)
        rules.append(f"- Failure subtype: **{subtype}**")
        if subtype == "observability_contract":
            rules.extend([
                "- Prioritize checker-contract alignment before semantic rewrites.",
                "- Ensure save statement uses exact checker-required signal names.",
                "- Keep one canonical save list; avoid alias/colon save syntax.",
            ])
        elif subtype == "simulation_artifact":
            rules.extend([
                "- Stabilize runability: valid tran setup, realistic maxstep, complete includes.",
                "- Eliminate compile/runtime blockers before changing behavior.",
            ])
        else:
            rules.extend([
                "- Preserve interface; focus on semantics.",
                "- Check threshold choice, edge direction, reset priority, initialization.",
                "- Drive transition target variable continuously (not inside conditional branch).",
            ])
    elif status == "FAIL_INFRA":
        rules.append("- Ensure all required files are present and code blocks are complete.")

    return "\n".join(rules)


def _classify_failure_subtype(notes: list) -> str:
    note_text = " ".join(str(n) for n in notes).lower()
    if any(m in note_text for m in ("missing ", "tran.csv missing", "too_few_", "insufficient_")):
        return "observability_contract"
    if any(m in note_text for m in ("timeout", "evas_timeout", "tb_not_executed")):
        return "simulation_artifact"
    return "behavior_semantic"


def _loop_progress(history: list[dict], current_status: str) -> str:
    lines = ["\n# Loop Progress\n"]
    status_chain = " → ".join(
        f"R{h.get('round', '?')}:{h.get('status', '?')}" for h in history[-5:]
    )
    lines.append(f"History: {status_chain} → R{len(history)+1}:{current_status}")
    for h in history[-3:]:
        lines.append(
            f"- R{h.get('round', '?')}: status={h.get('status', '?')}, "
            f"wt={h.get('scores', {}).get('weighted_total', '?'):.3f}, "
            f"transition={h.get('transition', '?')}"
        )
    return "\n".join(lines)


# ─── Utility helpers ─────────────────────────────────────────

def _read_meta(task_dir: Path) -> dict:
    import json
    meta_path = task_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _read_prompt_md(task_dir: Path) -> str:
    prompt_path = task_dir / "prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


def _read_buggy_dut(task_dir: Path) -> str | None:
    buggy_dir = task_dir / "buggy"
    if not buggy_dir.exists():
        return None
    va_files = sorted(buggy_dir.glob("*.va"))
    if va_files:
        return va_files[0].read_text(encoding="utf-8", errors="ignore")
    return None


def _find_gold_tb(gold_dir: Path) -> Path | None:
    preferred = sorted(gold_dir.glob("tb*_ref.scs"))
    if preferred:
        return preferred[0]
    fallback = sorted(gold_dir.glob("tb*.scs"))
    return fallback[0] if fallback else None


def _extract_save_signals(tb_text: str) -> list[str]:
    """Extract signal names from save statement in Spectre testbench."""
    for line in tb_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("save "):
            rest = stripped[5:].strip()
            if rest.lower() in ("all", "allpub"):
                return []
            signals = []
            for token in rest.split():
                token = token.strip().strip(",")
                if not token:
                    continue
                match = re.match(r"[vV]\s*\(\s*([^)]+)\s*\)", token)
                signals.append(match.group(1) if match else token)
            return signals
    return []


def _format_scores(scores: dict) -> str:
    return " ".join(
        f"{k[:2].upper()}={v:.1f}" for k, v in scores.items()
        if k in ("dut_compile", "tb_compile", "sim_correct", "weighted_total")
    )


def _read_candidate_files(sample_dir: Path) -> str:
    parts = []
    for f in sorted(sample_dir.glob("*.va")):
        content = f.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"```verilog-a\n{content}\n```")
    for f in sorted(sample_dir.glob("*.scs")):
        content = f.read_text(encoding="utf-8", errors="ignore")
        parts.append(f"```spectre\n{content}\n```")
    return "\n\n".join(parts) if parts else "  (no candidate files found)"
