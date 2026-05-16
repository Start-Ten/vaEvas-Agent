"""Skill loader — scans veriloga-skills directory structure and loads SKILL.md + category references."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SkillRef:
    """Reference to a skill file — name, path, and category hint."""
    name: str
    path: Path
    category: str           # e.g., "dac", "pll-clock", "comparator"
    source: str             # "category" | "skill" | "template"


# ─── Keyword → category mapping ──────────────────────────────

KEYWORD_INDEX: dict[str, str] = {
    # PLL / Clock
    "pll": "pll-clock",
    "adpll": "pll-clock",
    "cppll": "pll-clock",
    "pfd": "pll-clock",
    "vco": "pll-clock",
    "lock": "pll-clock",
    "bbpd": "pll-clock",
    # DAC
    "dac": "dac",
    "dwa": "dac",
    "therm": "dac",
    "binary_clk": "dac",
    # ADC / SAR
    "adc": "adc-sar",
    "sar": "adc-sar",
    "flash_adc": "adc-sar",
    # Comparator
    "comparator": "comparator",
    "cmp": "comparator",
    "strongarm": "comparator",
    "hysteresis": "comparator",
    "offset": "comparator",
    # Digital Logic
    "mux": "digital-logic",
    "divider": "digital-logic",
    "clk_div": "digital-logic",
    "counter": "digital-logic",
    "lfsr": "digital-logic",
    "prbs": "digital-logic",
    "gray": "digital-logic",
    "digital": "digital-logic",
    "gate": "digital-logic",
    "dff": "digital-logic",
    "flip": "digital-logic",
    "not_": "digital-logic",
    "and_": "digital-logic",
    "or_": "digital-logic",
    "d2b": "digital-logic",
    # Sample & Hold
    "sample_hold": "sample-hold",
    "sample": "sample-hold",
    "aperture": "sample-hold",
    # Amplifier / Filter
    "gain": "amplifier-filter",
    "filter": "amplifier-filter",
    "lpf": "amplifier-filter",
    "amplifier": "amplifier-filter",
    # Signal Source
    "noise": "signal-source",
    "ramp": "signal-source",
    "burst": "signal-source",
    "pulse": "signal-source",
    "sine": "signal-source",
    # Measurement
    "extraction": "measurement-helpers",
    # Calibration
    "calibration": "calibration",
    # Power / Switch
    "switch": "power-switch",
    "power": "power-switch",
    # Passive
    "passive": "passive-model",
    "rlc": "passive-model",
    # Testbench (tb-generation tasks)
    "tb_generation": "testbench-spectre",
    "testbench": "testbench-spectre",
}


CATEGORY_FILE_MAP: dict[str, str] = {
    "adc-sar": "adc-sar.md",
    "amplifier-filter": "amplifier-filter.md",
    "calibration": "calibration.md",
    "comparator": "comparator.md",
    "dac": "dac.md",
    "digital-logic": "digital-logic.md",
    "measurement-helpers": "measurement-helpers.md",
    "passive-model": "passive-model.md",
    "pll-clock": "pll-clock.md",
    "power-switch": "power-switch.md",
    "sample-hold": "sample-hold.md",
    "signal-source": "signal-source.md",
    "testbench-spectre": "testbench-spectre.md",
}


def resolve_skills_root(config_skills_path: str | None = None) -> Path | None:
    """Resolve the veriloga-skills root directory.

    Search order:
    1. config_skills_path (from AgentConfig)
    2. ../veriloga-skills relative to this file's package root
    3. Common absolute paths on Windows
    """
    candidates = []

    if config_skills_path:
        candidates.append(Path(config_skills_path).resolve())

    # Relative to agent package (vaEvas-Agent/vaevas_agent → vaEvas-Agent/..)
    pkg_root = Path(__file__).resolve().parent.parent.parent
    candidates.append((pkg_root / "veriloga-skills").resolve())
    candidates.append((pkg_root / ".." / "veriloga-skills").resolve())

    # Common Windows paths
    import os
    home = Path(os.path.expanduser("~"))
    candidates.append(home / "Desktop" / "WorkSpace" / "VerilogA" / "veriloga-skills")

    for c in candidates:
        try:
            if c.exists() and c.is_dir():
                return c
        except (OSError, PermissionError):
            continue
    return None


def list_categories(categories_dir: Path) -> dict[str, Path]:
    """Scan categories directory and return {category_name: file_path}."""
    result: dict[str, Path] = {}
    if not categories_dir.exists():
        return result
    for md_file in sorted(categories_dir.glob("*.md")):
        category = md_file.stem  # e.g., "dac", "pll-clock"
        result[category] = md_file
    return result


def load_category_content(file_path: Path, max_chars: int = 3000) -> str:
    """Load a category reference file, truncated to max_chars."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return ""

    if len(text) <= max_chars:
        return text.strip()

    # Truncate at nearest paragraph break
    truncated = text[:max_chars]
    last_break = max(truncated.rfind("\n\n"), truncated.rfind("\n## "))
    if last_break > max_chars // 2:
        truncated = text[:last_break]
    return truncated.strip() + "\n\n... (truncated)"
