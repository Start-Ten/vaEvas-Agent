"""Terminal display utilities — minimal, no external dependencies."""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import Sequence

# ─── ANSI helpers ────────────────────────────────────────────

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_GRAY = "\033[90m"

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if "NO_COLOR" in os.environ:
        return False
    return True


def _supports_unicode() -> bool:
    """Check if terminal supports unicode characters."""
    if "CI" in os.environ or "GITHUB_ACTIONS" in os.environ:
        return True
    try:
        "✓".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def _c(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"{code}{text}{_RESET}"


def green(text: str) -> str:  return _c(_GREEN, text)
def red(text: str) -> str:    return _c(_RED, text)
def yellow(text: str) -> str: return _c(_YELLOW, text)
def cyan(text: str) -> str:   return _c(_CYAN, text)
def bold(text: str) -> str:   return _c(_BOLD, text)
def dim(text: str) -> str:    return _c(_DIM, text)
def gray(text: str) -> str:   return _c(_GRAY, text)


# ─── Terminal width ──────────────────────────────────────────

def term_width() -> int:
    return shutil.get_terminal_size().columns or 80


# ─── Spinner ─────────────────────────────────────────────────

class Spinner:
    """Simple inline spinner for progress indication."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self._frame = 0
        self._active = False

    def _render(self, text: str) -> str:
        frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
        self._frame += 1
        return f"\r\033[K{self.prefix} {frame} {text}"

    def tick(self, text: str) -> None:
        if _supports_color():
            sys.stdout.write(self._render(text))
            sys.stdout.flush()
            self._active = True

    def done(self, text: str = "") -> None:
        if self._active:
            sys.stdout.write(f"\r\033[K{self.prefix} {text}\n" if text else "\n")
            sys.stdout.flush()
            self._active = False


# ─── Box drawing ─────────────────────────────────────────────

def _hline(char: str, width: int) -> str:
    return char * width


_BOX_CHARS = {
    True:  {"tl": "╭", "tr": "╮", "bl": "╰", "br": "╯", "h": "─", "v": "│"},
    False: {"tl": "+", "tr": "+", "bl": "+", "br": "+", "h": "-", "v": "|"},
}

def _box() -> dict:
    return _BOX_CHARS[_supports_unicode()]


def box_header(title: str) -> str:
    w = min(term_width() - 4, 66)
    inner = f"  {title}  "
    b = _box()
    pad = max(0, w - len(inner) - 2)
    return dim(f"{b['tl']}{_hline(b['h'], len(inner))}{_hline(b['h'], pad)}{b['tr']}")


def box_line(content: str) -> str:
    w = min(term_width() - 4, 66)
    b = _box()
    return dim(b["v"]) + f" {content}" + " " * max(0, w - len(content) - 2) + dim(b["v"])


def box_footer() -> str:
    w = min(term_width() - 4, 66)
    b = _box()
    return dim(f"{b['bl']}{_hline(b['h'], w - 2)}{b['br']}")


# ─── Status formatting ───────────────────────────────────────

def _uni_pass() -> str:
    return "PASS" if not _supports_unicode() else green("PASS")

def _uni_fail() -> str:
    return "FAIL" if not _supports_unicode() else red("FAIL")

def _uni_check(ok: bool) -> str:
    if not _supports_unicode():
        return "OK" if ok else "XX"
    return green("+") if ok else red("-")


def status_icon(status: str) -> str:
    if not _supports_unicode():
        prefix = "[OK]" if status == "PASS" else "[XX]"
        return f"{prefix} {status}"
    if status == "PASS":
        return green("PASS")
    elif status.startswith("FAIL"):
        return red(f"{status}")
    return yellow(f"? {status}")


def scores_line(scores: dict) -> str:
    parts = []
    for k in ("dut_compile", "tb_compile", "sim_correct"):
        v = scores.get(k, 1.0)
        icon = _uni_check(v >= 1.0)
        parts.append(f"{icon} {k[:2]}={v:.1f}")
    wt = scores.get("weighted_total", 0.0)
    parts.append(f"wt={wt:.3f}")
    return "  ".join(parts)


def round_header(round_idx: int, step: str, extra: str = "") -> str:
    prefix = f"[Round {round_idx}]"
    return f"{bold(prefix)} {step}{' ' + dim(extra) if extra else ''}"


def failure_detail(subtype: str, notes: Sequence[str]) -> str:
    prefix = "->" if not _supports_unicode() else dim("->")
    if not notes:
        return f"{prefix} {subtype}"
    first = str(notes[0])
    if len(first) > 100:
        first = first[:97] + "..."
    return f"{prefix} {subtype}: {first}"


def transition_label(transition: str) -> str:
    prefix = "->" if not _supports_unicode() else "->"
    colors = {"improved": green, "regressed": red, "stalled": yellow, "lateral": cyan}
    fn = colors.get(transition, dim)
    return fn(f"{prefix} {transition}")


def result_summary(status: str, rounds: int, total_tokens: int,
                   elapsed_s: float, output_dir: str) -> str:
    lines = [
        box_header(f"Result: {status}"),
        box_line(f"Rounds: {rounds}    Total tokens: {total_tokens:,}    "
                 f"Total time: {elapsed_s:.1f}s"),
        box_line(f"Output: {output_dir}"),
        box_footer(),
    ]
    return "\n".join(lines)


# ─── Doctor formatting ───────────────────────────────────────

def doctor_check(name: str, status: str, message: str, fixing: bool = False) -> str:
    if _supports_unicode():
        icons = {"pass": green("[OK]"), "fail": red("[XX]"), "warn": yellow("[!!]")}
    else:
        icons = {"pass": "[OK]", "fail": "[XX]", "warn": "[!!]"}
    icon = icons.get(status, "[??]")
    suffix = dim(f" -> Fixing: {message} ...") if fixing else f"  {dim(message)}"
    return f"  {icon}  {name:<22s}{suffix}"


def doctor_header() -> str:
    return box_header("vaEvas Agent Doctor")


def doctor_summary(passed: int, total: int, fixed: int = 0, has_failures: bool = False) -> str:
    if fixed > 0:
        return f"\n  Fixed {fixed} issues. {passed}/{total} checks passed. Ready to run."
    if has_failures:
        return f"\n  {passed}/{total} checks passed."
    if passed == total:
        return f"\n  All {passed} checks passed. Ready to run."
    return f"\n  {passed}/{total} checks passed."


# ─── Plain text fallback ─────────────────────────────────────

def print_section(title: str) -> None:
    print(f"\n{bold(title)}")
    print(dim("-" * 40))
