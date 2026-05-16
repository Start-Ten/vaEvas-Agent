"""Loop termination conditions."""
from __future__ import annotations

from .state import LoopState


class Terminator:
    """Decides when to stop the repair loop."""

    def __init__(self, max_rounds: int = 3, stall_limit: int = 2, regress_limit: int = 2):
        self.max_rounds = max_rounds
        self.stall_limit = stall_limit
        self.regress_limit = regress_limit

    def should_stop(self, state: LoopState) -> tuple[bool, str]:
        """Return (stop, reason)."""
        if state.is_pass():
            return True, "PASS"

        # max_rounds counts total rounds run. Round 0 is the first round.
        # With max_rounds=3, we run rounds 0,1,2 → stop when history has 3 entries.
        if len(state.history) >= self.max_rounds:
            return True, f"max_rounds ({self.max_rounds}) reached"

        if len(state.history) >= self.stall_limit:
            recent = state.history[-self.stall_limit:]
            if all(r.transition == "stalled" for r in recent):
                return True, f"stalled for {self.stall_limit} consecutive rounds"

        if len(state.history) >= self.regress_limit:
            recent = state.history[-self.regress_limit:]
            if all(r.transition == "regressed" for r in recent):
                return True, f"regressed for {self.regress_limit} consecutive rounds"

        return False, ""

    def summary(self, state: LoopState) -> str:
        """Return a one-line summary of why the loop stopped."""
        stopped, reason = self.should_stop(state)
        if not stopped:
            return "running"
        return reason
