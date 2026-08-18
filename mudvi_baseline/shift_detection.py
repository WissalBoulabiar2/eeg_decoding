"""Subject-shift detection interface (modular extension point).

Duan et al. Sec 3.3 proposes a kernel/MMD-based detector for the
subject-AGNOSTIC scenario, where the decoder does not know when a new
subject starts. This is NOT implemented here: this baseline is
explicitly subject-AWARE (the task hands subjects to the trainer in a
known order, A01 -> A09), so the real MMD detector is out of scope.

`OracleShiftDetector` below simply reports a shift at every subject
boundary using the ground-truth subject id, satisfying the same
interface a real detector (Duan's MMD detector, or the user's own
confidence-based detector from mudvi_gp_report.pdf) would need to
implement, so either can be dropped into `trainer.py` later without
changing the training loop.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class ShiftDetector(ABC):
    @abstractmethod
    def detect(self, subject_id: str) -> bool:
        """Return True if a subject shift is detected at this step."""
        raise NotImplementedError


class OracleShiftDetector(ShiftDetector):
    """Always correct: uses the known subject id directly. This is the
    baseline's stand-in for Duan et al.'s MMD detector (not implemented,
    see module docstring)."""

    def __init__(self):
        self._last_subject_id = None

    def detect(self, subject_id: str) -> bool:
        shifted = subject_id != self._last_subject_id
        self._last_subject_id = subject_id
        return shifted

    def state_dict(self) -> dict:
        return {"last_subject_id": self._last_subject_id}

    def load_state_dict(self, state: dict) -> None:
        self._last_subject_id = state["last_subject_id"]
