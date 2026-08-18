"""Addition 2 (mudvi_gp_report.pdf Sec 3.3): Confidence-Based
Relationship-Shift Detection.

Duan et al.'s detector (`shift_detection.py`) targets a change in the
EEG FEATURE/INPUT distribution (kernel-MMD in the real paper; stood in
here by `OracleShiftDetector` -- see that module's own docstring, this
baseline is subject-aware so the real MMD detector was out of scope).

This module adds a SECOND, independent signal: a change in the
RELATIONSHIP between EEG input and class prediction, using a frozen
reference model theta_hat_D1 -- trained once on the first subject/task
and NEVER updated again (Sec 3.3 text). It does not replace or modify
the existing detector; both run side by side (combined in
`trainer.py`).

For each incoming batch x_t:
  c_t = cross_entropy(reference_model(x_t), y_t)            [Eq. 6, labels available]
  c_t = max_k [softmax(reference_model(x_t))]_k             [Eq. 7, label-free]

producing a scalar stream c_1, c_2, ... fed into the NOT/GLR change-point
detector (`change_point.py`). The main experiment here uses Eq. 6
("loss"), since this baseline trains offline with labels available; Eq.
7 ("confidence") is implemented too, as the report's label-free
alternative, but is not the default.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .change_point import ChangePointDetector


def freeze_reference_model(model: torch.nn.Module) -> torch.nn.Module:
    """Deep-copy `model` and freeze it: theta_hat_D1. Trained once (by
    whatever trained `model` up to this point) and NEVER updated again."""
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad_(False)
    return reference_model


@torch.no_grad()
def compute_confidence_stream(reference_model: torch.nn.Module, X: np.ndarray,
                               y, device, signal_type: str = "loss") -> np.ndarray:
    """Per-sample c_t values for one batch: Eq. 6 (signal_type="loss",
    requires labels) or Eq. 7 (signal_type="confidence", label-free)."""
    xb = torch.from_numpy(X).to(device)
    logits = reference_model(xb)
    if signal_type == "loss":
        if y is None:
            raise ValueError("signal_type='loss' (Eq. 6) requires labels y_t.")
        yb = torch.from_numpy(y).to(device)
        c = F.cross_entropy(logits, yb, reduction="none")
    elif signal_type == "confidence":
        c = F.softmax(logits, dim=1).max(dim=1).values
    else:
        raise ValueError(f"Unknown signal_type: {signal_type!r}")
    return c.cpu().numpy()


@torch.no_grad()
def compute_null_stream(reference_model: torch.nn.Module, X: np.ndarray, y: np.ndarray,
                         device, signal_type: str = "loss", batch_size: int = 32,
                         n_epochs: int = 40, seed: int = 12345) -> np.ndarray:
    """Real, genuinely-shift-free calibration stream: the frozen reference
    model's per-batch confidence/loss values on data it did NOT train on
    but is still in-distribution -- pass subject 1's HELD-OUT TEST split
    here, not its training data.

    CORRECTNESS-CRITICAL: using the reference model's own TRAINING data
    for X, y was tried first and is WRONG -- the model was fit on that
    data, so it is artificially stable/low-loss there (a train/test
    generalization gap), which underestimates the true threshold needed.
    Measured: calibrating on subject 1's train split gave a 6.30%
    false-positive rate on genuinely held-out data (vs. a 1% target);
    calibrating on subject 1's OWN held-out test split (never seen during
    training) instead gave 0.41% on a further, disjoint held-out slice --
    matching the target. This matters because in real use, later
    subjects' "no genuine shift" behavior looks like the reference
    model's held-out behavior, not its training behavior -- both are data
    the reference model wasn't fit on.

    Replayed for `n_epochs` random-permutation passes to build enough raw
    material for the block-bootstrap threshold calibration
    (`change_point.calibrate_threshold_from_stream`) despite the smaller
    size of a held-out split. Batching (random permutation each pass,
    same `batch_size` as live training) mirrors exactly how live training
    constructs its mini-batches, so the resulting stream's
    autocorrelation/shape matches production use -- unlike synthetic
    i.i.d. noise."""
    rng = np.random.RandomState(seed)
    n = len(y)
    values = []
    for _ in range(n_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            c = compute_confidence_stream(
                reference_model, X[idx], y[idx], device, signal_type=signal_type)
            values.append(float(np.mean(c)))
    return np.asarray(values, dtype=np.float64)


class ConfidenceShiftDetector:
    """Wraps the frozen reference model + scalar stream + NOT/GLR
    change-point detector behind a simple per-batch `update()` call."""

    def __init__(self, reference_model: torch.nn.Module, device,
                 signal_type: str = "loss", window_size: int = 40,
                 confidence_level: float = 0.99, min_segment_length: int = 5,
                 n_calibration_simulations: int = 8000, calibration_seed: int = 12345,
                 calibration_data=None, calibration_batch_size: int = 32,
                 calibration_epochs: int = 40, precomputed_threshold: float | None = None):
        """calibration_data: optional (X, y) -- pass subject 1's HELD-OUT
        TEST split here, NOT its training data (see `compute_null_stream`
        docstring for why: calibrating on the reference model's own
        training data measurably under-calibrates the threshold, ~6.30%
        false positives vs. a 1% target). When given, the change-point
        threshold is calibrated via block bootstrap on the model's REAL
        confidence/loss stream on that data, which is more faithful to
        the real signal's autocorrelation/shape than synthetic i.i.d.
        noise (measured ~0.41-1.06% false positives on further held-out
        data, matching the target). When None, falls back to
        synthetic-noise calibration."""
        self.reference_model = reference_model
        self.device = device
        self.signal_type = signal_type

        null_stream = None
        if precomputed_threshold is None and calibration_data is not None:
            X_cal, y_cal = calibration_data
            null_stream = compute_null_stream(
                reference_model, X_cal, y_cal, device, signal_type=signal_type,
                batch_size=calibration_batch_size, n_epochs=calibration_epochs,
                seed=calibration_seed)

        self.change_point_detector = ChangePointDetector(
            window_size=window_size, confidence_level=confidence_level,
            min_segment_length=min_segment_length,
            n_calibration_simulations=n_calibration_simulations,
            calibration_seed=calibration_seed,
            null_stream=null_stream,
            precomputed_threshold=precomputed_threshold)

    def state_dict(self) -> dict:
        """Reference model weights + change-point detector state, for
        checkpointing. The reference model's ARCHITECTURE is not stored
        here (only its weights) -- the caller must reconstruct an
        identical `MudviCNN` instance before calling `load_state_dict`
        (see checkpoint.py, which owns model construction)."""
        return {
            "reference_model_state": self.reference_model.state_dict(),
            "signal_type": self.signal_type,
            "change_point_detector": self.change_point_detector.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.reference_model.load_state_dict(state["reference_model_state"])
        self.reference_model.eval()
        for p in self.reference_model.parameters():
            p.requires_grad_(False)
        assert self.signal_type == state["signal_type"]
        self.change_point_detector.load_state_dict(state["change_point_detector"])

    def update(self, X: np.ndarray, y=None) -> bool:
        """Feed one new batch's mean confidence/loss value c_t into the
        change-point detector; return True if a relationship shift is
        flagged. Batch-level c_t (mean over the batch) is used because
        training here proceeds in mini-batches rather than single samples
        (IMPLEMENTATION DECISION; granularity is not specified by the
        report, which describes a per-sample stream in the general
        single-sample streaming setting)."""
        c_values = compute_confidence_stream(
            self.reference_model, X, y, self.device, signal_type=self.signal_type)
        c_t = float(np.mean(c_values))
        return self.change_point_detector.update(c_t)
