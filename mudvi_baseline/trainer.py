"""Sequential continual-learning trainer.

Implements exactly the loop described in the task:

  A01 -> train initial model, construct memory buffer
  A02 -> train using A02 data + replay memory from A01, update memory
  A03 -> train using A03 data + replay memory, update memory
  ...
  A09

Each training step draws one mini-batch of new-subject data and one
class-balanced mini-batch from memory, concatenates them, and takes a
single optimizer step on their combined loss (standard experience
replay / "joint training with current data", matching Duan et al.
Sec 3, Def. 1 and the "sequential + memory buffer" description).

Extension points:
  - `use_gradient_projection`: Addition 1 (mudvi_gp_report.pdf Sec 3.2).
    When True (and the memory buffer is non-empty), `_train_step`
    dispatches to `_train_step_gradient_projection`, which computes the
    memory-batch gradient and new-data gradient SEPARATELY and combines
    them via `gradient_projection.project_conflicting_gradients` before a
    single `optimizer.step()`. When False (default), `_train_step`
    dispatches to `_train_step_baseline`, which is byte-for-byte the
    original combined-loss single-backward-pass update -- so the two
    modes are directly comparable and neither the memory buffer
    (`memory.py`) nor the EEG preprocessing (`data.py`) are touched by
    this flag.
  - `shift_detector`: any object implementing
    `shift_detection.ShiftDetector`. The baseline uses
    `OracleShiftDetector` (subject id is known; this is the current
    stand-in for Duan's real kernel-MMD feature-shift detector, which
    this subject-aware baseline never implemented -- see
    `shift_detection.py`'s own docstring). Detection here is purely
    informational (matches Duan et al.: the detector flags cluster
    boundaries, it never gates whether training happens).
  - `relationship_shift_detection`: Addition 2 (mudvi_gp_report.pdf Sec
    3.3). When True, a frozen reference model (theta_hat_D1, snapshotted
    right after the FIRST subject finishes training -- see `run()`) is
    used by `confidence_detector.ConfidenceShiftDetector` to track a
    SECOND, independent shift signal (a change in the input/prediction
    RELATIONSHIP, via a NOT/GLR change-point detector over a confidence/
    loss stream), combined with the existing `shift_detector` via
    `final_shift = mmd_shift OR confidence_shift` inside
    `train_on_subject`. Purely diagnostic/logged (`self.shift_log`),
    exactly like `shift_detector` above -- never gates training or the
    memory buffer. When False (default), none of this runs and
    `train_on_subject` is byte-for-byte the original baseline behavior.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn

from .memory import ClassBalancedMemory
from .metrics import evaluate
from .shift_detection import ShiftDetector, OracleShiftDetector
from .gradient_projection import (
    flatten_grads,
    assign_flat_grad,
    project_conflicting_gradients,
    GradientProjectionStats,
)
from .confidence_detector import ConfidenceShiftDetector, freeze_reference_model


class ContinualTrainer:
    def __init__(self, model: nn.Module, memory: ClassBalancedMemory, device,
                 lr: float = 1e-3, new_batch_size: int = 32, mem_batch_size: int = 32,
                 epochs_per_subject: int = 15, seed: int = 0,
                 shift_detector: ShiftDetector | None = None,
                 use_gradient_projection: bool = False,
                 relationship_shift_detection: bool = False,
                 confidence_signal_type: str = "loss",
                 confidence_window_size: int = 40,
                 confidence_min_segment_length: int = 5):
        self.model = model.to(device)
        self.memory = memory
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.new_batch_size = new_batch_size
        self.mem_batch_size = mem_batch_size
        self.epochs_per_subject = epochs_per_subject
        self.rng = np.random.RandomState(seed)
        self.shift_detector = shift_detector or OracleShiftDetector()
        self.use_gradient_projection = use_gradient_projection
        self.gp_stats = GradientProjectionStats() if use_gradient_projection else None

        # Addition 2 (Confidence-Based Relationship-Shift Detection).
        self.relationship_shift_detection = relationship_shift_detection
        self.confidence_signal_type = confidence_signal_type
        self.confidence_window_size = confidence_window_size
        self.confidence_min_segment_length = confidence_min_segment_length
        self.confidence_detector: ConfidenceShiftDetector | None = None  # set in run(), after subject 1
        self.shift_log: list[dict] = []

        self.test_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.acc_matrix: dict[int, dict[str, float]] = {}

    def _train_step(self, X_new: np.ndarray, y_new: np.ndarray,
                     X_mem: np.ndarray, y_mem: np.ndarray):
        # Gradient projection only makes sense once the memory buffer has
        # something in it; on the very first subject (empty memory) both
        # modes are identical, so fall back to the baseline step.
        if self.use_gradient_projection and len(y_mem) > 0:
            return self._train_step_gradient_projection(X_new, y_new, X_mem, y_mem)
        return self._train_step_baseline(X_new, y_new, X_mem, y_mem)

    def _train_step_baseline(self, X_new: np.ndarray, y_new: np.ndarray,
                              X_mem: np.ndarray, y_mem: np.ndarray):
        """Original baseline update: concatenate new-subject + memory-replay
        data into a single batch and take one combined-loss backward pass."""
        if len(y_mem) > 0:
            X = np.concatenate([X_new, X_mem], axis=0)
            y = np.concatenate([y_new, y_mem], axis=0)
        else:
            X, y = X_new, y_new

        perm = self.rng.permutation(len(y))
        xb = torch.from_numpy(X[perm]).to(self.device)
        yb = torch.from_numpy(y[perm]).to(self.device)

        self.model.train()
        self.optimizer.zero_grad()
        logits = self.model(xb)
        loss = self.criterion(logits, yb)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _train_step_gradient_projection(self, X_new: np.ndarray, y_new: np.ndarray,
                                         X_mem: np.ndarray, y_mem: np.ndarray):
        """Addition 1 (Constrained Replay via Gradient Projection),
        mudvi_gp_report.pdf Sec 3.2, Eq. 3 & Eq. 5. Cross-entropy
        classification loss throughout, same as the baseline."""
        self.model.train()
        params = [p for p in self.model.parameters() if p.requires_grad]

        xb_mem = torch.from_numpy(X_mem).to(self.device)
        yb_mem = torch.from_numpy(y_mem).to(self.device)
        xb_new = torch.from_numpy(X_new).to(self.device)
        yb_new = torch.from_numpy(y_new).to(self.device)

        # ---- MEMORY GRADIENT ----
        # g*_k = grad_theta R(theta, M), the replay-memory loss gradient,
        # evaluated at the current parameters theta_hat_k, in train() mode
        # (dropout + batch statistics) since this is the actual forward
        # pass used to compute the real training gradient -- matches how
        # the baseline's own single combined-loss step is computed. This
        # loss value is NOT used for the mem_loss_before/after diagnostic
        # (see below) since train()-mode dropout/BatchNorm would make that
        # comparison not apples-to-apples with the post-update measurement.
        logits_mem = self.model(xb_mem)
        mem_loss_for_grad = self.criterion(logits_mem, yb_mem)
        g_mem = torch.autograd.grad(mem_loss_for_grad, params, retain_graph=False, allow_unused=True)
        g_mem_flat = flatten_grads(g_mem, params)

        # ---- CURRENT-SUBJECT GRADIENT ----
        # g_{k+1} = grad_theta R(theta, D_{k+1}), the new-subject loss
        # gradient, evaluated at the SAME theta_hat_k (torch.autograd.grad
        # never writes to .grad or updates weights, so the parameters used
        # here are identical to the ones used for g_mem above).
        logits_new = self.model(xb_new)
        new_loss = self.criterion(logits_new, yb_new)
        g_new = torch.autograd.grad(new_loss, params, retain_graph=False, allow_unused=True)
        g_new_flat = flatten_grads(g_new, params)

        # ---- CONFLICT CHECK ----
        # dot(g*_k, g_{k+1}) < 0 means the two gradients disagree: moving
        # along their sum would increase the memory loss to first order.
        dot_before = torch.dot(g_mem_flat, g_new_flat)
        conflict = bool((dot_before < 0).item())

        # ---- PROJECTION ----
        # Eq. 3: if conflicting, the larger-magnitude gradient has its
        # component along the smaller-magnitude gradient removed
        # (orthogonal projection) -- output is that single modified
        # gradient, NOT a sum. Otherwise, use the normal combined gradient.
        if conflict:
            g_tilde_flat = project_conflicting_gradients(g_mem_flat, g_new_flat)
        else:
            g_tilde_flat = g_mem_flat + g_new_flat
        dot_after = torch.dot(g_tilde_flat, g_mem_flat).item()

        # Diagnostic-only memory-loss measurement, apples-to-apples: BOTH
        # "before" and "after" computed in eval() mode (dropout off,
        # BatchNorm uses running statistics, not this batch's own), on
        # the SAME xb_mem/yb_mem batch, so the only thing that differs
        # between them is the actual parameter update from
        # optimizer.step() below -- not a change in forward-pass mode.
        # (Previously mem_loss_before was taken from the train()-mode
        # forward pass above, which also has dropout/batch-stat noise
        # baked in, making that comparison not apples-to-apples.)
        mem_loss_before = None
        if self.gp_stats is not None:
            self.model.eval()
            with torch.no_grad():
                mem_loss_before = self.criterion(self.model(xb_mem), yb_mem).item()
            self.model.train()

        # ---- PARAMETER UPDATE ----
        # Single optimizer.step() using g_tilde, matching Algorithm 1 lines
        # 13-21 (one update per iteration, assigned via .grad).
        self.optimizer.zero_grad(set_to_none=True)
        assign_flat_grad(g_tilde_flat, params)
        self.optimizer.step()

        if self.gp_stats is not None:
            self.model.eval()
            with torch.no_grad():
                mem_loss_after = self.criterion(self.model(xb_mem), yb_mem).item()
            self.gp_stats.log_step(
                conflict=conflict,
                dot_before=dot_before.item(),
                dot_after=dot_after,
                mem_loss_before=mem_loss_before,
                mem_loss_after=mem_loss_after,
            )

        return new_loss.item()

    def train_on_subject(self, subject_id: str, X_train: np.ndarray, y_train: np.ndarray):
        """Train on one subject's data + replay from memory (empty on the
        very first subject, matching 'train initial model' in the task)."""
        # `OracleShiftDetector` is a one-shot state machine: it returns
        # True exactly once, the first time a given subject_id is seen,
        # then False on every later call with that same id. Capture that
        # single boundary signal here (unchanged from before) and reuse
        # it below only on the very first training step of this subject
        # -- calling .detect() again inside the loop would always see the
        # id as "already seen" and trivially return False every time.
        mmd_shift_at_subject_start = self.shift_detector.detect(subject_id)  # logged/available, not gating training

        n = len(y_train)
        if n == 0:
            return
        step_in_subject = 0
        for _ in range(self.epochs_per_subject):
            order = self.rng.permutation(n)
            for start in range(0, n, self.new_batch_size):
                idx = order[start:start + self.new_batch_size]
                X_new, y_new = X_train[idx], y_train[idx]
                X_mem, y_mem = self.memory.sample_balanced_batch(self.mem_batch_size)
                self._train_step(X_new, y_new, X_mem, y_mem)

                # ---- Addition 2: combined shift detection (diagnostic
                # only, never gates training/memory -- same principle as
                # `shift_detector.detect()` above). Only active once
                # relationship_shift_detection=True AND the frozen
                # reference model exists (i.e. from the 2nd subject on;
                # see `run()`). Entirely additive: when the flag is off
                # this block never executes and behavior is unchanged.
                if self.relationship_shift_detection and self.confidence_detector is not None:
                    mmd_shift = mmd_shift_at_subject_start if step_in_subject == 0 else False
                    confidence_shift = self.confidence_detector.update(X_new, y_new)
                    final_shift = mmd_shift or confidence_shift
                    self.shift_log.append({
                        "subject_id": subject_id,
                        "mmd_shift": mmd_shift,
                        "confidence_shift": confidence_shift,
                        "final_shift": final_shift,
                    })
                step_in_subject += 1

        self.memory.update(X_train, y_train)

    def evaluate_all_seen(self, step_index: int):
        accs = {}
        for sid, (X_test, y_test) in self.test_sets.items():
            accs[sid] = evaluate(self.model, X_test, y_test, self.device)
        self.acc_matrix[step_index] = accs
        return accs

    def run(self, subjects_data: list, start_step: int = 0, on_step_done=None):
        """subjects_data: ordered list of data.SubjectData, A01 -> A09.

        `start_step`: resume support (Lightning AI migration -- see
        checkpoint.py). Steps `[0, start_step)` are assumed already
        trained (their weights/memory/etc. were restored via
        `load_state_dict` before this call); this method still registers
        their test sets (needed so `evaluate_all_seen` can score ALL seen
        subjects at every later step) but skips re-training them. Default
        0 reproduces the original, non-resumable behavior exactly.

        `on_step_done`: optional callback `(step_index:int) -> None`,
        invoked after each step's evaluation -- the checkpointing hook
        point (see checkpoint.save_checkpoint). Not called for skipped
        (already-completed) steps.
        """
        for step_index, sd in enumerate(subjects_data):
            self.test_sets[sd.subject_id] = (sd.X_test, sd.y_test)
            if step_index < start_step:
                continue
            self.train_on_subject(sd.subject_id, sd.X_train, sd.y_train)

            if self.relationship_shift_detection and self.confidence_detector is None:
                # ---- FROZEN REFERENCE MODEL (Addition 2) ----
                # theta_hat_D1: deep copy of the model right after it
                # finished training on the FIRST subject in the sequence
                # (Sec 3.3 text: "Task 1"), frozen (requires_grad=False)
                # and NEVER updated again for the rest of the run.
                reference_model = freeze_reference_model(self.model)
                # Threshold calibration: synthetic i.i.d. N(0,1) Monte
                # Carlo (calibration_data left at its default None --
                # ConfidenceShiftDetector/ChangePointDetector fall back to
                # `change_point.calibrate_threshold`). A real-data
                # block-bootstrap alternative was tried
                # (confidence_detector.compute_null_stream +
                # ChangePointDetector's `null_stream` param) but reverted:
                # calibrating on subject 1's train data under-calibrated
                # the threshold (~6.30% false positives vs. a 1% target,
                # since the reference model is artificially stable on data
                # it was fit on), and the corrected test-data version was
                # not yet validated end-to-end before this revert. The
                # synthetic-noise calibration measured ~1.06% false
                # positives on held-out noise, matching the target, and
                # produced the best validated real-data result so far
                # (confidence_shifts=8 on the 3-subject smoke test).
                self.confidence_detector = ConfidenceShiftDetector(
                    reference_model, self.device,
                    signal_type=self.confidence_signal_type,
                    window_size=self.confidence_window_size,
                    min_segment_length=self.confidence_min_segment_length,
                )

            self.evaluate_all_seen(step_index)
            if on_step_done is not None:
                on_step_done(step_index)
        return self.acc_matrix

    def state_dict(self) -> dict:
        """Full trainer state for checkpointing (Lightning AI migration):
        model, optimizer, memory buffer contents, GP/RSD diagnostic
        state, accuracy matrix so far, and RNG state for the trainer's
        own batch-order stream + torch's global RNG (dropout, init).

        NOT covered (documented limitation, not silently assumed away):
        the memory buffer's OWN internal RNG stream (`self.memory.rng`)
        is not persisted, so resuming does not reproduce bit-identical
        reservoir-sampling decisions for the very next `memory.update()`
        call after a resume -- only the buffer's CONTENTS up to the
        checkpoint are exact. This is a deliberate scope decision
        ("restartable where practical", not "bit-exact across restarts")
        given the computational cost of the full run is unaffected either
        way.
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "memory": self.memory.state_dict(),
            "trainer_rng": self.rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "shift_detector": (
                self.shift_detector.state_dict()
                if hasattr(self.shift_detector, "state_dict") else None
            ),
            "acc_matrix": {k: dict(v) for k, v in self.acc_matrix.items()},
            "use_gradient_projection": self.use_gradient_projection,
            "relationship_shift_detection": self.relationship_shift_detection,
        }
        if torch.cuda.is_available():
            state["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        if self.gp_stats is not None:
            state["gp_stats"] = dataclasses.asdict(self.gp_stats)
        if self.confidence_detector is not None:
            state["confidence_detector"] = self.confidence_detector.state_dict()
            state["shift_log"] = list(self.shift_log)
        return state

    def load_state_dict(self, state: dict) -> None:
        assert state["use_gradient_projection"] == self.use_gradient_projection, (
            "Checkpoint was saved with a different use_gradient_projection "
            "setting than this trainer was constructed with."
        )
        assert state["relationship_shift_detection"] == self.relationship_shift_detection, (
            "Checkpoint was saved with a different relationship_shift_detection "
            "setting than this trainer was constructed with."
        )
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.memory.load_state_dict(state["memory"])
        self.rng.set_state(state["trainer_rng"])
        torch.set_rng_state(state["torch_rng"])
        if "torch_cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda_rng"])
        if state.get("shift_detector") is not None and hasattr(self.shift_detector, "load_state_dict"):
            self.shift_detector.load_state_dict(state["shift_detector"])
        self.acc_matrix = {int(k): dict(v) for k, v in state["acc_matrix"].items()}
        if "gp_stats" in state and self.gp_stats is not None:
            self.gp_stats = GradientProjectionStats(**state["gp_stats"])
        if "confidence_detector" in state:
            if self.confidence_detector is None:
                # Reconstruct an empty detector wrapping a fresh deep copy
                # of the (not-yet-loaded) model architecture, then load
                # the checkpointed reference weights + calibrated
                # threshold + sliding-window buffer into it.
                reference_model = freeze_reference_model(self.model)
                self.confidence_detector = ConfidenceShiftDetector(
                    reference_model, self.device,
                    signal_type=self.confidence_signal_type,
                    window_size=self.confidence_window_size,
                    min_segment_length=self.confidence_min_segment_length,
                    precomputed_threshold=0.0,  # placeholder, overwritten by load_state_dict below
                )
            self.confidence_detector.load_state_dict(state["confidence_detector"])
            self.shift_log = list(state.get("shift_log", []))
