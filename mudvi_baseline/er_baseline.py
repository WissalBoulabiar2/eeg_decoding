"""Experience Replay (ER) baseline, for comparison against MUDVI per Prof.
Duan's feedback ("more detailed comparison with established baselines such
as ER and MIR"). Not derived from Duan et al. or mudvi_gp_report.pdf --
this is the standard formulation from the continual-learning literature
(Rolnick et al. 2019, "Experience Replay for Continual Learning"; Chaudhry
et al. 2019, "On Tiny Episodic Memories in Continual Learning"): a single
reservoir-sampled replay buffer, interleaved with the incoming stream via
one combined-loss update per step.

`trainer.ContinualTrainer._train_step_baseline` is already, by its own
docstring, exactly this update rule (concatenate new-subject batch +
memory batch, single combined cross-entropy backward pass). The one
substantive difference between MUDVI's own baseline and textbook ER is
the memory POLICY: `memory.ClassBalancedMemory` partitions the buffer
into one reservoir per CLASS (an explicit simplification of Duan et al.'s
subject-cluster memory, see that module's docstring), whereas standard ER
uses a single global reservoir (Vitter 1985 algorithm R) over the whole
stream with uniform sampling. `ReservoirMemory` below implements that
global-buffer version. Everything else (model, optimizer, batch sizes,
epochs, training loop) is reused unchanged from `trainer.py` -- ER here is
literally `ContinualTrainer` wired up with this memory instead of
`ClassBalancedMemory`, per the "integrate into the existing pipeline"
requirement; no separate training loop is implemented.
"""
from __future__ import annotations

import numpy as np


class ReservoirMemory:
    """Single global reservoir buffer (Vitter 1985, algorithm R), fixed
    capacity, uniform sampling for replay. Same public interface as
    `memory.ClassBalancedMemory` so it drops into `ContinualTrainer`
    unmodified: `update(X, y)`, `sample_balanced_batch(batch_size)`
    (uniform here, not class-balanced -- name kept for interface
    compatibility with the trainer), `class_counts()`. `get_all()` is new,
    used by the MIR baseline to score every buffered sample."""

    def __init__(self, capacity: int, num_classes: int = 4, seg_shape=(22, 400), seed: int = 0):
        self.capacity = capacity
        self.num_classes = num_classes
        self.rng = np.random.RandomState(seed)

        self._X = np.zeros((0,) + seg_shape, dtype=np.float32)
        self._y = np.zeros((0,), dtype=np.int64)
        self._seen_count = 0

    def __len__(self):
        return len(self._y)

    def update(self, X: np.ndarray, y: np.ndarray):
        """Classic reservoir sampling: each incoming item is inserted while
        the buffer has free capacity; once full, item i (1-indexed by total
        items seen) replaces a uniformly random existing slot with
        probability capacity/i, giving every item seen so far an equal
        capacity/total_seen chance of being in the final buffer."""
        for i in range(len(y)):
            self._seen_count += 1
            if len(self._X) < self.capacity:
                self._X = np.concatenate([self._X, X[i:i + 1]], axis=0)
                self._y = np.concatenate([self._y, y[i:i + 1]], axis=0)
            else:
                j = self.rng.randint(0, self._seen_count)
                if j < self.capacity:
                    self._X[j] = X[i]
                    self._y[j] = y[i]

    def sample_balanced_batch(self, batch_size: int):
        """Uniform random sample from the single buffer (no class
        balancing -- that is precisely the MUDVI-vs-ER contrast)."""
        n = len(self._y)
        if n == 0:
            return (np.zeros((0,) + self._X.shape[1:], dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))
        take = min(batch_size, n)
        idx = self.rng.choice(n, size=take, replace=False)
        return self._X[idx], self._y[idx]

    def get_all(self):
        """Full buffer contents, used by MIR's interference scoring."""
        return self._X, self._y

    def sample_indexed_batch(self, batch_size: int):
        """Like `sample_balanced_batch`, but also returns the buffer
        indices sampled -- used by GMED (`gmed_baseline.py`), which needs
        to write edited versions of specific examples back into the
        buffer in place. Uses a fresh draw from the same `self.rng`
        stream (not a wrapper around `sample_balanced_batch`), so it does
        not disturb that method's own RNG draws for callers (ER/MIR) that
        only ever call one or the other."""
        n = len(self._y)
        if n == 0:
            return (np.zeros((0,) + self._X.shape[1:], dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.int64))
        take = min(batch_size, n)
        idx = self.rng.choice(n, size=take, replace=False)
        return self._X[idx].copy(), self._y[idx].copy(), idx

    def edit_at(self, idx: np.ndarray, X_edited: np.ndarray) -> None:
        """Overwrite buffer slots `idx` with edited inputs (labels
        unchanged -- GMED edits inputs, not labels). Used by
        `gmed_baseline.GMEDTrainer` to persist gradient-based memory
        edits into the buffer, per Jin et al. 2020."""
        self._X[idx] = X_edited

    def class_counts(self):
        return {c: int((self._y == c).sum()) for c in range(self.num_classes)}

    def state_dict(self) -> dict:
        """Full buffer contents + reservoir counter, for checkpointing."""
        return {
            "capacity": self.capacity,
            "num_classes": self.num_classes,
            "X": self._X,
            "y": self._y,
            "seen_count": self._seen_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self._X = state["X"]
        self._y = state["y"]
        self._seen_count = state["seen_count"]
