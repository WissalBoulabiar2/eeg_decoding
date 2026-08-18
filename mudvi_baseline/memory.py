"""Class-balanced replay memory buffer.

Duan et al.'s full memory mechanism (Sec 3.1-3.2, Eq. 1-4) balances the
buffer at the level of per-*subject* clusters, weighted by a gradient-
norm informativeness score, and requires knowing/detecting subject
identity. The task for this baseline instead asks explicitly for a
buffer "balanced across the 4 classes as much as possible" -- since
class identity here is fixed and known (4 MI classes, shared across all
subjects) this is implemented directly as an even, fixed-capacity
partition of the buffer across the 4 classes, with reservoir sampling
within each class partition. This is a deliberate SIMPLIFICATION of
Duan's subject-cluster mechanism (IMPLEMENTATION DECISION, not an
invented detail: reservoir sampling itself is the "conventional way to
update memory buffer" Duan et al. explicitly describe before proposing
their subject-cluster refinement).

Duan et al.'s informativeness-weighted replacement (Eq. 1) and
importance-weighted sampling (Eq. 4) are NOT implemented here; both are
noted as extension points below so a more faithful subject-cluster/
informativeness memory can be swapped in later without touching the
trainer.
"""
from __future__ import annotations

import numpy as np


class ClassBalancedMemory:
    def __init__(self, capacity: int, num_classes: int = 4, seg_shape=(22, 400), seed: int = 0):
        self.capacity = capacity
        self.num_classes = num_classes
        self.per_class_capacity = capacity // num_classes
        self.rng = np.random.RandomState(seed)

        self._X = {c: np.zeros((0,) + seg_shape, dtype=np.float32) for c in range(num_classes)}
        self._y = {c: np.zeros((0,), dtype=np.int64) for c in range(num_classes)}
        self._seen_count = {c: 0 for c in range(num_classes)}  # total items seen for reservoir sampling

    def __len__(self):
        return sum(len(self._y[c]) for c in range(self.num_classes))

    def update(self, X: np.ndarray, y: np.ndarray):
        """Insert new (subject, class) data into the buffer using per-class
        reservoir sampling. This is the extension point where Duan et al.'s
        Eq. 1-2 informativeness-weighted replacement could be substituted
        for plain uniform reservoir replacement."""
        for c in range(self.num_classes):
            idx = np.where(y == c)[0]
            if len(idx) == 0:
                continue
            Xc, yc = X[idx], y[idx]
            self._insert_class(c, Xc, yc)

    def _insert_class(self, c: int, Xc: np.ndarray, yc: np.ndarray):
        buf_X, buf_y = self._X[c], self._y[c]
        cap = self.per_class_capacity
        for i in range(len(yc)):
            self._seen_count[c] += 1
            if len(buf_X) < cap:
                buf_X = np.concatenate([buf_X, Xc[i:i + 1]], axis=0)
                buf_y = np.concatenate([buf_y, yc[i:i + 1]], axis=0)
            else:
                j = self.rng.randint(0, self._seen_count[c])
                if j < cap:
                    buf_X[j] = Xc[i]
                    buf_y[j] = yc[i]
        self._X[c], self._y[c] = buf_X, buf_y

    def sample_balanced_batch(self, batch_size: int):
        """Draw a batch as evenly as possible across the classes currently
        present in memory (uniform random sampling within each class).
        This is the extension point where Duan et al.'s Eq. 4
        importance-weighted sampling could replace uniform sampling."""
        present = [c for c in range(self.num_classes) if len(self._y[c]) > 0]
        if not present:
            return (np.zeros((0,) + next(iter(self._X.values())).shape[1:], dtype=np.float32),
                    np.zeros((0,), dtype=np.int64))

        per_class = max(1, batch_size // len(present))
        Xs, ys = [], []
        for c in present:
            n = len(self._y[c])
            take = min(per_class, n)
            idx = self.rng.choice(n, size=take, replace=False)
            Xs.append(self._X[c][idx])
            ys.append(self._y[c][idx])
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        return X, y

    def class_counts(self):
        return {c: len(self._y[c]) for c in range(self.num_classes)}

    def state_dict(self) -> dict:
        """Full buffer contents + reservoir counters, for checkpointing
        (Lightning AI migration: restartable runs need the exact buffer
        contents, not just its capacity)."""
        return {
            "capacity": self.capacity,
            "num_classes": self.num_classes,
            "X": {c: self._X[c] for c in range(self.num_classes)},
            "y": {c: self._y[c] for c in range(self.num_classes)},
            "seen_count": dict(self._seen_count),
        }

    def load_state_dict(self, state: dict) -> None:
        for c in range(self.num_classes):
            self._X[c] = state["X"][c]
            self._y[c] = state["y"][c]
        self._seen_count = dict(state["seen_count"])
