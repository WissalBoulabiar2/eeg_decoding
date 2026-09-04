"""Joint (offline) training baseline -- the upper-bound comparison point
for every continual-learning method in this project.

Not derived from Duan et al. or `mudvi_gp_report.pdf` (neither paper's
"joint training" mention -- e.g. Sec 4.6 "Effect of Offline Pre-training"
-- gives an implementation, only the concept). Per the user's explicit
instruction, this is verified against an established public
implementation rather than invented from scratch: ContinualAI's Avalanche
library, `avalanche.training.supervised.JointTraining`
(github.com/ContinualAI/avalanche, MIT licensed). Avalanche's own
docstring for that class: "JointTraining concatenates all the experiences
together and trains on all of them at the same time (a.k.a. offline
training)" -- `train_dataset_adaptation` literally concatenates every
experience's dataset up front, then a single ordinary supervised-training
loop runs over the pooled, shuffled data for `train_epochs` epochs. No
per-experience boundary, no replay memory, no continual-learning
mechanism of any kind. That is exactly what `JointTrainer.run` below
does, with "experience" = "subject".

This makes joint training "offline"/"upper bound" by construction: since
only one model is ever trained and it is trained once on every subject's
data at once, Acc(i,i) == Acc(N,i) for every subject trivially, so BWT
and forgetting (see `metrics.py`) come out to exactly 0 -- the expected,
correct property of a joint-training baseline, not a metrics bug. To let
`run_experiment.py` compute those metrics unmodified (it expects one
`acc_matrix` row per subject/step, indexed 0..N-1), the single evaluation
result is replicated across all N rows rather than reporting only one.

Reuses the same model architecture, optimizer (Adam, lr=cfg.lr) and batch
size as every other method for a fair comparison, per the fairness
convention documented in `run_experiment.build_trainer`'s docstring.
`epochs_per_subject` is reused as "epochs over the full pooled dataset"
(Avalanche's `train_epochs`) rather than adding a new CLI flag -- one full
pass over N subjects' worth of pooled data is the natural joint-training
analogue of N sequential per-subject passes.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _NullMemory:
    """Placeholder so `run_experiment.py`'s final
    `trainer.memory.class_counts()` print still works for joint training,
    which has no replay buffer by definition."""

    def class_counts(self) -> dict:
        return {}


class JointTrainer:
    """Same public interface as `trainer.ContinualTrainer` that
    `run_experiment.py` relies on (`.run()`, `.memory`, `.gp_stats`/
    `.ocar_stats`/`.ocarpp_stats`, `.state_dict()`/`.load_state_dict()`),
    so it drops into the existing runner unmodified. `gp_stats`/
    `ocar_stats`/`ocarpp_stats` stay `None` -- those are MUDVI-only
    extensions, not defined for joint training (enforced in
    `config.parse_config`)."""

    def __init__(self, model: nn.Module, device, lr: float = 1e-3,
                 new_batch_size: int = 32, epochs_per_subject: int = 15,
                 seed: int = 0, **_ignored_continual_only_kwargs):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = new_batch_size
        self.epochs = epochs_per_subject
        self.rng = np.random.RandomState(seed)

        self.memory = _NullMemory()
        self.gp_stats = None
        self.ocar_stats = None
        self.ocarpp_stats = None

        self.test_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.acc_matrix: dict[int, dict[str, float]] = {}

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        if len(y) == 0:
            return float("nan")
        self.model.eval()
        with torch.no_grad():
            xb = torch.from_numpy(X).to(self.device)
            preds = self.model(xb).argmax(dim=1).cpu().numpy()
        return float((preds == y).mean())

    def evaluate_all_seen(self, step_index: int) -> dict:
        accs = {sid: self._evaluate(X, y) for sid, (X, y) in self.test_sets.items()}
        self.acc_matrix[step_index] = accs
        return accs

    def run(self, subjects_data: list, start_step: int = 0, on_step_done=None):
        """Pools every subject's train set and does ONE training pass
        (Avalanche's concatenate-then-train-once semantics), then
        replicates the resulting accuracies across all N step indices so
        `run_experiment.py`'s per-subject reporting loop and
        `metrics.compute_bwt`/`compute_forgetting` work unmodified.

        `start_step`: joint training has no per-subject steps to resume
        mid-way through -- either it already ran to completion (checkpoint
        exists, `start_step > 0`, `load_state_dict` already restored
        `acc_matrix` in full) or it has not (`start_step == 0`)."""
        for sd in subjects_data:
            self.test_sets[sd.subject_id] = (sd.X_test, sd.y_test)

        if start_step > 0:
            return self.acc_matrix

        X_train = np.concatenate([sd.X_train for sd in subjects_data], axis=0)
        y_train = np.concatenate([sd.y_train for sd in subjects_data], axis=0)
        n = len(y_train)

        self.model.train()
        for _ in range(self.epochs):
            perm = self.rng.permutation(n)
            for batch_start in range(0, n, self.batch_size):
                idx = perm[batch_start:batch_start + self.batch_size]
                xb = torch.from_numpy(X_train[idx]).to(self.device)
                yb = torch.from_numpy(y_train[idx]).to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(xb), yb)
                loss.backward()
                self.optimizer.step()

        accs = self.evaluate_all_seen(step_index=0)
        for step_index in range(1, len(subjects_data)):
            self.acc_matrix[step_index] = dict(accs)

        if on_step_done is not None:
            on_step_done(len(subjects_data) - 1)
        return self.acc_matrix

    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "trainer_rng": self.rng.get_state(),
            "torch_rng": torch.get_rng_state(),
            "acc_matrix": {k: dict(v) for k, v in self.acc_matrix.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["trainer_rng"])
        torch.set_rng_state(state["torch_rng"])
        self.acc_matrix = {int(k): dict(v) for k, v in state["acc_matrix"].items()}
