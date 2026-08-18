"""Maximally Interfered Retrieval (MIR) baseline, for comparison against
MUDVI per Prof. Duan's feedback. Not derived from Duan et al. or
mudvi_gp_report.pdf -- standard formulation from Aljundi et al. 2019,
"Online Continual Learning with Maximally Interfered Retrieval" (NeurIPS
2019): instead of replaying a uniformly/class-balanced sampled memory
batch, take a one-step "virtual update" using the current mini-batch's
gradient, then replay the buffer samples whose loss would increase the
MOST under that virtual update (i.e. the samples most "interfered with"
by the incoming update), on the reasoning that those are the ones about
to be forgotten.

Storage uses the same `ReservoirMemory` as the ER baseline (same budget,
same policy) -- MIR's contribution is the RETRIEVAL strategy, not the
buffer itself (Aljundi et al. also pair MIR with a plain reservoir
buffer). `MIRTrainer` subclasses `trainer.ContinualTrainer` and overrides
only `_train_step`, so the actual parameter update
(`_train_step_baseline`: concatenate + single combined-loss backward
pass) is reused byte-for-byte from the base trainer -- MIR only changes
which memory samples are handed to it.

IMPLEMENTATION DECISIONS (standard efficient approximations, per the task
instruction to avoid large computational overhead):
  - The virtual update is a single **SGD** step (`theta - lr * grad`)
    using the trainer's own learning rate, not a replica of the full Adam
    optimizer state (running moment estimates). This is the standard
    simplification used in MIR reimplementations: replicating exact Adam
    state for a throwaway one-step lookahead adds bookkeeping without
    changing which samples rank highest by loss increase, since the
    ranking only needs the CHANGE in loss induced by roughly the same
    update direction.
  - The candidate pool scored for interference is the model's ENTIRE
    memory buffer (<=200 samples at this problem's scale), not a further
    random subsample of it. One extra forward pass over <=200 small-CNN
    (22x400) inputs is cheap relative to the training step itself, so no
    additional approximation is needed here specifically because of this
    dataset's small buffer size -- flagged explicitly rather than silently
    assumed.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .trainer import ContinualTrainer


class MIRTrainer(ContinualTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Persistent scratch copy for the virtual-update forward pass,
        # built once (not per step) to avoid repeated deep-copy overhead.
        self._virtual_model = copy.deepcopy(self.model).to(self.device)

    def _select_interfered(self, X_new: np.ndarray, y_new: np.ndarray,
                            X_cand: np.ndarray, y_cand: np.ndarray):
        """Aljundi et al. 2019 Sec 3.1: rank buffer candidates by the
        increase in loss they would suffer under a virtual SGD step taken
        on the incoming mini-batch's gradient, and return the top
        `mem_batch_size` most-interfered samples."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        param_names = [n for n, p in self.model.named_parameters() if p.requires_grad]

        xb_new = torch.from_numpy(X_new).to(self.device)
        yb_new = torch.from_numpy(y_new).to(self.device)
        self.model.train()
        logits_new = self.model(xb_new)
        loss_new = self.criterion(logits_new, yb_new)
        grads = torch.autograd.grad(loss_new, params, allow_unused=True)
        grad_by_name = dict(zip(param_names, grads))

        lr = self.optimizer.param_groups[0]["lr"]
        self._virtual_model.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for name, p_virtual in self._virtual_model.named_parameters():
                g = grad_by_name.get(name)
                if g is not None:
                    p_virtual -= lr * g

        xb_cand = torch.from_numpy(X_cand).to(self.device)
        yb_cand = torch.from_numpy(y_cand).to(self.device)
        self.model.eval()
        self._virtual_model.eval()
        with torch.no_grad():
            loss_before = F.cross_entropy(self.model(xb_cand), yb_cand, reduction="none")
            loss_after = F.cross_entropy(self._virtual_model(xb_cand), yb_cand, reduction="none")
        self.model.train()

        interference = (loss_after - loss_before).cpu().numpy()
        k = min(self.mem_batch_size, len(y_cand))
        top_idx = np.argsort(-interference)[:k]
        return X_cand[top_idx], y_cand[top_idx]

    def _train_step(self, X_new: np.ndarray, y_new: np.ndarray,
                     X_mem: np.ndarray, y_mem: np.ndarray):
        # X_mem/y_mem here is whatever `train_on_subject` sampled via
        # `self.memory.sample_balanced_batch(...)` (uniform, from
        # `ReservoirMemory`) -- MIR replaces that with its own
        # interference-based retrieval from the FULL buffer, then reuses
        # the base trainer's standard combined-loss update unchanged.
        X_cand, y_cand = self.memory.get_all()
        if len(y_cand) == 0:
            return self._train_step_baseline(X_new, y_new, X_cand, y_cand)
        X_mir, y_mir = self._select_interfered(X_new, y_new, X_cand, y_cand)
        return self._train_step_baseline(X_new, y_new, X_mir, y_mir)
