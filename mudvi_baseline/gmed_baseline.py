"""Gradient-based Memory Editing (GMED) baseline, for comparison against
MUDVI/ER/MIR per Prof. Duan's feedback ("more detailed comparison with
established baselines"). Not derived from Duan et al. or
mudvi_gp_report.pdf -- standard formulation from Jin et al. 2020,
"Gradient Based Memory Editing for Task-Free Continual Learning"
(arXiv:2006.15294): rather than only choosing WHICH buffered examples to
replay (MIR) or how to store them (ER's plain reservoir), GMED perturbs
buffered examples THEMSELVES in input space -- via a gradient step that
pushes each one toward the direction most likely to be forgotten by the
model's next update -- and persists that edit back into the buffer, on
the reasoning that a memory of "harder", more failure-representative
examples is more resistant to the loss of old knowledge as training
continues.

Storage reuses the same `er_baseline.ReservoirMemory` as ER/MIR (same
budget, same reservoir-sampling insertion policy) -- GMED's contribution
is editing memory CONTENTS in place, not the buffer's storage/insertion
policy, consistent with this project's fairness protocol (Section 9):
only the retrieval/replay-construction step differs between the ER/MIR/
GMED baselines. `GMEDTrainer` subclasses `trainer.ContinualTrainer`; the
actual parameter update (`_train_step_baseline`) is reused byte-for-byte
from the base trainer, exactly like `MIRTrainer`. Unlike MIR, GMED
overrides `train_on_subject` rather than just `_train_step`: editing
needs the buffer INDICES of the sampled memory batch (to write edits
back in place), which the base class's `_train_step(X_new, y_new, X_mem,
y_mem)` interface does not carry -- MIR avoids this by only ever needing
the full buffer contents (`memory.get_all()`), not indices into it.

IMPLEMENTATION DECISIONS (standard efficient approximations, documented
per this project's convention -- see `mir_baseline.py`'s equivalent
section -- rather than silently assumed):
  - The "future" model used to evaluate each candidate edit is the same
    one-step-SGD virtual update used by `MIRTrainer._select_interfered`
    (lr = trainer's own Adam lr, single step on the incoming mini-batch's
    gradient), not a replica of full Adam state. Duplicated here rather
    than imported since MIR's virtual-update logic is private to that
    class and this is the same well-established simplification used by
    both baselines' reimplementations, not a shared abstraction either
    original paper defines.
  - The edit itself is a single gradient-ASCENT step in INPUT space,
    x_edit = x + edit_lr * grad_x[ L(f_theta'(x), y) ], where theta' is
    the virtual-updated model -- one step, not a multi-step inner loop,
    since one step is what the paper's own ablation shows captures most
    of the benefit at much lower cost, and this project's problem scale
    (<=200-example buffer) does not need extra steps to stay cheap.
  - Per-example input gradients are computed with the model in eval()
    mode (BatchNorm running stats, dropout off) -- otherwise BatchNorm's
    train()-mode batch statistics would couple different memory
    examples' gradients together, corrupting the "per example" edit
    direction. Same reasoning as MIR's eval()-mode loss comparison.
  - Keep-vs-discard decision: an edited example replaces the original in
    the buffer only if it has a STRICTLY HIGHER loss under the virtual
    model than the original (i.e. the edit found a "harder" version of
    that example) -- otherwise the original is left untouched. This is a
    simplified single-criterion version of the paper's edit-acceptance
    rule (the full rule additionally budgets the edit's effect on
    current-batch loss; omitted here since the loss-increase criterion
    is the dominant term at this project's buffer scale).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .trainer import ContinualTrainer


class GMEDTrainer(ContinualTrainer):
    def __init__(self, *args, edit_lr: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.edit_lr = edit_lr
        # Persistent scratch copy for the virtual-update forward pass,
        # built once (not per step) -- same rationale as MIRTrainer's.
        self._virtual_model = copy.deepcopy(self.model).to(self.device)

    def _virtual_update(self, X_new: np.ndarray, y_new: np.ndarray) -> None:
        """One-step SGD lookahead on the incoming batch's gradient, written
        into `self._virtual_model` -- same simplification as
        `mir_baseline.MIRTrainer._select_interfered`."""
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

    def _edit_memory_batch(self, X_mem: np.ndarray, y_mem: np.ndarray):
        """Gradient-ascent edit of `X_mem` under `self._virtual_model`
        (must be called after `_virtual_update` for this step). Returns
        `(X_edited, accepted)`: the candidate edited inputs and a boolean
        mask of which ones increased virtual-model loss over the
        original (and are therefore worth keeping)."""
        self._virtual_model.eval()
        xb = torch.tensor(X_mem, dtype=torch.float32, device=self.device, requires_grad=True)
        yb = torch.from_numpy(y_mem).to(self.device)

        logits = self._virtual_model(xb)
        per_example_loss = F.cross_entropy(logits, yb, reduction="none")
        loss_before = per_example_loss.detach()
        grad_x, = torch.autograd.grad(per_example_loss.sum(), xb)

        with torch.no_grad():
            x_edit = xb + self.edit_lr * grad_x
            logits_edit = self._virtual_model(x_edit)
            loss_after = F.cross_entropy(logits_edit, yb, reduction="none")

        accepted = (loss_after > loss_before).cpu().numpy()
        X_edited = x_edit.detach().cpu().numpy()
        return X_edited, accepted

    def train_on_subject(self, subject_id: str, X_train: np.ndarray, y_train: np.ndarray):
        """Same overall structure as `ContinualTrainer.train_on_subject`
        (oracle shift-detector call kept for consistency with ER/MIR,
        which inherit it unchanged, even though it is purely informational
        and unused in GMED's output -- GP/RSD are MUDVI-only and asserted
        off for --method gmed in config.py, so those branches are
        correctly omitted here, not silently dropped). The one
        substantive difference: the memory batch is sampled WITH indices
        (`sample_indexed_batch`), edited, written back in place
        (`memory.edit_at`), and only then handed to the same combined-loss
        update (`_train_step_baseline`) every other method uses."""
        self.shift_detector.detect(subject_id)  # informational only, unused for gmed (matches er/mir)

        n = len(y_train)
        if n == 0:
            return
        for _ in range(self.epochs_per_subject):
            order = self.rng.permutation(n)
            for start in range(0, n, self.new_batch_size):
                idx = order[start:start + self.new_batch_size]
                X_new, y_new = X_train[idx], y_train[idx]
                X_mem, y_mem, mem_idx = self.memory.sample_indexed_batch(self.mem_batch_size)
                if len(y_mem) > 0:
                    self._virtual_update(X_new, y_new)
                    X_edited, accepted = self._edit_memory_batch(X_mem, y_mem)
                    if accepted.any():
                        self.memory.edit_at(mem_idx[accepted], X_edited[accepted])
                        X_mem = X_mem.copy()
                        X_mem[accepted] = X_edited[accepted]
                self._train_step_baseline(X_new, y_new, X_mem, y_mem)

        self.memory.update(X_train, y_train)
