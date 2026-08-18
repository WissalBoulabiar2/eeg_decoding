"""Addition 1 (mudvi_gp_report.pdf Sec 3.2): Constrained Replay via Gradient Projection.

Implements the two pieces of math the trainer needs (Eq. 5 -- separate
memory/new-data gradients from the same parameters theta_hat_k; Eq. 3 --
magnitude-aware orthogonal projection when the two gradients conflict).
Kept as a standalone module, imported by `trainer.py` only on the branch
where `use_gradient_projection=True`, so the baseline path is untouched.

Verified invariant (derived directly from Eq. 3, not invented; see
IMPLEMENTATION_MAPPING.md Sec 2): both branches guarantee
dot(g_tilde, g_mem) >= 0 -- the combined/projected update direction never
has an uphill component along the memory gradient, so a small enough step
cannot increase the replay-memory loss (first-order guarantee). Branch 1
(conflict, memory gradient is the smaller one) gives exactly 0 by
construction of orthogonal projection. Branch 2 (conflict, memory gradient
is the larger one) gives ||g_mem||^2 - dot(g_mem,g_new)^2/||g_new||^2 >= 0
by Cauchy-Schwarz.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def flatten_grads(grads, params) -> torch.Tensor:
    """Flatten a per-parameter gradient tuple (as returned by
    `torch.autograd.grad`, which may contain None for an unused parameter)
    into a single 1-D vector, in the same order as `params`, so the memory
    and new-data gradients can be combined/compared as whole vectors."""
    parts = []
    for g, p in zip(grads, params):
        parts.append(torch.zeros_like(p).reshape(-1) if g is None else g.reshape(-1))
    return torch.cat(parts)


def assign_flat_grad(flat: torch.Tensor, params) -> None:
    """Write a flattened gradient vector back into each parameter's
    `.grad`, matching Algorithm 1's 'assigned via .grad then single step'."""
    idx = 0
    for p in params:
        n = p.numel()
        p.grad = flat[idx:idx + n].view_as(p).clone()
        idx += n


def project_conflicting_gradients(g_mem: torch.Tensor, g_new: torch.Tensor,
                                   eps: float = 1e-12) -> torch.Tensor:
    """Eq. 3, conflict branch (dot(g_mem, g_new) < 0): the LARGER-magnitude
    gradient has its component along the smaller-magnitude gradient removed
    (orthogonal projection). The output is that single modified gradient,
    NOT a sum of the two."""
    norm_mem_sq = torch.dot(g_mem, g_mem)
    norm_new_sq = torch.dot(g_new, g_new)
    dot = torch.dot(g_mem, g_new)
    if norm_mem_sq >= norm_new_sq:
        # memory gradient is the larger one -> project IT onto the
        # orthogonal complement of the new-data gradient.
        return g_mem - (dot / (norm_new_sq + eps)) * g_new
    else:
        # new-data gradient is the larger one -> project IT onto the
        # orthogonal complement of the memory gradient.
        return g_new - (dot / (norm_mem_sq + eps)) * g_mem


@dataclass
class GradientProjectionStats:
    """Running log for Addition 1 (task requirement: log conflict count/%,
    dot product before/after projection, memory loss before/after)."""
    total_steps: int = 0
    conflict_steps: int = 0
    dot_before_sum: float = 0.0
    dot_after_sum: float = 0.0
    mem_loss_before_sum: float = 0.0
    mem_loss_after_sum: float = 0.0

    def log_step(self, conflict: bool, dot_before: float, dot_after: float,
                 mem_loss_before: float, mem_loss_after: float) -> None:
        self.total_steps += 1
        self.conflict_steps += int(conflict)
        self.dot_before_sum += dot_before
        self.dot_after_sum += dot_after
        self.mem_loss_before_sum += mem_loss_before
        self.mem_loss_after_sum += mem_loss_after

    def summary(self) -> dict:
        if self.total_steps == 0:
            return {"total_steps": 0}
        n = self.total_steps
        return {
            "total_steps": n,
            "conflict_steps": self.conflict_steps,
            "conflict_rate": self.conflict_steps / n,
            "avg_dot_before": self.dot_before_sum / n,
            "avg_dot_after": self.dot_after_sum / n,
            "avg_mem_loss_before": self.mem_loss_before_sum / n,
            "avg_mem_loss_after": self.mem_loss_after_sum / n,
        }
