"""OCAR -- Online Curvature-Aware Replay (Urettini & Carta, ICML 2025,
arXiv:2502.01866, official code: https://github.com/edo-urettini/CL_stability).

Third gradient-combination strategy for `trainer.ContinualTrainer`,
alongside the plain combined-loss baseline (`_train_step_baseline`) and
Addition 1 / Gradient Projection (`_train_step_gradient_projection`,
`gradient_projection.py`). Where GP resolves *directional* conflict
between the new-subject gradient and the memory-replay gradient (a
first-order, sign-of-dot-product signal), OCAR instead rescales the
combined update using a running estimate of each layer's *curvature*
(Fisher information), via Kronecker-Factored Approximate Curvature
(K-FAC; Martens & Grosse 2015, "Optimizing Neural Networks with
Kronecker-factored Approximate Curvature"; Grosse & Martens 2016 for the
convolutional extension used here) -- a natural-gradient-style step,
estimated online from the same new-subject and memory-replay batches the
trainer already draws, with no task ID and no extra data.

SCOPE (disclosed simplification, same documentation policy as this
project's `memory.py` and `shift_detection.py`): full Kronecker
factorization is implemented for the three GENUINE matrix-multiply
layers in `MudviCNN` -- `temporal_conv[0]` (ordinary Conv2d),
`pointwise_conv[0]` (Conv2d with a 1x1 kernel, which reduces the K-FAC
convolutional formula to the plain-linear-layer case), and `classifier`
(nn.Linear). `depthwise_conv[0]` uses `groups=F1` (a genuinely grouped/
depthwise convolution, one independent weight block per input channel);
the standard K-FAC derivation assumes a single shared weight matrix per
layer and does not directly generalize to per-group independent blocks,
so that layer -- along with every BatchNorm affine parameter and the
classifier's bias -- uses a diagonal (per-parameter, Adagrad/Adam-style
EMA-of-squared-gradient) curvature proxy instead. This is an
intentionally simpler substitute for those specific parameters, not an
invented detail: it is disclosed here exactly once, the same way
`memory.py`'s docstring discloses substituting uniform reservoir
sampling for Duan et al.'s informativeness-weighted replacement.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad(k: int) -> tuple[int, int]:
    """(left, right) padding matching `nn.Conv2d(..., padding="same")`
    for stride=1, dilation=1: total = k-1, more padding on the right/
    bottom when k is even (matches PyTorch's own convention)."""
    total = k - 1
    left = total // 2
    return left, total - left


def _unfold_conv_input(x: torch.Tensor, kernel_size: tuple[int, int]) -> torch.Tensor:
    """x: (B, Cin, H, W), a "same"-padded, stride-1 conv's input.
    Returns (B*L, Cin*kh*kw) patches, L = H*W (output spatial size equals
    input spatial size under same padding + stride 1), matching the
    Kronecker-factored-convolution (KFC) treatment of Grosse & Martens
    2016: each output spatial location is treated as one additional
    "sample" for the input-patch covariance factor A."""
    kh, kw = kernel_size
    pad_h_l, pad_h_r = _same_pad(kh)
    pad_w_l, pad_w_r = _same_pad(kw)
    x_padded = F.pad(x, (pad_w_l, pad_w_r, pad_h_l, pad_h_r))
    patches = F.unfold(x_padded, kernel_size=(kh, kw))  # (B, Cin*kh*kw, L)
    b, d, l = patches.shape
    return patches.permute(0, 2, 1).reshape(b * l, d)


class _KFACFactors:
    """Running EMA Kronecker factors A (input/patch covariance) and G
    (output-gradient covariance) for one ordinary (non-grouped) conv or
    linear layer, plus damped-inverse preconditioning of that layer's
    weight gradient: g_tilde = G_inv @ g @ A_inv (Martens & Grosse 2015,
    Eq. 4-ish natural-gradient approximation)."""

    def __init__(self, in_dim: int, out_dim: int, damping: float, device):
        self.in_dim, self.out_dim = in_dim, out_dim
        self.damping = damping
        self.A = torch.eye(in_dim, device=device)
        self.G = torch.eye(out_dim, device=device)
        self.initialized = False

    def update(self, a_batch: torch.Tensor, g_batch: torch.Tensor, ema_decay: float) -> None:
        if not self.initialized:
            self.A = a_batch.detach().clone()
            self.G = g_batch.detach().clone()
            self.initialized = True
        else:
            self.A.mul_(ema_decay).add_(a_batch.detach(), alpha=1 - ema_decay)
            self.G.mul_(ema_decay).add_(g_batch.detach(), alpha=1 - ema_decay)

    def precondition(self, grad_out_in: torch.Tensor) -> torch.Tensor:
        """grad_out_in: (out_dim, in_dim) weight-gradient matrix."""
        eye_a = torch.eye(self.in_dim, device=self.A.device)
        eye_g = torch.eye(self.out_dim, device=self.G.device)
        a_inv = torch.linalg.inv(self.A + self.damping * eye_a)
        g_inv = torch.linalg.inv(self.G + self.damping * eye_g)
        return g_inv @ grad_out_in @ a_inv

    def condition_number(self) -> float:
        eye_a = torch.eye(self.in_dim, device=self.A.device)
        eye_g = torch.eye(self.out_dim, device=self.G.device)
        cond_a = torch.linalg.cond(self.A + self.damping * eye_a).item()
        cond_g = torch.linalg.cond(self.G + self.damping * eye_g).item()
        return 0.5 * (cond_a + cond_g)

    def state_dict(self) -> dict:
        return {"A": self.A.cpu(), "G": self.G.cpu(), "initialized": self.initialized}

    def load_state_dict(self, state: dict, device) -> None:
        self.A = state["A"].to(device)
        self.G = state["G"].to(device)
        self.initialized = state["initialized"]


class _DiagFactors:
    """Diagonal (per-parameter) EMA-of-squared-gradient curvature proxy,
    for the grouped depthwise-conv weight, BatchNorm affine parameters,
    and the classifier bias -- see module docstring for why these do not
    get full K-FAC."""

    def __init__(self, numel: int, damping: float, device):
        self.v = torch.zeros(numel, device=device)
        self.damping = damping

    def update(self, grad_flat: torch.Tensor, ema_decay: float) -> None:
        self.v.mul_(ema_decay).add_(grad_flat.detach().pow(2), alpha=1 - ema_decay)

    def precondition(self, grad_flat: torch.Tensor) -> torch.Tensor:
        return grad_flat / (self.v.sqrt() + self.damping)

    def state_dict(self) -> dict:
        return {"v": self.v.cpu()}

    def load_state_dict(self, state: dict, device) -> None:
        self.v = state["v"].to(device)


class KFACPreconditioner:
    """Curvature state + preconditioning for `MudviCNN`'s learnable
    parameters. Owns forward/full-backward hooks on the three K-FAC'd
    submodules to capture their per-step input activations and output
    gradients; `observe_pass` folds one autograd.grad() pass's captured
    hook data (K-FAC layers) and gradient values (diagonal-fallback
    layers) into the running EMA curvature estimate, and `precondition`
    applies the current estimate to a combined new+replay gradient.

    Called twice per training step by
    `trainer.ContinualTrainer._train_step_ocar` (once for the
    memory-replay pass, once for the new-subject pass) before the two
    gradients are combined -- both passes' activation/output-gradient
    statistics contribute to the same running curvature estimate, since
    curvature is a property of the loss landscape at the current
    parameters, not of which batch produced a given gradient.
    """

    def __init__(self, model: nn.Module, device, ema_decay: float = 0.95, damping: float = 1e-3):
        self.device = device
        self.ema_decay = ema_decay
        self.damping = damping

        self._kfac_modules = {
            "temporal_conv.0.weight": model.temporal_conv[0],
            "pointwise_conv.0.weight": model.pointwise_conv[0],
            "classifier.weight": model.classifier,
        }
        self._captured: dict[str, torch.Tensor] = {}
        self._hooks = []
        for name, module in self._kfac_modules.items():
            self._hooks.append(module.register_forward_hook(self._make_fwd_hook(name)))
            self._hooks.append(module.register_full_backward_hook(self._make_bwd_hook(name)))

        self._factors = {
            "temporal_conv.0.weight": _KFACFactors(
                in_dim=model.temporal_conv[0].in_channels * model.temporal_conv[0].kernel_size[0]
                * model.temporal_conv[0].kernel_size[1],
                out_dim=model.temporal_conv[0].out_channels, damping=damping, device=device),
            "pointwise_conv.0.weight": _KFACFactors(
                in_dim=model.pointwise_conv[0].in_channels, out_dim=model.pointwise_conv[0].out_channels,
                damping=damping, device=device),
            "classifier.weight": _KFACFactors(
                in_dim=model.classifier.in_features, out_dim=model.classifier.out_features,
                damping=damping, device=device),
        }

        kfac_names = set(self._kfac_modules.keys())
        self._diag = {
            name: _DiagFactors(p.numel(), damping, device)
            for name, p in model.named_parameters() if name not in kfac_names
        }

    def _make_fwd_hook(self, name: str):
        def hook(module, inputs, output):
            self._captured[f"{name}::input"] = inputs[0].detach()
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            self._captured[f"{name}::grad_output"] = grad_output[0].detach()
        return hook

    def observe_pass(self, named_params: list[str], grads: tuple) -> None:
        """named_params/grads: parallel lists from one
        `torch.autograd.grad(loss, params, ...)` call. Updates the K-FAC
        factors (from this pass's hook-captured tensors) and the
        diagonal factors (from the grad values themselves)."""
        for name, module in self._kfac_modules.items():
            x = self._captured.get(f"{name}::input")
            go = self._captured.get(f"{name}::grad_output")
            if x is None or go is None:
                continue
            if isinstance(module, nn.Linear):
                n = x.shape[0]
                a_batch = x.t() @ x / n
                g_batch = go.t() @ go / n
            else:  # Conv2d (temporal_conv / pointwise_conv -- non-grouped)
                patches = _unfold_conv_input(x, module.kernel_size)
                b, cout, h, w = go.shape
                go_flat = go.permute(0, 2, 3, 1).reshape(-1, cout)
                n = patches.shape[0]
                a_batch = patches.t() @ patches / n
                g_batch = go_flat.t() @ go_flat / n
            self._factors[name].update(a_batch, g_batch, self.ema_decay)

        for name, g in zip(named_params, grads):
            if g is not None and name in self._diag:
                self._diag[name].update(g.reshape(-1), self.ema_decay)

    def precondition(self, named_params: list[str], params: list, grad_flat: torch.Tensor) -> torch.Tensor:
        """Splits the flat combined (new+replay) gradient back into one
        tensor per parameter, preconditions each individually (K-FAC for
        the 3 designated weights, diagonal EMA for everything else), and
        re-flattens -- same slicing convention as
        `gradient_projection.flatten_grads`/`assign_flat_grad`, so the
        result drops directly into that module's `assign_flat_grad`."""
        parts = []
        idx = 0
        for name, p in zip(named_params, params):
            n = p.numel()
            g = grad_flat[idx:idx + n].view_as(p)
            idx += n
            if name in self._factors:
                out_ch = p.shape[0]
                g_tilde = self._factors[name].precondition(g.reshape(out_ch, -1)).reshape_as(g)
            elif name in self._diag:
                g_tilde = self._diag[name].precondition(g.reshape(-1)).reshape_as(g)
            else:
                g_tilde = g
            parts.append(g_tilde.reshape(-1))
        return torch.cat(parts)

    def avg_condition_number(self) -> float:
        vals = [f.condition_number() for f in self._factors.values() if f.initialized]
        return sum(vals) / len(vals) if vals else float("nan")

    def state_dict(self) -> dict:
        return {
            "factors": {name: f.state_dict() for name, f in self._factors.items()},
            "diag": {name: d.state_dict() for name, d in self._diag.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        for name, f in self._factors.items():
            f.load_state_dict(state["factors"][name], self.device)
        for name, d in self._diag.items():
            d.load_state_dict(state["diag"][name], self.device)


@dataclass
class OCARStats:
    """Running log for OCAR (task requirement, parallel to
    `gradient_projection.GradientProjectionStats`): cosine similarity
    between the raw combined gradient and its curvature-preconditioned
    version (how much OCAR actually changes the step -- the OCAR analog
    of GP's binary conflict flag), the average K-FAC condition number
    (is the curvature estimate well- or ill-conditioned), and memory
    loss before/after the update (same apples-to-apples eval-mode
    convention GP already uses)."""
    total_steps: int = 0
    cosine_sim_sum: float = 0.0
    condition_number_sum: float = 0.0
    mem_loss_before_sum: float = 0.0
    mem_loss_after_sum: float = 0.0

    def log_step(self, cosine_sim: float, condition_number: float,
                 mem_loss_before: float, mem_loss_after: float) -> None:
        self.total_steps += 1
        self.cosine_sim_sum += cosine_sim
        self.condition_number_sum += condition_number
        self.mem_loss_before_sum += mem_loss_before
        self.mem_loss_after_sum += mem_loss_after

    def summary(self) -> dict:
        if self.total_steps == 0:
            return {"total_steps": 0}
        n = self.total_steps
        return {
            "total_steps": n,
            "avg_cosine_sim_raw_vs_preconditioned": self.cosine_sim_sum / n,
            "avg_condition_number": self.condition_number_sum / n,
            "avg_mem_loss_before": self.mem_loss_before_sum / n,
            "avg_mem_loss_after": self.mem_loss_after_sum / n,
        }
