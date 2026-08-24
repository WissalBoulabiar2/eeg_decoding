"""OCAR -- Online Curvature-Aware Replay (Urettini & Carta, "Online
Curvature-Aware Replay: Leveraging 2nd Order Information for Online
Continual Learning", ICML 2025, arXiv:2502.01866,
https://openreview.net/forum?id=ek5a5WC4TW). Official code:
https://github.com/edo-urettini/CL_stability (built on Avalanche for
the continual-learning training loop and a MODIFIED fork of nngeometry
for Fisher-Information-Matrix computation).

This module reimplements the ALGORITHM their code actually runs --
verified by reading `ocl_survey/src/strategies/robust_grad.py`'s
`SignSGDPlugin.before_update` directly (that class name is misleading;
per the repo's own README ["ng #added OCAR strategy"] and its
`method_factory.py` wiring [`name == "robust_grad"` -> `ReplayPlugin` +
`SignSGDPlugin`], THIS is the OCAR gradient-preconditioning mechanism)
-- WITHOUT taking on Avalanche or nngeometry as dependencies: this
project has neither, and pulling in a full continual-learning framework
plus a ~4000-line Fisher-information library for one gradient-
combination strategy would be a disproportionate addition. The VERIFIED
algorithm, exactly as their `before_update` runs it:

  1. ONE combined batch (new-subject + memory-replay samples
     concatenated), a single forward pass and standard `loss.backward()`
     -- NOT two separate gradients summed (that separate-gradient
     structure belongs to this project's own Gradient Projection (GP)
     addition, not to OCAR).
  2. Periodically, recompute the Fisher Information Matrix as a
     Kronecker-factored (K-FAC) approximation over that combined batch
     (their code: `nngeometry.metrics.FIM(..., representation=
     PMatKFAC)`; every `train_epochs` mini-batches -- in their usual
     ONLINE-CL benchmark configs `train_epochs=1`, which collapses this
     to "every step"; see `fim_update_every` below for why their literal
     semantics do not transfer to this project and what we do instead).
  3. EMA-blend with the previous Fisher estimate, where `alpha_ema` is
     the weight on the NEW estimate (their published "robust_grad"
     config sets `alpha_ema=1.0` -- i.e. NO blending, always use the
     freshly computed Fisher; kept tunable here, defaulting to their
     published value).
  4. Invert the damped Fisher and multiply the model's ACTUAL `.grad`
     (already populated by the standard backward pass) by that inverse
     -- a natural-gradient step -- then `optimizer.step()`.
  5. Damping grows over training: `tau` starts at the learning rate and
     increases by a fixed `regul` every recompute cycle (their code;
     the commented-out Levenberg-Marquardt adaptive-tau branch in their
     own file is dead and is not reproduced here either).

DISCLOSED DEVIATIONS (verified against their actual code, not invented
-- same documentation policy as this project's `memory.py`/
`shift_detection.py`):

  - Their nngeometry fork's `layercollection.py` sets
    `_ignore_modules = ["BatchNorm2d"]` (BatchNorm parameters are
    excluded from the Fisher computation and preconditioning entirely
    -- raw, unscaled gradient passes through for them), and its
    `Conv2dLayer` construction never reads `mod.groups`, meaning a
    GROUPED convolution (like this project's `depthwise_conv`,
    `groups=F1`) is not supported by their code at all -- passing one
    in would silently build a wrong-shaped Fisher block rather than a
    documented fallback. We reproduce the SAME behavior their own code
    already exhibits for layers it does not support: `depthwise_conv.0.
    weight` and every BatchNorm2d affine parameter get the RAW
    gradient, completely untouched by preconditioning -- not a
    diagonal approximation (an earlier version of this module used one
    before this file was checked against their actual source; this
    version is the more faithful plain pass-through).
  - Their per-sample loss reweighting (`robust_grad.py`:
    `weights[buffer_idx] = n_known / n_new`) exists specifically to
    compensate for CLASS-INCREMENTAL imbalance (few examples of
    already-seen classes vs. many of a newly-introduced class this
    experience). This project's setting is DOMAIN-incremental with a
    FIXED 4-class label space -- every subject has all 4 motor-imagery
    classes from the start, so there is no growing "known classes"
    count for that formula to act on, and it has no well-defined analog
    here. It is NOT ported: the combined batch uses plain (unweighted)
    cross-entropy, and class balance is instead handled the way the
    rest of this project already handles it, via `ClassBalancedMemory`'s
    balanced buffer sampling.
  - `regul_last`/`alpha_ema_last` (their per-last-layer variant, and
    explicitly marked "not used in the current implementation" in
    their own docstring) and the commented-out Levenberg-Marquardt
    branch are dead in their own published default config and are not
    reproduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _same_pad(k: int) -> tuple[int, int]:
    """(left, right) padding matching `nn.Conv2d(..., padding="same")`
    for stride=1, dilation=1."""
    total = k - 1
    left = total // 2
    return left, total - left


def _unfold_conv_input(x: torch.Tensor, kernel_size: tuple[int, int]) -> torch.Tensor:
    """x: (B, Cin, H, W), a "same"-padded, stride-1 conv's input.
    Returns (B*L, Cin*kh*kw) patches, L = H*W."""
    kh, kw = kernel_size
    pad_h_l, pad_h_r = _same_pad(kh)
    pad_w_l, pad_w_r = _same_pad(kw)
    x_padded = F.pad(x, (pad_w_l, pad_w_r, pad_h_l, pad_h_r))
    patches = F.unfold(x_padded, kernel_size=(kh, kw))  # (B, Cin*kh*kw, L)
    b, d, l = patches.shape
    return patches.permute(0, 2, 1).reshape(b * l, d)


class _KFACLayer:
    """One K-FAC'd layer's running Kronecker factors A (input/patch
    covariance, bias-augmented with a constant-1 column when the layer
    has a bias -- the standard K-FAC treatment of bias, Martens &
    Grosse 2015) and G (output-gradient covariance), plus damped-
    inverse preconditioning of that layer's weight+bias gradient."""

    def __init__(self, in_dim: int, out_dim: int, has_bias: bool, device):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.has_bias = has_bias
        a_dim = in_dim + (1 if has_bias else 0)
        self.A = torch.eye(a_dim, device=device)
        self.G = torch.eye(out_dim, device=device)
        self.initialized = False

    def compute_factors(self, x_flat: torch.Tensor, go_flat: torch.Tensor):
        if self.has_bias:
            ones = torch.ones(x_flat.shape[0], 1, device=x_flat.device, dtype=x_flat.dtype)
            x_flat = torch.cat([x_flat, ones], dim=1)
        n = x_flat.shape[0]
        a_batch = x_flat.t() @ x_flat / n
        g_batch = go_flat.t() @ go_flat / n
        return a_batch, g_batch

    def update(self, a_batch: torch.Tensor, g_batch: torch.Tensor, alpha_ema: float) -> None:
        if not self.initialized:
            self.A = a_batch.detach().clone()
            self.G = g_batch.detach().clone()
            self.initialized = True
        else:
            self.A = (1 - alpha_ema) * self.A + alpha_ema * a_batch.detach()
            self.G = (1 - alpha_ema) * self.G + alpha_ema * g_batch.detach()

    def precondition(self, grad_weight: torch.Tensor, grad_bias, tau: float):
        """grad_weight: any shape whose total size is out_dim*in_dim
        (e.g. a Conv2d weight's native (out_ch, in_ch, kh, kw) -- flattens
        to (out_dim, in_dim) in the SAME (Cin, kh, kw) row-major order the
        A factor's patches were built in via `_unfold_conv_input`, so this
        reshape is consistent with the Kronecker factors, not arbitrary).
        grad_bias: (out_dim,) or None (must match `self.has_bias`).
        Returns (g_tilde_weight, g_tilde_bias), reshaped back to the
        ORIGINAL grad_weight/grad_bias shapes."""
        g_w = grad_weight.reshape(self.out_dim, self.in_dim)
        if self.has_bias:
            g = torch.cat([g_w, grad_bias.reshape(-1, 1)], dim=1)
        else:
            g = g_w
        eye_a = torch.eye(self.A.shape[0], device=self.A.device)
        eye_g = torch.eye(self.G.shape[0], device=self.G.device)
        a_inv = torch.linalg.inv(self.A + tau * eye_a)
        g_inv = torch.linalg.inv(self.G + tau * eye_g)
        g_tilde = g_inv @ g @ a_inv
        if self.has_bias:
            return g_tilde[:, :-1].reshape(grad_weight.shape), g_tilde[:, -1].reshape_as(grad_bias)
        return g_tilde.reshape_as(grad_weight), None

    def condition_number(self, tau: float) -> float:
        eye_a = torch.eye(self.A.shape[0], device=self.A.device)
        eye_g = torch.eye(self.G.shape[0], device=self.G.device)
        cond_a = torch.linalg.cond(self.A + tau * eye_a).item()
        cond_g = torch.linalg.cond(self.G + tau * eye_g).item()
        return 0.5 * (cond_a + cond_g)

    def state_dict(self) -> dict:
        return {"A": self.A.cpu(), "G": self.G.cpu(), "initialized": self.initialized}

    def load_state_dict(self, state: dict, device) -> None:
        self.A = state["A"].to(device)
        self.G = state["G"].to(device)
        self.initialized = state["initialized"]


class OCARPreconditioner:
    """Curvature (Fisher/K-FAC) state and preconditioning for
    `MudviCNN`, matching the verified mechanism of the official OCAR
    repo's `SignSGDPlugin.before_update` -- see module docstring for
    the full algorithm and the two disclosed deviations (BatchNorm2d /
    grouped-conv excluded from preconditioning; no class-frequency loss
    reweighting).

    Primary entry point: `precondition_grad_(model)`, which rescales
    `.grad` IN PLACE after a normal `loss.backward()` on one combined
    (new + replay) batch -- the official algorithm's actual structure.
    `precondition_flat` is a secondary entry point used ONLY by the
    MUDVI+OCAR+GP ablation combination in `trainer.py`, which is NOT
    part of the published OCAR algorithm (see that method's docstring).
    """

    def __init__(self, model: nn.Module, device, lr: float,
                 alpha_ema: float = 1.0, regul: float = 0.01,
                 fim_update_every: int = 1):
        self.device = device
        self.alpha_ema = alpha_ema
        self.regul = regul
        self.fim_update_every = fim_update_every
        self.tau = lr  # matches their `if self.tau == 0: self.tau = optimizer lr`
        self._step_count = 0

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

        self._layers = {
            "temporal_conv.0.weight": _KFACLayer(
                in_dim=model.temporal_conv[0].in_channels * model.temporal_conv[0].kernel_size[0]
                * model.temporal_conv[0].kernel_size[1],
                out_dim=model.temporal_conv[0].out_channels,
                has_bias=model.temporal_conv[0].bias is not None, device=device),
            "pointwise_conv.0.weight": _KFACLayer(
                in_dim=model.pointwise_conv[0].in_channels,
                out_dim=model.pointwise_conv[0].out_channels,
                has_bias=model.pointwise_conv[0].bias is not None, device=device),
            "classifier.weight": _KFACLayer(
                in_dim=model.classifier.in_features, out_dim=model.classifier.out_features,
                has_bias=model.classifier.bias is not None, device=device),
        }

    def _make_fwd_hook(self, name: str):
        def hook(module, inputs, output):
            self._captured[f"{name}::input"] = inputs[0].detach()
        return hook

    def _make_bwd_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            self._captured[f"{name}::grad_output"] = grad_output[0].detach()
        return hook

    def _extract_batch_factors(self, name: str, module):
        x = self._captured.get(f"{name}::input")
        go = self._captured.get(f"{name}::grad_output")
        if x is None or go is None:
            return None
        if isinstance(module, nn.Linear):
            return x, go
        patches = _unfold_conv_input(x, module.kernel_size)
        b, cout, h, w = go.shape
        go_flat = go.permute(0, 2, 3, 1).reshape(-1, cout)
        return patches, go_flat

    def maybe_update_fisher(self) -> None:
        """Called once per training step (after the forward/backward
        pass(es) whose hooks populated `self._captured`). Matches the
        official code's `if self.iterations % train_epochs == 0` gating,
        generalized to a configurable `fim_update_every` -- their
        literal `train_epochs` (num. local epochs per experience, which
        collapses their gate to "every step" in the online-CL configs
        their paper reports) does not have the same meaning here, where
        `epochs_per_subject=15` is a different axis (this project trains
        multiple local epochs per subject even in its "online" setting).
        Default `fim_update_every=1` reproduces "every step", matching
        the effective behavior their reported experiments actually run
        under; set higher only for an explicit periodicity ablation."""
        should_update = (self._step_count % self.fim_update_every == 0)
        self._step_count += 1
        if not should_update:
            return
        for name, module in self._kfac_modules.items():
            factors = self._extract_batch_factors(name, module)
            if factors is None:
                continue
            x_flat, go_flat = factors
            a_batch, g_batch = self._layers[name].compute_factors(x_flat, go_flat)
            self._layers[name].update(a_batch, g_batch, self.alpha_ema)
        self.tau = self.tau + self.regul

    def precondition_grad_(self, model: nn.Module) -> None:
        """Rescales `.grad` in place for the 3 K-FAC'd layers (weight
        and bias jointly, via the bias-augmented Kronecker factor);
        every OTHER parameter's `.grad` (the grouped depthwise-conv
        weight, all BatchNorm2d affine parameters) is left completely
        untouched -- matching the official code's BatchNorm2d exclusion
        and its lack of grouped-conv support (see module docstring)."""
        named = dict(model.named_parameters())
        for name, layer in self._layers.items():
            weight = named[name]
            if weight.grad is None:
                continue
            bias_name = name.rsplit(".", 1)[0] + ".bias"
            bias = named.get(bias_name) if layer.has_bias else None
            bias_grad = bias.grad if bias is not None else None
            g_tilde_w, g_tilde_b = layer.precondition(weight.grad, bias_grad, self.tau)
            weight.grad = g_tilde_w
            if bias is not None and g_tilde_b is not None:
                bias.grad = g_tilde_b

    def precondition_flat(self, named_params: list[str], params: list, grad_flat: torch.Tensor) -> torch.Tensor:
        """Same per-layer preconditioning as `precondition_grad_`, but
        operating on a flat gradient vector instead of `.grad` in place.
        Used ONLY by the MUDVI+OCAR+GP ablation combination in
        `trainer.py`, which is NOT part of the published OCAR algorithm
        (that algorithm has no gradient-conflict-projection step) and
        needs a flat vector because it must compose with GP's own
        flat-vector projection first -- see
        `ContinualTrainer._train_step_ocar`'s docstring."""
        parts = {}
        idx = 0
        for name, p in zip(named_params, params):
            n = p.numel()
            parts[name] = grad_flat[idx:idx + n].view_as(p)
            idx += n
        for name, layer in self._layers.items():
            bias_name = name.rsplit(".", 1)[0] + ".bias"
            weight_grad = parts[name]
            bias_grad = parts.get(bias_name) if layer.has_bias else None
            g_tilde_w, g_tilde_b = layer.precondition(weight_grad, bias_grad, self.tau)
            parts[name] = g_tilde_w
            if bias_grad is not None:
                parts[bias_name] = g_tilde_b
        return torch.cat([parts[name].reshape(-1) for name in named_params])

    @property
    def layer_names(self) -> list[str]:
        """Read-only accessor: the K-FAC'd parameter names this
        preconditioner tracks (`temporal_conv.0.weight`,
        `pointwise_conv.0.weight`, `classifier.weight`). Added so
        `ocarpp.py` can reuse this class's Fisher/K-FAC state (hooks,
        running A/G factors, EMA update) by composition instead of
        duplicating it -- purely additive, does not change OCAR's own
        algorithm or state."""
        return list(self._layers.keys())

    def get_factors(self, name: str):
        """Read-only accessor: (A, G, has_bias, initialized) for one
        tracked layer's current running K-FAC factors. See `layer_names`
        docstring for why this exists."""
        layer = self._layers[name]
        return layer.A, layer.G, layer.has_bias, layer.initialized

    def avg_condition_number(self) -> float:
        vals = [l.condition_number(self.tau) for l in self._layers.values() if l.initialized]
        return sum(vals) / len(vals) if vals else float("nan")

    def state_dict(self) -> dict:
        return {
            "layers": {name: l.state_dict() for name, l in self._layers.items()},
            "tau": self.tau,
            "step_count": self._step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        for name, l in self._layers.items():
            l.load_state_dict(state["layers"][name], self.device)
        self.tau = state["tau"]
        self._step_count = state["step_count"]


@dataclass
class OCARStats:
    """Running log for OCAR (task requirement, parallel to
    `gradient_projection.GradientProjectionStats`): cosine similarity
    between the raw combined-batch gradient and its curvature-
    preconditioned version, the average K-FAC condition number, and
    memory loss before/after the update (same apples-to-apples
    eval-mode convention GP already uses)."""
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
