"""OCAR++ -- this project's own extension of OCAR (`ocar.py`; Urettini &
Carta, "Online Curvature-Aware Replay", ICML 2025, arXiv:2502.01866,
official code github.com/edo-urettini/CL_stability). OCAR++ is NOT part
of the published paper or its official repo -- it is an additional
mechanism layered on top of OCAR's verified K-FAC/Fisher machinery,
kept in its own module and its own trainer code path (`trainer.py`'s
`_train_step_ocarpp`) so that `method="ocar"` (`ocar.py`,
`OCARPreconditioner`) stays byte-for-byte the verified baseline and is
never touched by this file.

WHAT OCAR++ CHANGES, CONCEPTUALLY:

  OCAR original:  gradient -> Fisher/K-FAC preconditioning -> update
  OCAR++:         gradient -> Fisher importance + historical drift
                   + directional conflict -> adaptive protection -> update

OCAR's own preconditioning (`OCARPreconditioner.precondition_grad_`)
rescales the gradient by the FULL damped Fisher inverse in every
direction, regardless of whether that direction is actually drifting
away from where the model has been consolidated. OCAR++ instead:

  1. Maintains a SLOW CONSOLIDATION ANCHOR theta*_t (an EMA of the
     parameters themselves, not of the gradient or the Fisher), for the
     same K-FAC'd layers OCAR already tracks curvature for:
        theta*_0 = theta_0
        theta*_{t+1} = beta_a * theta*_t + (1 - beta_a) * theta_{t+1}
     The drift delta_t = theta_t - theta*_t measures how far the CURRENT
     parameters have wandered from their slowly-consolidated anchor.

  2. Reuses OCAR's own running K-FAC factors A_l, G_l for each tracked
     layer (via `OCARPreconditioner.get_factors`, read-only -- this file
     never recomputes or duplicates the Fisher estimate) and, ONLY when
     it needs the eigenbasis, spectrally decomposes them:
        A_l = U_A Lambda_A U_A^T,  G_l = U_G Lambda_G U_G^T
     Both matrices are exact running covariances (x^T x / n, go^T go / n
     started from an identity prior), hence PSD by construction; a
     `.clamp(min=0)` on the eigenvalues only guards against float noise,
     it is not a modeling assumption.

  3. In that eigenbasis, projects both the gradient and the drift for
     each direction i:  g~_i = (U_G^T g U_A)_i,  delta~_i = (U_G^T
     delta U_A)_i (the bias column is folded into the A-side basis via
     the SAME bias-augmented-A convention `ocar.py`'s `_KFACLayer`
     already uses, so this stays consistent with the tracked factors).

  4. Computes a per-direction normalized conflict c_i = g~_i delta~_i /
     (|g~_i||delta~_i| + eps), keeps only the "gradient pushes further
     along the existing drift" sign via C_i = max(0, c_i), weights it by
     that direction's Fisher importance I_i = lambda_i / (lambda_i +
     eps) (lambda_i = lambda_G lambda_A for a Kronecker-factored
     direction), and protects high-importance/high-conflict directions
     more: kappa_i = 1 + gamma * I_i * C_i, g~'_i = g~_i / kappa_i.
     Directions with high importance but NO conflict (the gradient is
     not fighting the anchor -- e.g. it agrees with the drift's return
     to the anchor, or the drift is ~0) keep kappa_i ~= 1: full
     plasticity, unlike OCAR's original preconditioning, which damps
     every direction by the SAME Fisher-inverse regardless of conflict.

  5. Maps back g' = U_G g~' U_A^T, writes it into `.grad` in place (the
     same "rescale .grad, then let the trainer's normal optimizer.step()
     run" convention `OCARPreconditioner.precondition_grad_` uses), and
     the anchor is advanced with the POST-step parameters once
     `optimizer.step()` has run (see `trainer.py`).

Every parameter OCAR itself does not precondition (BatchNorm2d affine
params, the grouped depthwise-conv weight -- see `ocar.py`'s module
docstring for why) is likewise left untouched here: OCAR++ only adds a
conflict-aware rescaling on top of the SAME set of K-FAC'd layers OCAR
already covers, it does not extend K-FAC coverage to layers the
official algorithm does not support.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .ocar import OCARPreconditioner


def _eigh_psd(M: torch.Tensor):
    """Symmetric eigendecomposition of a PSD covariance matrix (M is
    always x^T x / n or go^T go / n, or the identity prior -- PSD by
    construction). Returns (eigvals, eigvecs), eigvals clamped to >= 0
    to guard against float noise around 0, not a modeling choice."""
    eigvals, eigvecs = torch.linalg.eigh(M)
    return eigvals.clamp(min=0.0), eigvecs


class _OCARPlusPlusLayer:
    """Per-layer OCAR++ state: the slow consolidation anchor for this
    layer's weight (+ bias, bias-augmented the same way `ocar.py`'s
    `_KFACLayer` augments A) and the cached eigenbasis of its current
    K-FAC factors."""

    def __init__(self, weight: torch.Tensor, bias, has_bias: bool):
        self.theta_star_w = weight.detach().clone()
        self.theta_star_b = bias.detach().clone() if has_bias else None
        self.has_bias = has_bias
        self.U_A = None
        self.lambda_A = None
        self.U_G = None
        self.lambda_G = None

    def refresh_eigenbasis(self, A: torch.Tensor, G: torch.Tensor) -> None:
        self.lambda_A, self.U_A = _eigh_psd(A)
        self.lambda_G, self.U_G = _eigh_psd(G)

    def drift(self, weight: torch.Tensor, bias) -> torch.Tensor:
        """delta_t = theta_t - theta*_t, bias-augmented (out_dim,
        in_dim + 1 if has_bias else in_dim), matching how the weight
        gradient itself gets bias-augmented before basis projection."""
        d_w = (weight.detach() - self.theta_star_w).reshape(self.U_G.shape[0], -1)
        if self.has_bias:
            d_b = (bias.detach() - self.theta_star_b).reshape(-1, 1)
            return torch.cat([d_w, d_b], dim=1)
        return d_w

    def update_anchor_(self, weight: torch.Tensor, bias, beta_a: float) -> None:
        self.theta_star_w.mul_(beta_a).add_(weight.detach(), alpha=1 - beta_a)
        if self.has_bias:
            self.theta_star_b.mul_(beta_a).add_(bias.detach(), alpha=1 - beta_a)

    def state_dict(self) -> dict:
        return {
            "theta_star_w": self.theta_star_w.cpu(),
            "theta_star_b": self.theta_star_b.cpu() if self.has_bias else None,
        }

    def load_state_dict(self, state: dict, device) -> None:
        self.theta_star_w = state["theta_star_w"].to(device)
        if self.has_bias:
            self.theta_star_b = state["theta_star_b"].to(device)


class OCARPlusPlusPreconditioner:
    """Conflict-Aware Fisher Preconditioning (OCAR++). Wraps an
    `OCARPreconditioner` (`ocar.py`) BY COMPOSITION to reuse its hooks
    and running K-FAC factors -- this class never recomputes the Fisher
    estimate itself, it only reads `self._base`'s factors and applies
    the additional drift/conflict-aware rescaling described in this
    module's docstring.

    `_base`'s own `precondition_grad_`/`precondition_flat` (the
    original OCAR rescaling) are never called by this class -- only
    `maybe_update_fisher()` (curvature estimation) and `get_factors()`
    (read-only access) are reused.
    """

    def __init__(self, model: nn.Module, device, lr: float,
                 alpha_ema: float = 1.0, regul: float = 0.01,
                 fim_update_every: int = 1,
                 beta_anchor: float = 0.999, gamma: float = 1.0,
                 eps: float = 1e-8):
        self.device = device
        self.beta_anchor = beta_anchor
        self.gamma = gamma
        self.eps = eps

        self._base = OCARPreconditioner(
            model, device, lr=lr, alpha_ema=alpha_ema, regul=regul,
            fim_update_every=fim_update_every,
        )
        self._last_conflict_sum = 0.0
        self._last_kappa_sum = 0.0
        self._last_direction_count = 0

        named = dict(model.named_parameters())
        self._layers: dict[str, _OCARPlusPlusLayer] = {}
        for name in self._base.layer_names:
            _, _, has_bias, _ = self._base.get_factors(name)
            weight = named[name]
            bias_name = name.rsplit(".", 1)[0] + ".bias"
            bias = named.get(bias_name) if has_bias else None
            self._layers[name] = _OCARPlusPlusLayer(weight, bias, has_bias)

    def maybe_update_fisher(self) -> None:
        """Advances OCAR's own Fisher/K-FAC estimate (unchanged
        mechanism, reused as-is), then refreshes OCAR++'s cached
        eigenbasis from the (possibly just-updated) factors."""
        self._base.maybe_update_fisher()
        for name, layer in self._layers.items():
            A, G, _, initialized = self._base.get_factors(name)
            if not initialized:
                continue
            layer.refresh_eigenbasis(A, G)

    def precondition_grad_(self, model: nn.Module) -> None:
        """Conflict-Aware Fisher Preconditioning: rescales `.grad` in
        place for each K-FAC'd layer using that direction's Fisher
        importance AND its conflict with the drift from the slow
        consolidation anchor -- see module docstring, steps 3-5.
        Layers without an initialized eigenbasis yet (first call,
        before any Fisher estimate exists) are left with their raw
        gradient, same convention as OCAR's own preconditioner before
        its first Fisher update."""
        named = dict(model.named_parameters())
        self._last_conflict_sum = 0.0
        self._last_kappa_sum = 0.0
        self._last_direction_count = 0
        for name, layer in self._layers.items():
            if layer.U_A is None:
                continue
            weight = named[name]
            if weight.grad is None:
                continue
            bias_name = name.rsplit(".", 1)[0] + ".bias"
            bias = named.get(bias_name) if layer.has_bias else None
            out_dim = layer.U_G.shape[0]
            a_dim = layer.U_A.shape[0]  # bias-augmented in_dim (in_dim+1 if has_bias)

            g_w = weight.grad.reshape(out_dim, -1)
            if layer.has_bias:
                g = torch.cat([g_w, bias.grad.reshape(-1, 1)], dim=1)
            else:
                g = g_w
            delta = layer.drift(weight, bias)

            g_tilde = layer.U_G.t() @ g @ layer.U_A
            delta_tilde = layer.U_G.t() @ delta @ layer.U_A

            c = (g_tilde * delta_tilde) / (g_tilde.abs() * delta_tilde.abs() + self.eps)
            C = c.clamp(min=0.0)
            lam = layer.lambda_G.unsqueeze(1) * layer.lambda_A.unsqueeze(0)
            I = lam / (lam + self.eps)
            kappa = 1.0 + self.gamma * I * C
            g_tilde_prime = g_tilde / kappa

            self._last_conflict_sum += C.sum().item()
            self._last_kappa_sum += kappa.sum().item()
            self._last_direction_count += C.numel()

            g_prime = layer.U_G @ g_tilde_prime @ layer.U_A.t()
            if layer.has_bias:
                weight.grad = g_prime[:, :a_dim - 1].reshape(weight.shape).contiguous()
                bias.grad = g_prime[:, -1].reshape_as(bias).contiguous()
            else:
                weight.grad = g_prime.reshape(weight.shape).contiguous()

    def update_anchors_(self, model: nn.Module) -> None:
        """theta*_{t+1} = beta_a * theta*_t + (1 - beta_a) * theta_{t+1}.
        Call AFTER `optimizer.step()` so theta_{t+1} is the post-update
        parameter, matching the formula in this module's docstring."""
        named = dict(model.named_parameters())
        for name, layer in self._layers.items():
            weight = named[name]
            bias_name = name.rsplit(".", 1)[0] + ".bias"
            bias = named.get(bias_name) if layer.has_bias else None
            layer.update_anchor_(weight, bias, self.beta_anchor)

    def avg_condition_number(self) -> float:
        return self._base.avg_condition_number()

    def last_diagnostics(self) -> tuple[float, float]:
        """(avg_directional_conflict, avg_protection_kappa) averaged
        over every direction of every K-FAC'd layer in the most recent
        `precondition_grad_` call. (0.0, 1.0) before any call has run
        (no directions preconditioned yet -- kappa's identity value)."""
        if self._last_direction_count == 0:
            return 0.0, 1.0
        n = self._last_direction_count
        return self._last_conflict_sum / n, self._last_kappa_sum / n

    def state_dict(self) -> dict:
        return {
            "base": self._base.state_dict(),
            "layers": {name: l.state_dict() for name, l in self._layers.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        self._base.load_state_dict(state["base"])
        for name, l in self._layers.items():
            l.load_state_dict(state["layers"][name], self.device)
        for name, layer in self._layers.items():
            A, G, _, initialized = self._base.get_factors(name)
            if initialized:
                layer.refresh_eigenbasis(A, G)


@dataclass
class OCARPlusPlusStats:
    """Running log for OCAR++, parallel to `ocar.OCARStats`: cosine
    similarity between the raw and conflict-aware-preconditioned
    gradient, the average K-FAC condition number (reused from OCAR's
    own diagnostic), the average directional conflict C and protection
    factor kappa actually applied, and memory loss before/after
    (same apples-to-apples eval-mode convention as OCAR/GP)."""
    total_steps: int = 0
    cosine_sim_sum: float = 0.0
    condition_number_sum: float = 0.0
    avg_conflict_sum: float = 0.0
    avg_kappa_sum: float = 0.0
    mem_loss_before_sum: float = 0.0
    mem_loss_after_sum: float = 0.0

    def log_step(self, cosine_sim: float, condition_number: float,
                 avg_conflict: float, avg_kappa: float,
                 mem_loss_before: float, mem_loss_after: float) -> None:
        self.total_steps += 1
        self.cosine_sim_sum += cosine_sim
        self.condition_number_sum += condition_number
        self.avg_conflict_sum += avg_conflict
        self.avg_kappa_sum += avg_kappa
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
            "avg_directional_conflict": self.avg_conflict_sum / n,
            "avg_protection_kappa": self.avg_kappa_sum / n,
            "avg_mem_loss_before": self.mem_loss_before_sum / n,
            "avg_mem_loss_after": self.mem_loss_after_sum / n,
        }
