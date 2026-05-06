r"""
SMFE-specific loss functions for two-stream HGT training.

There are three losses here. The full training objective is

.. math::

    \mathcal{L}_{\mathrm{HGT}}
    = \mathcal{L}_{\mathrm{task}}
    + \lambda_{\mathrm{align}}\,\mathcal{L}_{\mathrm{align}}
    + \lambda_{\mathrm{xcov}}\,\mathcal{L}_{\mathrm{xcov,HGT}}
    + \lambda_{\mathrm{inv}}\,\mathcal{L}_{\mathrm{inv}}

where

* :math:`\mathcal{L}_{\mathrm{align}}` keeps the HGT output linearly
  decodable back to the SMFE factors :math:`(s_v, m_v)`. Stop-gradient on
  the target prevents collapse of the SMFE embeddings.
* :math:`\mathcal{L}_{\mathrm{xcov,HGT}}` enforces decorrelation between
  the recovered state and mechanism subspaces in a mini-batch.
* :math:`\mathcal{L}_{\mathrm{inv}}` is an optional IRMv1 penalty across
  environments, applied to a chosen logit / loss path.
"""

from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Alignment: HGT output should linearly recover SMFE factors
# ---------------------------------------------------------------------

def alignment_loss(
    s_pred: torch.Tensor,
    m_pred: torch.Tensor,
    s_target: torch.Tensor,
    m_target: torch.Tensor,
    stop_grad_target: bool = True,
) -> torch.Tensor:
    r"""
    .. math::
        \mathcal{L}_{\mathrm{align}}
        = \frac{1}{|B|} \sum_{v\in B}
            \big\| P_S\,h_v - \mathrm{sg}(s_v) \big\|_2^2
          + \big\| P_M\,h_v - \mathrm{sg}(m_v) \big\|_2^2

    Args
    ----
    s_pred  : (N, d_S) probe output P_S(h)
    m_pred  : (N, d_M) probe output P_M(h)
    s_target: (N, d_S) frozen SMFE state factor s_v
    m_target: (N, d_M) frozen SMFE mechanism factor m_v
    stop_grad_target : if True, gradient does not flow into the targets.

    Notes
    -----
    The stop-gradient is critical. Without it, gradients flow back into
    :math:`s_v, m_v` and the SMFE objective is undermined: HGT can satisfy
    alignment by pushing the SMFE embeddings toward HGT's outputs, which
    is the opposite of what we want.
    """
    if stop_grad_target:
        s_target = s_target.detach()
        m_target = m_target.detach()
    return F.mse_loss(s_pred, s_target) + F.mse_loss(m_pred, m_target)


# ---------------------------------------------------------------------
# Output-level cross-covariance penalty
# ---------------------------------------------------------------------

def xcov_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    Frobenius-norm cross-covariance penalty between two centered batches:

    .. math::
        \mathcal{L}_{\mathrm{xcov}} = \big\| \tfrac{1}{|B|} A^\top B \big\|_F^2

    where columns of :math:`A,B` are coordinate-wise centered.

    Driving this to zero kills all linear cross-coordinate correlation
    between :math:`A` and :math:`B`. This is exactly what we want at the
    HGT output: the recovered state subspace (a = P_S h) and the recovered
    mechanism subspace (b = P_M h) should not share linear structure.

    For nonlinear independence, swap to :func:`hsic_loss` (kernelized).
    """
    if a.size(0) != b.size(0):
        raise ValueError(f"batch dim mismatch: {a.size(0)} vs {b.size(0)}")
    n = a.size(0)
    if n < 2:
        return a.new_zeros(())
    a_c = a - a.mean(dim=0, keepdim=True)
    b_c = b - b.mean(dim=0, keepdim=True)
    cov = (a_c.transpose(0, 1) @ b_c) / max(n, 1)
    return (cov * cov).sum() + eps * 0.0  # eps kept as a placeholder for AMP-stability tweaks


def xcov_at_output_loss(
    s_pred: torch.Tensor,
    m_pred: torch.Tensor,
) -> torch.Tensor:
    """Convenience wrapper: cross-covariance between probe outputs."""
    return xcov_loss(s_pred, m_pred)


# ---------------------------------------------------------------------
# (Optional) HSIC penalty -- kernelized independence
# ---------------------------------------------------------------------

def _gaussian_kernel(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sq = (x * x).sum(dim=1, keepdim=True)
    d2 = sq + sq.transpose(0, 1) - 2.0 * (x @ x.transpose(0, 1))
    d2 = d2.clamp(min=0.0)
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def hsic_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    sigma_a: float = 1.0,
    sigma_b: float = 1.0,
) -> torch.Tensor:
    r"""
    HSIC :math:`(B-1)^{-2} \mathrm{tr}(K_S H K_M H)` with Gaussian kernels
    and centering :math:`H = I - B^{-1}\mathbf{1}\mathbf{1}^\top`.

    Use this in place of :func:`xcov_loss` when nonlinear independence is
    needed; it has a stronger guarantee (a characteristic-kernel HSIC = 0
    iff independent) but costs O(B^2) memory.
    """
    n = a.size(0)
    if n < 2:
        return a.new_zeros(())
    Ka = _gaussian_kernel(a, sigma_a)
    Kb = _gaussian_kernel(b, sigma_b)
    H = torch.eye(n, device=a.device, dtype=a.dtype) - 1.0 / n
    KaH = Ka @ H
    KbH = Kb @ H
    return (KaH * KbH.transpose(0, 1)).sum() / ((n - 1) ** 2)


# ---------------------------------------------------------------------
# IRMv1 invariance penalty
# ---------------------------------------------------------------------

def irmv1_penalty(
    env_logits: Sequence[torch.Tensor],
    env_targets: Sequence[torch.Tensor],
    loss_fn=F.cross_entropy,
) -> torch.Tensor:
    r"""
    Standard IRMv1 penalty (Arjovsky et al., 2019, eq. 6 of the paper):

    .. math::
        \mathcal{L}_{\mathrm{IRM}}
        = \sum_{e\in\mathcal{Q}} \big\| \nabla_{w}\big|_{w=1} \mathcal{R}_e(w\cdot \hat{f}) \big\|_2^2

    where :math:`\hat{f}` is the model's logits and :math:`w` is a dummy
    scalar pinned at 1.0 used only to take a gradient. Per-environment
    risks are computed with the same loss function.

    Args
    ----
    env_logits  : list of logits tensors, one per environment.
    env_targets : list of target tensors, one per environment.
    loss_fn     : per-environment loss; default cross_entropy.

    Returns
    -------
    scalar tensor.
    """
    if len(env_logits) != len(env_targets):
        raise ValueError("env_logits and env_targets must have same length")
    pen = env_logits[0].new_zeros(())
    for logits, y in zip(env_logits, env_targets):
        scale = torch.tensor(1.0, requires_grad=True, device=logits.device, dtype=logits.dtype)
        loss = loss_fn(logits * scale, y)
        g = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        pen = pen + (g * g).sum()
    return pen


# ---------------------------------------------------------------------
# Combined SMFE loss helper
# ---------------------------------------------------------------------

class SMFELossWeights:
    """Hyperparameter container for the SMFE add-on losses."""
    def __init__(
        self,
        lambda_align: float = 1.0,
        lambda_xcov: float = 1e-2,
        lambda_inv: float = 0.0,
    ):
        self.lambda_align = lambda_align
        self.lambda_xcov = lambda_xcov
        self.lambda_inv = lambda_inv


def smfe_total_loss(
    task_loss: torch.Tensor,
    s_pred: torch.Tensor,
    m_pred: torch.Tensor,
    s_target: torch.Tensor,
    m_target: torch.Tensor,
    weights: SMFELossWeights,
    inv_penalty: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, dict]:
    """
    Aggregate task loss + SMFE add-on losses with the given weights.

    Returns
    -------
    total : scalar loss tensor used for .backward()
    parts : dict of detached scalars for logging
    """
    al = alignment_loss(s_pred, m_pred, s_target, m_target, stop_grad_target=True)
    xc = xcov_at_output_loss(s_pred, m_pred)
    total = task_loss + weights.lambda_align * al + weights.lambda_xcov * xc
    if weights.lambda_inv > 0.0 and inv_penalty is not None:
        total = total + weights.lambda_inv * inv_penalty

    parts = {
        "task": float(task_loss.detach().item()),
        "align": float(al.detach().item()),
        "xcov": float(xc.detach().item()),
        "inv": float(inv_penalty.detach().item()) if inv_penalty is not None else 0.0,
        "total": float(total.detach().item()),
    }
    return total, parts
