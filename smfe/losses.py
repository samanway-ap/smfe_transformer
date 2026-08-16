r"""
SMFE-specific loss functions for two-stream HGT training.

The full training objective is

.. math::

    \mathcal{L}_{\mathrm{HGT}}
    = \mathcal{L}_{\mathrm{task}}
    + \lambda_{\mathrm{align}}\,\mathcal{L}_{\mathrm{align}}
    + \lambda_{\mathrm{indep}}\,\mathcal{L}_{\mathrm{indep}}
    + \lambda_{\mathrm{inv}}\,\mathcal{L}_{\mathrm{inv}}

where

* :math:`\mathcal{L}_{\mathrm{align}}` keeps the HGT output linearly
  decodable back to the SMFE factors :math:`(s_v, m_v)`. Stop-gradient on
  the target prevents collapse of the SMFE embeddings.
* :math:`\mathcal{L}_{\mathrm{indep}}` penalizes dependence between the
  recovered state and mechanism subspaces in a mini-batch. The default is
  **HSIC** under an RBF kernel (:func:`hsic_loss`), which is zero exactly
  at independence under a characteristic kernel. The linear proxy
  :func:`xcov_loss` (``penalty="xcov"``) is its special case at a linear
  kernel: it removes only the *linear* cross-covariance and is constant in
  any dependence that leaves the cross-covariance at zero. HSIC is the
  penalty the SMFE correspondence adopts because driving the bilinear form
  to zero leaves arbitrary higher-order dependence intact.
* :math:`\mathcal{L}_{\mathrm{inv}}` is an optional IRMv1 penalty across
  environments, applied to a chosen logit / loss path.

Notes on the HSIC estimator (following the correspondence, App. B / E):

* the RBF bandwidth on each block is the **median pairwise squared
  distance of the sample**, recomputed per call (``sigma=None``);
* the training penalty is estimated on a **subsample** of at most
  ``subsample`` entities per step (default 4096), using the biased
  V-statistic so it is differentiable;
* :func:`hsic_unbiased` provides the unbiased U-statistic used for
  post-hoc measurement of achieved dependence on a checkpoint.

Because the kernel penalty's gradient norm is much larger than the
bilinear one's, the training loop clips the data gradient and the penalty
gradient *separately* (a single joint clip lets the penalty consume the
whole budget); see ``train_smfe_hgt.py``.
"""

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple
import warnings
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
    between :math:`A` and :math:`B`, but a bilinear form can vanish with
    arbitrary higher-order dependence intact: it is the special case of
    HSIC at a *linear* kernel. It is kept as the ``penalty="xcov"`` option
    (and for reproducing the linear-penalty arm of the correspondence),
    but the default independence penalty is now :func:`hsic_loss`, which is
    zero exactly at independence under a characteristic kernel.
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
# HSIC penalty -- kernelized independence (the default independence term)
# ---------------------------------------------------------------------

def _pairwise_sq_dists(x: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance matrix, numerically floored at 0."""
    sq = (x * x).sum(dim=1, keepdim=True)
    d2 = sq + sq.transpose(0, 1) - 2.0 * (x @ x.transpose(0, 1))
    return d2.clamp(min=0.0)


def _median_bandwidth(d2: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""
    Median-heuristic bandwidth: the median of the *off-diagonal* pairwise
    squared distances. Recomputed per call so the kernel tracks the scale
    of the batch (App. B: "bandwidth is the median pairwise squared
    distance of the sample, recomputed per call"). Floored away from zero
    so a degenerate (all-identical) block does not divide by zero.
    """
    n = d2.size(0)
    if n < 2:
        return d2.new_tensor(1.0)
    iu = torch.triu_indices(n, n, offset=1, device=d2.device)
    med = d2[iu[0], iu[1]].median()
    return med.clamp(min=eps)


def _rbf_kernel(x: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    r"""
    RBF (Gaussian) kernel :math:`\exp(-\lVert x_i-x_j\rVert^2 / \text{bw})`.

    ``sigma is None`` selects the median heuristic, with the bandwidth
    ``bw`` set equal to the median off-diagonal squared distance. A float
    ``sigma`` uses the classic ``bw = 2\,\sigma^2`` form, matching the old
    fixed-bandwidth behavior.
    """
    d2 = _pairwise_sq_dists(x)
    if sigma is None:
        bw = _median_bandwidth(d2)
    else:
        bw = d2.new_tensor(2.0 * sigma * sigma)
    return torch.exp(-d2 / bw)


def _subsample_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    subsample: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Take a common random row subset of ``a``/``b`` when ``n > subsample``."""
    n = a.size(0)
    if subsample is None or n <= subsample:
        return a, b
    idx = torch.randperm(n, device=a.device)[:subsample]
    return a[idx], b[idx]


def hsic_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    sigma_a: Optional[float] = None,
    sigma_b: Optional[float] = None,
    subsample: Optional[int] = 4096,
) -> torch.Tensor:
    r"""
    Biased (V-statistic) HSIC used as the training independence penalty:

    .. math::
        \widehat{\mathrm{HSIC}} = (n-1)^{-2}\,\mathrm{tr}(K_a H K_b H),
        \qquad H = I - n^{-1}\mathbf{1}\mathbf{1}^\top,

    with RBF kernels :math:`K_a, K_b`. Under a characteristic kernel HSIC
    is zero exactly at independence, so unlike :func:`xcov_loss` it is a
    function of the whole joint law of ``(a, b)`` and not only its bilinear
    part. This is the penalty the SMFE correspondence adopts for the
    independence term.

    Args
    ----
    a, b     : (N, d) batches (e.g. recovered state / mechanism subspaces).
    sigma_a, sigma_b : RBF bandwidths. ``None`` (default) uses the median
        heuristic, recomputed per block per call.
    subsample : if ``N`` exceeds this, HSIC is estimated on a random subset
        of ``subsample`` rows (default 4096, per the correspondence). Pass
        ``None`` to use the full batch. Kernels cost ``O(min(N, subsample)^2)``.

    Notes
    -----
    The V-statistic is differentiable and is what the training loop
    descends on; for post-hoc measurement of achieved dependence use the
    unbiased U-statistic :func:`hsic_unbiased`.
    """
    if a.size(0) != b.size(0):
        raise ValueError(f"batch dim mismatch: {a.size(0)} vs {b.size(0)}")
    a, b = _subsample_rows(a, b, subsample)
    n = a.size(0)
    if n < 2:
        return a.new_zeros(())
    Ka = _rbf_kernel(a, sigma_a)
    Kb = _rbf_kernel(b, sigma_b)
    H = torch.eye(n, device=a.device, dtype=a.dtype) - 1.0 / n
    KaH = Ka @ H
    KbH = Kb @ H
    return (KaH * KbH.transpose(0, 1)).sum() / ((n - 1) ** 2)


@torch.no_grad()
def hsic_unbiased(
    a: torch.Tensor,
    b: torch.Tensor,
    sigma_a: Optional[float] = None,
    sigma_b: Optional[float] = None,
    subsample: Optional[int] = 8192,
) -> torch.Tensor:
    r"""
    Unbiased (U-statistic) HSIC estimator, for **measuring** the achieved
    dependence on a trained checkpoint (Table 7 reports HSIC "unbiased at
    n = 8192"). Not intended as a training loss: it is not guaranteed
    non-negative and can return small values of either sign near
    independence.

    Uses the Song et al. (2007) estimator on kernels with zeroed
    diagonals:

    .. math::
        \frac{1}{n(n-3)}\Big[\mathrm{tr}(\tilde K\tilde L)
        + \frac{\mathbf{1}^\top\tilde K\mathbf{1}\,\mathbf{1}^\top\tilde L\mathbf{1}}{(n-1)(n-2)}
        - \frac{2}{n-2}\,\mathbf{1}^\top\tilde K\tilde L\mathbf{1}\Big].
    """
    if a.size(0) != b.size(0):
        raise ValueError(f"batch dim mismatch: {a.size(0)} vs {b.size(0)}")
    a, b = _subsample_rows(a, b, subsample)
    n = a.size(0)
    if n < 4:
        return a.new_zeros(())
    K = _rbf_kernel(a, sigma_a).clone()
    L = _rbf_kernel(b, sigma_b).clone()
    K.fill_diagonal_(0.0)
    L.fill_diagonal_(0.0)
    ones = a.new_ones(n)
    KL_trace = (K * L.transpose(0, 1)).sum()          # tr(K L)
    sumK = ones @ K @ ones
    sumL = ones @ L @ ones
    KL_row = (K @ ones) @ (L @ ones)                  # 1^T K L 1
    term = (
        KL_trace
        + (sumK * sumL) / ((n - 1) * (n - 2))
        - 2.0 / (n - 2) * KL_row
    )
    return term / (n * (n - 3))


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
    """
    Hyperparameter container for the SMFE add-on losses.

    Parameters
    ----------
    lambda_align : weight on the alignment loss.
    lambda_indep : weight on the independence penalty (whichever is
        selected by ``penalty``).
    lambda_inv   : weight on the optional IRMv1 penalty.
    penalty      : ``"hsic"`` (default; kernelized, the correspondence's
        choice) or ``"xcov"`` (linear special case).
    hsic_sigma   : RBF bandwidth for the HSIC penalty; ``None`` uses the
        median heuristic.
    hsic_subsample : entities sampled per step for the HSIC estimate.
    lambda_xcov  : **deprecated** alias. If given, selects ``penalty="xcov"``
        and sets ``lambda_indep`` to its value, preserving the old
        linear-penalty behavior.
    """
    def __init__(
        self,
        lambda_align: float = 1.0,
        lambda_indep: float = 1e-2,
        lambda_inv: float = 0.0,
        penalty: str = "hsic",
        hsic_sigma: Optional[float] = None,
        hsic_subsample: Optional[int] = 4096,
        lambda_xcov: Optional[float] = None,
    ):
        if lambda_xcov is not None:
            warnings.warn(
                "lambda_xcov is deprecated; the default independence penalty "
                "is now HSIC. Passing lambda_xcov selects the linear proxy "
                "(penalty='xcov') for backward compatibility. Use "
                "lambda_indep together with penalty='hsic'/'xcov' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            penalty = "xcov"
            lambda_indep = lambda_xcov
        if penalty not in ("hsic", "xcov"):
            raise ValueError(f"penalty must be 'hsic' or 'xcov', got {penalty!r}")
        self.lambda_align = lambda_align
        self.lambda_indep = lambda_indep
        self.lambda_inv = lambda_inv
        self.penalty = penalty
        self.hsic_sigma = hsic_sigma
        self.hsic_subsample = hsic_subsample

    @property
    def lambda_xcov(self) -> float:
        """Back-compat read accessor: the independence weight when the
        linear proxy is active, else 0.0."""
        return self.lambda_indep if self.penalty == "xcov" else 0.0


def independence_penalty(
    s_pred: torch.Tensor,
    m_pred: torch.Tensor,
    weights: SMFELossWeights,
) -> torch.Tensor:
    """Independence penalty selected by ``weights.penalty`` (HSIC or xcov)."""
    if weights.penalty == "hsic":
        return hsic_loss(
            s_pred, m_pred,
            sigma_a=weights.hsic_sigma, sigma_b=weights.hsic_sigma,
            subsample=weights.hsic_subsample,
        )
    return xcov_at_output_loss(s_pred, m_pred)


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

    The independence term is HSIC by default (``weights.penalty == "hsic"``)
    and the linear cross-covariance proxy when ``weights.penalty == "xcov"``.

    Returns
    -------
    total : scalar loss tensor used for .backward()
    parts : dict of detached scalars for logging. Always carries an
        ``"indep"`` key (the active penalty's value); the per-penalty keys
        ``"hsic"`` / ``"xcov"`` mirror it and are 0.0 when inactive.
    """
    al = alignment_loss(s_pred, m_pred, s_target, m_target, stop_grad_target=True)
    indep = independence_penalty(s_pred, m_pred, weights)
    total = task_loss + weights.lambda_align * al + weights.lambda_indep * indep
    if weights.lambda_inv > 0.0 and inv_penalty is not None:
        total = total + weights.lambda_inv * inv_penalty

    indep_val = float(indep.detach().item())
    parts = {
        "task": float(task_loss.detach().item()),
        "align": float(al.detach().item()),
        "indep": indep_val,
        "penalty": weights.penalty,
        "hsic": indep_val if weights.penalty == "hsic" else 0.0,
        "xcov": indep_val if weights.penalty == "xcov" else 0.0,
        "inv": float(inv_penalty.detach().item()) if inv_penalty is not None else 0.0,
        "total": float(total.detach().item()),
    }
    return total, parts
