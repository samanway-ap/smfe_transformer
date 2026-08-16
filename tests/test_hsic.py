"""
Unit tests for the HSIC independence penalty that replaced the linear
cross-covariance proxy as the SMFE independence term.

These exercise the loss functions only (pure PyTorch), so they run without
PyTorch-Geometric / the vendored pyHGT stack. We import ``smfe.losses``
directly by file path to avoid triggering ``smfe/__init__`` (which pulls in
the model and therefore torch_geometric).

What is checked:

  1) Median-heuristic bandwidth is used when sigma is None, and it tracks
     the scale of the batch (x vs 10x reads as the same dependence).
  2) HSIC registers a nonlinear (even-function) coupling that has zero
     linear cross-covariance -- exactly the case the linear proxy is blind
     to. This is the whole reason for the swap.
  3) The unbiased estimator sits at the floor (~0) on independent blocks.
  4) Subsampling caps the kernel size and stays finite / differentiable.
  5) SMFELossWeights defaults to HSIC; the deprecated lambda_xcov alias
     still selects the linear proxy.

Run:
    python -m tests.test_hsic
"""
import os
import importlib.util
import warnings

import torch

# Load smfe/losses.py directly, without importing the smfe package (whose
# __init__ imports the model and needs torch_geometric).
_LOSSES_PATH = os.path.join(os.path.dirname(__file__), "..", "smfe", "losses.py")
_spec = importlib.util.spec_from_file_location("smfe_losses_under_test", _LOSSES_PATH)
losses = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(losses)

hsic_loss = losses.hsic_loss
hsic_unbiased = losses.hsic_unbiased
xcov_loss = losses.xcov_loss
SMFELossWeights = losses.SMFELossWeights
independence_penalty = losses.independence_penalty


def test_median_bandwidth_is_scale_equivariant():
    print("\n[hsic 1] median-heuristic bandwidth tracks batch scale")
    torch.manual_seed(0)
    n, d = 256, 8
    a = torch.randn(n, d)
    # b is a deterministic (nonlinear) function of a -> fully dependent.
    b = torch.tanh(a) + 0.1 * a**2

    h1 = hsic_loss(a, b).item()
    # Rescale both blocks by 10x; the median heuristic should rescale the
    # bandwidth with them, leaving the statistic essentially unchanged.
    h10 = hsic_loss(10.0 * a, 10.0 * b).item()
    print(f"  HSIC(x) = {h1:.6e} | HSIC(10x) = {h10:.6e}")
    assert h1 > 0
    rel = abs(h1 - h10) / max(h1, 1e-12)
    assert rel < 1e-3, f"median bandwidth not scale-equivariant: rel diff {rel:.3e}"
    print("  PASS")


def test_hsic_sees_nonlinear_coupling_that_xcov_misses():
    print("\n[hsic 2] HSIC detects a coupling the linear proxy is blind to")
    torch.manual_seed(1)
    n = 1024
    s = torch.randn(n, 1)
    # m depends on s only through an even function -> Cov(s, m) ~ 0 but the
    # variables are strongly dependent. This is the correspondence's
    # "bilinear form vanishes with higher-order dependence intact" case.
    m = (s**2 - 1.0) / (2.0**0.5)

    xc = xcov_loss(s, m).item()
    hs = hsic_loss(s, m).item()

    # A genuinely independent control at matched shapes.
    m_indep = torch.randn(n, 1)
    hs_indep = hsic_loss(s, m_indep).item()

    print(f"  xcov(s, s^2) = {xc:.3e}  (linear proxy ~ blind)")
    print(f"  HSIC(s, s^2) = {hs:.3e}  vs  HSIC(s, indep) = {hs_indep:.3e}")
    assert xc < 1e-3, f"expected near-zero linear cross-cov, got {xc:.3e}"
    assert hs > 10 * max(hs_indep, 1e-12), \
        "HSIC failed to separate nonlinear dependence from independence"
    print("  PASS")


def test_unbiased_hsic_floor_on_independent_blocks():
    print("\n[hsic 3] unbiased HSIC ~ 0 on independent blocks")
    torch.manual_seed(2)
    n, d = 512, 6
    a = torch.randn(n, d)
    b = torch.randn(n, d)
    u_indep = hsic_unbiased(a, b).item()
    # Dependent blocks should read clearly above the floor.
    b_dep = torch.tanh(a)
    u_dep = hsic_unbiased(a, b_dep).item()
    print(f"  unbiased HSIC indep = {u_indep:.3e} | dep = {u_dep:.3e}")
    assert abs(u_indep) < 1e-2, f"unbiased HSIC not near zero on independent data: {u_indep:.3e}"
    assert u_dep > abs(u_indep), "unbiased HSIC did not rise on dependent data"
    print("  PASS")


def test_subsample_caps_kernel_and_is_differentiable():
    print("\n[hsic 4] subsampling caps kernel size and keeps gradients")
    torch.manual_seed(3)
    n, d = 5000, 4
    a = torch.randn(n, d, requires_grad=True)
    b = torch.randn(n, d, requires_grad=True)
    val = hsic_loss(a, b, subsample=256)
    assert torch.isfinite(val), "HSIC returned a non-finite value"
    val.backward()
    # Exactly the sampled rows (<= 256) should carry gradient on each side.
    nz_a = int((a.grad.abs().sum(dim=1) > 0).sum())
    print(f"  n={n}, subsample=256 -> rows with grad on a: {nz_a}")
    assert 0 < nz_a <= 256, f"subsample did not cap the estimate: {nz_a} rows"
    print("  PASS")


def test_weights_default_to_hsic_and_alias_is_backcompat():
    print("\n[hsic 5] SMFELossWeights defaults to HSIC; lambda_xcov alias works")
    w = SMFELossWeights(lambda_align=1.0, lambda_indep=1e-2)
    assert w.penalty == "hsic"
    assert w.lambda_xcov == 0.0  # inactive linear proxy reads as 0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        w_old = SMFELossWeights(lambda_align=1.0, lambda_xcov=5e-3)
    assert w_old.penalty == "xcov", "lambda_xcov alias did not select the linear proxy"
    assert abs(w_old.lambda_indep - 5e-3) < 1e-12
    assert any(issubclass(c.category, DeprecationWarning) for c in caught), \
        "expected a DeprecationWarning for lambda_xcov"

    # independence_penalty should route to the selected penalty. With
    # n < subsample there is no random subsampling, so the dispatched value
    # matches the underlying estimator exactly.
    torch.manual_seed(4)
    s = torch.randn(128, 3)
    m = s**2 - 1.0
    assert torch.allclose(independence_penalty(s, m, w), hsic_loss(s, m)), \
        "penalty='hsic' did not route to hsic_loss"
    assert torch.allclose(independence_penalty(s, m, w_old), xcov_loss(s, m)), \
        "penalty='xcov' did not route to xcov_loss"
    print("  PASS")


if __name__ == "__main__":
    test_median_bandwidth_is_scale_equivariant()
    test_hsic_sees_nonlinear_coupling_that_xcov_misses()
    test_unbiased_hsic_floor_on_independent_blocks()
    test_subsample_caps_kernel_and_is_differentiable()
    test_weights_default_to_hsic_and_alias_is_backcompat()
    print("\nAll HSIC tests passed.")
