"""
Cross-stream attention readout.

After both HGT streams produce per-node hidden vectors

    h_S[v] in R^{d_S}      (state stream)
    h_M[v] in R^{d_M}      (mechanism stream)

we need to combine them into a single vector h[v] in R^{d_O} that the
downstream task head consumes.

A naive sum h = h_S + h_M ties the two streams' contributions together with
no learned tradeoff and (worse) reintroduces entanglement at layer L just
after we paid for keeping them separate at layers 0..L-1. A gated combiner
g_r * h_S + (1-g_r) * h_M is interpretable but produces a single scalar mix
per relation, which is too coarse when different nodes need different
mixes.

Cross-stream attention solves both problems. We treat (h_S, h_M) as a
length-2 token sequence per node, and let the model learn how much weight
to give each stream, per node, per attention head, via a learnable global
query. Because softmax is over only 2 tokens, the operation reduces to a
sigmoid-like gate per head, but with the proper attention scaling and a
multi-head ensemble.

Lipschitz note. The fused output is
    h[v] = sum_h alpha_S^{(h)}(v) * z_S^{(h)}(v) + alpha_M^{(h)}(v) * z_M^{(h)}(v)
with alpha_S + alpha_M = 1 per head. Hence the Lipschitz constant of h
w.r.t. the mechanism factor is bounded by max_h alpha_M^{(h)} times the
projection norm. This gives a per-batch, per-node *measurable* upper bound
on mechanism reliance, which we expose via `forward(..., return_attn=True)`.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class CrossStreamAttentionReadout(nn.Module):
    """
    Fuse (h_S, h_M) -> h via two-token multi-head attention.

    Args
    ----
    d_S, d_M  : input dims of the state and mechanism streams
    d_O       : output dim (must be divisible by n_heads)
    n_heads   : number of attention heads
    dropout   : dropout applied after the output projection
    use_norm  : if True, applies LayerNorm to the fused output
    """

    def __init__(
        self,
        d_S: int,
        d_M: int,
        d_O: int,
        n_heads: int = 4,
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        super().__init__()
        if d_O % n_heads != 0:
            raise ValueError(
                f"d_O ({d_O}) must be divisible by n_heads ({n_heads})"
            )
        self.d_S = d_S
        self.d_M = d_M
        self.d_O = d_O
        self.n_heads = n_heads
        self.d_head = d_O // n_heads

        # Per-stream projections into a common d_O space, split across heads.
        self.proj_S = nn.Linear(d_S, d_O)
        self.proj_M = nn.Linear(d_M, d_O)

        # Learnable per-head global query. Small init for stable softmax start.
        self.q = nn.Parameter(torch.randn(n_heads, self.d_head) * 0.02)

        # Output projection after concat.
        self.out = nn.Linear(d_O, d_O)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_O) if use_norm else nn.Identity()

        self._sqrt_dh = math.sqrt(self.d_head)

    def forward(
        self,
        h_S: torch.Tensor,
        h_M: torch.Tensor,
        return_attn: bool = False,
    ):
        """
        Args
        ----
        h_S : (N, d_S)
        h_M : (N, d_M)

        Returns
        -------
        h          : (N, d_O)
        attn       : (N, 2, n_heads), only if return_attn=True.
                     attn[:, 0, h] = state weight in head h
                     attn[:, 1, h] = mechanism weight in head h
        """
        if h_S.size(0) != h_M.size(0):
            raise ValueError(
                f"h_S and h_M must have same N; got {h_S.size(0)} vs {h_M.size(0)}"
            )
        N = h_S.size(0)

        # Project and split into heads: (N, n_heads, d_head)
        zS = self.proj_S(h_S).view(N, self.n_heads, self.d_head)
        zM = self.proj_M(h_M).view(N, self.n_heads, self.d_head)

        # Stack as length-2 sequence: (N, 2, n_heads, d_head)
        Z = torch.stack([zS, zM], dim=1)

        # Score per (node, stream, head): broadcast q over N and stream axis.
        # q: (n_heads, d_head) -> (1, 1, n_heads, d_head)
        q = self.q.view(1, 1, self.n_heads, self.d_head)
        scores = (Z * q).sum(dim=-1) / self._sqrt_dh   # (N, 2, n_heads)
        attn = torch.softmax(scores, dim=1)            # softmax over 2 streams

        # Weighted sum across the 2-token axis: (N, n_heads, d_head)
        fused = (attn.unsqueeze(-1) * Z).sum(dim=1)
        fused = fused.reshape(N, self.d_O)

        out = self.norm(self.drop(self.out(fused)))

        if return_attn:
            return out, attn
        return out

    def extra_repr(self) -> str:
        return (
            f"d_S={self.d_S}, d_M={self.d_M}, d_O={self.d_O}, "
            f"n_heads={self.n_heads}"
        )
