"""
Two-stream SMFE-coupled Heterogeneous Graph Transformer.

Architecture
------------
                            +---------------------+
            s_v  ---->      |  STATE HGT (intra)  |  ---->  h_S in R^{d_S}
                            +---------------------+
                                                                 \\
                                                                  >--->  CrossStreamAttn  --->  h
                                                                 /
                            +---------------------+
            m_v  ---->      |  MECH HGT  (inter)  |  ---->  h_M in R^{d_M}
                            +---------------------+

Key choices
-----------
* The two HGTs are completely independent modules. They have independent
  type-specific projections, independent relation matrices, independent
  per-edge softmax. There is no parameter sharing and no cross-talk in
  message passing -- exactly the property we need for the OOD risk bound
  on the mechanism stream to be informative.

* The state stream sees only intra-domain edges; the mechanism stream sees
  only inter-domain edges. Both streams see the full node set.

* Fusion happens only at the very end via cross-stream attention. The
  fusion is a per-node, per-head soft mixture of the two streams; the
  attention weights are returned for diagnostics.

* The model exposes h_S, h_M, and attn alongside the fused h, so the
  caller can apply alignment and decorrelation losses (see smfe.losses).
"""

from __future__ import annotations
from typing import Optional, Sequence
import torch
import torch.nn as nn

from pyHGT.model import GNN  # original UCLA-DM HGT, vendored unchanged
from .partition import partition_edges, make_node_type_to_domain
from .readout import CrossStreamAttentionReadout


class TwoStreamSMFEHGT(nn.Module):
    """
    Two-stream HGT with cross-stream attention readout.

    Args
    ----
    d_S_in, d_M_in : input feature dims for the SMFE state / mechanism
                     embeddings (i.e. dim of s_v and m_v).
    n_hid_S        : hidden / output dim of the state stream.
    n_hid_M        : hidden / output dim of the mechanism stream.
    n_hid_out      : fused output dim after the readout.
    num_types      : number of HGT node types.
    num_relations  : number of HGT relation types.
    n_heads_hgt    : attention heads inside each HGT stream.
    n_heads_readout: heads in the cross-stream readout.
    n_layers       : depth of each HGT stream.
    dropout        : dropout used inside HGT and the readout.
    prev_norm,
    last_norm      : LayerNorm flags inside HGT (see UCLA-DM/pyHGT).
    use_RTE        : enable Relative Temporal Encoding inside HGT.
    node_type_to_domain : optional length-num_types int sequence mapping
                     each node type id to its semantic domain id. If None,
                     each node type is its own domain (so all heterogeneous
                     edges are 'inter').
    """

    def __init__(
        self,
        d_S_in: int,
        d_M_in: int,
        n_hid_S: int = 128,
        n_hid_M: int = 128,
        n_hid_out: int = 256,
        num_types: int = 5,
        num_relations: int = 8,
        n_heads_hgt: int = 8,
        n_heads_readout: int = 4,
        n_layers: int = 3,
        dropout: float = 0.2,
        prev_norm: bool = True,
        last_norm: bool = True,
        use_RTE: bool = True,
        node_type_to_domain: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.d_S_in = d_S_in
        self.d_M_in = d_M_in
        self.n_hid_S = n_hid_S
        self.n_hid_M = n_hid_M
        self.n_hid_out = n_hid_out
        self.num_types = num_types
        self.num_relations = num_relations

        # Two independent HGT stacks. Same architecture, different params.
        self.state_gnn = GNN(
            in_dim=d_S_in,
            n_hid=n_hid_S,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=n_heads_hgt,
            n_layers=n_layers,
            dropout=dropout,
            conv_name="hgt",
            prev_norm=prev_norm,
            last_norm=last_norm,
            use_RTE=use_RTE,
        )
        self.mech_gnn = GNN(
            in_dim=d_M_in,
            n_hid=n_hid_M,
            num_types=num_types,
            num_relations=num_relations,
            n_heads=n_heads_hgt,
            n_layers=n_layers,
            dropout=dropout,
            conv_name="hgt",
            prev_norm=prev_norm,
            last_norm=last_norm,
            use_RTE=use_RTE,
        )

        self.readout = CrossStreamAttentionReadout(
            d_S=n_hid_S,
            d_M=n_hid_M,
            d_O=n_hid_out,
            n_heads=n_heads_readout,
            dropout=dropout,
            use_norm=True,
        )

        nt2d = make_node_type_to_domain(num_types, node_type_to_domain)
        # Buffer so it travels with the model (.to(device), .state_dict()).
        self.register_buffer("nt2d", nt2d)

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(
        self,
        s_feat: torch.Tensor,         # (N, d_S_in)
        m_feat: torch.Tensor,         # (N, d_M_in)
        node_type: torch.Tensor,      # (N,)
        edge_index: torch.Tensor,     # (2, E)
        edge_type: torch.Tensor,      # (E,)
        edge_time: torch.Tensor,      # (E,)
        return_intermediate: bool = False,
    ):
        """
        Returns
        -------
        if return_intermediate=False:
            h           : (N, n_hid_out)
        if return_intermediate=True:
            dict with keys:
                'h'      : fused output (N, n_hid_out)
                'h_S'    : state-stream output (N, n_hid_S)
                'h_M'    : mech-stream output (N, n_hid_M)
                'attn'   : (N, 2, n_heads_readout) cross-stream attention
                'masks'  : (intra_mask, inter_mask) edge masks for logging
        """
        intra, inter, masks = partition_edges(
            node_type, edge_index, edge_type, edge_time, self.nt2d
        )
        ei_S, et_S, ed_S = intra
        ei_M, et_M, ed_M = inter

        # Note: GNN.forward signature is
        #   (node_feature, node_type, edge_time, edge_index, edge_type)
        # -- edge_time precedes edge_index, which is unusual. Keep the order.
        h_S = self.state_gnn(s_feat, node_type, ed_S, ei_S, et_S)
        h_M = self.mech_gnn(m_feat, node_type, ed_M, ei_M, et_M)

        if return_intermediate:
            h, attn = self.readout(h_S, h_M, return_attn=True)
            return {
                "h": h,
                "h_S": h_S,
                "h_M": h_M,
                "attn": attn,
                "masks": masks,
            }
        return self.readout(h_S, h_M, return_attn=False)


# ---------------------------------------------------------------------
# Linear probes for the SMFE alignment loss
# ---------------------------------------------------------------------

class SMFEProbes(nn.Module):
    """
    Linear probes from the fused HGT output back to the SMFE factors.

    These probes participate in `smfe.losses.alignment_loss` and
    `smfe.losses.xcov_at_output_loss`. They are *trainable* but applied
    against a stop-gradient'd target s_v / m_v so that gradients flow
    HGT -> probes, not into the SMFE embeddings themselves.
    """

    def __init__(self, d_O: int, d_S: int, d_M: int):
        super().__init__()
        self.proj_S = nn.Linear(d_O, d_S)
        self.proj_M = nn.Linear(d_O, d_M)

    def forward(self, h: torch.Tensor):
        return self.proj_S(h), self.proj_M(h)
