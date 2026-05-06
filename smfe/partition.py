"""
Edge partitioning for the two-stream SMFE-HGT.

Given a PyG-style heterogeneous batch (node_type, edge_index, edge_type,
edge_time) and a node_type -> semantic_domain map, split the edge tensors
into two disjoint subsets:

    intra (state)      : edges where domain[src] == domain[tgt]
    inter (mechanism)  : edges where domain[src] != domain[tgt]

Both subsets share the same node tensor; only the edge views differ.

Notes on isolated nodes
-----------------------
A node may have no edges in one of the two views (e.g. a Field node with no
within-domain Field-Field edges). The HGTConv `update()` step still runs on
it, with aggregated message = 0, so the node receives the bias of
a_linears[t] mixed in via the residual gate. This is acceptable: it's a
learnable per-type contribution, not a stale value. We document it here so
users do not assume that isolated-in-stream nodes are passthrough.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Optional
import torch


def make_node_type_to_domain(
    num_types: int,
    mapping: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Build a long tensor of length `num_types` where entry t is the semantic
    domain id of node type t. Default: identity (every type is its own domain).

    Pass a custom mapping when several node types belong to the same
    semantic domain. For example, in OAG you may want to lump
    {Author, Institute} into a single 'people' domain so that
    Author-Institute edges count as intra rather than inter.
    """
    if mapping is None:
        return torch.arange(num_types, dtype=torch.long)
    if len(mapping) != num_types:
        raise ValueError(
            f"mapping length {len(mapping)} != num_types {num_types}"
        )
    return torch.as_tensor(mapping, dtype=torch.long)


def partition_edges(
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    edge_time: torch.Tensor,
    nt2d: torch.Tensor,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Split (edge_index, edge_type, edge_time) into intra and inter views.

    Args
    ----
    node_type : (N,)        long tensor of node-type ids
    edge_index: (2, E)      long tensor; row 0 is source, row 1 is target
    edge_type : (E,)        long tensor of relation ids
    edge_time : (E,)        long tensor of edge timestamps (used by RTE)
    nt2d      : (T,)        long tensor mapping node type -> semantic domain

    Returns
    -------
    intra  : (edge_index_intra, edge_type_intra, edge_time_intra)
    inter  : (edge_index_inter, edge_type_inter, edge_time_inter)
    masks  : (intra_mask, inter_mask)  -- bool tensors of shape (E,)
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must be (2, E); got shape {tuple(edge_index.shape)}"
        )
    src = edge_index[0]
    tgt = edge_index[1]
    nt2d = nt2d.to(node_type.device)
    d_src = nt2d[node_type[src]]
    d_tgt = nt2d[node_type[tgt]]

    intra_mask = d_src.eq(d_tgt)
    inter_mask = ~intra_mask

    intra = (
        edge_index[:, intra_mask],
        edge_type[intra_mask],
        edge_time[intra_mask],
    )
    inter = (
        edge_index[:, inter_mask],
        edge_type[inter_mask],
        edge_time[inter_mask],
    )
    return intra, inter, (intra_mask, inter_mask)


def edge_partition_stats(
    node_type: torch.Tensor,
    edge_index: torch.Tensor,
    nt2d: torch.Tensor,
) -> dict:
    """Diagnostic counts for a sanity check / logging."""
    src = edge_index[0]
    tgt = edge_index[1]
    nt2d = nt2d.to(node_type.device)
    d_src = nt2d[node_type[src]]
    d_tgt = nt2d[node_type[tgt]]
    intra = d_src.eq(d_tgt).sum().item()
    inter = (~d_src.eq(d_tgt)).sum().item()
    n_isolated_intra = 0
    n_isolated_inter = 0
    if edge_index.numel() > 0:
        # node has at least one intra edge if it is endpoint of any intra edge
        intra_mask = d_src.eq(d_tgt)
        in_intra = torch.zeros(node_type.size(0), dtype=torch.bool, device=node_type.device)
        in_intra[src[intra_mask]] = True
        in_intra[tgt[intra_mask]] = True
        in_inter = torch.zeros(node_type.size(0), dtype=torch.bool, device=node_type.device)
        in_inter[src[~intra_mask]] = True
        in_inter[tgt[~intra_mask]] = True
        n_isolated_intra = int((~in_intra).sum().item())
        n_isolated_inter = int((~in_inter).sum().item())
    return {
        "n_intra_edges": intra,
        "n_inter_edges": inter,
        "n_nodes_isolated_in_intra": n_isolated_intra,
        "n_nodes_isolated_in_inter": n_isolated_inter,
    }
