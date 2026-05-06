"""
Canonical input format for the two-stream SMFE-HGT.

A single batch (or whole graph) is represented by an `SMFEBatch`. This is
the contract every loader must satisfy. It bundles the heterogeneous
graph (nodes, edges, types, timestamps), the semantic-domain definition,
and the precomputed SMFE state / mechanism embeddings for every node in
one place so the model has exactly one input shape to validate against.

Field contract
--------------
    s_feat              : (N, d_S_in)   float, SMFE state embedding s_v
    m_feat              : (N, d_M_in)   float, SMFE mechanism embedding m_v
    node_type           : (N,)          long,  node-type id in [0, num_types)
    edge_index          : (2, E)        long,  row 0 = source, row 1 = target
    edge_type           : (E,)          long,  relation id in [0, num_relations)
    edge_time           : (E,)          long,  timestamp (used by RTE; pass zeros if unused)
    node_type_to_domain : (num_types,)  long,  optional; type -> semantic domain id

Shapes must agree across fields:
    s_feat.size(0) == m_feat.size(0) == node_type.size(0)        = N
    edge_type.size(0) == edge_time.size(0) == edge_index.size(1) = E

`SMFEBatch.validate()` enforces this. Use `SMFEBatch.from_tensors(...)`
to construct from positional tensors, or build the dataclass directly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class SMFEBatch:
    s_feat: torch.Tensor
    m_feat: torch.Tensor
    node_type: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    edge_time: torch.Tensor
    node_type_to_domain: Optional[torch.Tensor] = None

    @classmethod
    def from_tensors(
        cls,
        s_feat: torch.Tensor,
        m_feat: torch.Tensor,
        node_type: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_time: torch.Tensor,
        node_type_to_domain: Optional[torch.Tensor] = None,
    ) -> "SMFEBatch":
        b = cls(
            s_feat=s_feat,
            m_feat=m_feat,
            node_type=node_type,
            edge_index=edge_index,
            edge_type=edge_type,
            edge_time=edge_time,
            node_type_to_domain=node_type_to_domain,
        )
        b.validate()
        return b

    @property
    def num_nodes(self) -> int:
        return self.node_type.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    @property
    def d_S_in(self) -> int:
        return self.s_feat.size(-1)

    @property
    def d_M_in(self) -> int:
        return self.m_feat.size(-1)

    def to(self, device) -> "SMFEBatch":
        return SMFEBatch(
            s_feat=self.s_feat.to(device),
            m_feat=self.m_feat.to(device),
            node_type=self.node_type.to(device),
            edge_index=self.edge_index.to(device),
            edge_type=self.edge_type.to(device),
            edge_time=self.edge_time.to(device),
            node_type_to_domain=(
                self.node_type_to_domain.to(device)
                if self.node_type_to_domain is not None else None
            ),
        )

    def validate(self) -> None:
        N = self.node_type.size(0)
        E = self.edge_index.size(1)
        if self.s_feat.dim() != 2 or self.s_feat.size(0) != N:
            raise ValueError(
                f"s_feat must be (N, d_S_in) with N={N}; got {tuple(self.s_feat.shape)}"
            )
        if self.m_feat.dim() != 2 or self.m_feat.size(0) != N:
            raise ValueError(
                f"m_feat must be (N, d_M_in) with N={N}; got {tuple(self.m_feat.shape)}"
            )
        if self.edge_index.dim() != 2 or self.edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must be (2, E); got {tuple(self.edge_index.shape)}"
            )
        if self.edge_type.shape != (E,):
            raise ValueError(f"edge_type must be (E,) with E={E}; got {tuple(self.edge_type.shape)}")
        if self.edge_time.shape != (E,):
            raise ValueError(f"edge_time must be (E,) with E={E}; got {tuple(self.edge_time.shape)}")
        for name, t in (
            ("node_type", self.node_type),
            ("edge_index", self.edge_index),
            ("edge_type", self.edge_type),
            ("edge_time", self.edge_time),
        ):
            if t.dtype != torch.long:
                raise TypeError(f"{name} must be long; got {t.dtype}")
        if self.node_type_to_domain is not None:
            if self.node_type_to_domain.dim() != 1:
                raise ValueError("node_type_to_domain must be 1-D")
            if self.node_type_to_domain.dtype != torch.long:
                raise TypeError("node_type_to_domain must be long")
