"""
SMFE add-ons for two-stream HGT.

Public API
----------
TwoStreamSMFEHGT       -- the model
SMFEProbes             -- linear probes from HGT output back to (s, m)
CrossStreamAttentionReadout -- exposed for advanced reuse / ablation

partition_edges        -- split edge tensors into intra / inter
make_node_type_to_domain

alignment_loss         -- HGT output should linearly recover (s_v, m_v)
hsic_loss              -- kernelized independence penalty (the DEFAULT)
hsic_unbiased          -- unbiased HSIC U-statistic for post-hoc measurement
xcov_loss / xcov_at_output_loss  -- linear proxy (HSIC's linear special case)
independence_penalty   -- dispatch HSIC / xcov from SMFELossWeights
irmv1_penalty          -- per-environment invariance term

SMFELossWeights, smfe_total_loss -- training-step helpers
"""

from .model import TwoStreamSMFEHGT, SMFEProbes
from .readout import CrossStreamAttentionReadout
from .partition import partition_edges, make_node_type_to_domain, edge_partition_stats
from .types import SMFEBatch
from .losses import (
    alignment_loss,
    xcov_loss,
    xcov_at_output_loss,
    hsic_loss,
    hsic_unbiased,
    independence_penalty,
    irmv1_penalty,
    SMFELossWeights,
    smfe_total_loss,
)

__all__ = [
    "TwoStreamSMFEHGT",
    "SMFEProbes",
    "CrossStreamAttentionReadout",
    "SMFEBatch",
    "partition_edges",
    "make_node_type_to_domain",
    "edge_partition_stats",
    "alignment_loss",
    "xcov_loss",
    "xcov_at_output_loss",
    "hsic_loss",
    "hsic_unbiased",
    "independence_penalty",
    "irmv1_penalty",
    "SMFELossWeights",
    "smfe_total_loss",
]
