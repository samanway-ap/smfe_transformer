"""
Targeted tests for stream-isolation properties of the two-stream HGT.

  test_stream_isolation_via_edges:
      With identical features fed to both streams, the only thing that can
      make the two streams produce different outputs is the edge partition.
      So we drive the state stream with intra-only edges and the mechanism
      stream with inter-only edges, and check that h_S != h_M except in
      degenerate cases.

  test_no_inter_edges_means_mech_is_pure_bias:
      With p_inter = 0, the mechanism stream gets no message passing.
      Its output should depend only on the per-type adapt_ws + the HGT
      update bias terms applied to m_feat. We verify this by comparing
      to a forward pass with all-zero inter edges.

Run:
    python -m tests.test_isolation
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from smfe import TwoStreamSMFEHGT, edge_partition_stats, make_node_type_to_domain
from tests.test_two_stream import build_synth_graph


def test_stream_isolation_via_edges():
    print("\n[isolation 1] with identical features, h_S != h_M because edges differ")
    torch.manual_seed(7)
    device = "cpu"

    num_types = 3
    num_relations = 4
    d_S = d_M = 16
    n_hid = 32

    node_type, edge_index, edge_type, edge_time = build_synth_graph(
        n_per_type=(12, 12, 12), num_relations=num_relations,
        p_intra=0.15, p_inter=0.10, device=device,
    )
    N = node_type.numel()

    model = TwoStreamSMFEHGT(
        d_S_in=d_S, d_M_in=d_M,
        n_hid_S=n_hid, n_hid_M=n_hid, n_hid_out=n_hid,
        num_types=num_types, num_relations=num_relations,
        n_heads_hgt=2, n_heads_readout=2, n_layers=2,
        dropout=0.0, use_RTE=False,
    ).to(device).eval()

    # Identical features.
    feat = torch.randn(N, d_S, device=device)
    out = model(feat, feat, node_type, edge_index, edge_type, edge_time,
                return_intermediate=True)

    diff = (out["h_S"] - out["h_M"]).abs().mean().item()
    print(f"  mean |h_S - h_M| = {diff:.4f}")
    assert diff > 1e-3, "h_S and h_M coincide; streams are not isolated"
    print("  PASS")


def test_no_inter_edges_means_no_mech_message_passing():
    print("\n[isolation 2] zero inter edges -> mech stream uses no neighbor info")
    torch.manual_seed(8)
    device = "cpu"

    num_types = 3
    num_relations = 4
    d_S = d_M = 16
    n_hid = 32

    # First graph with inter edges.
    nt1, ei1, et1, ed1 = build_synth_graph(
        n_per_type=(10, 10, 10), num_relations=num_relations,
        p_intra=0.15, p_inter=0.20, device=device, seed=8,
    )
    nt2d = make_node_type_to_domain(num_types)
    s1 = edge_partition_stats(nt1, ei1, nt2d)

    # Second graph: same nodes, only intra edges retained.
    intra_mask = nt2d[nt1[ei1[0]]].eq(nt2d[nt1[ei1[1]]])
    ei2 = ei1[:, intra_mask]
    et2 = et1[intra_mask]
    ed2 = ed1[intra_mask]

    model = TwoStreamSMFEHGT(
        d_S_in=d_S, d_M_in=d_M,
        n_hid_S=n_hid, n_hid_M=n_hid, n_hid_out=n_hid,
        num_types=num_types, num_relations=num_relations,
        n_heads_hgt=2, n_heads_readout=2, n_layers=2,
        dropout=0.0, use_RTE=False,
    ).to(device).eval()

    s_feat = torch.randn(nt1.numel(), d_S, device=device)
    m_feat = torch.randn(nt1.numel(), d_M, device=device)

    with torch.no_grad():
        out_full  = model(s_feat, m_feat, nt1, ei1, et1, ed1, return_intermediate=True)
        out_intra = model(s_feat, m_feat, nt1, ei2, et2, ed2, return_intermediate=True)

    # State stream sees only intra edges, which are the same in both graphs ->
    # h_S should be (numerically) identical.
    state_diff = (out_full["h_S"] - out_intra["h_S"]).abs().mean().item()
    # Mech stream sees inter edges only -> in graph 2 those are gone, so h_M
    # should differ from the full-graph case.
    mech_diff  = (out_full["h_M"] - out_intra["h_M"]).abs().mean().item()
    print(f"  state stream diff (should be ~0): {state_diff:.6f}")
    print(f"  mech  stream diff (should be > 0): {mech_diff:.4f}")
    print(f"  full-graph stats: {s1}")
    assert state_diff < 1e-5, "state stream changed when only inter edges were removed"
    assert mech_diff  > 1e-3, "mech stream output unchanged when its edges were removed"
    print("  PASS")


if __name__ == "__main__":
    test_stream_isolation_via_edges()
    test_no_inter_edges_means_no_mech_message_passing()
    print("\nAll isolation tests passed.")
