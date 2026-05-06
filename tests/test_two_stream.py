"""
Smoke test: build a tiny synthetic heterogeneous graph, run the two-stream
SMFE-HGT forward and backward, and verify:

  1) Output shapes are correct.
  2) Edge partition stats match expectations.
  3) Cross-stream attention weights are valid (non-negative, sum to 1).
  4) Gradients flow into BOTH the state stream and the mechanism stream.
  5) The full SMFE loss (task + align + xcov) runs and decreases on a
     deliberately-easy fitting problem.
  6) When one stream is given uninformative inputs, the readout's attention
     learns to down-weight it.

Run:
    cd /path/to/smfe-hgt
    python -m tests.test_two_stream
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.nn.functional as F

from smfe import (
    TwoStreamSMFEHGT,
    SMFEProbes,
    alignment_loss,
    xcov_at_output_loss,
    smfe_total_loss,
    SMFELossWeights,
    edge_partition_stats,
    make_node_type_to_domain,
)


# -----------------------------------------------------------------------
# Synthetic heterogeneous graph
# -----------------------------------------------------------------------

def build_synth_graph(
    n_per_type=(20, 30, 25, 15),     # 4 node types
    p_intra=0.10,                    # within-type edge prob (sparser is more realistic)
    p_inter=0.04,                    # cross-type edge prob
    num_relations=6,
    max_time=20,
    seed=0,
    device="cpu",
):
    g = torch.Generator(device="cpu").manual_seed(seed)
    types = []
    for t, n in enumerate(n_per_type):
        types.extend([t] * n)
    node_type = torch.tensor(types, dtype=torch.long, device=device)
    N = node_type.numel()

    src_list, tgt_list, rel_list, time_list = [], [], [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            same = node_type[i] == node_type[j]
            p = p_intra if same else p_inter
            if torch.rand((), generator=g).item() < p:
                src_list.append(i)
                tgt_list.append(j)
                rel_list.append(int(torch.randint(0, num_relations, (1,), generator=g)))
                time_list.append(int(torch.randint(0, max_time, (1,), generator=g)))

    edge_index = torch.tensor([src_list, tgt_list], dtype=torch.long, device=device)
    edge_type = torch.tensor(rel_list, dtype=torch.long, device=device)
    edge_time = torch.tensor(time_list, dtype=torch.long, device=device)
    return node_type, edge_index, edge_type, edge_time


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

def test_forward_shapes_and_partition():
    print("\n[test 1] forward shapes + partition stats")
    torch.manual_seed(0)
    device = "cpu"

    n_per_type = (20, 30, 25, 15)
    num_types = len(n_per_type)
    num_relations = 6
    d_S, d_M = 32, 24
    n_hid_S, n_hid_M, n_hid_out = 64, 64, 96

    node_type, edge_index, edge_type, edge_time = build_synth_graph(
        n_per_type=n_per_type, num_relations=num_relations, device=device
    )
    N = node_type.numel()

    nt2d = make_node_type_to_domain(num_types)  # identity: each type is its own domain
    stats = edge_partition_stats(node_type, edge_index, nt2d.to(device))
    print(" partition stats:", stats)
    assert stats["n_intra_edges"] + stats["n_inter_edges"] == edge_index.size(1)

    model = TwoStreamSMFEHGT(
        d_S_in=d_S, d_M_in=d_M,
        n_hid_S=n_hid_S, n_hid_M=n_hid_M, n_hid_out=n_hid_out,
        num_types=num_types, num_relations=num_relations,
        n_heads_hgt=4, n_heads_readout=4, n_layers=2,
        dropout=0.0, prev_norm=True, last_norm=True, use_RTE=True,
    ).to(device)

    s_feat = torch.randn(N, d_S, device=device)
    m_feat = torch.randn(N, d_M, device=device)

    out = model(s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
                return_intermediate=True)

    assert out["h"].shape   == (N, n_hid_out), out["h"].shape
    assert out["h_S"].shape == (N, n_hid_S),   out["h_S"].shape
    assert out["h_M"].shape == (N, n_hid_M),   out["h_M"].shape
    assert out["attn"].shape == (N, 2, 4),      out["attn"].shape

    # softmax sanity
    s = out["attn"].sum(dim=1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-5), \
        f"attn does not sum to 1 across streams: max dev = {(s-1).abs().max()}"

    print(f" h: {out['h'].shape}, h_S: {out['h_S'].shape}, h_M: {out['h_M'].shape}, attn: {out['attn'].shape}")
    print(f" attn sums to 1 across streams: {torch.allclose(s, torch.ones_like(s), atol=1e-5)}")
    print(" PASS")


def test_gradients_into_both_streams():
    print("\n[test 2] gradients flow into both streams")
    torch.manual_seed(1)
    device = "cpu"

    num_types = 3
    num_relations = 4
    d_S = d_M = 16
    n_hid = 32

    node_type, edge_index, edge_type, edge_time = build_synth_graph(
        n_per_type=(10, 12, 8), num_relations=num_relations, device=device
    )
    N = node_type.numel()

    model = TwoStreamSMFEHGT(
        d_S_in=d_S, d_M_in=d_M,
        n_hid_S=n_hid, n_hid_M=n_hid, n_hid_out=n_hid,
        num_types=num_types, num_relations=num_relations,
        n_heads_hgt=2, n_heads_readout=2, n_layers=2,
        dropout=0.0, use_RTE=True,
    ).to(device)

    s_feat = torch.randn(N, d_S, device=device, requires_grad=False)
    m_feat = torch.randn(N, d_M, device=device, requires_grad=False)

    out = model(s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
                return_intermediate=True)

    # Trivial target; just make sure backward works and reaches all params.
    target = torch.randn(N, n_hid, device=device)
    loss = F.mse_loss(out["h"], target)
    loss.backward()

    n_state_params_with_grad = sum(
        1 for p in model.state_gnn.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_mech_params_with_grad = sum(
        1 for p in model.mech_gnn.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    n_readout_params_with_grad = sum(
        1 for p in model.readout.parameters() if p.grad is not None and p.grad.abs().sum() > 0
    )
    print(f" state_gnn params with grad: {n_state_params_with_grad}")
    print(f" mech_gnn  params with grad: {n_mech_params_with_grad}")
    print(f" readout   params with grad: {n_readout_params_with_grad}")
    assert n_state_params_with_grad > 0, "state stream got no gradient"
    assert n_mech_params_with_grad  > 0, "mechanism stream got no gradient"
    assert n_readout_params_with_grad > 0, "readout got no gradient"
    print(" PASS")


def test_smfe_total_loss_runs_and_decreases():
    print("\n[test 3] task + align + xcov loss decreases on an easy fit")
    torch.manual_seed(2)
    device = "cpu"

    num_types = 3
    num_relations = 4
    d_S = d_M = 16
    n_hid = 32
    n_classes = 4

    node_type, edge_index, edge_type, edge_time = build_synth_graph(
        n_per_type=(15, 15, 15), num_relations=num_relations, device=device
    )
    N = node_type.numel()

    # Easy fit: labels = node type. The state stream alone should suffice.
    y = node_type.clone()

    model = TwoStreamSMFEHGT(
        d_S_in=d_S, d_M_in=d_M,
        n_hid_S=n_hid, n_hid_M=n_hid, n_hid_out=n_hid,
        num_types=num_types, num_relations=num_relations,
        n_heads_hgt=2, n_heads_readout=2, n_layers=2,
        dropout=0.0, use_RTE=False,    # turn off RTE for determinism
    ).to(device)

    probes = SMFEProbes(d_O=n_hid, d_S=d_S, d_M=d_M).to(device)
    classifier = nn.Linear(n_hid, n_classes).to(device)

    s_feat = torch.randn(N, d_S, device=device)
    m_feat = torch.randn(N, d_M, device=device)

    weights = SMFELossWeights(lambda_align=1.0, lambda_xcov=1e-2, lambda_inv=0.0)

    opt = torch.optim.Adam(
        list(model.parameters()) + list(probes.parameters()) + list(classifier.parameters()),
        lr=5e-3,
    )

    losses = []
    for step in range(30):
        opt.zero_grad()
        out = model(s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
                    return_intermediate=True)
        logits = classifier(out["h"])
        task = F.cross_entropy(logits, y)
        s_pred, m_pred = probes(out["h"])
        total, parts = smfe_total_loss(
            task_loss=task,
            s_pred=s_pred, m_pred=m_pred,
            s_target=s_feat, m_target=m_feat,
            weights=weights,
        )
        total.backward()
        opt.step()
        losses.append(parts["total"])
        if step % 5 == 0:
            print(f" step {step:2d} | task {parts['task']:.4f} | align {parts['align']:.4f} "
                  f"| xcov {parts['xcov']:.4f} | total {parts['total']:.4f}")

    print(f" first total {losses[0]:.4f} -> last total {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "loss did not decrease"
    print(" PASS")


def test_attention_downweights_uninformative_stream():
    print("\n[test 4] readout learns to down-weight an uninformative stream")
    torch.manual_seed(3)
    device = "cpu"

    num_types = 3
    num_relations = 4
    d_S = d_M = 16
    n_hid = 32
    n_classes = 4

    node_type, edge_index, edge_type, edge_time = build_synth_graph(
        n_per_type=(15, 15, 15), num_relations=num_relations, device=device
    )
    N = node_type.numel()
    y = node_type.clone()  # label = node type, recoverable from state alone

    model = TwoStreamSMFEHGT(
        d_S_in=d_S, d_M_in=d_M,
        n_hid_S=n_hid, n_hid_M=n_hid, n_hid_out=n_hid,
        num_types=num_types, num_relations=num_relations,
        n_heads_hgt=2, n_heads_readout=2, n_layers=2,
        dropout=0.0, use_RTE=False,
    ).to(device)
    classifier = nn.Linear(n_hid, n_classes).to(device)

    # Informative state input: includes node-type signal.
    type_onehot = F.one_hot(node_type, num_classes=num_types).float()
    s_feat = torch.cat([type_onehot, torch.randn(N, d_S - num_types)], dim=1)
    # Uninformative mech input: pure noise, redrawn each step (no signal at all).

    opt = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()), lr=1e-2
    )

    for step in range(80):
        opt.zero_grad()
        m_feat = torch.randn(N, d_M, device=device)  # noise, redrawn
        out = model(s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
                    return_intermediate=True)
        logits = classifier(out["h"])
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()

    # After training, look at average attention on the mech stream.
    model.eval()
    with torch.no_grad():
        m_feat = torch.randn(N, d_M, device=device)
        out = model(s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
                    return_intermediate=True)
        attn = out["attn"]                 # (N, 2, n_heads_readout)
        mean_state = attn[:, 0, :].mean().item()
        mean_mech  = attn[:, 1, :].mean().item()
    print(f" mean attention -> state: {mean_state:.3f}, mech: {mean_mech:.3f}")
    # We don't require mech weight to be tiny -- just lower than state, since
    # the model has no incentive to put weight on noise.
    assert mean_state > mean_mech, \
        f"expected state > mech attention, got {mean_state:.3f} vs {mean_mech:.3f}"
    print(" PASS")


if __name__ == "__main__":
    test_forward_shapes_and_partition()
    test_gradients_into_both_streams()
    test_smfe_total_loss_runs_and_decreases()
    test_attention_downweights_uninformative_stream()
    print("\nAll smoke tests passed.")
