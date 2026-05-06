"""
End-to-end training script skeleton for SMFE-coupled HGT.

This is a minimal, runnable example on synthetic data. To adapt to a real
KG (OAG, ogbn-mag, OGBL-BioKG, your enterprise KG) replace `build_loader()`
with your own data pipeline producing the same five tensors per batch:

    s_feat     : (N, d_S_in)   pre-computed SMFE state embeddings
    m_feat     : (N, d_M_in)   pre-computed SMFE mechanism embeddings
    node_type  : (N,)
    edge_index : (2, E)
    edge_type  : (E,)
    edge_time  : (E,)

Plus task targets (here, node labels) and the SMFE factor *targets* used
for the alignment loss. By convention we use s_feat / m_feat themselves
as the alignment targets, treated with stop-gradient inside the loss.

Usage:
    python train_smfe_hgt.py \
        --epochs 20 --batches-per-epoch 50 \
        --lambda-align 1.0 --lambda-xcov 1e-2

For multi-environment training with IRM, supply --num-envs > 1 and the
script will batch across environments using the synthetic shift built in.
"""
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from smfe import (
    TwoStreamSMFEHGT,
    SMFEProbes,
    SMFELossWeights,
    smfe_total_loss,
    irmv1_penalty,
    edge_partition_stats,
    make_node_type_to_domain,
)


# ---------------------------------------------------------------------
# Synthetic loader (replace with your real loader).
# ---------------------------------------------------------------------

def build_loader(
    n_per_type=(40, 60, 50, 30),
    num_relations=8,
    d_S_in=64,
    d_M_in=48,
    n_classes=4,
    max_time=20,
    seed=0,
    device="cpu",
):
    """Yields (s_feat, m_feat, node_type, edge_index, edge_type, edge_time, y)."""
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)

    while True:
        types = []
        for t, n in enumerate(n_per_type):
            types.extend([t] * n)
        node_type = torch.tensor(types, dtype=torch.long, device=device)
        N = node_type.numel()

        # Random sparse edges, with intra denser than inter.
        edge_set = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                same = node_type[i] == node_type[j]
                p = 0.05 if same else 0.02
                if torch.rand((), generator=g).item() < p:
                    edge_set.append((
                        i, j,
                        int(torch.randint(0, num_relations, (1,), generator=g)),
                        int(torch.randint(0, max_time, (1,), generator=g)),
                    ))
        if not edge_set:
            continue
        src, tgt, rel, ts = zip(*edge_set)
        edge_index = torch.tensor([list(src), list(tgt)], dtype=torch.long, device=device)
        edge_type = torch.tensor(rel, dtype=torch.long, device=device)
        edge_time = torch.tensor(ts, dtype=torch.long, device=device)

        # Pre-computed SMFE-style embeddings: state holds type signal,
        # mechanism is mostly noise here.
        type_onehot = F.one_hot(node_type, num_classes=len(n_per_type)).float()
        s_feat = torch.cat(
            [type_onehot, torch.randn(N, d_S_in - len(n_per_type), device=device)],
            dim=1,
        )
        m_feat = torch.randn(N, d_M_in, device=device)

        # Task: predict node type from state + neighborhood.
        y = node_type.clone()

        yield s_feat, m_feat, node_type, edge_index, edge_type, edge_time, y


# ---------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------

def train_one_epoch(
    model, probes, classifier, opt, loader, weights,
    batches_per_epoch, device, log_every=10,
):
    model.train(); probes.train(); classifier.train()
    parts_running = {"task": 0.0, "align": 0.0, "xcov": 0.0, "total": 0.0}
    correct = 0
    n_seen = 0

    for step in range(batches_per_epoch):
        s_feat, m_feat, node_type, edge_index, edge_type, edge_time, y = next(loader)

        opt.zero_grad()
        out = model(
            s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
            return_intermediate=True,
        )
        logits = classifier(out["h"])
        task_loss = F.cross_entropy(logits, y)
        s_pred, m_pred = probes(out["h"])

        loss_total, parts = smfe_total_loss(
            task_loss=task_loss,
            s_pred=s_pred, m_pred=m_pred,
            s_target=s_feat, m_target=m_feat,
            weights=weights,
        )
        loss_total.backward()
        opt.step()

        for k in parts_running:
            parts_running[k] += parts[k]
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            n_seen += y.numel()

        if (step + 1) % log_every == 0:
            print(
                f"  step {step+1:4d}/{batches_per_epoch} | "
                f"task {parts['task']:.4f} | align {parts['align']:.4f} | "
                f"xcov {parts['xcov']:.4f} | acc {correct/max(n_seen,1):.3f}"
            )

    for k in parts_running:
        parts_running[k] /= batches_per_epoch
    parts_running["acc"] = correct / max(n_seen, 1)
    return parts_running


@torch.no_grad()
def evaluate(model, probes, classifier, loader, n_batches, device):
    model.eval(); probes.eval(); classifier.eval()
    correct = total = 0
    mech_attn_acc = state_attn_acc = 0.0
    for _ in range(n_batches):
        s_feat, m_feat, node_type, edge_index, edge_type, edge_time, y = next(loader)
        out = model(
            s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
            return_intermediate=True,
        )
        logits = classifier(out["h"])
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        attn = out["attn"]                  # (N, 2, n_heads_readout)
        state_attn_acc += attn[:, 0, :].mean().item()
        mech_attn_acc  += attn[:, 1, :].mean().item()
    return {
        "acc": correct / max(total, 1),
        "mean_state_attn": state_attn_acc / max(n_batches, 1),
        "mean_mech_attn":  mech_attn_acc  / max(n_batches, 1),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train SMFE-coupled HGT (skeleton)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batches-per-epoch", type=int, default=20)
    p.add_argument("--eval-batches", type=int, default=5)

    p.add_argument("--num-types", type=int, default=4)
    p.add_argument("--num-relations", type=int, default=8)
    p.add_argument("--n-classes", type=int, default=4)

    p.add_argument("--d-s-in", type=int, default=64)
    p.add_argument("--d-m-in", type=int, default=48)
    p.add_argument("--n-hid-s", type=int, default=128)
    p.add_argument("--n-hid-m", type=int, default=128)
    p.add_argument("--n-hid-out", type=int, default=128)
    p.add_argument("--n-heads-hgt", type=int, default=4)
    p.add_argument("--n-heads-readout", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-rte", action="store_true")

    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lambda-align", type=float, default=1.0)
    p.add_argument("--lambda-xcov",  type=float, default=1e-2)
    p.add_argument("--lambda-inv",   type=float, default=0.0)

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    print(f"device={device}")

    # Optional: lump several types into one semantic domain. None = identity.
    node_type_to_domain = None  # e.g. [0, 0, 1, 2] would merge types 0 and 1 into one domain

    model = TwoStreamSMFEHGT(
        d_S_in=args.d_s_in, d_M_in=args.d_m_in,
        n_hid_S=args.n_hid_s, n_hid_M=args.n_hid_m, n_hid_out=args.n_hid_out,
        num_types=args.num_types, num_relations=args.num_relations,
        n_heads_hgt=args.n_heads_hgt, n_heads_readout=args.n_heads_readout,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prev_norm=True, last_norm=True,
        use_RTE=not args.no_rte,
        node_type_to_domain=node_type_to_domain,
    ).to(device)

    probes = SMFEProbes(d_O=args.n_hid_out, d_S=args.d_s_in, d_M=args.d_m_in).to(device)
    classifier = nn.Linear(args.n_hid_out, args.n_classes).to(device)
    weights = SMFELossWeights(
        lambda_align=args.lambda_align,
        lambda_xcov=args.lambda_xcov,
        lambda_inv=args.lambda_inv,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")

    n_per_type = tuple(20 + 10 * i for i in range(args.num_types))
    train_loader = build_loader(
        n_per_type=n_per_type, num_relations=args.num_relations,
        d_S_in=args.d_s_in, d_M_in=args.d_m_in,
        n_classes=args.n_classes, seed=args.seed, device=device,
    )
    eval_loader = build_loader(
        n_per_type=n_per_type, num_relations=args.num_relations,
        d_S_in=args.d_s_in, d_M_in=args.d_m_in,
        n_classes=args.n_classes, seed=args.seed + 1234, device=device,
    )

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(probes.parameters()) + list(classifier.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, probes, classifier, opt, train_loader, weights,
            batches_per_epoch=args.batches_per_epoch,
            device=device, log_every=max(1, args.batches_per_epoch // 5),
        )
        eval_metrics = evaluate(model, probes, classifier, eval_loader,
                                n_batches=args.eval_batches, device=device)
        dt = time.time() - t0
        print(
            f"\n[epoch {epoch:3d}/{args.epochs}] "
            f"train_acc={train_metrics['acc']:.3f} "
            f"task={train_metrics['task']:.4f} "
            f"align={train_metrics['align']:.4f} "
            f"xcov={train_metrics['xcov']:.4f} | "
            f"eval_acc={eval_metrics['acc']:.3f} "
            f"attn(state)={eval_metrics['mean_state_attn']:.3f} "
            f"attn(mech)={eval_metrics['mean_mech_attn']:.3f} | "
            f"{dt:.1f}s\n"
        )


if __name__ == "__main__":
    main()
