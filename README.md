# SMFE-Coupled Heterogeneous Graph Transformer

Two-stream HGT for State–Mechanism Factorized Embeddings (SMFE), with
cross-stream attention readout and SMFE-aware training losses.

## What's here

```
smfe-hgt/
├── pyHGT/                 # Vendored UCLA-DM/pyHGT (ogbn-mag flavor), unchanged.
│   ├── conv.py            #   HGTConv, DenseHGTConv, RelTemporalEncoding, GeneralConv
│   ├── model.py           #   GNN, Classifier, Matcher
│   ├── data.py / utils.py
│   └── __init__.py
├── smfe/
│   ├── partition.py       # Edge partitioning into intra (state) / inter (mechanism)
│   ├── readout.py         # CrossStreamAttentionReadout (2-token multi-head attention)
│   ├── model.py           # TwoStreamSMFEHGT, SMFEProbes
│   ├── losses.py          # alignment_loss, xcov_loss, hsic_loss, irmv1_penalty, smfe_total_loss
│   └── __init__.py
├── tests/
│   ├── test_two_stream.py # forward shapes, gradients, full SMFE loss decreases
│   └── test_isolation.py  # h_S == h_S' when only inter edges are removed; mech ≠ state
├── train_smfe_hgt.py      # Runnable end-to-end skeleton on synthetic data
└── requirements.txt
```

## Install

```bash
# Use a fresh venv. PyG and its companion libs are version-sensitive.
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

You also need PyTorch matching your CUDA version (install separately
following pytorch.org instructions). The vendored pyHGT was written
against PyG 2.x and works with current PyG.

## Smoke tests

```bash
python tests/test_two_stream.py   # 4 tests: shapes, gradients, loss-down, attn down-weights noise
python tests/test_isolation.py    # 2 tests: stream isolation properties
```

All six tests should pass on CPU in under a minute.

## End-to-end skeleton

```bash
python train_smfe_hgt.py \
    --epochs 10 --batches-per-epoch 20 \
    --d-s-in 64 --d-m-in 48 \
    --n-hid-s 128 --n-hid-m 128 --n-hid-out 128 \
    --n-heads-hgt 4 --n-heads-readout 4 --n-layers 2 \
    --lambda-align 1.0 --lambda-xcov 1e-2
```

To plug in a real KG (OGBL-BioKG, OAG, ogbn-mag, your enterprise KG):
replace `build_loader()` in `train_smfe_hgt.py` with your pipeline. The
model interface expects six tensors per batch:

```
s_feat     : (N, d_S_in)   pre-computed SMFE state embeddings
m_feat     : (N, d_M_in)   pre-computed SMFE mechanism embeddings
node_type  : (N,)
edge_index : (2, E)        row 0 = source, row 1 = target
edge_type  : (E,)
edge_time  : (E,)
```

## Architecture

```
        s_v  ──►  STATE HGT (intra edges only)  ──►  h_S ∈ R^{d_S}
                                                          \
                                                           ─►  CrossStreamAttn  ──►  h
                                                          /
        m_v  ──►  MECH HGT  (inter edges only)  ──►  h_M ∈ R^{d_M}
```

Two independent HGT stacks: independent type-specific projections,
independent relation matrices, **independent per-edge softmax**. No
parameter sharing, no message-passing cross-talk. Fusion happens only at
the readout, via a per-node, per-head soft mixture whose weights are
exposed for diagnostics.

## SMFE losses

Total objective:

    L_HGT = L_task
          + λ_align · L_align          # HGT output should linearly recover (s_v, m_v)
          + λ_xcov  · L_xcov_HGT       # decorrelate recovered state and mechanism subspaces
          + λ_inv   · L_inv            # IRMv1 across environments (optional)

Stop-gradient is applied to (s_v, m_v) inside `alignment_loss`, so HGT
moves toward SMFE — never the reverse.

## Configuring the semantic-domain map

Edge partitioning uses a `node_type → semantic_domain` map. Default is
identity (each HGT type is its own domain), so all heterogeneous edges
land in the mechanism stream. To merge several types into one semantic
domain — e.g. merge {Author, Institute} into a 'people' domain — pass
`node_type_to_domain=[...]` to `TwoStreamSMFEHGT(...)`.

## Notes on extensions

* `smfe.losses.hsic_loss` is a kernelized drop-in replacement for
  `xcov_loss` if linear decorrelation is insufficient (costs O(B²) memory).
* `smfe.losses.irmv1_penalty` plugs in when you have explicit
  environments. Apply it on the task logits computed per environment.
* `CrossStreamAttentionReadout(return_attn=True)` returns `(N, 2,
  n_heads_readout)` weights — log these as a robustness diagnostic
  alongside `B_K` from bridge discovery.
