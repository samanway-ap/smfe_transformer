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
│   ├── types.py           # SMFEBatch — canonical input format
│   ├── partition.py       # Edge partitioning into intra (state) / inter (mechanism)
│   ├── readout.py         # CrossStreamAttentionReadout (2-token multi-head attention)
│   ├── model.py           # TwoStreamSMFEHGT, SMFEProbes
│   ├── losses.py          # alignment_loss, hsic_loss (default indep penalty), hsic_unbiased, xcov_loss, irmv1_penalty, smfe_total_loss
│   └── __init__.py
├── tests/
│   ├── test_two_stream.py # forward shapes, gradients, full SMFE loss decreases
│   └── test_isolation.py  # h_S == h_S' when only inter edges are removed; mech ≠ state
├── train_smfe_hgt.py      # Runnable end-to-end skeleton on synthetic data
└── requirements.txt
```

## Install

### Option A — Docker (recommended)

The repo ships a CPU-only image that pins compatible PyTorch / PyG
versions, avoiding the manual CUDA-matched install dance.

```bash
# Build
docker build -f docker/Dockerfile -t smfe-hgt .

# Run the smoke tests
docker run --rm smfe-hgt \
    sh -c "python tests/test_two_stream.py && python tests/test_isolation.py"

# Or use compose for the predefined train / test services
docker compose -f docker/docker-compose.yml run --rm test
docker compose -f docker/docker-compose.yml run --rm train
```

The compose services mount the repo at `/app`, so local edits are
picked up without rebuilding.

### Option B — Local venv

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

## Hyperparameters

Every stream-shaped hyperparameter can be set independently for the
state HGT and the mechanism HGT. Joint args (e.g. `n_layers=3`) apply to
both streams; the matching `_S` / `_M` overrides take precedence when
supplied. This means the two streams can have **different depths,
different head counts, different dropout, different normalization, and
independent RTE settings**.

### Module-level

| Hyperparameter   | Joint arg            | State override     | Mech override      | Default | Meaning |
|------------------|----------------------|--------------------|--------------------|---------|---------|
| Depth            | `n_layers`           | `n_layers_S`       | `n_layers_M`       | `3`     | Number of HGT layers in the stream |
| HGT heads        | `n_heads_hgt`        | `n_heads_hgt_S`    | `n_heads_hgt_M`    | `8`     | Multi-head count inside each HGTConv |
| Dropout          | `dropout`            | `dropout_S`        | `dropout_M`        | `0.2`   | Dropout in HGTConv + readout |
| Pre-layer norm   | `prev_norm`          | `prev_norm_S`      | `prev_norm_M`      | `True`  | LayerNorm before each HGT layer |
| Last-layer norm  | `last_norm`          | `last_norm_S`      | `last_norm_M`      | `True`  | LayerNorm after the final HGT layer |
| Use RTE          | `use_RTE`            | `use_RTE_S`        | `use_RTE_M`        | `True`  | Relative Temporal Encoding on edges |

| Hyperparameter   | Arg                  | Default | Meaning |
|------------------|----------------------|---------|---------|
| State input dim  | `d_S_in`             | —       | Width of the per-node SMFE state vector `s_v` |
| Mech input dim   | `d_M_in`             | —       | Width of the per-node SMFE mechanism vector `m_v` |
| State hidden dim | `n_hid_S`            | `128`   | Hidden + output dim of the state HGT |
| Mech hidden dim  | `n_hid_M`            | `128`   | Hidden + output dim of the mech HGT |
| Fused output dim | `n_hid_out`          | `256`   | Output dim after the cross-stream readout |
| Readout heads    | `n_heads_readout`    | `4`     | Multi-head count in the cross-stream attention |
| Readout dropout  | `readout_dropout`    | `dropout` | Falls back to `dropout` when `None` |
| #Node types      | `num_types`          | —       | HGT node-type cardinality |
| #Relations       | `num_relations`      | —       | HGT relation-type cardinality |
| Domain map       | `node_type_to_domain`| identity| Length-`num_types` int seq mapping each type to a semantic-domain id |

### CLI args (train_smfe_hgt.py)

Every per-stream override is exposed:

```bash
python train_smfe_hgt.py \
    --d-s-in 64 --d-m-in 48 \
    --n-hid-s 128 --n-hid-m 128 --n-hid-out 256 \
    --n-layers 2 --n-layers-s 4 --n-layers-m 1 \
    --n-heads-hgt 4 --n-heads-hgt-s 8 --n-heads-hgt-m 2 \
    --dropout 0.2 --dropout-s 0.3 --dropout-m 0.1 \
    --n-heads-readout 4 \
    --epochs 10 --batches-per-epoch 20 \
    --lambda-align 1.0 --penalty hsic --lambda-indep 1e-2
```

The independence penalty defaults to **HSIC** (kernelized). Pass
`--penalty xcov` to use the linear cross-covariance proxy instead, and
`--lambda-indep` to weight whichever penalty is active. `--hsic-sigma`
(default: median heuristic) and `--hsic-subsample` (default 4096) tune the
kernel estimate.

Drop the `_S` / `_M` flags to use the joint defaults.

## Input format

The model consumes a single `SMFEBatch` per call. This is the **only**
format guaranteed to be supported: a loader's job is to produce one
`SMFEBatch` per batch. The dataclass is shape-checked by `validate()`
before every forward pass.

```python
from smfe import SMFEBatch

batch = SMFEBatch(
    s_feat     = s_feat,      # (N, d_S_in)       float32 — SMFE state embedding s_v per node
    m_feat     = m_feat,      # (N, d_M_in)       float32 — SMFE mechanism embedding m_v per node
    node_type  = node_type,   # (N,)              long    — node-type id in [0, num_types)
    edge_index = edge_index,  # (2, E)            long    — row 0: source, row 1: target
    edge_type  = edge_type,   # (E,)              long    — relation id in [0, num_relations)
    edge_time  = edge_time,   # (E,)              long    — timestamp; use zeros if unused
    node_type_to_domain = nt2d,  # optional (num_types,) long
)
batch.validate()                         # raises if shapes/dtypes disagree
out = model.forward_batch(batch, return_intermediate=True)
# out["h"]    : (N, n_hid_out)
# out["h_S"]  : (N, n_hid_S)        state-stream embedding
# out["h_M"]  : (N, n_hid_M)        mechanism-stream embedding
# out["attn"] : (N, 2, n_heads_readout)   per-node fusion weights
```

### Field contract

* `s_feat`, `m_feat` are the **pre-computed SMFE factors**, one row per
  node, in the same row order as `node_type`. The model never recomputes
  them; the alignment loss applies stop-gradient to `s_feat` / `m_feat`,
  so SMFE is a fixed target and HGT moves toward it.
* `node_type[i] ∈ [0, num_types)` is the heterogeneous node-type label.
* `edge_index, edge_type, edge_time` follow PyG conventions. The
  `(source, target)` order matches the vendored `pyHGT.GNN.forward`.
* `node_type_to_domain` (length `num_types`) groups node types into
  semantic domains. Edges are partitioned by this map: if
  `domain[src] == domain[tgt]` the edge feeds the **state** stream
  (intra-domain), otherwise the **mechanism** stream (inter-domain). The
  default is identity, so without overriding, every node type is its own
  domain and all heterogeneous edges land in the mechanism stream.

### Constructing from existing PyG / pyHGT pipelines

If you already have an OAG / ogbn-mag / OGBL-BioKG pipeline emitting the
six tensors, wrap them:

```python
batch = SMFEBatch.from_tensors(
    s_feat, m_feat, node_type, edge_index, edge_type, edge_time,
    node_type_to_domain=nt2d,   # optional
)
```

`from_tensors` runs `validate()` for you. To plug into a real KG,
replace `build_loader()` in `train_smfe_hgt.py` with a generator yielding
`(SMFEBatch, targets)` tuples.

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
          + λ_indep · L_indep          # independence of recovered state / mechanism subspaces
          + λ_inv   · L_inv            # IRMv1 across environments (optional)

`L_indep` is **HSIC** under an RBF kernel by default — the Hilbert–Schmidt
Independence Criterion, which is zero exactly at independence under a
characteristic kernel. It replaced the linear cross-covariance proxy
‖Σ_SM‖²_F (`xcov_loss`, still available via `penalty="xcov"`), which only
removes *linear* cross-covariance and is its special case at a linear
kernel: a bilinear form can vanish with arbitrary higher-order dependence
intact. The RBF bandwidth is the median pairwise squared distance of the
batch (recomputed per call), and the penalty is estimated on a subsample of
4096 entities per step.

Because the kernel penalty's gradient norm dwarfs the data term's, the
training loop clips the **data gradient and the penalty gradient
separately** (each to `--clip-norm`, default 1.0); a single joint clip lets
the penalty consume the whole budget and starves the task.

Stop-gradient is applied to (s_v, m_v) inside `alignment_loss`, so HGT
moves toward SMFE — never the reverse.

## Configuring the semantic-domain map

Edge partitioning uses a `node_type → semantic_domain` map. Default is
identity (each HGT type is its own domain), so all heterogeneous edges
land in the mechanism stream. To merge several types into one semantic
domain — e.g. merge {Author, Institute} into a 'people' domain — pass
`node_type_to_domain=[...]` to `TwoStreamSMFEHGT(...)`.

## Notes on extensions

* `smfe.losses.hsic_loss` is the **default** independence penalty
  (kernelized, O(B²) memory on the subsample). `smfe.losses.xcov_loss` is
  the linear special case, kept for the `penalty="xcov"` ablation.
  `smfe.losses.hsic_unbiased` is the unbiased U-statistic for measuring
  achieved dependence on a trained checkpoint (not a training loss).
* `smfe.losses.irmv1_penalty` plugs in when you have explicit
  environments. Apply it on the task logits computed per environment.
* `CrossStreamAttentionReadout(return_attn=True)` returns `(N, 2,
  n_heads_readout)` weights — log these as a robustness diagnostic
  alongside `B_K` from bridge discovery.
