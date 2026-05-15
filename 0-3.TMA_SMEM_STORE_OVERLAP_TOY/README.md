# TMA vs Reg-to-SMEM Store Overlap Toy

This benchmark checks whether TMA global-to-SMEM writes and normal register-to-SMEM
stores contend for shared-memory write bandwidth.

It runs three kernels for each `(store_warps, store_tiles_per_tma)` point:

- `TMA-only`: warp 0 repeatedly TMA-loads one 32 KiB tile into SMEM.
- `store-only`: selected store warps repeatedly write register values into a separate
  32 KiB SMEM tile.
- `overlap`: both loops run at the same time, using disjoint SMEM regions.

The store path supports `--store-vec 1|2|4`, which maps to scalar,
`st.shared.v2.u32`, and `st.shared.v4.u32` writes. Use `--store-vec 4` when the
goal is to drive the register-to-SMEM write path harder.

The CSV reports:

- `expected_no_share_ms = max(tma_only_ms, store_only_ms)`
- `overlap_extra_ms = overlap_ms - expected_no_share_ms`
- `overlap_efficiency = expected_no_share_ms / overlap_ms`

If TMA and stores do not share the limiting path, overlap time should be close to
`expected_no_share_ms`. If they share a write path, `overlap_ms` grows and
`overlap_efficiency` drops.

Build:

```bash
cd /home/snu_avq1/workspace/chaewon/benchmark
make tma_smem_store_overlap_toy
```

Smoke:

```bash
./build/0-3.TMA_SMEM_STORE_OVERLAP_TOY/tma_smem_store_overlap_toy \
  --blocks 64 --repeats 256 --source-tiles 1024 \
  --warmup 1 --iters 2 \
  --store-warps 8 --store-tiles 4 \
  --store-vec 4 \
  --csv /tmp/tma_smem_store_overlap_smoke.csv
```

Full sweep:

```bash
./build/0-3.TMA_SMEM_STORE_OVERLAP_TOY/tma_smem_store_overlap_toy \
  --blocks 4096 --repeats 8192 --source-tiles 8192 \
  --warmup 2 --iters 5 \
  --store-vec 4 \
  --csv 0-3.TMA_SMEM_STORE_OVERLAP_TOY/tma_smem_store_overlap_summary.csv
```
