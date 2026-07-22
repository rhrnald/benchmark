# 16K cluster-N-fast scheduler ablation

## Conditions

- Date: 2026-07-22
- GPU: NVIDIA B200, Vast.ai instance `45481495`, 1000 W power limit
- Definition commit: `50224fa`
- BF16 random `[0,1)`, M=N=K=16384, full FP32 C output
- 256x256 CTA, K64/S3, split-B TMA, two-CTA B multicast, phase 0/0
- Static persistent scheduler; one process per case, W1/I5, three rotated
  processes

Every variant passed the 512 pattern validation bit-exactly.  Temperature was
31 C before and 34 C after the sweep.

## Results

| local scheduler | CTAs | TFLOP/s | versus baseline |
|---|---:|---:|---:|
| existing M-fast 16x16 | 148 | **1783.871 +/- 2.085** | baseline |
| cluster-N-fast 32x8 | 148 | 1756.695 +/- 1.252 | -1.523% |
| cluster-N-fast 16x16 | 148 | 1754.691 +/- 0.621 | -1.636% |
| cluster-N-fast 8x32 | 148 | 1690.133 +/- 0.898 | -5.255% |
| cluster-N-fast 4x64 | 148 | 1615.893 +/- 0.311 | -9.416% |
| cluster-N-fast explicit 16x9 wave | 144 | 1580.579 +/- 0.705 | -11.396% |

Cluster ranks remain adjacent M tiles and still share B through multicast.
Only cluster IDs change from M-fast to N-fast, causing many clusters to request
the same A panels close together.  Contrary to the partial-repeat ceiling,
this order is slower, and the loss grows as the N span increases.  The reason
is that the existing M-fast order also lets many clusters consume the same B
panel from L2; full N-fast destroys that already-effective B locality.  Merely
making A requests simultaneous does not guarantee that the first request
fills L2 early enough to serve the other SMs.

Reproduce with:

```bash
./run_b200_gemm_cluster_nfast_ablation.sh \
  /workspace/benchmark/5.GEMM /workspace/gemm_cluster_nfast_ablation
```

Raw CSVs, logs, validation, hashes, and source snapshots are in this directory.
