# 16K hybrid cluster N-group ablation

## Conditions

- Date: 2026-07-22
- GPU: NVIDIA B200, Vast.ai instance `45481495`, 1000 W power limit
- Definition commit: `6210083`
- Same dense BF16 16K GEMM and selected 256x256 B-multicast K64/S3 kernel as
  the full cluster-N-fast experiment
- Static 16x16 macro, 148 persistent CTAs, one process per case, W1/I5, three
  rotated processes

All variants passed the 512 pattern validation bit-exactly.  Temperature moved
from 31 C to 34 C.

## Results

N-group is the number of adjacent N tiles visited for one M pair before the
scheduler advances to the next M pair.  Group 1 is algebraically the original
M-fast order; group 16 is the previously measured full N-fast 16x16 order.

| N group | TFLOP/s | versus group 1 |
|---:|---:|---:|
| 1 | **1786.590 +/- 0.364** | baseline |
| 2 | 1777.966 +/- 1.452 | -0.483% |
| 4 | 1772.420 +/- 1.278 | -0.793% |
| 8 | 1757.388 +/- 0.312 | -1.635% |
| 16, prior full N-fast run | 1754.691 +/- 0.621 | -1.786% |

The monotonic trend rejects even a small N-first strip.  The selected order
remains group 1 / M-fast.  The partial A-repeat result is therefore an operand
residency ceiling, not evidence that reversing CTA order will realize it.
Future A work must retain the successful B-panel traversal and reduce actual A
transactions, for example by multicast/cluster sharing in a topology that does
not exchange B reuse for A reuse.

Reproduce with:

```bash
./run_b200_gemm_cluster_ngroup_ablation.sh \
  /workspace/benchmark/5.GEMM /workspace/gemm_cluster_ngroup_ablation
```

Raw CSVs, logs, validation, hashes, and source snapshots are in this directory.
