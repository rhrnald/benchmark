# Dense GEMM asynchronous persistent CTA

> The asynchronous-barrier correction remains current.  Dense-address L2
> tuning was refined on 2026-07-21; see
> `../gemm_dense_l2_persistent_tuning_20260721_b200_45447842/summary.md` for the
> selected cache schedule.

- Date: 2026-07-20
- Vast.ai instance: `45378714`
- GPU: NVIDIA B200, 1000 W power limit
- Math: BF16 uniform `[0,1)` A/B, FP32 TMA C store
- GEMM: square 8K/16K/32K, CTA `256x256`, K stage 64, three stages
- Persistent grid: 148 worker CTAs

## Corrected mainloop

The previous implementation used one completion mbarrier per pipe and waited
for every commit in the issuer warp.  The corrected implementation has a
completion mbarrier for every `[pipe][stage]`.  Issuers commit without an
immediate completion wait; producers wait before reusing that shared-memory
stage, and issuers drain only the final stage before the epilogue.

The persistent scheduler uses a global work queue ordered by output-tile
macroblocks.  M varies fastest for B reuse.  The selected shapes are `16x16`
at 8K/16K and `8x18` at 32K.

Dense 512 validation passed exactly in both normal mode and with one
persistent CTA dynamically processing four C tiles: `bad=0`, `max_abs=0`.

## Paired dense results

Six AB/BA passes, six timed iterations per pass:

| size | normal TFLOP/s | persistent TFLOP/s | gain |
|---:|---:|---:|---:|
| 8192 | 1674.335 | **1697.631** | +1.39% |
| 16384 | 1757.211 | **1777.128** | +1.13% |
| 32768 | 1512.945 | **1615.074** | +6.75% |

## Same-address control

| size | normal TFLOP/s | persistent TFLOP/s |
|---:|---:|---:|
| 8192 | 1877.233 | 1874.692 |
| 16384 | 1953.929 | 1955.753 |
| 32768 | 1781.176 | 1785.165 |

The matched same-address result shows that persistent tile transitions no
longer impose the previous approximately 1-PFLOP/s ceiling.

Reproduce from `5.GEMM`:

```bash
make build-l2 build-persistent NVCC=/usr/local/cuda-12.9/bin/nvcc
./gemm256_tma_tcgen05_persistent \
  --validate --validate-size 512 --validate-pattern pattern \
  --persistent-ctas 1
./gemm256_tma_tcgen05_persistent \
  --sizes 8192,16384,32768 --warmup 3 --iters 6 \
  --input-init random --persistent-ctas 148 \
  --csv persistent.csv
```
