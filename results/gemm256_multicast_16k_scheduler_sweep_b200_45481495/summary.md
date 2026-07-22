# 16K two-CTA multicast macro/scheduler sweep

## Conditions

- Date: 2026-07-22
- GPU: NVIDIA B200, Vast.ai instance `45481495`, 1000 W power limit
- Definition commit: `9fcae48`
- GEMM: BF16 random `[0,1)`, M=N=K=16384, complete FP32 TMA C store
- Kernel: CTA `256x256`, K64, three stages, 148 persistent CTAs grouped
  into 74 clusters of two
- B load: rank-0 TMA multicast to both CTAs; A and `tcgen05 cta_group::1`
  MMA remain CTA-local
- Ordering: local M-fast and macro N-fast
- Macro cross product: M/N in `{8,16,32}`; all divide the 64x64 CTA tile
  grid exactly, so no padded edge tasks differ between configurations
- Scheduler: cluster-leader dynamic atomic allocation versus fixed
  grid-stride static allocation
- Measurement: one process per case, warmup 1, five timed launches, three
  rotated processes per configuration

All 18 configurations passed the 512 pattern validation bit-exactly.

## Results

Event TFLOP/s, mean and sample standard deviation across three processes:

| macro | dynamic | static | static vs dynamic |
|---:|---:|---:|---:|
| 8x8 | 1752.756 +/- 1.831 | 1783.083 +/- 0.392 | +1.730% |
| 8x16 | 1753.843 +/- 1.332 | **1784.075 +/- 0.924** | +1.724% |
| 8x32 | 1753.457 +/- 0.784 | 1783.371 +/- 0.966 | +1.706% |
| 16x8 | 1771.626 +/- 1.357 | 1783.477 +/- 0.329 | +0.669% |
| 16x16 | **1772.407 +/- 0.984** | **1783.590 +/- 0.247** | +0.631% |
| 16x32 | 1771.766 +/- 0.236 | 1783.738 +/- 0.488 | +0.676% |
| 32x8 | 1722.044 +/- 0.868 | 1691.152 +/- 1.009 | -1.794% |
| 32x16 | 1722.082 +/- 0.801 | 1692.586 +/- 2.923 | -1.713% |
| 32x32 | 1721.882 +/- 0.220 | 1688.941 +/- 1.460 | -1.913% |

## Conclusions

`16x16` is the best dynamic shape, but is not a unique overall optimum once
the atomic queue is removed. Static shapes with macro M=8 or 16 form a tight
1783--1784 TFLOP/s plateau. The highest mean is static `8x16`, only 0.485
TFLOP/s (0.027%) above static `16x16`, well inside run-to-run variation.
Static `16x16` has the smallest standard deviation of the entire sweep and is
therefore selected as the robust 16K default.

Static scheduling improves the matched `16x16` configuration by 0.631%.
Eliminating one cluster-level atomic allocation and two cluster syncs per
output-tile pair is beneficial because all 16K tiles have identical K work.
The result reverses for macro M=32: its grid-stride wave mapping harms
locality enough to overwhelm scheduler savings. Macro M=32 should therefore
not be used for this 148-CTA launch.

This static scheduler is CUTLASS-inspired, not CUTLASS-identical. Both are
persistent and cluster-aware, but this kernel uses a custom 2D macro mapping
and a simple fixed grid stride. CUTLASS represents swizzle/raster order in its
tile scheduler and Blackwell kernels can use Cluster Launch Control rather
than either this static mapping or a global atomic queue.

GPU temperature was 33 C before and 36 C after the run. The Vast instance was
stopped after artifact download.
