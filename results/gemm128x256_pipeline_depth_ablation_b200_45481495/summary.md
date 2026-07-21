# `128x256` distinct-B pipeline-depth ablation (B200)

This experiment asks whether the 1712-TFLOP/s repeated-address baseline can
be improved by changing only pipeline granularity and depth.  It does not use
the shared-B/broadcast shortcut: the left and right `128x128` output halves
load distinct B panels in every variant.

## Fixed conditions

- B200 instance `45481495`, 148 persistent CTAs
- CTA output tile `128x256`, random BF16 `[0,1)` inputs, TMA FP32 C store
- repeated global-memory addresses (`GEMM_REPEAT_INPUT=1`)
- identical total A/B bytes and identical MMA work for every variant
- one case per process, one warmup and five timed launches
- three process-level samples in rotated variant order
- event TFLOP/s below is the mean and sample standard deviation of those
  three process-level results
- 512 pattern validation before measurement; all variants were bit-exact

The existing phase settings were held fixed: pipe-1 phase 128 cycles, TMA
phase 96 cycles at 8K and 128 cycles at 16K/32K, MMA phase 0 cycles.

## Result

| variant | dynamic SMEM | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| `K128`, 2 stages | 197632 B | 1600.490 +/- 0.515 | 1711.930 +/- 0.969 | 1556.790 +/- 1.137 |
| `K64`, 3 stages | 148480 B | 1642.320 +/- 1.160 | **1753.739 +/- 0.744** | 1600.875 +/- 1.915 |
| `K64`, 4 stages | 197632 B | **1645.157 +/- 1.616** | 1751.973 +/- 0.547 | **1604.700 +/- 1.772** |

All throughput values are TFLOP/s.

Relative to `K128`, 2 stages:

| size | `K64`, 3 stages | `K64`, 4 stages |
|---:|---:|---:|
| 8K | +2.614% | **+2.791%** |
| 16K | **+2.442%** | +2.339% |
| 32K | +2.832% | **+3.078%** |

## Interpretation

There is a real pipeline-only gain.  Splitting each `K=128` step into two
`K=64` steps lets TMA production and MMA consumption interleave more finely,
raising 16K from 1711.930 to 1753.739 TFLOP/s without reducing communication
or changing the mathematical work.

A fourth stage provides no consistent additional gain over three stages: it
is 0.10% slower at 16K and only 0.17--0.24% faster at 8K/32K.  Therefore the
material improvement comes primarily from `K=64` granularity; a three-stage
ring is already sufficient to cover most of the latency.  The remaining gap
to the historical 1797/1800 result cannot be attributed solely to pipeline
depth because that historical kernel also reused one B panel for both output
halves and transferred less B data.

## Reproduction

```bash
./run_b200_gemm128x256_pipeline_depth_ablation.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm128x256_pipeline_depth_ablation
```

The experiment definition is Git commit `e5a31ee`.  Raw CSVs, validation
logs, source snapshot, binary hashes, and pre/post `nvidia-smi` snapshots are
stored beside this summary.
