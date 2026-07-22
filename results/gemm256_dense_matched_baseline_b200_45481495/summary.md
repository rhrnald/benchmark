# Matched dense-address `256x256` GEMM baseline (B200)

This changes only `GEMM_REPEAT_INPUT=1 -> 0` relative to the reproduced
same-address pipeline.  Pipeline depth, phase settings, tile shape, input,
epilogue, and C store remain fixed.  The persistent variant schedules output
tiles in cache-local macroblocks so neighboring workers reuse A/B through L2.

## Conditions

- B200 instance `45481495`, 1000 W power limit
- valid BF16 `256x256x64` CTA tile, distinct B halves
- three shared-memory stages, two TMA/MMA pipes
- real A/B coordinates and full FP32 TMA C store
- BF16 uniform `[0,1)` inputs
- one case per process, warmup 1, five timed launches
- three process samples in rotated grouped/persistent order
- both variants passed the 512 pattern validation bit-exactly

## Result

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| grouped grid | 1694.096 +/- 1.724 | 1761.240 +/- 1.007 | 1585.165 +/- 3.710 |
| persistent, 148 CTAs | **1712.740 +/- 0.306** | **1780.463 +/- 0.734** | 1572.202 +/- 22.749 |

Values are mean +/- sample standard deviation in TFLOP/s.  Persistent improves
8K by 1.10% and 16K by 1.09%.  Its 32K samples were 1587.550, 1582.990, and
1546.066 TFLOP/s; the last sample is an isolated slowdown that makes the mean
unsuitable for selecting a 32K schedule.  The next L2 sweep must carry this
baseline as a paired control.

Compared with the persistent same-address means, real coordinates reduce
throughput by 5.58% at 8K and 8.68% at 16K.  This is the exposed cost that L2
tile scheduling should target.

## Reproduction

```bash
./run_b200_gemm256_dense_matched_baseline.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm256_dense_matched_baseline
```

The experiment definition is Git commit `b2037a6`.  Raw CSVs, validation
logs, source snapshot, hashes, and GPU snapshots are stored beside this file.
