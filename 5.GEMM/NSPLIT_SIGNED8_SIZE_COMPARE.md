# Recent direct N-split signed8 size comparison

This experiment extends the exact recent direct N-split source that measured
`1807.204 TFLOP/s` on `[0,1)` and `1607.658 TFLOP/s` on `[-8,8)` at 16K.
The canonical source and the phase-shift artifact baseline have identical
SHA-256 `cd595ba3...`.

The 8K and 32K ports change only:

- the compile-time square problem size;
- the persistent macro shape.

The mainloop remains the recent direct N-split dataflow: one `256x64` A TMA,
two independent `64x128` B TMA producers, two `256x128` MMA consumers,
three K64 stages, and full FP32 TMA C store. At 8K and 32K, `16x16`, `12x12`,
and `8x18` macros are measured in the same balanced run. The winning mean is
reported as ours for that size. The 16K cell uses the exact canonical
`16x16` source.

All performance cells use deterministic BF16 uniform `[-8,8)`, one case per
process, one warmup, five timed launches, and three process samples.

## Measured result

The run used one 1,000 W B200, driver `580.126.09`, CUDA `12.9.86`, and
definition commit `a472f81`. Every size/macro binary passed both pattern and
ones full-C validation with zero error.

| size | `16x16` | `12x12` | `8x18` | selected |
|---:|---:|---:|---:|---:|
| 8K | 1566.498 +/- 1.885 | 1565.857 +/- 2.559 | **1568.114 +/- 1.792** | `8x18` |
| 16K | **1600.626 +/- 3.408** | -- | -- | `16x16` canonical |
| 32K | 1298.032 +/- 22.721 | 1386.478 +/- 13.173 | **1406.844 +/- 9.481** | `8x18` |

The 8K macro means differ by only 0.14%, so `8x18` is the measured winner but
not a strong scheduler conclusion. At 32K, `8x18` is 8.38% faster than the
direct `16x16` size port and 1.47% faster than `12x12`.

| size | CUTLASS | cuBLAS | ours, recent N-split best | ours vs cuBLAS |
|---:|---:|---:|---:|---:|
| 8K | 1485.807 +/- 0.471 | 1610.837 +/- 2.735 | **1568.114 +/- 1.792** | -2.652% |
| 16K | 1309.717 +/- 8.656 | 1683.090 +/- 1.349 | **1600.626 +/- 3.408** | -4.900% |
| 32K | 1138.963 +/- 3.752 | 1422.276 +/- 4.309 | **1406.844 +/- 9.481** | -1.085% |

The canonical 16K result is 0.438% below its earlier same-source
`1607.658 TFLOP/s` measurement, which is a normal session-level shift. The
earlier `99.97%` ratio used a cuBLAS value from another session; the new
same-session ratio is `95.10%`, so the latter is the valid direct comparison.

Artifact:
`../results/gemm_nsplit_signed8_compare_b200_45481495_20260727_a472f81/`.
