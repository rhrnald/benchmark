# Static macro orientation comparison

16K square BF16 GEMM, uniform `[-8,8)`, FP32 accumulation/output,
148 persistent CTAs, static grid-stride ownership, W1/I5, four
ABBA-position-balanced independent process samples per variant.

| static macro | samples (TFLOP/s) | mean +/- sample SD | vs `8x16` |
|---:|---:|---:|---:|
| `8x16` | 1628.617 / 1630.365 / 1627.641 / 1625.577 | **1628.050 +/- 1.997** | +0.000% |
| `16x8` | 1600.183 / 1600.802 / 1596.822 / 1603.149 | **1600.239 +/- 2.612** | -1.708% |
