# Static 8x16 local-order comparison

16K square BF16 GEMM, FP32 accumulation/output, 148 persistent CTAs,
static grid-stride ownership, W1/I5, four ABBA-position-balanced
independent process samples per cell.

| input | M-fast | N-fast | N-fast vs M-fast |
|---|---:|---:|---:|
| `[0,1)` | 1839.429 +/- 1.177 | 1781.671 +/- 1.489 | -3.140% |
| `[-8,8)` | 1630.438 +/- 1.106 | 1568.869 +/- 2.581 | -3.776% |
