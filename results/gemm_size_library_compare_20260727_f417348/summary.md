# BF16 GEMM size/library comparison

Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.
Each cell is mean +/- sample SD across three independent processes;
each process uses one warmup and five timed launches. Method order is
cyclic position-balanced.

| size | input | ours (static `8x16`) | cuBLAS | selected CUTLASS | ours/cuBLAS |
|---:|---|---:|---:|---:|---:|
| 8K | `[0,1)` | 1772.068 +/- 3.328 | 1807.121 +/- 0.902 | 1667.820 +/- 2.070 | 98.060% |
| 8K | `[-8,8)` | 1594.164 +/- 1.235 | 1613.517 +/- 1.106 | 1471.757 +/- 26.765 | 98.801% |
| 16K | `[0,1)` | 1835.421 +/- 0.961 | 1902.320 +/- 1.022 | 1443.860 +/- 0.826 | 96.483% |
| 16K | `[-8,8)` | 1624.303 +/- 3.847 | 1685.414 +/- 1.087 | 1311.863 +/- 7.713 | 96.374% |
| 32K | `[0,1)` | 1602.280 +/- 21.458 | 1617.230 +/- 18.678 | 1282.530 +/- 2.216 | 99.076% |
| 32K | `[-8,8)` | 1391.279 +/- 9.220 | 1420.659 +/- 5.608 | 1136.013 +/- 3.323 | 97.932% |
