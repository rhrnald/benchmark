# BF16 GEMM ours vs cuBLAS remeasurement

Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.
Each cell is mean +/- sample SD across four independent processes;
each process uses one warmup and five timed launches. The two methods
occupy the first and second positions exactly twice per cell.

| size | input | ours (static `8x16`) | cuBLAS | ours/cuBLAS |
|---:|---|---:|---:|---:|
| 8K | `[0,1)` | 1766.749 +/- 1.109 | 1799.487 +/- 4.022 | 98.181% |
| 8K | `[-8,8)` | 1590.216 +/- 2.410 | 1609.157 +/- 1.251 | 98.823% |
| 16K | `[0,1)` | 1829.322 +/- 2.935 | 1893.726 +/- 0.828 | 96.599% |
| 16K | `[-8,8)` | 1620.620 +/- 3.089 | 1680.376 +/- 2.541 | 96.444% |
| 32K | `[0,1)` | 1580.013 +/- 9.121 | 1603.139 +/- 11.766 | 98.557% |
| 32K | `[-8,8)` | 1371.745 +/- 4.459 | 1417.911 +/- 13.582 | 96.744% |
