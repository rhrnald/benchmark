# Recent direct N-split BF16 uniform `[-8,8)` comparison

Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.
Each cell is mean +/- sample SD across three independent processes;
each process used one warmup and five timed launches.

## Ours scheduler selection

| size | macro | TFLOP/s |
|---:|---:|---:|
| 8K | `16x16` | 1566.498 +/- 1.885 |
| 8K | `12x12` | 1565.857 +/- 2.559 |
| 8K | `8x18` | 1568.114 +/- 1.792 **selected** |
| 16K | `16x16` | 1600.626 +/- 3.408 **selected** |
| 32K | `16x16` | 1298.032 +/- 22.721 |
| 32K | `12x12` | 1386.478 +/- 13.173 |
| 32K | `8x18` | 1406.844 +/- 9.481 **selected** |

## Selected comparison

| size | CUTLASS | cuBLAS | ours (recent N-split best) | ours vs cuBLAS |
|---:|---:|---:|---:|---:|
| 8K | 1485.807 +/- 0.471 | 1610.837 +/- 2.735 | 1568.114 +/- 1.792 (`8x18`) | -2.652% |
| 16K | 1309.717 +/- 8.656 | 1683.090 +/- 1.349 | 1600.626 +/- 3.408 (`16x16`) | -4.900% |
| 32K | 1138.963 +/- 3.752 | 1422.276 +/- 4.309 | 1406.844 +/- 9.481 (`8x18`) | -1.085% |
