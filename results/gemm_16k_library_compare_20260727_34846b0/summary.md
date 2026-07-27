# 16K BF16 GEMM: ours vs cuBLAS vs CUTLASS

Square row-major `C=A*B`, BF16 A/B, FP32 accumulation/output.
Each cell is mean +/- sample SD across six independent processes.
Each process uses one warmup and five timed launches. The six method
orders are fully position-balanced.

| input | ours (`fixed_sink`) | cuBLAS | CUTLASS selected CLC | ours/cuBLAS |
|---|---:|---:|---:|---:|
| `[0,1)` | 1834.342 +/- 2.432 | 1897.885 +/- 3.031 | 1443.285 +/- 1.724 | 96.652% |
| `[-8,8)` | 1623.016 +/- 1.993 | 1683.923 +/- 2.435 | 1305.937 +/- 5.014 | 96.383% |
