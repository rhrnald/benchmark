# Canonical B-first N-split vs cuBLAS

Measured 2026-07-31 on Vast B200 instance `46376869`, after promoting
consumer `wait B -> wait A` to the canonical kernel.

## Protocol

| item | value |
|---|---|
| GEMM | square row-major `C=A*B` |
| types | BF16 A/B, FP32 accumulation and output |
| input distributions | BF16 uniform `[0,1)` and `[-8,8)` using identical seeds |
| custom kernel | persistent 148 CTA, static `8x16`, `256x256x64`, 3 stages |
| consumer order | W2 `B0 -> A`, W3 `B1 -> A` |
| timing | one warmup, five timed launches |
| processes | four independent processes per cell |
| ordering | ours/cuBLAS alternate; each occupies each position twice |
| device | NVIDIA B200, driver `595.71.05` |
| CUDA/cuBLAS | CUDA 12.9, `cublasGetVersion() = 120901` |
| definition | canonical `899f940`, pinned runner commit `3583cda` |

All three custom size ports pass full-C 512 `pattern` and `ones` validation.
The table reports mean TFLOP/s plus sample standard deviation.

## Result

| size | input | ours, B-first | cuBLAS | ours/cuBLAS |
|---:|---|---:|---:|---:|
| 8K | `[0,1)` | 1763.599 ± 1.260 | 1822.687 ± 3.414 | 96.758% |
| 8K | `[-8,8)` | 1592.419 ± 3.685 | 1619.840 ± 6.199 | **98.307%** |
| 16K | `[0,1)` | 1854.043 ± 1.457 | 1924.352 ± 1.688 | 96.346% |
| 16K | `[-8,8)` | 1648.123 ± 3.490 | 1708.077 ± 2.013 | 96.490% |
| 32K | `[0,1)` | 1666.094 ± 12.326 | 1698.081 ± 10.980 | **98.116%** |
| 32K | `[-8,8)` | 1435.286 ± 15.128 | 1466.391 ± 1.892 | **97.879%** |

The closest cells are 8K signed at 98.307% and 32K unit at 98.116%.
The largest remaining gap is 16K unit, where ours reaches 96.346% of cuBLAS.
The 32K custom process variance is materially larger than at 8K/16K, so its
absolute mean should not be over-interpreted without more processes.

Complete generated source, binaries, validation, environment, sequence, raw
logs, and CSVs are stored in
[`gemm_bfirst_cublas_3583cda.tar.gz`](../results/gemm_bfirst_cublas_3583cda.tar.gz).
