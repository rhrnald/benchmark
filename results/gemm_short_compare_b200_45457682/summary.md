# BF16 GEMM short-window comparison

- Date: 2026-07-21
- Vast.ai instance: `45457682`
- GPU: NVIDIA B200, driver 580.126.09, 1000 W power limit
- Input: bit-identical BF16 uniform `[0,1)` A/B
- Operation: square `C=A*B`, FP32 accumulation and FP32 output
- Protocol: one case per process, 3 warmups, 6 timed iterations, 3 passes
- Order: implementation and size order rotated between passes

## Results

| size | implementation | three process results (TFLOP/s) | mean |
|---:|---|---|---:|
| 8192 | custom persistent | 1714.929 / 1710.944 / 1715.842 | **1713.905** |
| 8192 | cuBLAS GemmEx | 1790.098 / 1784.365 / 1787.584 | **1787.349** |
| 8192 | CUTLASS | 1636.700 / 1631.310 / 1637.520 | **1635.177** |
| 16384 | custom persistent | 1778.312 / 1782.250 / 1771.802 | **1777.455** |
| 16384 | cuBLAS GemmEx | 1868.511 / 1880.441 / 1879.254 | **1876.069** |
| 16384 | CUTLASS | 1403.330 / 1420.350 / 1416.220 | **1413.300** |
| 32768 | custom persistent | 1619.021 / 1623.895 / 1616.578 | **1619.831** |
| 32768 | cuBLAS GemmEx | 1627.448 / 1627.671 / 1619.254 | **1624.791** |
| 32768 | CUTLASS | 1230.000 / 1233.860 / 1224.200 | **1229.353** |

| size | custom versus cuBLAS | custom versus CUTLASS |
|---:|---:|---:|
| 8192 | -4.11% | +4.81% |
| 16384 | -5.26% | +25.77% |
| 32768 | -0.31% | +31.76% |

The custom kernel reproduces the earlier short-window result, including
`1777.455 TFLOP/s` at 16K. cuBLAS is fastest at 8K and 16K; custom and cuBLAS
are effectively tied at 32K within the observed run-to-run spread.

## Implementations

- Custom: persistent 148-CTA `256x256x64`, three-stage kernel.
- cuBLAS: `cublasGemmEx`, BF16 A/B, FP32 compute and C, `alpha=1`, `beta=0`.
- CUTLASS 8K: `256x256x64`, dynamic `2x1` cluster, 2-SM five-stage
  direct-store Stream-K kernel in heuristic mode.
- CUTLASS 16K/32K: `256x256x64`, static `4x1` cluster, 2-SM five-stage
  direct-store CLC scheduler.
- CUTLASS revision: `e8ecfad75b44d1ad56264f5001d877e9e47fe080`.

`raw.log` contains all process output. The custom CSV files are stored in this
directory. CUTLASS's example driver was extended only to accept an explicit
`--warmup` count so that all three implementations use the same 3/6 protocol.
