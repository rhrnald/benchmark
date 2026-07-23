# Default persistent GEMM vs cuBLAS and CUTLASS

- Date: 2026-07-21
- Vast.ai instance: `45460466` (destroyed after artifact collection)
- GPU: NVIDIA B200, 148 SM, driver 580.126.09, 1000 W power limit,
  maximum SM clock 1965 MHz
- Operation: square row-major `C=A*B`, BF16 A/B, FP32 accumulation and FP32 C
- Inputs: bit-identical deterministic uniform `[0,1)` and `[-8,8)` A/B
- Protocol: one case per process, one warmup plus five timed launches, three
  processes per cell
- Order: method, size, and distribution order rotated across the three passes
- Pre-case temperature snapshots: 28--34 C

The custom 512 validation passed exactly before measurement: `bad=0`,
`max_abs=0`, `max_rel=0`.

## Uniform `[0,1)`

| size | custom persistent | cuBLAS | CUTLASS | custom vs cuBLAS | custom vs CUTLASS |
|---:|---:|---:|---:|---:|---:|
| 8192 | **1713.253 ± 0.420** | **1787.475 ± 2.912** | **1649.037 ± 2.098** | -4.15% | +3.89% |
| 16384 | **1783.963 ± 0.918** | **1873.248 ± 1.987** | **1433.417 ± 0.671** | -4.77% | +24.46% |
| 32768 | **1625.957 ± 2.371** | **1629.405 ± 22.572** | **1269.063 ± 6.500** | -0.21% | +28.12% |

## Uniform `[-8,8)`

| size | custom persistent | cuBLAS | CUTLASS | custom vs cuBLAS | custom vs CUTLASS |
|---:|---:|---:|---:|---:|---:|
| 8192 | **1546.731 ± 2.572** | **1593.261 ± 0.909** | **1459.173 ± 2.580** | -2.92% | +6.00% |
| 16384 | **1585.642 ± 2.353** | **1665.146 ± 2.631** | **1293.517 ± 5.059** | -4.77% | +22.58% |
| 32768 | **1396.409 ± 7.926** | **1410.183 ± 19.077** | **1102.067 ± 2.386** | -0.98% | +26.71% |

Values are the arithmetic mean and sample standard deviation of three
process-level TFLOP/s results.  Each process result is computed from the
aggregate CUDA-event time of its five timed launches.

## Input-range effect

| size | custom | cuBLAS | CUTLASS |
|---:|---:|---:|---:|
| 8192 | -9.72% | -10.87% | -11.51% |
| 16384 | -11.12% | -11.11% | -9.76% |
| 32768 | -14.12% | -13.45% | -13.16% |

All three implementations slow similarly when the distribution changes from
`[0,1)` to `[-8,8)`, consistent with the previously observed data-dependent
power/clock effect.

## Implementations

- Custom: default recovered persistent kernel, CTA `256x256x64`, three stages,
  148 workers, `16x16` macroblocks at 8K/16K and `8x18` at 32K.
- cuBLAS: `cublasGemmEx`, BF16 inputs, FP32 compute/output, `alpha=1`, `beta=0`.
- CUTLASS 8K: `256x256x64`, dynamic `2x1` cluster, 2-SM five-stage direct-store
  Stream-K kernel in heuristic/deterministic mode.
- CUTLASS 16K/32K: `256x256x64`, static `4x1` cluster, 2-SM five-stage
  direct-store CLC kernel.
- CUTLASS revision: `e8ecfad75b44d1ad56264f5001d877e9e47fe080`.

The CUTLASS rows are the selected targeted kernels above, not a claim about
the best possible kernel in every CUTLASS configuration.  The 32K cuBLAS cell
has noticeably higher process-to-process variance; rotating order prevents it
from systematically favoring one implementation, but more repetitions would
be appropriate for a final paper confidence interval.

`aggregate.csv` contains all process results and statistics.
`comparison_raw.log` contains command output and pre-case temperature/power/
clock snapshots.  `results/` contains the custom kernel CSVs, and
`binaries.sha256` records the library benchmark binaries.  The measured custom
binary SHA-256 was
`f4a90515d549adcf9a8967788bcab9efe99c5e304d8a6a5654d35ce9c0251ab5`.
