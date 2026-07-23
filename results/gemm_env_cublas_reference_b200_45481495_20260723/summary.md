# 16K GEMM cuBLAS environment reference

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 148 SM, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Runtime cuBLAS: `libcublas.so.12.9.1.4`
- Definition commit: `ca9b31b282e6c9c52e4960e371c0059d05fe8c13`
- Problem: row-major `C=A*B`, `M=N=K=16384`, BF16 A/B, FP32
  accumulation and FP32 C
- Protocol: one case per process, one warmup, five timed launches, three
  position-balanced process samples per method and input distribution

The preserved `p0`, current source-backed E7a, and cuBLAS were measured in one
activation session.  The cuBLAS executable is byte-identical to the binary
used by the earlier standard 1/5 comparison on instance `45460466`:

```text
1f47799b1ffd8d815f457aa908be4ded8c8a3539d6a11fcc8053410612f5148b
```

It dynamically loaded CUDA 12.9.1.4 cuBLAS on this instance.  A/B initialization
uses the same hash generator, seeds, and BF16 conversion in all three methods.
The current and `p0` kernels both passed the full 512 pattern validation with
`bad=0`, `max_abs=0`, and `max_rel=0`.

## Same-session results

All values are CUDA-event TFLOP/s.  `of cuBLAS` is the ratio of process means.

| input | method | three process samples | mean ± sample SD | of cuBLAS |
|---|---|---|---:|---:|
| `[0,1)` | preserved `p0` | 1736.619 / 1736.469 / 1738.303 | **1737.130 ± 1.018** | 94.568% |
| `[0,1)` | current E7a | 1769.857 / 1771.251 / 1770.039 | **1770.382 ± 0.758** | **96.378%** |
| `[0,1)` | cuBLAS | 1837.552 / 1837.043 / 1836.123 | **1836.906 ± 0.724** | 100% |
| `[-8,8)` | preserved `p0` | 1512.006 / 1509.472 / 1515.028 | **1512.169 ± 2.782** | 94.030% |
| `[-8,8)` | current E7a | 1525.394 / 1533.533 / 1530.209 | **1529.712 ± 4.092** | **95.121%** |
| `[-8,8)` | cuBLAS | 1611.974 / 1610.166 / 1602.366 | **1608.169 ± 5.106** | 100% |

Current E7a is 1.914% faster than `p0` for `[0,1)` and 1.160% faster for
`[-8,8)` in this session.  Moving from `[0,1)` to `[-8,8)` reduces throughput
by 12.950% for `p0`, 13.594% for E7a, and 12.452% for cuBLAS.

The run also reproduces the earlier measurements on this instance.  The new
`p0` mean is only 0.114% below 1739.110 TFLOP/s, and the new E7a mean is only
0.177% below its canonical 1773.523 TFLOP/s result.  Every pre-case snapshot
reported a 1965 MHz SM clock and 38--39 C GPU temperature; the full run moved
from 36 C to 39 C.

## Historical normalization

There is no cuBLAS measurement from the exact historical `p0` session on
instance `45465499`.  The nearest standard 1/5 reference is a separate B200,
instance `45460466`, with driver 580.126.09 and the exact same cuBLAS
executable.  It measured 1873.248 TFLOP/s for `[0,1)` and 1665.146 TFLOP/s for
`[-8,8)`.

| comparison | `[0,1)` |
|---|---:|
| historical `p0` on `45465499` / indirect cuBLAS on `45460466` | 1806.657 / 1873.248 = **96.445%** |
| current `p0` / current same-session cuBLAS | 1737.130 / 1836.906 = **94.568%** |
| current E7a / current same-session cuBLAS | 1770.382 / 1836.906 = **96.378%** |
| current cuBLAS versus indirect prior cuBLAS | **-1.940%** |
| current `p0` versus historical `p0` | **-3.848%** |
| current E7a versus historical `p0` | **-2.008%** |

The current host runs the same cuBLAS binary 1.940% below the prior
different-instance reference, so a material machine/environment component is
real.  It does not prove that the full 3.848% `p0` decrease is environmental:
the old `p0` and old cuBLAS numbers were not collected on the same GPU.

As a reference-only normalization, current E7a reaches 96.378% of current
cuBLAS, only 0.067 percentage point below the historical cross-instance
`p0`/cuBLAS ratio of 96.445%.  The analogous signed-input E7a/cuBLAS ratio is
95.121%, close to the prior different-kernel standard comparison's 95.225%.
This supports the narrower conclusion that the source-backed E7a has recovered
roughly the historical normalized efficiency, not that 1806.657 has been
directly reproduced.

## Artifacts

- `aggregate.csv`: process values and ratios
- `sequence.tsv`: balanced execution order
- `logs/`: individual output, validation, environment, library hashes, and
  temperature/clock snapshots
- `source/`: exact current-kernel source and driver-script snapshots; for
  cuBLAS, the measured preserved binary SHA above is the authoritative
  provenance

The Vast instance was stopped after artifact download and confirmed as
`actual_status=exited`, `intended_status=stopped`.
