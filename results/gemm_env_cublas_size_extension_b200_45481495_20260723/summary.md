# 8K and 32K GEMM cuBLAS environment reference

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 148 SM, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Runtime cuBLAS: `libcublas.so.12.9.1.4`
- Initial definition commit: `e47d494997a2692c814d7d1669f288f058366232`
- 8K signed confirmation definition:
  `15c919ec6fc0ef5e3df99f607fc8e370f0be6e6d`
- Problem: row-major `C=A*B`, square M=N=K, BF16 A/B, FP32
  accumulation and FP32 C
- Protocol: one case per process, one warmup, five timed launches

The initial activation measured three position-balanced process samples per
method at 8K. At 32K, four methods were each placed in every one of four
sequence positions once. The extra method is the direct E7a `16x16` size port;
the reported size-tuned E7a uses the historical 32K `8x18` macro shape.

The preserved `p0`, generated E7a kernels, and cuBLAS use the same deterministic
BF16 input generator and seeds. The preserved `p0` and cuBLAS executables are
byte-identical to the earlier references:

```text
p0     166044d6690b52dc448befefb79baabb1c9cb56950b16a0536454249d623ff72
cuBLAS 1f47799b1ffd8d815f457aa908be4ded8c8a3539d6a11fcc8053410612f5148b
```

## Size overview

All values are CUDA-event TFLOP/s and sample SD across independent processes.
Each row compares methods from the same activation. The 16K rows are brought
in from the preceding environment-reference run on the same GPU and software
stack.

### BF16 uniform `[0,1)`

| size / session | E7a schedule | preserved `p0` | E7a | cuBLAS | E7a vs `p0` | E7a/cuBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8K initial, n=3 | `16x16` | 1674.848 +/- 0.241 | **1704.894 +/- 1.270** | 1737.851 +/- 1.589 | +1.794% | **98.104%** |
| 16K prior, n=3 | `16x16` | 1737.130 +/- 1.018 | **1770.382 +/- 0.758** | 1836.906 +/- 0.724 | +1.914% | **96.378%** |
| 32K initial, n=4 | `8x18` | 1591.566 +/- 4.728 | **1608.133 +/- 10.369** | 1625.373 +/- 20.788 | +1.041% | **98.939%** |

### BF16 uniform `[-8,8)`

| size / session | E7a schedule | preserved `p0` | E7a | cuBLAS | E7a vs `p0` | E7a/cuBLAS |
|---|---|---:|---:|---:|---:|---:|
| 8K confirmation, n=3 | `16x16` | 1534.717 +/- 1.239 | **1555.989 +/- 1.043** | 1580.738 +/- 1.507 | +1.386% | **98.434%** |
| 16K prior, n=3 | `16x16` | 1512.169 +/- 2.782 | **1529.712 +/- 4.092** | 1608.169 +/- 5.106 | +1.160% | **95.121%** |
| 32K initial, n=4 | `8x18` | 1316.549 +/- 7.034 | **1338.924 +/- 8.584** | 1332.195 +/- 17.252 | +1.699% | **100.505%** |

The 32K signed E7a/cuBLAS crossing is only 0.505%, while cuBLAS has a
17.252-TFLOP/s sample SD. It should be treated as a tie at this protocol, not
as evidence that E7a is definitively faster than cuBLAS.

## 32K scheduler ablation

Only the persistent macro shape differs between these two generated E7a
sources. Both have REG 172, STACK 0, no spill/local memory, 1184 B static
shared memory, and 197632 B dynamic shared memory.

| input | direct size port `16x16` | size-tuned `8x18` | change | paired signs |
|---|---:|---:|---:|---:|
| `[0,1)` | 1595.511 +/- 12.227 | **1608.133 +/- 10.369** | +0.791% | 3/4 positive |
| `[-8,8)` | 1278.397 +/- 17.720 | **1338.924 +/- 8.584** | +4.735% | 4/4 positive |

The unit-input gain is small and noisy. The signed-input gain is large and
consistent, so `8x18` is the better 32K configuration for the overview.

## 8K signed confirmation

The initial 8K signed cuBLAS samples were
`1482.058 / 1519.939 / 1519.831` TFLOP/s. The first value was 2.489% below the
other two despite the same reported 1965 MHz SM clock and 38 C temperature.
It was not removed after the fact.

A separate activation repeated a complete three-position rotation for all
three methods:

| method | confirmation samples | mean +/- sample SD |
|---|---|---:|
| preserved `p0` | 1533.670 / 1534.396 / 1536.084 | 1534.717 +/- 1.239 |
| E7a `16x16` | 1555.852 / 1555.021 / 1557.093 | 1555.989 +/- 1.043 |
| cuBLAS | 1581.372 / 1579.018 / 1581.824 | 1580.738 +/- 1.507 |

All three methods shifted upward in the new activation, so the two activations
are not pooled into one six-sample absolute mean. The confirmation's
same-activation E7a/cuBLAS ratio is 98.434%, and its three per-pass ratios are
98.386%, 98.480%, and 98.437%. This replaces the misleading initial apparent
100.494% crossing in the overview.

## Correctness and provenance

- The 8K E7a changes only the compile-time problem size and symbol/file names
  from canonical 16K E7a; its scheduler remains `16x16`.
- The direct 32K E7a similarly changes only size and names.
- The tuned 32K E7a additionally changes only the persistent macro from
  `16x16` to `8x18`.
- All three generated kernels use REG 172, STACK 0, no spills, and no local
  memory.
- E7a 8K passed 512 pattern and ones full-C validation bit-exactly.
- Direct 32K passed the 512 pattern validation; tuned 32K passed pattern and
  ones. Every recorded validation has `bad=0`, `max_abs=0`, `max_rel=0`.
- All 50 initial pre-case snapshots reported 1965 MHz SM and 3996 MHz memory
  clocks. The initial run moved from 36 C to 43 C; method position was balanced
  within each size and distribution.

The first attempted confirmation definition, commit `70173a4`, stopped at its
SHA preflight because of a truncated expected hash and executed no performance
case. Commit `15c919e` fixes only that preflight literal and is the definition
that produced the confirmation data.

## Artifacts

- `aggregate.csv`: every process value, session label, SD, and ratio
- `sequence.tsv`: initial balanced execution order
- `sequence_8k_signed_confirmation.tsv`: confirmation execution order
- `logs/`: individual output, validation, resources, environment, hashes, and
  temperature/clock snapshots
- `source/`: exact generated E7a sources and driver-script snapshots

The Vast instance was stopped after each artifact download.
