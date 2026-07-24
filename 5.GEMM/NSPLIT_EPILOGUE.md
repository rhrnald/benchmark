# N-split transpose epilogue ablation

Last updated: 2026-07-24

## Fixed mainloop contract

This experiment keeps the requested N-split dataflow unchanged:

- one A TMA load moves a shared `256x64` BF16 tile, or 32 KiB;
- B0 and B1 independently load `64x128` BF16 tiles, or 16 KiB each;
- consumer warps 2 and 3 own logical `256x128` output halves;
- each consumer accumulates every K slice into its own half;
- A is shared by both consumers, while B0/B1 split only the N dimension.

The transpose-compute form evaluates

```text
C_p^T[128x256] = B_p^T[128x64] A^T[64x256]
```

for each N half `p`. It addresses the same TMA payloads as the direct form,
but uses one `m128n256k16` operation per consumer and K16. Therefore the CTA
issues eight MMA operations per K64 instead of sixteen.

The first B200 decomposition proved that the mainloop change is useful:

| path | `[0,1)` TFLOP/s | `[-8,8)` TFLOP/s |
|---|---:|---:|
| direct N-split E2E | 1799.869 | 1599.595 |
| transpose scalar E2E | 1808.175 | 1598.322 |
| direct N-split no-store | 1866.203 | 1662.674 |
| transpose no-store | 1882.926 | 1670.525 |

The no-store transpose path improved by `+0.8963%/+0.4722%`, but its scalar
transpose epilogue added about `0.019/0.029 ms` relative to the direct
epilogue. This ablation changes only that transpose epilogue.

## Why the naive vec2 is not sufficient

With `tcgen05.ld.32x32b`, one lane initially owns one logical N column. A
naive 2x2 register transpose pairs lanes `(2g,2g+1)` and lets both lanes issue
one 64-bit shared store. The output values and alignment are correct, but an
`STS.64` warp request is divided into two half-warp transactions. Within each
half warp, the even- and odd-M rows use the same 16 bank pairs, creating a
two-way conflict.

The conflict-free vec2 mapping instead assigns:

```text
destination M-row parity = lane >> 4
destination N-pair       = lane & 15
```

Lanes 0--15 write the even M row and lanes 16--31 write the odd M row. Each
half warp writes one complete 128-byte row and covers all 32 banks exactly
once. A two-shuffle Latin permutation gathers both adjacent N values.

The `cf1_x32` alternative retains one XOR shuffle. Within every group of four
M-row pairs, it schedules the odd row two pairs after the even row. Their
SW128 masks have an XOR delta of 20 words, so the interleaved lane stores also
cover all banks once.

The vec4 form assigns one M row to each quarter warp. Four indexed shuffles
form a 4x4 register transpose; every `STS.128` quarter-warp transaction covers
all 32 banks once.

Every conflict-free generator performs an exhaustive host audit before it
writes CUDA:

- all 16,384 coordinates of one `128x128` chunk are produced exactly once;
- all SW128 word offsets are unique;
- vec2/vec4 physical words remain adjacent and naturally aligned;
- where inline PTX is used, its addresses equal
  `cstore_sw128_float_word_offset`;
- every half-warp or quarter-warp transaction uses 32 unique banks.

## Local `sm_100a` codegen gate

CUDA 12.9, `-O3 -std=c++17 -lineinfo -Xptxas=-v`:

| epilogue | normalized ops | registers | spill/local | static TMEM load | static shuffle | static shared store |
|---|---:|---:|---:|---:|---:|---:|
| scalar x64 | 2229 | 174 | 0 | 8 x64 | 0 | 512 x 32-bit |
| vec2 x32, adjacent control | 2290 | 115 | 0 | 4 x32 | 64 | 64 x 64-bit |
| vec2 CF1 x32, one shuffle | 2574 | 116 | 0 | 4 x32 | 64 | 64 x 64-bit |
| vec2 CF2 x32, two shuffle | 2421 | 111 | 0 | 4 x32 | 128 | 64 x 64-bit |
| vec2 CF2 x64, fused PTX | 2706 | 128 | 0 | 4 x64 | 256 | 128 x 64-bit |
| vec4 CF x64, fused PTX | 3162 | 128 | 0 | 4 x64 | 256 | 64 x 128-bit |

The x32 variants execute a four-iteration TMEM-load loop; the fused x64
variants execute a two-iteration loop. The static SASS count above intentionally
does not multiply those runtime trip counts.

For one full `256x256` output tile, the shared-store model is:

| epilogue | store warp instructions | shuffle warp instructions | expected 128-byte bank wavefronts |
|---|---:|---:|---:|
| scalar | 2048 | 0 | 2048 |
| vec2 x32 adjacent | 1024 | 1024 | 4096 |
| vec2 CF1 x32 | 1024 | 1024 | 2048 |
| vec2 CF2 x32/x64 | 1024 | 2048 | 2048 |
| vec4 CF x64 | 512 | 2048 | 2048 |

Two local candidates are rejected before B200 timing:

- naive, fully unrolled x64 vec2: 250 registers and the two-way bank conflict;
- `tcgen05.ld.16x64b` vec2: 255 registers, stack/local memory, and spills.

## B200 matched experiment

The timed variants are:

1. `nsplit_exact`: direct-MMA N-split E2E reference;
2. `nsplit_transpose_nostore`: shared transpose mainloop upper bound;
3. `nsplit_transpose_scalar`: scalar transpose-epilogue control;
4. `nsplit_transpose_vec2_x32`: bank-conflicted vec2 control;
5. `nsplit_transpose_vec2_cf1_x32`: one-shuffle conflict-free vec2;
6. `nsplit_transpose_vec2_cf_x32`: two-shuffle conflict-free vec2;
7. `nsplit_transpose_vec2_cf_x64`: fused x64 conflict-free vec2;
8. `nsplit_transpose_vec4_cf_x64`: fused x64 conflict-free vec4.

Protocol:

- dense `16384x16384x16384` BF16-input/FP32-output GEMM;
- inputs `[0,1)` and `[-8,8)`;
- one warmup and the mean of five timed launches per process;
- exactly one input/variant case per process;
- eight Williams-balanced passes per input, so every variant occupies every
  execution position once and, within the eight pass orders, every ordered
  non-self adjacent pair occurs exactly once;
- process-level temperature, power, and SM-clock samples before and after;
- full-C `pattern` 256/512 and `ones` 512 validation for every E2E transpose
  variant;
- source, binary, full/normalized SASS, compiler resources, execution order,
  CSV, validation, telemetry, and SHA-256 manifests preserved.

Run on the B200 instance with a new output directory:

```bash
cd /workspace/benchmark
./run_b200_gemm_nsplit_epilogue.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_epilogue_b200
```

An epilogue candidate advances only when the paired 95% confidence interval
against `nsplit_transpose_scalar` is positive for both input distributions.
The sweep winner then receives a separately committed confirmatory matched
A/B run. It replaces the direct `nsplit_exact` canonical kernel only if that
confirmation shows a paired mean gain over exact of at least 0.5% for both
inputs with no negative confidence interval.

## B200 result

The Williams-balanced run completed on Vast.ai instance `45481495` from
definition commit `e4f5061`. Each cell below is the mean of eight independent
W1/I5 processes; `vs scalar` is the pass-paired mean and 95% confidence
interval.

| variant | `[0,1)` TFLOP/s | vs scalar | `[-8,8)` TFLOP/s | vs scalar |
|---|---:|---:|---:|---:|
| direct N-split exact | 1758.641 +/- 1.232 | -0.4453% [-0.5088,-0.3817] | 1564.629 +/- 1.840 | -0.5949% [-0.8586,-0.3313] |
| transpose no-store | 1839.790 +/- 1.020 | +4.1485% [+4.0799,+4.2171] | 1631.399 +/- 2.513 | +3.6471% [+3.3679,+3.9263] |
| transpose scalar | **1766.507 +/- 1.345** | reference | **1574.004 +/- 4.138** | reference |
| vec2 x32 adjacent | 1763.815 +/- 1.235 | -0.1523% [-0.2087,-0.0960] | 1573.723 +/- 4.553 | -0.0172% [-0.3576,+0.3233] |
| vec2 CF1 x32 | 1763.000 +/- 1.241 | -0.1984% [-0.3090,-0.0878] | 1570.659 +/- 3.969 | -0.2121% [-0.4306,+0.0063] |
| vec2 CF2 x32 | 1761.743 +/- 0.749 | -0.2696% [-0.3184,-0.2209] | 1572.339 +/- 4.840 | -0.1054% [-0.3696,+0.1587] |
| vec2 CF2 x64 | 1765.241 +/- 0.989 | -0.0716% [-0.1097,-0.0336] | 1573.960 +/- 2.387 | -0.0020% [-0.3063,+0.3023] |
| vec4 CF x64 | 1761.078 +/- 0.897 | -0.3073% [-0.3550,-0.2597] | 1568.183 +/- 3.095 | -0.3693% [-0.5965,-0.1421] |

The scalar transpose path beat direct exact by `+0.4473%`
`[+0.3832,+0.5114]` for `[0,1)` and `+0.5994%`
`[+0.3320,+0.8667]` for `[-8,8)`. The first mean remains below the
predeclared `+0.5%` two-input adoption threshold, so direct exact remains the
working canonical.

No vector candidate had a positive confidence interval against scalar on
both inputs. The best vector candidate, CF2 x64, was slightly slower than
scalar: `-0.0716%/-0.0020%`. A conflict-free 128-byte shared-memory
wavefront is already the minimum transaction for each output row segment.
Vector forms reduce store instruction count but not the 256 KiB tile's
minimum 2,048 wavefronts, and add shuffle work. This closes register-transpose
vector width as the next optimization axis.

All 20 full-C validations reported `status=ok`, all eight binaries passed the
expected MMA/TMA/TMEM/store SASS gates, and every kernel had zero stack,
local memory, and spill traffic. Telemetry stayed at 1965 MHz and 32--33 C.
This activation used GPU UUID `GPU-2a1b935d-fd9c-ec3c-8ce7-e6d92149f96f`,
which differs from the earlier transpose run, so only the same-session paired
effects are used for selection.

Artifacts:

- [`gemm_nsplit_epilogue_b200_45481495_20260724_e4f5061`](../results/gemm_nsplit_epilogue_b200_45481495_20260724_e4f5061/)
- archive SHA-256:
  `e68073e5bf509da3959a7df6710aee723d5b40435d8c8e3311cf4047ac703849`
