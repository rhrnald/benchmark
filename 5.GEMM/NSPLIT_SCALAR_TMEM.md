# N-split scalar TMEM-load granularity ablation

Last updated: 2026-07-24

## Motivation

The vector-epilogue sweep showed that the scalar transpose epilogue already
uses the minimum 2,048 conflict-free 128-byte shared-memory wavefronts for a
`256x256` FP32 output tile. Vec2/vec4 reduced store instruction count but not
shared bytes or bank service, and their shuffle dependency chains made every
candidate slower than scalar.

This experiment keeps the scalar mapping and changes only the TMEM fragment
width:

- `scalar_x64`: audited transpose control with two unrolled x64 loads per
  `128x128` chunk;
- `scalar_x32`: four streamed x32 loads per chunk, each immediately followed
  by 32 conflict-free scalar store instructions.

The x32 path introduces no shuffle and writes the same physical SW128
addresses. It trades twice as many dynamic TMEM warp-load instructions for a
smaller live fragment and a reusable store body.

## Fixed kernel contract

Both scalar paths preserve:

- one shared BF16 A TMA load of `256x64`, or 32 KiB, per K64;
- independent BF16 B0/B1 TMA loads of `64x128`, or 16 KiB each;
- W2/W3 ownership of logical `256x128` N halves;
- `C_p^T = B_p^T A^T` and one `m128n256k16` MMA per consumer/K16;
- eight CTA MMA issues per K64;
- three 64 KiB shared-memory mainloop stages;
- 148 persistent CTAs and the same `16x16`, local-M-fast tile order;
- four FP32 `128x128` TMA C stores and identical global A/B/C coordinates.

The scalar-x32 generator is hash-gated to scalar-x64 SHA-256
`a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc`.
It reconstructs the input exactly after replacing only the C-stage function
and host-visible epilogue label. Its host audit checks one `128x128` chunk:
all 16,384 SW128 output and TMEM words occur once, all 512 warp-store
transactions are present, and every warp store covers all 32 banks. The
generated stage body applies this audited mapping to all four chunk offsets.

## Local `sm_100a` gate

CUDA 12.9.86, `-O3 -std=c++17 -lineinfo -Xptxas=-v`:

| path | source SHA-256 | ops | registers | stack/local/spill | static TMEM load | static scalar store | shuffle |
|---|---|---:|---:|---:|---:|---:|---:|
| scalar x64 | `a6c31fb0...211cc` | 2229 | 174 | 0 | 8 x64 | 512 | 0 |
| scalar x32 | `be19af8e...7006` | 2011 | 91 | 0 | 4 x32 | 128 | 0 |

The x32 outer load loop is deliberately `unroll 1`. Static SASS therefore has
one x32 load and 32 stores at each of four inlined chunk sites. At runtime,
one full output tile executes:

| path | TMEM warp loads | scalar store warp instructions | expected 128-byte bank wavefronts |
|---|---:|---:|---:|
| scalar x64 | 32 | 2048 | 2048 |
| scalar x32 | 64 | 2048 | 2048 |

Both paths retain MMA 4, TMA load/store 21/4, phase-check 48, one UTCBAR,
static shared 1184 B, and no shuffle. The local gate confirms x32 has zero
stack, local memory, or spill traffic.

## B200 matched protocol

Timed variants:

1. `nsplit_exact`: direct-MMA working canonical;
2. `nsplit_transpose_scalar_x64`: transpose/scalar control;
3. `nsplit_transpose_scalar_x32`: the sole candidate.

For each of `[0,1)` and `[-8,8)`, all six permutations are executed. Thus
every variant occupies every position twice and every ordered non-self
within-order adjacent pair occurs twice.

Every cell uses:

- dense `16384x16384x16384` BF16-input/FP32-output GEMM;
- exactly one process per input/variant case;
- one warmup and five timed launches;
- process-level temperature, power, and SM-clock snapshots;
- full-C pattern 256/512 and ones 512 validation for both transpose paths,
  plus pattern/ones 512 for exact;
- source/binary/SASS/resource hashes and a complete SHA-256 manifest.

Run only from its committed definition:

```bash
cd /workspace/benchmark
./run_b200_gemm_nsplit_scalar_x32.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_scalar_x32_b200
```

Scalar x32 advances only if its pass-paired 95% confidence interval against
scalar x64 is above zero for both inputs. A passing candidate receives a
separately committed counterbalanced x64/x32 confirmation before adoption.
It replaces direct exact only if the confirmed paired mean improvement over
exact is at least `+0.5%` for both inputs with no negative confidence
interval. If x32 does not beat x64, x16 is not sent to B200 and the next axis
is cross-tile one-stage prefetch.

## B200 result

The committed definition `6e138f0` was measured on Vast instance `45481495`
with GPU UUID `GPU-2a1b935d-fd9c-ec3c-8ce7-e6d92149f96f`, CUDA 12.9.86,
driver 580.126.09, and a reported 1965 MHz SM clock. Temperature moved from
30 C before the run to 33 C after it. All 36 independent W1/I5 processes and
all eight full-C validation cases completed successfully.

| path | `[0,1)` TFLOP/s | `[-8,8)` TFLOP/s |
|---|---:|---:|
| direct exact | 1758.255 +/- 1.236 | 1567.368 +/- 2.940 |
| transpose scalar x64 | 1766.621 +/- 1.077 | 1574.829 +/- 3.793 |
| transpose scalar x32 | 1765.736 +/- 0.856 | 1573.798 +/- 1.520 |

The pass-paired comparisons were:

| comparison | `[0,1)` paired change, 95% CI | `[-8,8)` paired change, 95% CI |
|---|---:|---:|
| scalar x32 vs scalar x64 | -0.0501% [-0.1521%, +0.0520%] | -0.0652% [-0.2287%, +0.0984%] |
| scalar x64 vs direct exact | +0.4759% [+0.4146%, +0.5371%] | +0.4762% [+0.2033%, +0.7492%] |
| scalar x32 vs direct exact | +0.4255% [+0.3251%, +0.5260%] | +0.4106% [+0.1820%, +0.6392%] |

Reducing the live register count from 174 to 91 therefore did not improve
throughput. The x32 path leaves the 2,048 scalar shared stores and 2,048
128-byte bank wavefronts unchanged while doubling dynamic TMEM warp loads
from 32 to 64. Because `tcgen05` already limits residency to one CTA per SM,
the register reduction does not create an occupancy benefit.

Scalar x32 fails the screening gate because neither input has a positive
lower confidence bound against x64. Scalar x64 also remains below the
two-input `+0.5%` canonical-replacement threshold against direct exact.
Consequently:

- direct exact remains the canonical source;
- x32 is rejected and x16 will not be measured;
- the next one-factor axis is cross-tile one-stage prefetch.

The curated result is in
[`gemm_nsplit_scalar_x32_b200_45481495_20260724_6e138f0`](../results/gemm_nsplit_scalar_x32_b200_45481495_20260724_6e138f0/).
It excludes binaries and full SASS dumps. The separately retained full archive
has SHA-256
`2eb0f6c459fbae0be8eb32c4b9028e9c2feb465d92413efc0f9d09c5ff81d254`.
