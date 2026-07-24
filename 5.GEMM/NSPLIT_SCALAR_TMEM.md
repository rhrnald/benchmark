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
and host-visible epilogue label. Its exhaustive host audit proves all 65,536
output coordinates and all `128x512` TMEM source words occur once, each
chunk's 16,384 SW128 words occur once, and every warp store covers all 32
banks.

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
