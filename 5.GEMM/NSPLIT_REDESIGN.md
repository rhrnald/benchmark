# 256x256 N-split GEMM redesign

## Goal

The optimization target is a single-CTA `256x256x64` mainloop with the
following ownership:

| object | stage load / ownership |
|---|---|
| A | one `256x64` BF16 TMA load (32 KiB), shared by both MMA warps |
| B0 | one `64x128` BF16 TMA load (16 KiB), N columns `[0,128)` |
| B1 | one `64x128` BF16 TMA load (16 KiB), N columns `[128,256)` |
| warp 2 | accumulate output `C[0:256, 0:128]` |
| warp 3 | accumulate output `C[0:256, 128:256]` |

This is an N split, not the E7a K split.  B0 and B1 each contain the complete
K64 depth for one N128 output half.  A is loaded once and consumed by both
warps.

`tcgen05.mma` CTA-group-1 does not provide an M256 instruction.  Each
consumer therefore realizes its logical `256x128` output as two consecutive
`m128n128k16` operations, one for each M128 half.  For every K64 stage:

```text
W0: wait reuse(B0, B1) -> TMA A 256x64 -> TMA B0 64x128
W1: wait reuse(B1)     -> TMA B1 64x128
W2: wait A/B0 -> 4 K16 x 2 M128 MMA -> commit pipe 0
W3: wait A/B1 -> 4 K16 x 2 M128 MMA -> commit pipe 1
```

The CTA consequently issues 16 dynamic MMA instructions per K64.  The four
physical TMEM result tiles remain `128x128`, so the existing full FP32
SW128/TMA epilogue is unchanged.

The exact starting source is the previously validated macro-free E2a source:

- Git source commit: `3d2d0a4`
- canonical source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- pattern and ones size-512 full-C validation: exact
- historical codegen: 174 registers, zero stack/local/spill

The former E7a M-split source is retained as an exact comparison source at
`results/gemm_e7a_phase_redesign_b200_45481495_20260724_99246ad0/source/baseline.cu`.

## First experiment

The first run separates topology recovery from optimization:

| variant | only changed mechanism |
|---|---|
| `e7a_exact` | archived exact E7a M-split reference |
| `nsplit_exact` | exact requested N-split implementation |
| `nsplit_consumer_u1` | codegen-only control: add `#pragma unroll 1` only to the consumers' outer K64 loop |
| `nsplit_u1_suspend_prod` | above source control plus CUTLASS-style suspend hint only on producer stage-reuse waits |

`consumer_u1` is a compiler-policy control: E7a uses that policy, while the
audited E2a source allowed compiler auto-unrolling.  The suspend candidate
does not alter the consumer readiness waits or the final MMA completion
wait.  It changes only producer waits for an SMEM stage that is still owned
by a previous MMA epoch.

The local CUDA 12.9 SM100a audit found the normalized 1,913-instruction main
kernel sequence of `nsplit_consumer_u1` to be byte-for-byte identical to
`nsplit_exact`; both use 174 registers with no stack or spill.  The control is
still compiled and validated, but is omitted from timed B200 runs.  The
suspend candidate is therefore paired directly with `nsplit_exact`.

Simple TMA/MMA phase delays are excluded from this run.  The corrected
N-split phase resweep already found TMA delays of 32--128 cycles to be
approximately 0.04--0.08% slower and MMA delays of 64/128 cycles to be
approximately 0.70/0.62% slower than phase `0/0`.

## Measurement protocol

- one NVIDIA B200, 148 persistent CTAs
- dense `M=N=K=16384`, complete real A/B coordinates
- row-major BF16 A/B, FP32 accumulation and complete FP32 C store
- BF16 uniform `[0,1)` and uniform `[-8,8)` inputs
- one case per process, one warmup and five timed launches
- three position-rotated passes per input and timed variant
- pattern and ones size-512 full-C validation for every binary
- source/binary hashes, compiler resource report, and SASS retained
- decision based on pass-matched ratios from one activation

An N-split mechanism advances when it improves both distributions by at
least 0.5% relative to its matched N-split control.  Replacing the E7a
performance reference additionally requires beating `e7a_exact` in both
distributions.  Smaller consistent changes are diagnostics, not adoption
claims.

## First B200 result

The experiment ran on instance `45481495` from definition commit
`616d6c029facfac4b00f0be1209fe3e9d9a9f62f`.  Each value below is the mean of
three position-balanced W1/I5 processes; the uncertainty is sample standard
deviation.

| variant | `[0,1)` TFLOP/s | vs E7a | `[-8,8)` TFLOP/s | vs E7a |
|---|---:|---:|---:|---:|
| `e7a_exact` | 1817.023 +/- 1.632 | -- | 1612.664 +/- 4.797 | -- |
| `nsplit_exact` | 1797.175 +/- 3.595 | -1.0924% | 1593.776 +/- 2.177 | -1.1712% |
| `nsplit_u1_suspend_prod` | 1796.770 +/- 1.533 | -1.1146% | 1601.205 +/- 4.090 | -0.7106% |

Relative to `nsplit_exact`, producer-only suspend changed `[0,1)` by
`-0.0224%` paired and `[-8,8)` by `+0.4660%` paired.  It does not improve
both distributions and misses the 0.5% gate, so it is rejected.

All four compiled binaries, including the untimed `consumer_u1` control,
passed pattern and ones size-512 full-C validation with zero error.  E7a used
172 registers; all N-split variants used 174.  Every variant had zero
stack/local memory/spill and 1184 B static shared memory.  The compiler again
produced identical normalized main-kernel SASS for `nsplit_exact` and
`nsplit_consumer_u1`.

The GPU remained at a recorded maximum SM clock of 1965 MHz and warmed only
from 32 C to 34 C.  The E7a/N-split difference is therefore a same-session
topology result, not a comparison across different instance temperatures.
The requested N-split remains the optimization working source, while E7a
remains the performance reference.

Exact CSVs, generated sources, validation logs, resource reports, normalized
kernel instruction streams, telemetry, and hashes are retained in
[`../results/gemm_nsplit_redesign_b200_45481495_20260724_616d6c0/`](../results/gemm_nsplit_redesign_b200_45481495_20260724_616d6c0/).

## Reproduction

```bash
./run_b200_gemm_nsplit_redesign.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_redesign_b200
```

The runner refuses to execute if either exact source hash differs from the
audited value.
