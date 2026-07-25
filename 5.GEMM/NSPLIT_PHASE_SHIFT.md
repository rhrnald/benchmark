# Direct N-split phase-shift ablation

Last updated: 2026-07-25

## Goal

Test whether phase shifting helps the direct N-split dense GEMM without any
cross-tile prefetch.  The source remains the requested dataflow:

- A `256x64` TMA load shared by both consumer warps;
- B0/B1 each `64x128`;
- warp 2/3 each accumulate one logical `256x128` output half;
- full FP32 C store through the existing SW128 TMA epilogue.

## Variants

| variant | changed mechanism |
|---|---|
| `baseline` | direct N-split source |
| `cta4` | one-time CTA startup offset `(blockIdx.x % 4) * 256` cycles |
| `cta8` | one-time CTA startup offset `(blockIdx.x % 8) * 128` cycles |
| `b1_gap0` | codegen control at warp-1 B1 issue site |
| `b1_gap32` | `nanosleep.u32 32` before each B1 TMA issue |
| `b1_gap64` | `nanosleep.u32 64` before each B1 TMA issue |
| `pipe1_gap0` | codegen control before pipe-1 consumer MMA |
| `pipe1_gap32` | `nanosleep.u32 32` before pipe-1 consumer MMA after ready waits |
| `pipe1_gap64` | `nanosleep.u32 64` before pipe-1 consumer MMA after ready waits |

`b1_gap32/64` are compared against `b1_gap0`, and `pipe1_gap32/64` against
`pipe1_gap0`, so the effect is not confused with the extra instruction site.

## Run

```bash
./run_b200_gemm_nsplit_phase_shift.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_phase_shift_b200
```

For local codegen-only validation:

```bash
GEMM_CODEGEN_ONLY=1 ./run_b200_gemm_nsplit_phase_shift.sh \
  /home/chaewon/benchmark \
  /tmp/gemm_nsplit_phase_shift_codegen
```

## Decision

Use the standard dense 16K protocol: one case per process, warmup 1, timed 5,
both BF16 uniform `[0,1)` and `[-8,8)`.  Adopt only if both distributions are
at least `+0.5%` faster against the matched control.  Smaller positive values
are diagnostic only.

## B200 result

Definition commit: `01a3d17`.  The run used instance `45481495`, CUDA 12.9,
one B200, dense 16K GEMM, real A/B/C addresses, and W1/I5 with three rotated
processes per variant and input.  All 18 size-512 full-C validations passed
with `bad=0`.

| variant | `[0,1)` TFLOP/s | vs matched control | `[-8,8)` TFLOP/s | vs matched control |
|---|---:|---:|---:|---:|
| `baseline` | 1807.204 +/- 3.255 | -- | 1607.658 +/- 7.150 | -- |
| `cta4` | 1805.725 +/- 1.172 | -0.0816% vs baseline | 1605.531 +/- 5.690 | -0.1310% vs baseline |
| `cta8` | 1806.918 +/- 0.927 | -0.0156% vs baseline | 1601.467 +/- 1.560 | -0.3836% vs baseline |
| `b1_gap0` | 1809.522 +/- 2.057 | +0.1286% vs baseline | 1607.710 +/- 4.992 | +0.0050% vs baseline |
| `b1_gap32` | 1803.850 +/- 12.560 | -0.3137% vs `b1_gap0` | 1601.359 +/- 4.281 | -0.3949% vs `b1_gap0` |
| `b1_gap64` | 1811.195 +/- 2.071 | +0.0926% vs `b1_gap0` | 1608.805 +/- 5.580 | +0.0681% vs `b1_gap0` |
| `pipe1_gap0` | 1743.847 +/- 1.587 | -3.5056% vs baseline | 1563.106 +/- 11.076 | -2.7719% vs baseline |
| `pipe1_gap32` | 1719.213 +/- 3.346 | -1.4127% vs `pipe1_gap0` | 1528.315 +/- 0.619 | -2.2224% vs `pipe1_gap0` |
| `pipe1_gap64` | 1671.931 +/- 1.226 | -4.1239% vs `pipe1_gap0` | 1451.694 +/- 7.701 | -7.1251% vs `pipe1_gap0` |

No candidate meets the `+0.5%` two-input adoption gate.  CTA startup
staggering is neutral to negative.  Pipe-1 consumer delay is clearly harmful,
including the codegen control.  `b1_gap64` is the only directionally positive
phase shift, but the effect is only `+0.0926%/+0.0681%`; it remains diagnostic
and is not adopted.

Artifact:
[`../results/gemm_nsplit_phase_shift_b200_45481495_20260725_01a3d17/`](../results/gemm_nsplit_phase_shift_b200_45481495_20260725_01a3d17/).
The full archive SHA-256 is
`5adea263f86ed9cb22462efbafb797cb5c652fcce181c96f3c4565da0321c605`.
