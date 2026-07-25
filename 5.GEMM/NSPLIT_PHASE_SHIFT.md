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
