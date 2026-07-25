# Clean N-split A-locality scheduler sweep

Last updated: 2026-07-26

## Goal

Try to recover the large A-locality gap seen in the earlier partial-reuse
diagnostic without changing the arithmetic kernel.

The diagnostic showed that repeating only A source addresses improved the
dense 16K pipeline by `+6.520%`, while repeating only B improved it by
`+1.648%`.  The current clean N-split scheduler is local M-fast inside a
`16x16` macroblock, which makes neighboring workers share B coordinates first.
This sweep instead makes local N fast so a wave of persistent CTAs consumes
more output tiles with the same M coordinate and therefore the same A panels.

## Fixed kernel contract

All variants keep:

- dense `16384 x 16384 x 16384` BF16 A/B and FP32 C;
- CTA tile `256 x 256`;
- K stage `64`, three shared-memory stages;
- A TMA `256x64`, B0/B1 TMA `64x128` each;
- direct N-split MMA ownership, warp 2/3 each computing one `256x128` C half;
- full FP32 C TMA store;
- 148 persistent CTAs;
- no phase shift, no L2 promotion, no multicast.

Only the mapping from the persistent task id to `(tile_m, tile_n)` changes.

## Variants

| variant | macro shape | local order | macro order | intended locality |
|---|---:|---|---|---|
| `baseline` | `16x16` | M-fast | N-fast | current B-first schedule |
| `nfast_16x16` | `16x16` | N-fast | M-fast | same macro shape, A-first |
| `nfast_8x32` | `8x32` | N-fast | M-fast | stronger A reuse per wave |
| `nfast_4x64` | `4x64` | N-fast | M-fast | maximum 16K-row A reuse inside one N sweep |

The generator is hash-gated against the current clean source SHA-256
`cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`.

## Local codegen gate

The codegen-only run passed locally with CUDA 12.9 and `sm_100a`.

| variant | normalized ops | registers | static shared | UTCHMMA | TMA load sites | TMA store sites | phase checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 1913 | 174 | 1184 | 8 | 21 | 4 | 48 |
| `nfast_16x16` | 1913 | 174 | 1184 | 8 | 21 | 4 | 48 |
| `nfast_8x32` | 1913 | 174 | 1184 | 8 | 21 | 4 | 48 |
| `nfast_4x64` | 1913 | 174 | 1184 | 8 | 21 | 4 | 48 |

All variants have stack/local/spill 0.  Therefore a performance difference is
interpretable as scheduler/cache behavior rather than mainloop codegen drift.

## Run

Inside the B200 instance:

```bash
cd /workspace/benchmark
./run_b200_gemm_nsplit_a_locality.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_a_locality_b200
```

For local codegen-only validation:

```bash
out_dir=$(mktemp -d /tmp/gemm-a-locality-codegen.XXXXXX)
GEMM_CODEGEN_ONLY=1 \
  ./run_b200_gemm_nsplit_a_locality.sh \
  /home/chaewon/benchmark \
  "$out_dir/result"
```

## Decision gate

Use the standard dense 16K protocol:

- one case per process;
- warmup 1, timed 5;
- four position-balanced processes per input;
- inputs `[0,1)` and `[-8,8)`;
- full-C validation for `pattern` and `ones` before timing.

Promote a candidate only if both input distributions improve by at least
`+0.5%` over baseline and all validation/resource gates pass.  A result below
that threshold remains diagnostic.  The user-facing 5% target requires a much
larger positive result than previous local-order sweeps, so this run is a
screening test for whether stronger A-locality macro shapes matter.
