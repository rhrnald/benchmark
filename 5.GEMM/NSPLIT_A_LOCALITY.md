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

## B200 result

Definition commit: `6014916`.  The run used Vast instance `45481495`, CUDA
12.9, one B200, dense 16K GEMM, real A/B/C addresses, and W1/I5 with four
position-balanced processes per input.  All `pattern` and `ones` full-C
validations passed.

| variant | `[0,1)` TFLOP/s | vs baseline | `[-8,8)` TFLOP/s | vs baseline |
|---|---:|---:|---:|---:|
| `baseline` | 1753.998 +/- 0.313 | -- | 1521.339 +/- 6.957 | -- |
| `nfast_16x16` | 1765.930 +/- 0.574 | +0.6803% | 1524.371 +/- 3.632 | +0.1993% |
| `nfast_8x32` | 1711.153 +/- 15.678 | -2.4427% | 1425.926 +/- 3.545 | -6.2717% |
| `nfast_4x64` | 1511.402 +/- 14.112 | -13.8310% | 1274.197 +/- 19.193 | -16.2451% |

`nfast_16x16` confirms a small benefit from making same-A N neighbors
consecutive, but it fails the two-input `+0.5%` promotion gate because the
signed input improves by only `+0.1993%`.  Stronger `8x32` and `4x64` A
locality are clearly harmful.  The likely reason is that the wider N sweep
destroys too much B locality and produces a less favorable persistent-wave
interleaving; the repeated-A diagnostic cannot be realized by simply making
larger N-fast macroblocks.

Decision: do not promote any A-locality scheduler variant as the default.
Keep `nfast_16x16` as a possible component for a future combined experiment,
but it is not a 5% path by itself.

Artifact:
[`../results/gemm_nsplit_a_locality_b200_45481495_20260726_6014916/`](../results/gemm_nsplit_a_locality_b200_45481495_20260726_6014916/).
The full archive SHA-256 is
`f8a3a90697fa13ab475bd8479198a806eb2350af84d3243201b81935026f37d9`.
