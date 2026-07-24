# E7a phase-shift redesign experiment

## Question

Can phase staggering improve the current `256x256x64` E7a dense GEMM without
reusing the old N-split pipeline's one-time `96/0`, `128/0`, or larger
TMA/MMA delays?

The current E7a topology differs from that older kernel:

- warp 0 waits for shared-stage reuse, then issues A and late B1;
- warp 1 waits for the same reuse dependencies, then issues early B0;
- warps 2 and 3 consume the same A/B stage and compute the upper/lower
  `128x256` M halves with `m128n256k16`;
- all three TMA transactions and both consumers are therefore coupled again
  on every three-stage ring reuse.

The per-stage trace found that B0/B1 are normally hidden, while almost all
excess consumer wait time is in A. It also found a roughly 1,064-cycle K64
cadence. Consequently the experiment spaces work that has measured slack
instead of delaying a complete producer or consumer loop.

## Variants

| variant | only changed mechanism |
|---|---|
| `baseline` | exact hash-gated canonical E7a |
| `cta4` | one-time CTA startup offset `(blockIdx.x % 4) * 256` cycles |
| `cta8` | one-time CTA startup offset `(blockIdx.x % 8) * 128` cycles |
| `b1_gap0` | one `nanosleep.u32 0` at the B1-gap site; codegen control |
| `b1_gap32` | one `nanosleep.u32 32` after each A TMA issue and before late B1 |
| `b1_gap64` | one `nanosleep.u32 64` after each A TMA issue and before late B1 |
| `b1_cross` | W3 performs its existing B1 wait before its first MMA half; W2 keeps the original order |

The CTA variants spread 148 persistent CTAs across approximately one K64
cadence and reuse the existing post-TMEM-allocation CTA barrier. They add no
steady-state wait. The B1-gap variants act after stage reuse and therefore
cannot be erased by the next shared reuse wait. They use one scheduler-
suspending instruction rather than a clock-polling loop, avoiding both active
polling power and a major K-loop code-generation change. `b1_cross` adds no
wait and preserves K accumulation order; it only moves W3's existing B1 wait
so W2 and W3 occupy opposite halves of the wait/MMA sequence.

## Protocol

- GPU: one NVIDIA B200, one persistent CTA per SM (`148` CTAs)
- GEMM: `M=N=K=16384`, BF16 A/B, FP32 accumulate/output
- inputs: uniform `[0,1)` and uniform `[-8,8)`
- complete dense A/B addresses and complete SW128/TMA C store
- one case per process
- one warmup and five timed launches
- three rotated passes per input and variant
- pattern and ones size-512 full-C validation before timing
- decision uses same-session paired ratios, not historical absolute TFLOP/s

Adoption requires at least `+0.5%` on both input distributions. A smaller
consistent gain is treated as a diagnostic and must be extended before it can
change the canonical source.

## Reproduction

```bash
./run_b200_gemm_e7a_phase_redesign.sh \
  /workspace/benchmark \
  /workspace/gemm_e7a_phase_redesign_b200
```

The generator refuses to patch a canonical source whose SHA-256 differs from
`37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a`.

## B200 result

The experiment ran on instance `45481495` from definition commit
`99246ad06ac7ded30b3363f65fd407dbf3e2b8cb`.  The primary result contains
three position-rotated W1/I5 processes per variant and input.  Values below
are event TFLOP/s; percentages are the mean of pass-matched changes from the
exact canonical baseline.

| variant | `[0,1)` | paired change | `[-8,8)` | paired change |
|---|---:|---:|---:|---:|
| `baseline` | 1819.500 +/- 0.641 | -- | 1613.465 +/- 1.236 | -- |
| `cta4` | 1819.517 +/- 3.568 | +0.001% | 1614.732 +/- 1.310 | +0.079% |
| `cta8` | 1817.449 +/- 3.290 | -0.113% | 1612.529 +/- 0.880 | -0.058% |
| `b1_gap0` | 1805.425 +/- 21.350 | -0.774% | 1612.341 +/- 1.017 | -0.070% |
| `b1_gap32` | 1817.785 +/- 0.605 | -0.094% | 1610.263 +/- 9.186 | -0.199% |
| `b1_gap64` | 1817.348 +/- 3.711 | -0.118% | 1609.389 +/- 6.527 | -0.253% |
| `b1_cross` | 1816.851 +/- 0.858 | -0.146% | 1613.433 +/- 0.350 | -0.002% |

`b1_gap0` produced one isolated `[0,1)` low sample, `1780.775`, while its
other primary samples were `1818.120` and `1817.379`.  The adjacent telemetry
remained at 35 C and 1965 MHz.  The sample was retained, and no primary file
was overwritten.  Three additional rotated processes were then collected
for `baseline`, `b1_gap0`, `b1_gap32`, and `b1_gap64` under both distributions.

Across all six processes, the B1 candidates remained slower than the exact
baseline:

| variant | `[0,1)` | paired vs baseline | `[-8,8)` | paired vs baseline |
|---|---:|---:|---:|---:|
| `baseline` | 1819.161 +/- 0.997 | -- | 1614.696 +/- 2.404 | -- |
| `b1_gap0` | 1810.922 +/- 14.853 | -0.453% | 1611.485 +/- 4.679 | -0.199% |
| `b1_gap32` | 1815.853 +/- 2.593 | -0.182% | 1611.968 +/- 6.256 | -0.169% |
| `b1_gap64` | 1816.715 +/- 2.626 | -0.134% | 1611.333 +/- 5.013 | -0.208% |

The apparent six-process `b1_gap32/64` changes relative to `b1_gap0` were
`+0.278%/+0.325%` for `[0,1)` but only `+0.031%/-0.009%` for `[-8,8)`.
They are driven by the retained `b1_gap0` outlier and fail the requirement for
a gain on both distributions.  The six-process medians lead to the same
decision: all three B1 variants are 0.07--0.14% below the baseline median.

All seven variants passed both size-512 pattern and ones full-C validation
with zero error.  CTA staggering preserved the baseline's 172 registers.
The B1 gap control and delayed variants used 180 registers, and `b1_cross`
used 174; all had zero stack, local memory, and spills.

No phase variant meets the `+0.5%` two-distribution adoption gate.  Keep the
canonical E7a unchanged at phase `0/0`.  The result closes simple one-time
cross-CTA startup staggering, per-stage B1 `nanosleep` spacing, and relocation
of W3's existing B1 wait for this topology.

Exact primary and extension CSVs, generated sources, validation output,
resource/SASS dumps, telemetry, and hashes are retained in
[`../results/gemm_e7a_phase_redesign_b200_45481495_20260724_99246ad0/`](../results/gemm_e7a_phase_redesign_b200_45481495_20260724_99246ad0/).
