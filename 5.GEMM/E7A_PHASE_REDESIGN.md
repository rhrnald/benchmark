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
