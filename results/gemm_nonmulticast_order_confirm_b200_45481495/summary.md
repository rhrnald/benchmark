# Non-multicast scheduler order confirmation

## Outcome

The focused confirmation did **not** promote `order_nm`.  Its paired gain over
the current `order_mn` default was `+0.077% +/- 0.857%` (mean plus sample
standard deviation), with a two-sided paired 95% t interval of `-0.822%` to
`+0.976%`.  Five of six pairs were positive, but pair 4 was `-1.654%`.

This fails every predeclared promotion gate except the BA-first subgroup:

- all six pairs positive: fail (5/6)
- paired mean at least 0.5%: fail (`+0.077%`)
- both execution-order subgroup means positive: fail (AB `-0.230%`, BA `+0.385%`)
- paired 95% interval excludes zero: fail

The one negative sample is retained.  As a non-decisive sensitivity check,
the other five samples average only `+0.424%` and the six-pair median is
`+0.422%`, both still below the 0.5% gate.  The compile-time default therefore
remains local M-fast plus macro N-fast (`order_mn`).

## Provenance and fixed conditions

- Definition commit: `ac71b14`
- Kernel base commit: `f669b6f`
- CUDA source SHA-256:
  `ae8799c19d79db0bf9ce0f07043dc660fec210fcd94817fa974f48cf41bdabef`
- Vast instance: `45481495`
- GPU: NVIDIA B200, 148 SMs, 1000 W power limit
- Shape: BF16 `16384x16384x16384`, FP32 accumulation and complete FP32 C
- Input generation: deterministic pseudo-uniform FP32 `[0,1)` converted to BF16
- CTA: `256x256`; K64; three stages; split `64x128` B TMA loads
- Scheduler: 148 dynamic persistent CTAs, `16x16` macro, no multicast
- Effective phase: TMA/MMA `0/0`; no A/B/C TMA L2 promotion
- Timing: one case per process, warmup 1, five timed launches
- Sampling: six adjacent, counterbalanced AB/BA pairs (six processes per variant)

Here A is `order_mn` (local M-fast, macro N-fast) and B is `order_nm`
(local N-fast, macro M-fast).  Pair order was `AB, BA, BA, AB, AB, BA`, so
each variant occupied the first and second position three times and both
variants had the same aggregate sequence position.

## Results

| pair | process order | `order_mn` | `order_nm` | B - A | paired change |
|---:|---|---:|---:|---:|---:|
| 1 | AB | 1773.511 | 1782.280 | +8.769 | +0.494% |
| 2 | BA | 1776.562 | 1780.378 | +3.816 | +0.215% |
| 3 | BA | 1771.703 | 1778.346 | +6.643 | +0.375% |
| 4 | AB | 1773.644 | 1744.313 | -29.331 | -1.654% |
| 5 | AB | 1770.290 | 1778.597 | +8.307 | +0.469% |
| 6 | BA | 1773.211 | 1783.224 | +10.013 | +0.565% |

All performance values are event TFLOP/s.  Process-level summaries are:

| variant | mean TFLOP/s | sample SD | min | max |
|---|---:|---:|---:|---:|
| `order_mn` | 1773.154 | 2.112 | 1770.290 | 1776.562 |
| `order_nm` | 1774.523 | 14.927 | 1744.313 | 1783.224 |

The paired difference is `+1.370 +/- 15.191 TFLOP/s`; the percentage median
is `+0.422%`.  Each W1/I5 process is one statistical sample; the five timed
launches inside it are not treated as independent samples.

## Correctness and environment

- Both binaries passed the 512 pattern reference with `max_abs=0`,
  `max_rel=0`, and `bad=0`.
- The host mapping test verified exact coverage for all four local/macro order
  combinations and checked the 64x64 `order_mn` and `order_nm` boundaries.
- The result manifest passed `sha256sum -c` on the remote host before download.
- Temperature was 30 C before and 31 C after.  Pair-start telemetry stayed at
  30--31 C and reported the idle SM clock at 1965 MHz.  No thermal or hardware
  slowdown was reported.
- Per-kernel clocks and L2/DRAM counters are unavailable, so pair 4's transient
  regression cannot be attributed to cache, clock, or power from these data.

## Decision

Keep `order_mn` as the default and do not spend GPU time retuning macro shapes
under `order_nm`.  Snake and N-strip were already rejected in round one.  The
next planned work is a code-level feasibility audit for K-outer two-output A
stage reuse, which can reduce requested A transactions directly rather than
depending on an 8-MiB L2 reuse distance between completed output tiles.

## Reproduction

```bash
DEFINITION_COMMIT=ac71b14 \
  ./run_b200_gemm_nonmulticast_order_confirm.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_order_confirm
```

Raw CSVs, validation logs, mapping output, exact compile commands, execution
sequence, pair-start telemetry, source snapshots, and hashes are retained in
this directory.
