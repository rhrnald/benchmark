# Incremental stage-ring ablation

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Baseline definition/result: `8add902` / `ae74998`
- Candidate definition: `6d8fef3`
- Baseline binary SHA-256:
  `c8cf7f415a8933c516d90725f46eceb14e66fc0168f0fffe45566d048e904667`
- Candidate source SHA-256:
  `bd1d56dba7d067e7b5a61ebbaa7c815c31838a0f33497af7764fbfe1334bf24a`
- Candidate binary SHA-256:
  `d68328a1167ba94219128becdc5e3bf62a712508c5c42893caf2d64ba51bba6e`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

The candidate replaces every per-K-stage `% 3` and `/ 3` barrier-ring
calculation with persistent thread-local `(stage, phase)` cursors.  TMA, MMA,
commit, wait, scheduler, and epilogue work are unchanged.

## Correctness and codegen

Both 512 pattern and ones full-C validations passed with `max_abs=0` and
`max_rel=0`.

| metric | baseline | incremental ring |
|---|---:|---:|
| registers | 178 | 184 |
| stack / spills | 16 B / 0 | 16 B / 0 |
| static instructions | 1936 | 1976 |
| divide-by-3 reciprocal sequences in the GEMM paths | 30 | 0 |
| `SEL` | 2 | 48 |
| `R2UR.BROADCAST` | 65 | 104 |
| `BSSY` / `BSYNC` | 13 / 13 | 28 / 28 |

The core operation counts are identical: 7 A TMA loads, 14 B TMA loads,
8 static MMA instructions, 21 barrier arrives, 48 barrier waits, and 4 C TMA
stores.  The compiler removed the reciprocal divisions but materialized the
loop-carried cursor with more selects, broadcasts, reconvergence, and six more
registers.

## Performance

All values are event TFLOP/s.  Delta is E1 relative to the same-pass baseline.

| input | baseline samples | baseline mean | E1 samples | E1 mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1740.531, 1737.418, 1737.051 | **1738.333** | 1728.679, 1728.911, 1729.534 | **1729.041** | **-0.5345%** | -0.5344% |
| BF16 uniform `[-8,8)` | 1514.704, 1510.059, 1508.738 | **1511.167** | 1500.335, 1505.022, 1509.872 | **1505.076** | **-0.4030%** | -0.4023% |

Decision: reject and revert E1.  It does not improve either input distribution,
and the primary `[0,1)` regression exceeds the predefined 0.5% gate.  The next
arithmetic candidate, if revisited, should keep epoch state local and common the
producer quotient instead of carrying a ring cursor across the epilogue.
