# Producer-only suspended `mbarrier` wait

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `dd5b2e2` (source-identical to `ef7ebca`)
- Candidate definition: `5480fb1`
- Protocol: one case per process, warmup 1, timed launches 5, six
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `9b4099e5e1cf9df1395b215c895c1be63874c341c4c818e9cfd28ca81ec34cf1`
- Clean / candidate binary SHA-256:
  `84b3e638d797ca731c59249d7526fada590a64a1a36edb53f35b8f1f02c24f41` /
  `f06a8b3647a2a9f4f1a7072f99720491d141fe04cd0b88995e24ebf6e9e512a4`

The broad E8d candidate suspended every barrier wait.  This candidate keeps
consumer A/B readiness and final-drain waits as the original tight polling
loop, and applies CUTLASS's `0x989680` suspend hint only to the two producers'
stage-reuse waits on `mma_done`.  TMA/MMA work, bytes and addresses, barrier
topology, scheduler, and epilogue are otherwise unchanged.

Pattern and ones full-C validations at size 512 both passed exactly.

## Code generation

| metric | clean dual-wide | suspend producer waits |
|---|---:|---:|
| registers / spills | 172 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| all SASS instruction lines | 4192 | 4368 |
| main-kernel instruction lines | 1952 | 2040 |
| static `SYNCS.PHASECHK` | 64 | 92 |
| static `NANOSLEEP.SYNCS` | 0 | 28 |
| MMA / A TMA / B TMA / C TMA store | 4 / 7 / 14 / 4 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.  The first three pairs triggered the
predefined 0.5--1.0% extension rule on signed input, so three additional
order-balanced pairs were collected without rebuilding either binary.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1774.260, 1770.035, 1770.418, 1770.092, 1769.764, 1769.798 | **1770.728** | 1778.307, 1776.321, 1778.590, 1776.367, 1776.498, 1776.248 | **1777.055** | **+0.3573%** | **+0.3574%** |
| BF16 uniform `[-8,8)` | 1530.017, 1527.115, 1525.684, 1527.923, 1528.638, 1530.523 | **1528.317** | 1538.285, 1531.816, 1538.693, 1532.403, 1536.750, 1536.668 | **1535.769** | **+0.4876%** | **+0.4877%** |

All twelve paired deltas are positive.  They are +0.2281%, +0.3551%,
+0.4616%, +0.3545%, +0.3805%, and +0.3644% for `[0,1)`, and +0.5404%,
+0.3078%, +0.8527%, +0.2932%, +0.5307%, and +0.4015% for `[-8,8)`.
Temperature stayed at 37--39 C.  The first extension candidate's pre-run
snapshot was still at the idle 120 MHz clock, but its mandatory warmup raised
the timed run and its 1776.367 TFLOP/s agrees with the other candidate samples;
all other pre-run snapshots were at 1965 MHz.

Restricting sleep to producer waits is consistently better than tight polling
and slightly better than broad E8d, but the final means remain below the
pre-registered 0.5% materiality gate in both distributions.  The candidate is
therefore retained as a positive diagnostic and not adopted into the working
default.  The next experiment targets the larger A readiness latency directly
by splitting the 32 KiB A transaction into two concurrent 16 KiB transactions.
