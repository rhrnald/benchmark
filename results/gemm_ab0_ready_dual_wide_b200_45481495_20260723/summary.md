# Combined dual-wide A and B0 readiness

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `883f85b` (source-identical to `ef7ebca`)
- Candidate definition: `4e13fa4`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `4d53e0ca210cb8074287869436882337222591b79495ecf6db42f07c72e6c0ef`
- Clean / candidate binary SHA-256:
  `e880794137b42ec859e8141d25d9480ae26fbfd3bc1f8d7a56f252ebebc1a19e` /
  `15426117e08db1435e5dac2fdc5c7218f3e4228605a5135ef9e5a9f453fad8c7`

Warp 0's A transaction and warp 1's early B0 transaction now arrive on one
per-stage barrier initialized for two arrivals.  Both wide-MMA consumers wait
once for that common dependency before their first two K16 MMAs.  Late B1
keeps its independent barrier between the first and second MMA pairs.  TMA
issue order, bytes, addresses, MMA work, scheduler, and epilogue are unchanged.

Three independent pattern and three independent ones full-C validations at
size 512 all passed exactly.

## Code generation

| metric | clean dual-wide | combined A+B0 readiness |
|---|---:|---:|
| registers / spills | 172 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | 0 B / 1152 B |
| all SASS instruction lines | 4192 | 4112 |
| main-kernel instruction lines | 1952 | 1912 |
| static `SYNCS.PHASECHK` / `SYNCS.ARRIVE` | 64 / 21 | 62 / 21 |
| MMA / A TMA / B TMA / C TMA store | 4 / 7 / 14 / 4 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1767.979, 1771.461, 1769.882 | **1769.774** | 1774.120, 1774.264, 1773.022 | **1773.802** | **+0.2276%** | **+0.2277%** |
| BF16 uniform `[-8,8)` | 1524.133, 1530.862, 1529.494 | **1528.163** | 1528.620, 1533.019, 1525.696 | **1529.112** | **+0.0621%** | **+0.0623%** |

All three `[0,1)` pairs are positive, but the mean gain is below the predefined
0.5% noise gate.  Signed-input pairs have mixed direction and a near-zero
mean.  The candidate is correct but not adopted.  The large wait aggregates
in the phase trace therefore mostly represent real readiness latency and
fixed synchronization-path cost that one fewer wait does not remove.
