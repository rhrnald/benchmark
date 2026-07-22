# Shared dual-wide MMA completion barrier

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `6d08a6e` (source-identical to `ef7ebca`)
- Candidate definition: `dadc40c`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `ebecf145187656827047b6629907da7f563a9e96dab6c00d5757310b6c4e329e`
- Clean / candidate binary SHA-256:
  `47f9c474989cfb8c951a4acf9636602dc779f5e8185138fd2b8303524fdecf4d` /
  `9458b4da4fc3cafe25ea44285c7c5308a39e780101a35509fb1c784fed7c7ecc`

The candidate replaces two completion barriers per stage with one barrier
initialized for two arrivals.  Both M-split consumer warps commit to the same
stage barrier.  Each producer therefore waits once, rather than once for each
consumer, before reusing that shared-memory stage.  MMA/TMA work, addresses,
scheduler, and epilogue are unchanged.

Three independent pattern and three independent ones full-C validations at
size 512 all passed exactly.

## Code generation

| metric | clean dual-wide | shared completion barrier |
|---|---:|---:|
| registers / spills | 172 / 0 | 170 / 0 |
| stack / static shared | 0 B / 1184 B | 0 B / 1152 B |
| all SASS instruction lines | 4192 | 3968 |
| main-kernel instruction lines | 1952 | 1840 |
| static `SYNCS.PHASECHK` | 64 | 36 |
| MMA / A TMA / B TMA / C TMA store | 4 / 7 / 14 / 4 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1770.625, 1771.646, 1770.329 | **1770.867** | 1770.089, 1769.151, 1769.160 | **1769.467** | **-0.0791%** | **-0.0790%** |
| BF16 uniform `[-8,8)` | 1530.911, 1531.666, 1526.626 | **1529.734** | 1531.393, 1528.518, 1531.259 | **1530.390** | **+0.0429%** | **+0.0431%** |

Both deltas are well inside the 0.5% noise gate.  All three `[0,1)` pairs are
slightly negative, while signed-input pairs have mixed direction.  The large
static instruction reduction does not improve the critical path.  This agrees
with the phase trace: producers already finish about 2.9K cycles before the
consumers, so removing their duplicate stage-reuse waits is not useful.
Reject the candidate and keep per-consumer completion barriers.
