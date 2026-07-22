# Producer-loop `unroll 1` ablation

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `0885dca` (source-identical to `ef7ebca`)
- Candidate definition: `cbb9aed`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `8b039d3ed7800d7c054b9523d938060eff4c8e4fdd1f2e876bb2089b1ce782a3`
- Clean / candidate binary SHA-256:
  `8f56509008f43538cf7fa235d80b6ea7c571de25d1790dffb5977f21ab182bb7` /
  `b59f35ba2880bc202fe23c40d6a6efeec4b6b2b6cfdfdb94256e45154d906aab`

The candidate adds `#pragma unroll 1` only to the warp-0 and warp-1 producer
K loops.  The already-adopted consumer-loop `unroll 1`, TMA bytes and order,
barriers, MMA issue sequence, scheduler, and epilogue are unchanged.

Pattern and ones full-C validations at size 512 both passed exactly.

## Code generation

| metric | clean dual-wide | producer-loop u1 |
|---|---:|---:|
| registers / spills | 172 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | 0 B / 1184 B |
| all SASS instruction lines | 4192 | 2576 |
| main-kernel instruction lines | 1952 | 1144 |
| static `SYNCS.PHASECHK` / `SYNCS.ARRIVE` | 64 / 21 | 16 / 3 |
| static MMA / A TMA / B TMA / C TMA store | 4 / 7 / 14 / 4 | 4 / 1 / 2 / 4 |

The reduced static TMA and barrier counts are loop-site counts, not reduced
dynamic work.  Dynamic TMA transactions, barrier operations, and MMA work per
tile remain identical.

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1770.505, 1769.627, 1770.525 | **1770.219** | 1761.329, 1762.797, 1760.471 | **1761.532** | **-0.4907%** | **-0.4907%** |
| BF16 uniform `[-8,8)` | 1528.681, 1525.444, 1530.790 | **1528.305** | 1524.405, 1527.965, 1527.356 | **1526.575** | **-0.1132%** | **-0.1129%** |

The three `[0,1)` pair deltas are -0.5183%, -0.3860%, and -0.5679%.
The signed-input pair deltas are -0.2797%, +0.1653%, and -0.2243%.
The candidate is not adopted: it does not clear the 0.5% gate, both means are
lower, and the phase trace already shows both producers completing about 2.9K
cycles before the consumers.  Producer code size is therefore not the current
critical path.
