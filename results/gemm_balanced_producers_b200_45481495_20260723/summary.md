# Balanced A/B producer ownership

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `794d29f` (source-identical to `ef7ebca`)
- Candidate definition: `0da6a82`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `0cfead9f51afff96b9af0174309a827bcaacfe6f75987167f6bf79bb79efa56b`
- Clean / candidate binary SHA-256:
  `cadf16e13e34b92de2f6535568744f7a24a826f8db21ac27413a99588a3e5d58` /
  `549c416fb9fd7410ff623c9093cbb82203ccb48675890f0a77ea0655e67f246a`

The clean producer ownership is warp 0 issuing A then late B1 (48 KiB total)
and warp 1 issuing early B0 (16 KiB).  This candidate assigns warp 0 only A
and warp 1 B0 then B1, giving 32 KiB to each producer and starting B1 without
waiting behind A.  Transaction count, bytes and addresses, barrier topology,
consumer waits and MMA issue, scheduler, and epilogue are unchanged.

Pattern and ones full-C validations at size 512 both passed exactly.

## Code generation

| metric | clean dual-wide | balanced producers |
|---|---:|---:|
| registers / spills | 172 / 0 | 176 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| all SASS instruction lines | 4192 | 4208 |
| main-kernel instruction lines | 1952 | 1960 |
| static A / B TMA sites | 7 / 14 | unchanged |
| static `SYNCS.PHASECHK` / `SYNCS.ARRIVE` | 64 / 21 | unchanged |
| MMA / C TMA store | 4 / 4 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1770.060, 1769.855, 1769.308 | **1769.741** | 1767.194, 1768.670, 1767.631 | **1767.832** | **-0.1079%** | **-0.1079%** |
| BF16 uniform `[-8,8)` | 1530.903, 1531.261, 1530.584 | **1530.916** | 1528.298, 1524.491, 1525.090 | **1525.960** | **-0.3237%** | **-0.3237%** |

All six pairs regress.  Starting B1 earlier does not improve the critical
path; serializing both B transactions onto one warp and increasing register
use from 172 to 176 is slightly worse.  The candidate is not adopted, and the
clean kernel's two independently progressing B producer streams are retained.
