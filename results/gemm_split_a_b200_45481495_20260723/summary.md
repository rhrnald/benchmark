# Split-A dual-wide GEMM pipeline

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `b8080d2` (source-identical to `ef7ebca`)
- Candidate definition: `8fd2d98`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `ca0cb98bb38cde1fa2e9182bfdc5b4a00e4afc14447c11899f533615106fd23e`
- Clean / candidate binary SHA-256:
  `1271f849260da5d32668a23bed1773fa5601872fbf2bd446288973e2293797c0` /
  `ab664eb057acaa83354fd6f4f3758efef3b894b6c0e9cb627c351834795cdebd`

The candidate splits A's one M256 x K64 32 KiB TMA transaction into two
M128 x K64 16 KiB transactions with independent readiness barriers.  Warp 0
issues A0 then B1; warp 1 issues early B0 then A1.  Consumer warp 2 waits for
A0+B0 and warp 3 waits for A1+B0.  Shared payload layout, total A/B bytes,
B transactions, MMA work and destinations, scheduler, and epilogue are
unchanged.  Dynamic TMA transaction count increases from three to four per
K64 stage.

Pattern and ones full-C validations at size 512 both passed exactly.

## Code generation

| metric | clean dual-wide | split A |
|---|---:|---:|
| registers / spills | 172 / 0 | 174 / 0 |
| stack / static shared | 0 B / 1184 B | 0 B / 1200 B |
| all SASS instruction lines | 4192 | 4512 |
| main-kernel instruction lines | 1952 | 2112 |
| static A / B TMA sites | 7 / 14 | 14 / 14 |
| static `SYNCS.PHASECHK` / `SYNCS.ARRIVE` | 64 / 21 | 64 / 28 |
| MMA / C TMA store | 4 / 4 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1769.615, 1768.623, 1770.208 | **1769.482** | 1767.558, 1767.179, 1766.125 | **1766.954** | **-0.1429%** | **-0.1428%** |
| BF16 uniform `[-8,8)` | 1527.899, 1523.465, 1528.756 | **1526.707** | 1524.028, 1526.587, 1527.332 | **1525.982** | **-0.0474%** | **-0.0472%** |

All three `[0,1)` pairs regress; signed-input pairs are mixed.  The candidate
is correct but not adopted.  Reducing each A transaction's completion granule
does not repay the extra TMA command, barrier arrival, two registers, and
consumer skew.  Because the result is below the 0.5% gate and non-positive,
the alternative split-A issue order is not measured.
