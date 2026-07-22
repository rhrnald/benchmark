# CUTLASS-style suspended `mbarrier` wait

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide parent: `119b36a` (source-identical to `ef7ebca`)
- Candidate definition: `8a24602`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Clean / candidate source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` /
  `66dc91ff0badb7a07dbaea3d803ecc3dac65ae44e25fdc278abc76202d4084c4`
- Clean / candidate binary SHA-256:
  `dabe6069010876dab9aee974da9377a9868aef8e411fdd3fb0aceeeb9da9b6e0` /
  `010e37511d8a2b8dfd239b71b45b2b62041f74e49b1aa51bf87d94b35269970a`

The clean kernel repeatedly executes a hint-free
`mbarrier.try_wait.parity` while a barrier is incomplete.  This candidate
matches CUTLASS's barrier wait by adding the third-operand suspend hint
`0x989680` to every producer-reuse, consumer-readiness, and final-drain wait.
TMA/MMA work, bytes and addresses, barriers, scheduler, and epilogue are
otherwise unchanged.

Pattern and ones full-C validations at size 512 both passed exactly.

## Code generation

| metric | clean dual-wide | suspend all waits |
|---|---:|---:|
| registers / spills | 172 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| all SASS instruction lines | 4192 | 4384 |
| main-kernel instruction lines | 1952 | 2048 |
| static `SYNCS.PHASECHK` | 64 | 96 |
| static `NANOSLEEP.SYNCS` | 0 | 32 |
| MMA / A TMA / B TMA / C TMA store | 4 / 7 / 14 / 4 | unchanged |

The extra `SYNCS.PHASECHK` and `NANOSLEEP.SYNCS 0x989680` are the expected
lowering of the suspend-time operand.  Register, spill, shared-memory, and
dynamic arithmetic properties remain unchanged.

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass clean dual-wide parent.

| input | clean samples | clean mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1769.978, 1768.295, 1769.962 | **1769.412** | 1774.883, 1775.245, 1773.232 | **1774.453** | **+0.2849%** | **+0.2850%** |
| BF16 uniform `[-8,8)` | 1532.115, 1532.485, 1529.303 | **1531.301** | 1534.581, 1535.687, 1534.698 | **1534.989** | **+0.2408%** | **+0.2409%** |

All six pairs are positive.  The per-pair deltas are +0.2771%, +0.3930%,
+0.1847% for `[0,1)` and +0.1610%, +0.2089%, +0.3528% for `[-8,8)`.
Temperature stayed at 38--39 C and every measured process ran at 1965 MHz.

The direction is consistent but both gains are below the predefined 0.5%
noise gate, so the broad candidate is not adopted as the working default.
Because short consumer waits may pay wake-up latency while long producer
reuse waits benefit from sleeping, the next isolated ablation applies the
same hint only to producer `mma_done` waits.
