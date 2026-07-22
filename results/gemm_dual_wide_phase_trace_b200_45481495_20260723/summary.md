# Dual-wide GEMM phase trace

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Clean dual-wide default: `735c6e0` (same kernel as adoption `ef7ebca`)
- Trace definition: `87b908b`
- Target: block 0, ninth valid persistent output tile (`tile_iter=8`)
- Protocol: one untraced warmup followed by one traced launch in each process,
  five independent processes per input distribution
- Trace source / binary SHA-256:
  `d742bc0331e5faa3b6bbebe4686cdd7222ab97c9a6a416322f69da54c5f4645d` /
  `fcfab5804e7cd044445168b4fab135de620da5c5aa195a1cc8b4677277ab0253`

The ten samples all ran block 0 on SM 142.  The dynamic scheduler assigned
different real output tiles, spanning linear tile IDs 1190--1241, so the trace
does not depend on one repeated global-memory tile.  Pattern and ones full-C
validations at size 512 passed exactly before collection.

## Diagnostic code generation

| metric | clean dual-wide u1 | phase trace |
|---|---:|---:|
| registers / spills | 172 / 0 | 186 / 0 |
| stack | 0 B | 0 B |
| static shared | 1184 B | 3280 B |
| dynamic shared request | 197632 B | 197632 B |
| static MMA / A TMA / B TMA / C TMA store | 4 / 7 / 14 / 4 | unchanged |

Absolute timing belongs only to the instrumented kernel.  The trace retains
the GEMM work and is used to compare intervals inside the same sampled CTA,
not to report TFLOP/s.

## Median phase timing

All values are device `clock64` cycles.  Percentages use the median complete
tile envelope.  Aggregate consumer rows use the slower of warp 2 and warp 3
in each run, followed by the median of five runs.

| interval | `[0,1)` cycles | tile % | `[-8,8)` cycles | tile % |
|---|---:|---:|---:|---:|
| complete tile envelope | **279398** | 100% | **277012** | 100% |
| scheduler atomic/barrier/decode envelope | 839 | 0.300% | 804 | 0.290% |
| mainloop start through CTA join | 267983 | 95.914% | 265604 | 95.882% |
| consumer A-wait aggregate | 40590 | 14.528% | 37260 | 13.451% |
| consumer B0-wait aggregate | 25755 | 9.218% | 25715 | 9.283% |
| consumer B1-wait aggregate | 21593 | 7.728% | 21583 | 7.791% |
| first two wide-MMA issue aggregate | 45056 | 16.126% | 45056 | 16.265% |
| last two wide-MMA issue aggregate | 40704 | 14.569% | 40704 | 14.694% |
| MMA commit aggregate | 17152 | 6.139% | 17152 | 6.192% |
| final MMA drain | 713 | 0.255% | 713 | 0.257% |
| complete C epilogue | **10077** | **3.607%** | **10078** | **3.638%** |

Timestamp overhead is 2 cycles.  Aggregate wait rows cover 256 K64 stages;
their totals include both genuine readiness stalls and the cost of executing
the synchronization path.  They must not be read as independently removable
percentages or summed into a projected speedup.

## Cross-warp observations and decisions

- The two producer loops finish only 34/35 cycles apart, but both finish about
  2927/2880 cycles before the last consumer.  Producer load imbalance and
  scheduler prefetch remain low priority.
- The two consumers finish 106 cycles apart.  Final MMA drain is only 713
  cycles; the main opportunity is repeated per-stage readiness/issue overhead,
  not end-of-mainloop completion.
- Scheduler work is about 0.3% of a tile.  Moving the atomic earlier cannot
  materially close the current gap by itself.
- The 3.61--3.64% traced epilogue agrees with the separate same-binary
  C-store-off ceiling of 3.19--3.30%.  Epilogue overlap remains bounded and
  should follow mainloop synchronization work.

Next measure, as separate ablations: (1) one stage completion barrier with
arrival count 2 for the two consumer commits; (2) one A+B0 readiness barrier
with arrival count 2, since both wide consumers need both transactions; and
(3) producer-loop-only u1 codegen.  Combine only changes that independently
improve both input distributions.
