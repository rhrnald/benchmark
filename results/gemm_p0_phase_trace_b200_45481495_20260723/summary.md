# P0 coarse GEMM phase trace

## Configuration

- Definition commit: `6a4e893` (`Define coarse GEMM phase trace`)
- Performance parent: E2a at `93933c0`
- GPU: NVIDIA B200, Vast.ai instance `45481495`
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Power limit / observed SM clock: 1000 W / 1965 MHz
- Problem: dense BF16 `16384 x 16384 x 16384`, FP32 C
- Kernel: 148 persistent CTAs, CTA `256 x 256`, K64/S3
- Inputs: BF16 uniform `[0,1)` and `[-8,8)`
- Sampling: one untraced full launch, then block 0 at valid `tile_iter=8`
- Repetitions: five independent processes per input distribution
- Trace source SHA-256:
  `1554265a8e5c5847387895e57adf5fdd0376aa5dfc370ac45aa908aae2b3077f`
- Trace binary SHA-256:
  `f41622eb8e3ecc49b16b7869b75a710f1a720bdb5e7df01f5b0c0001b6149c9e`

The trace path adds timestamp branches and scratch state, so its performance is
not a benchmark result.  Relative timestamps are only compared inside one CTA
and one SM.  The performance parent uses 174 registers; this diagnostic binary
uses 184 registers, zero stack, zero spills, and 1648 B static shared memory.

Both 512 pattern and 512 ones full-C validations passed exactly before trace
collection.

## Results

All values below are device `clock64` cycles.  `range` is the minimum and
maximum of the five independent traces.

| interval | `[0,1)` median | `[0,1)` range | `[-8,8)` median | `[-8,8)` range |
|---|---:|---:|---:|---:|
| observed tile, atomic start to final barrier end | 278843 | 278010--279806 | 276861 | 275486--278243 |
| scheduler atomic | 294 | 282--411 | 279 | 269--284 |
| scheduler decode | 397 | 397--397 | 397 | 397--397 |
| producer warp 0 mainloop | 264611 | 264148--265626 | 262747 | 261617--264144 |
| producer warp 1 mainloop | 264634 | 263238--265619 | 262497 | 260717--264179 |
| consumer warp 2 mainloop + final drain | 267286 | 266632--268240 | 265321 | 264111--266628 |
| consumer warp 3 mainloop + final drain | 267455 | 266098--268409 | 265335 | 263577--266861 |
| mainloop envelope | 267573 | 266748--268527 | 265608 | 264227--266979 |
| last-consumer tail after last producer | 2760 | 2594--2902 | 2740 | 2594--2861 |
| producer completion skew, warp 0 minus warp 1 | 89 | -60--879 | 13 | -72--863 |
| mainloop end to C-epilogue start | 244 | 244--244 | 244 | 244--244 |
| complete C epilogue envelope | 10035 | 10034--10035 | 10035 | 10034--10035 |
| back-to-back clock-read overhead | 2 | 2--2 | 2 | 2--2 |

The C epilogue occupies a median 3.598% of the observed tile for `[0,1)` and
3.624% for `[-8,8)`.  This agrees with the independent same-binary C-store-off
upper bound of +3.813% and +3.256%, respectively: the epilogue is a real,
mostly serial secondary bottleneck.

The nominal producer byte split is asymmetric (`48/16 KiB` per K64 stage), but
the two producer completion times are almost coincident.  Their median signed
skew is only 89 and 13 cycles, while both finish roughly 2.7K cycles before the
last consumer.  The producer loops are paced by buffer reuse rather than by the
raw number of TMA issue instructions.  Reassigning the work to `32/32 KiB` is
therefore lower priority, especially because its static audit raises registers
from 174 to 178 and adds waits/control instructions.

The atomic and decode intervals are each below 0.15% of the observed tile.
Task-ID prefetch cannot provide a large standalone gain in this configuration.

Individual barrier instruction durations are not interpreted as stall time:
Blackwell may lower synchronization to deferred blocking, and warp arrival
timestamps are more informative.  The interval envelopes above remain valid.

## Decision

Test the E4 C-store overlap candidates next, in this order:

1. two buffers, one commit per 128x128 chunk, `wait_group.read 1` before reuse;
2. three buffers, one commit per chunk, `wait_group.read 2` before reuse;
3. final `wait_group.read 0` before the mainloop reuses the shared payload.

The audited candidates both reduce code generation from 174 to 172 registers
without spills.  E5 producer balancing and E6 task prefetch remain deferred
unless E4 changes the phase balance.
