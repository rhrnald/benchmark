# 16K multicast, pipeline-depth, and epilogue ablation

## Conditions

- Date: 2026-07-22
- GPU: NVIDIA B200, Vast.ai instance `45481495`, 1000 W power limit
- Definition commits: `8b85539`, `72a8360`, `635a5ff`
- GEMM: BF16 random `[0,1)`, M=N=K=16384, complete FP32 C output
- Scheduler: 148 persistent CTAs, static 16x16 macro schedule
- Multicast cases: 74 two-CTA clusters
- Measurement: one process per case, warmup 1, five timed launches, three
  independent processes in rotated order
- Phase shift: effective TMA/MMA `0/0` cycles

All nine variants passed the 512 pattern validation bit-exactly
(`max_abs=0`, `max_rel=0`, `bad=0`). Event TFLOP/s below is the mean and
sample standard deviation across three processes.

## 256x256 multicast and mainloop results

| multicast operand / B transaction | K stage | stages | dynamic SMEM | TFLOP/s | versus B split K64/S3 |
|---|---:|---:|---:|---:|---:|
| B, two `64x128` loads | 64 | 3 | 197632 B | **1783.800 +/- 1.301** | baseline |
| A, one `256x64` load | 64 | 3 | 197632 B | 1730.416 +/- 0.424 | -2.993% |
| B, one `64x256` load | 64 | 3 | 197632 B | 1740.478 +/- 0.338 | -2.429% |
| B, two `32x128` loads | 32 | 4 | 132096 B | 1593.331 +/- 0.583 | -10.678% |
| B, two `32x128` loads | 32 | 5 | 164864 B | 1664.468 +/- 1.057 | -6.690% |
| A, one `256x32` load | 32 | 4 | 132096 B | 1544.366 +/- 0.329 | -10.752% vs A K64/S3 |
| A, one `256x32` load | 32 | 5 | 164864 B | 1646.775 +/- 0.623 | -4.834% vs A K64/S3 |

The original split-B K64/S3 path remains the selected 256x256 kernel. A
multicast uses one transaction and one shared reuse barrier, but changes the
cluster raster from adjacent M tiles to adjacent N tiles. The measured 2.99%
loss shows that the simpler control path does not offset the locality and/or
cluster-direction cost at 16K.

Combining the two B halves removes one producer warp's TMA stream and halves
the remote ready/reuse barrier families, but is 2.43% slower. The two
independent `64x128` producer streams therefore provide useful issue-level
parallelism that outweighs their extra barrier instructions.

K32 doubles the number of TMA, barrier, and MMA-loop stage epochs. Five stages
recover 4.5--6.6 percentage points over four stages, showing that the deeper
ring hides some latency, but neither orientation approaches K64/S3. K32 uses
a correct SW64 A layout and SW128 B layout; the performance loss is not a
layout or numerical-correctness artifact.

## Dense 128x256 epilogue overlap

| epilogue | TFLOP/s | change |
|---|---:|---:|
| E0: serialized SMEM staging + TMA store | **1379.437 +/- 4.295** | baseline |
| E2: TMEM ping-pong + dedicated four-warp direct store | 1342.131 +/- 1.264 | -2.704% |

This is the dense-address counterpart of the earlier repeated-address
epilogue experiment. E2 overlaps tile `i` direct stores with tile `i+1`
compute, but its extra warpgroup and direct-global-store path remain slower
than the optimized TMA-store epilogue. It is not selected.

## Conclusion

None of the four requested directions improves the current 16K baseline.
The robust choice remains B multicast with two 128-wide producer streams,
K64, three stages, static 16x16 scheduling, and phase shift 0/0. The next
high-leverage work should measure and reduce cluster/DSM synchronization cost
without changing the successful B producer geometry.

GPU temperature moved from 33 C to 36 C. Raw CSVs, validation logs, source
snapshot, binary hashes, and pre/post `nvidia-smi` snapshots are stored in
this directory.
