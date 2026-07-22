# Dual-wide K-loop auto-unroll ablation

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Dual-wide u1 parent: `4e1ac27`
- Auto-unroll definition: `521b523`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- U1 / auto source SHA-256:
  `fdd34cecdd75890e7281551d8e8af9a29f99fdc7848f5f4e8027d44c5225281c` /
  `b134bdc7c70e0b1503f8e86303713f4bff75b309bc77ac3783c612c5b758ea65`
- U1 / auto binary SHA-256:
  `0c34e046c8b788448742560df8475182d2688f578577e38d68d885971271899c` /
  `a31b555992ce9cae8583f62c9975b8b02090057d1fb80c240c4faeacdeaf7d27`

The candidate removes only `#pragma unroll 1` from the consumer K-tile loop.
Logical work, dynamic MMA count, TMA traffic, shared-memory layout, scheduler,
and epilogue are identical.  The compiler duplicates the loop body in SASS,
increasing static MMA and commit sites, while register use remains unchanged.

Candidate pattern and ones full-C validations at size 512 passed exactly.

## Code generation

| metric | dual-wide u1 | compiler auto-unroll |
|---|---:|---:|
| registers / spills | 172 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| static `UTCHMMA` | 4 | 12 |
| static `UTCBAR` | 1 | 3 |
| A 2D / B 4D TMA loads | 7 / 14 | unchanged |
| C TMA stores / `BAR.SYNC` | 4 / 20 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is auto-unroll relative to the same-pass
u1 parent.

| input | u1 samples | u1 mean | auto samples | auto mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1772.729, 1774.044, 1773.097 | **1773.290** | 1772.505, 1773.324, 1772.592 | **1772.807** | **-0.0272%** | **-0.0272%** |
| BF16 uniform `[-8,8)` | 1528.899, 1527.794, 1534.407 | **1530.367** | 1527.673, 1525.283, 1528.564 | **1527.173** | **-0.2087%** | **-0.2084%** |

All six same-pass comparisons favor u1.  The random result is effectively
neutral, but signed input has a consistent regression.  Reject compiler
auto-unroll and retain the explicit u1 loop as the clean working default.
