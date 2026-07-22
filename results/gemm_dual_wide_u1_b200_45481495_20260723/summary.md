# Dual-wide staggered M-split GEMM mainloop

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Macro-free E2a parent: `3d2d0a4` (working source restored at `6ce6926`)
- Candidate definition: `4e1ac27`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Baseline source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Candidate source SHA-256:
  `fdd34cecdd75890e7281551d8e8af9a29f99fdc7848f5f4e8027d44c5225281c`
- Baseline / candidate binary SHA-256:
  `5d4127162b83ec0584b760290d4f7542b71d7588838a012fa4201e74cd7f6023` /
  `0c34e046c8b788448742560df8475182d2688f578577e38d68d885971271899c`

The two MMA consumer warps now split the output tile in M rather than N:
warp 2 computes the upper `128x256` half and warp 3 computes the lower
`128x256` half using `tcgen05.mma` `m128n256k16`.  This halves dynamic MMA
issues per CTA and K64 stage from 16 to 8.

The total TMA traffic is unchanged.  A remains one 32 KiB transaction.  B is
still two 16 KiB transactions, but the split is K0:32 and K32:64 rather than
two N128 panels.  Warp 1 issues the early B0 transaction; warp 0 issues A and
B1.  Each consumer waits for A+B0, issues two K16 MMAs, waits for B1, issues
two more MMAs, then commits.  The outer K-tile loop uses `#pragma unroll 1`.

Candidate pattern and ones full-C validations at size 512 passed exactly.

## Code generation

| metric | macro-free E2a | dual-wide u1 |
|---|---:|---:|
| registers / spills | 174 / 0 | 172 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| static `UTCHMMA` | 8 | 4 |
| dynamic MMA issues / CTA / K64 | 16 | 8 |
| A 2D / B 4D TMA loads | 7 / 14 | 7 / 14 |
| C TMA stores / C vector stores | 4 / 128 | unchanged |
| `UTCBAR` / `BAR.SYNC` | 1 / 20 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass parent.

| input | parent samples | parent mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1749.841, 1750.552, 1749.997 | **1750.130** | 1773.166, 1772.978, 1774.425 | **1773.523** | **+1.3366%** | **+1.3366%** |
| BF16 uniform `[-8,8)` | 1508.877, 1514.230, 1513.590 | **1512.232** | 1529.860, 1535.384, 1529.976 | **1531.740** | **+1.2900%** | **+1.2901%** |

All six pairs are positive, and both input distributions improve by more than
1%.  The topology therefore passes the adoption gate.  Before making it the
documented default, compare the otherwise identical compiler-auto-unrolled
K-tile loop directly against this u1 version.
