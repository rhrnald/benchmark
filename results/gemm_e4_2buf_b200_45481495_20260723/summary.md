# E4a two-buffer C-store pipeline

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Macro-free E2a parent: `3d2d0a4`
- E4a definition: `686b3d3`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Baseline source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Candidate source SHA-256:
  `1fb47c73b2658772db3bb9699fa7143904d792c97476911092eef4cc539bdc33`
- Baseline / candidate binary SHA-256:
  `5d4127162b83ec0584b760290d4f7542b71d7588838a012fa4201e74cd7f6023` /
  `8aa5aa88454a8be771a33a216343f50076cc77f25e31b577e757782a44a2210b`

The candidate keeps two 128x128 FP32 shared-memory buffers but commits every C
chunk as a separate bulk group.  Before reusing a buffer it executes
`cp.async.bulk.wait_group.read 1`, and after all four chunks it executes
`wait_group.read 0`.  Work, C layout, TMA bytes, and output tile order remain
unchanged.

Baseline pattern validation and candidate pattern/ones full-C validations at
size 512 all passed exactly.

## Code generation

| metric | macro-free E2a | E4a two-buffer |
|---|---:|---:|
| registers / spills | 174 / 0 | 172 / 0 |
| stack | 0 B | 0 B |
| static shared | 1184 B | 1184 B |
| C TMA stores / commits | 4 / 2 | 4 / 4 |
| source-read waits | full wait 0 x2 | read 1 x2 + read 0 x1 |

## Performance

All values are event TFLOP/s.  Delta is E4a relative to the same-pass parent.

| input | parent samples | parent mean | E4a samples | E4a mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1749.558, 1747.753, 1748.354 | **1748.555** | 1741.371, 1740.544, 1739.769 | **1740.561** | **-0.4572%** | **-0.4572%** |
| BF16 uniform `[-8,8)` | 1513.600, 1514.526, 1516.439 | **1514.855** | 1512.840, 1513.782, 1510.565 | **1512.396** | **-0.1623%** | **-0.1622%** |

All six paired deltas are negative.  The extra commits and reuse barriers cost
more than the overlap exposed by two buffers, so E4a is rejected without an
extended six-pair run.  E4b three-buffer remains a separate test because it
requires only one reuse wait/barrier rather than two.
