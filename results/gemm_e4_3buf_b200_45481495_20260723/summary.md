# E4b three-buffer C-store pipeline

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Macro-free E2a parent: `3d2d0a4`
- E4b definition: `525979c`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Baseline source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Candidate source SHA-256:
  `8f4fc8c34c31a92075ffacf384433fdc4f0dd837b8d8e260f481bf88d2c5bb03`
- Baseline / candidate binary SHA-256:
  `5d4127162b83ec0584b760290d4f7542b71d7588838a012fa4201e74cd7f6023` /
  `b7f185a23da13d83a1035f4aa28ec41a3ad44cb593321c353d4c68080eff6269`

The candidate uses three 128x128 FP32 shared-memory buffers.  Each of the four
C chunks is a separate bulk group, chunk 3 waits with
`cp.async.bulk.wait_group.read 2` before reusing buffer 0, and the epilogue ends
with `wait_group.read 0`.  Three buffers occupy 196608 B, exactly the existing
mainloop payload, so the dynamic allocation remains 197632 B.

Candidate pattern and ones full-C validations at size 512 passed exactly.

## Code generation

| metric | macro-free E2a | E4b three-buffer |
|---|---:|---:|
| registers / spills | 174 / 0 | 172 / 0 |
| stack | 0 B | 0 B |
| static shared | 1184 B | 1184 B |
| C TMA stores / commits | 4 / 2 | 4 / 4 |
| source-read waits | full wait 0 x2 | read 2 x1 + read 0 x1 |

## Performance

All values are event TFLOP/s.  Delta is E4b relative to the same-pass parent.

| input | parent samples | parent mean | E4b samples | E4b mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1748.741, 1747.985, 1747.196 | **1747.974** | 1739.425, 1739.159, 1738.941 | **1739.175** | **-0.5034%** | **-0.5034%** |
| BF16 uniform `[-8,8)` | 1513.341, 1514.719, 1516.344 | **1514.801** | 1510.775, 1513.880, 1509.055 | **1511.237** | **-0.2353%** | **-0.2352%** |

All six paired deltas are negative.  Even with only one reuse barrier, the
per-chunk commit/read pipeline is slower than the parent's two-chunk groups.
E4b is rejected without an extended run.  The 3.6% epilogue ceiling remains,
but it must be attacked in TMEM-to-SMEM staging/code generation rather than by
adding TMA store groups.
