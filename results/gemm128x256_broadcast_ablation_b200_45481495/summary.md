# Repeated-B broadcast and pipeline-depth ablation

- GPU: NVIDIA B200, Vast instance `45481495`
- Input: BF16 uniform `[0,1)`, repeated global A/B panels
- Output: full row-major FP32 C through the TMA-store epilogue
- Kernel: persistent CTA `128x256`, K-stage 128, two MMA issuer warps
- Launch: 148 persistent CTAs
- Timing: one warmup and five timed launches per process, three rotated passes
- Validation: all three variants bit-exact at size 512 (`bad=0`)

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| B0: distinct B halves, 2 stages | 1599.919 ± 0.909 | 1700.903 ± 15.239 | 1556.285 ± 1.745 |
| B1: shared B panel, 2 stages | 1673.163 ± 0.484 | 1824.972 ± 0.566 | 1661.418 ± 2.103 |
| B2: shared B panel, 3 stages | **1787.751 ± 0.536** | **1913.033 ± 0.362** | **1773.949 ± 6.036** |

Values are mean ± population standard deviation in TFLOP/s. The first B0 16K
sample was 1679.351; the other two were 1711.703 and 1711.654, consistent with
the preceding 1712.793 baseline. Candidate conclusions do not depend on that
single low baseline sample because B1/B2 were stable.

Relative mean changes:

| size | B1/B0: B reuse | B2/B1: third stage | B2/B0 total |
|---:|---:|---:|---:|
| 8K | +4.578% | +6.849% | +11.740% |
| 16K | +7.294% | +4.825% | +12.472% |
| 32K | +6.755% | +6.773% | +13.986% |

The integrated kernel therefore recovers and exceeds the historical
1797.227-TFLOP/s repeated-tile target. This optimization is specific to the
implicit tiled-B ceiling experiment: both `128x128` output halves intentionally
use the same B panel. A dense GEMM with different N coordinates cannot apply
this broadcast without changing the mathematical result.
