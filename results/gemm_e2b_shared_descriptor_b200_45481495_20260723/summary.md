# E2b shared-descriptor scalar cleanup

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Parent E2a definition/result: `2c44f23` / `a3975f2`
- E2b definition: `748cfa6`
- Parent binary SHA-256:
  `72cccca123a689013b9edc0bd43808f4e40b11cb92b78b79523e1fdf0f61f926`
- Candidate source SHA-256:
  `b6f8e25c6a409fc589b9af76641fb913dda7eb0fcadb6ca4aa58fbf379792133`
- Candidate binary SHA-256:
  `2c34f3890c03f4d04fe955e393c31ff470d2ff94bddcb851e48dff9c51d8dab8`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

E2b changes only the consumer descriptor path.  It converts each stage shared
pointer once and derives A/B descriptor addresses with 32-bit byte offsets.
The TMA, MMA, barriers, scheduler, epilogue, and output addresses are unchanged.

Both 512 pattern and ones full-C validations passed with `max_abs=0` and
`max_rel=0`.

## Codegen

| metric | E2a parent | E2b |
|---|---:|---:|
| registers / spills | 174 / 0 | 174 / 0 |
| stack | 0 B | 0 B |
| static text slots | 1928 | 1920 |
| MMA / TMA / waits / barriers | parent | unchanged |
| `R2UR.BROADCAST` | 65 | 65 |

## Performance

All values are event TFLOP/s.  Delta is E2b relative to same-pass E2a.

| input | E2a samples | E2a mean | E2b samples | E2b mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1748.276, 1752.531, 1752.000 | **1750.936** | 1751.589, 1749.455, 1750.705 | **1750.583** | -0.0201% | -0.0200% |
| BF16 uniform `[-8,8)` | 1521.157, 1517.288, 1517.341 | **1518.595** | 1516.146, 1515.825, 1516.434 | **1516.135** | -0.1620% | -0.1619% |

Decision: neutral/reject.  The generated code is smaller but the removed
scalar instructions were not on the measured critical path.  Revert E2b and
retain E2a as the current performance candidate.
