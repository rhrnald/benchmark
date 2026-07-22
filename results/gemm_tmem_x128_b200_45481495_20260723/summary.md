# E4d x128 TMEM epilogue loads

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Macro-free E2a parent: `3d2d0a4`
- E4d definition: `cf37a8f`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Baseline source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Candidate source SHA-256:
  `a1e364cc206ff19102783b3e78d412cae12f62f80bac6ac944671c6ab78621bc`
- Baseline / candidate binary SHA-256:
  `5d4127162b83ec0584b760290d4f7542b71d7588838a012fa4201e74cd7f6023` /
  `8408b07266371abb3c1f0498ccee4501af2e52e60f3a233d84896f6aaefa5694`

Each pair of `tcgen05.ld.32x32b.x64` operations for one 128-column C chunk is
replaced with one official CUDA PTX-wrapper x128 load.  The register-to-SW128
shared-store mapping and all 128 vector stores are unchanged.

Candidate pattern and ones full-C validations at size 512 passed exactly.

## Code generation

| metric | macro-free E2a | x128 TMEM load |
|---|---:|---:|
| registers / spills | 174 / 0 | 174 / 0 |
| stack / static shared | 0 B / 1184 B | unchanged |
| `LDTM.x64` / `LDTM.x128` | 8 / 0 | 0 / 4 |
| TMEM load waits | 8 | 4 |
| kernel text | 30848 B | 30592 B |
| MMA / TMA load / TMA store / C vector stores | 8 / 21 / 4 / 128 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass parent.

| input | parent samples | parent mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1747.791, 1749.019, 1748.069 | **1748.293** | 1747.480, 1748.601, 1749.852 | **1748.644** | **+0.0201%** | **+0.0201%** |
| BF16 uniform `[-8,8)` | 1519.116, 1519.173, 1519.567 | **1519.285** | 1514.229, 1513.973, 1515.681 | **1514.628** | **-0.3066%** | **-0.3066%** |

The `[0,1)` result is noise-level neutral, while all three signed-input pairs
are negative.  Halving TMEM load/wait instructions does not reduce the full
epilogue enough to improve both distributions.  E4d is rejected without an
extended run under the two-distribution adoption rule.
