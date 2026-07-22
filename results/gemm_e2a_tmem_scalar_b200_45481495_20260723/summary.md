# E2a TMEM scalar-address cleanup

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Clean baseline definition/result: `8add902` / `ae74998`
- E2a definition: `2c44f23`
- Baseline binary SHA-256:
  `c8cf7f415a8933c516d90725f46eceb14e66fc0168f0fffe45566d048e904667`
- Candidate source SHA-256:
  `06a23f5207e9434ad4d4d7aa8fd406fd8bb4de0e8351968ebb6de294ec93e851`
- Candidate binary SHA-256:
  `72cccca123a689013b9edc0bd43808f4e40b11cb92b78b79523e1fdf0f61f926`
- Protocol: one case per process, warmup 1, timed launches 5, six
  order-balanced process pairs per input distribution

E2a removes the CTA-local `tmem_tile_addr[4]` and per-output-tile
`c_taddr[4]` arrays.  MMA and epilogue helpers instead form the address as
`tmem_base + tile * 128`.  The arithmetic describes exactly the same four
TMEM quadrants and does not alter work, data movement, scheduler order, or C
storage.

Both 512 pattern and ones full-C validations passed with `max_abs=0` and
`max_rel=0`.

## Codegen

| metric | clean baseline | E2a |
|---|---:|---:|
| registers / spills | 178 / 0 | 174 / 0 |
| stack | 16 B | 0 B |
| static text slots | 1936 | 1928 |
| local loads `LDL` | 2 | 0 |
| MMA / TMA / waits / barriers | baseline | unchanged |
| `R2UR.BROADCAST` | 65 | 65 |

## Performance

All values are event TFLOP/s.  Delta is E2a relative to the same-pass clean
baseline.

| input | baseline samples | baseline mean | E2a samples | E2a mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1738.180, 1741.985, 1743.463, 1740.798, 1742.241, 1740.571 | **1741.206** | 1752.932, 1752.697, 1753.433, 1747.482, 1752.053, 1752.822 | **1751.903** | +0.6143% | +0.6144% |
| BF16 uniform `[-8,8)` | 1512.286, 1515.573, 1509.386, 1513.470, 1510.760, 1516.318 | **1512.966** | 1522.827, 1517.229, 1517.382, 1516.193, 1517.492, 1522.348 | **1518.912** | +0.3930% | +0.3932% |

Decision: promising but not yet a standalone default.  All six paired deltas
are positive for both distributions and codegen is strictly smaller, while
the average gain is modest and below the 1% adoption rule.  Retain E2a as the
parent for E2b shared-descriptor cleanup; adopt or revert the pair based on the
combined result.
