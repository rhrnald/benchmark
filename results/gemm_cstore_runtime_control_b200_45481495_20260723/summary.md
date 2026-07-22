# Same-binary C-store runtime control

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Clean baseline definition/result: `8add902` / `ae74998`
- Dual-mode definition: `41a6acb`
- Baseline binary SHA-256:
  `c8cf7f415a8933c516d90725f46eceb14e66fc0168f0fffe45566d048e904667`
- Dual-mode source SHA-256:
  `e41bdba2ccb6459053191653ecafeb3bcb59fc3d3b560e444b68e098e2ab94e7`
- Dual-mode binary SHA-256:
  `ee85a02c4b7b60800d6c79a30d3d938fc95ac2516bec584212876eb7002456c1`
- Protocol: one case per process, warmup 1, timed launches 5, three cyclically
  ordered clean/store-on/store-off triples per input distribution

The same cubin takes one uniform runtime argument.  Store-on executes the
normal TMEM-to-SMEM and TMA C-store path; store-off skips only that path.  The
caller CTA barrier, final MMA completion waits, persistent atomic scheduler,
sink publication, register allocation, and code layout are common.  Store-on
passed 512 pattern and ones full-C validation with zero error.  Store-off does
not produce C and its reported TFLOP/s is a diagnostic nominal rate.

## Codegen

| metric | clean baseline | dual-mode binary |
|---|---:|---:|
| registers / spills | 178 / 0 | 178 / 0 |
| stack | 16 B | 16 B |
| static text slots | 1936 | 1936 |
| A/B TMA / MMA / waits | identical | identical |
| C `LDTM` / `UTMASTG` | 8 / 4 | 8 / 4 |
| `BAR.SYNC` | 20 | 20 |

## Store-on equivalence gate

| input | clean mean | dual store-on mean | ratio-of-means delta | paired delta | gate |
|---|---:|---:|---:|---:|---|
| BF16 uniform `[0,1)` | **1741.541** | **1734.387** | -0.4108% | -0.4108% | pass (<0.5%) |
| BF16 uniform `[-8,8)` | **1511.271** | **1511.037** | -0.0155% | -0.0154% | pass |

## Strict store-off upper bound

Delta is same-binary store-off relative to same-pass store-on.

| input | store-on samples | store-on mean | store-off samples | store-off mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1737.119, 1729.449, 1736.594 | **1734.387** | 1797.826, 1802.828, 1800.898 | **1800.517** | +3.8129% | +3.8135% |
| BF16 uniform `[-8,8)` | 1510.991, 1510.808, 1511.311 | **1511.037** | 1559.837, 1561.852, 1559.035 | **1560.241** | +3.2564% | +3.2564% |

Interpretation: with register allocation and all non-store work held constant,
the complete C epilogue costs at most about 3.3--3.8% of throughput at 16K.
This is large enough to justify a genuinely overlapped epilogue design, but
small fixed barrier removals cannot close the much larger gap to 1.9 PFLOP/s.
Mainloop scalar/TMA/MMA scheduling remains the first optimization target; a
3-buffer C pipeline is secondary and should aim to recover only part of this
measured ceiling.
