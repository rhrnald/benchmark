# Compile-time C-store-off aggressive upper bound

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Clean baseline definition/result: `8add902` / `ae74998`
- Store-off definition: `19c1029`
- Baseline binary SHA-256:
  `c8cf7f415a8933c516d90725f46eceb14e66fc0168f0fffe45566d048e904667`
- Candidate source SHA-256:
  `0549f75b94bf9e7c5195f8ca787812da847b2d412153abfcacbcb0c88af3f674`
- Candidate binary SHA-256:
  `14a5a7130a3e5c07f0e669a9633a9b4e91cbed11a1a69aadce0e567336df1e95`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

This diagnostic retains all A/B TMA loads, all MMA issue/commit work, the final
MMA completion waits, the persistent atomic scheduler, and sink publication.
It removes the complete TMEM-to-SMEM staging and global C TMA-store path.  It
therefore does not produce C and is not an end-to-end GEMM result; C-reference
validation is intentionally inapplicable.

## Codegen caveat

| metric | clean baseline | compile-time store-off |
|---|---:|---:|
| registers / spills | 178 / 0 | 54 / 0 |
| static text slots | 1936 | 1640 |
| A/B `UTMALDG` / `UTCHMMA` / `SYNCS.PHASECHK` | 21 / 8 / 48 | 21 / 8 / 48 |
| C `LDTM` / `UTMASTG` | 8 / 4 | 0 / 0 |
| `BAR.SYNC` | 20 | 9 |

The large register reduction can change mainloop scheduling even though
tcgen05 still limits residency to one CTA per SM.  Thus this experiment is an
aggressive upper bound on epilogue plus compiled register-pressure/code-layout
cost, not an exact isolated C-store time.  A same-binary runtime on/off control
is required for the strict bound.

## Performance

All values are nominal event TFLOP/s using the GEMM FLOP count.  Delta is
compile-time store-off relative to the same-pass clean end-to-end baseline.

| input | baseline samples | baseline mean | store-off samples | store-off mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1736.658, 1742.979, 1738.462 | **1739.366** | 1790.428, 1791.121, 1787.398 | **1789.649** | +2.8909% | +2.8910% |
| BF16 uniform `[-8,8)` | 1519.431, 1509.224, 1507.828 | **1512.161** | 1548.857, 1550.577, 1548.261 | **1549.232** | +2.4515% | +2.4527% |

Interpretation: even this optimistic removal raises throughput only about
2.5--2.9%.  The epilogue is a real secondary cost, but it cannot by itself
explain the full gap to 1.9 PFLOP/s.  Follow with a dual-mode same-binary
control before assigning an exact epilogue fraction.
