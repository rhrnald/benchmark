# E7a phase-redesign ablation

All results are 16K dense BF16-input/FP32-output GEMM event TFLOP/s.
Each sample is a separate process with one warmup and five timed
launches.  Three rotated passes were collected for each input.

The clean E7a baseline is compiled directly from the audited canonical
source. Candidate sources are hash-gated generated variants.

## Input `random`

| variant | samples | mean +/- sample SD | vs baseline | vs matched control | paired control delta |
|---|---|---:|---:|---:|---:|
| baseline | 1819.092, 1820.239, 1819.169 | 1819.500 +/- 0.641 | +0.0000% | baseline +0.0000% | +0.0000% |
| cta4 | 1815.715, 1820.044, 1822.793 | 1819.517 +/- 3.568 | +0.0010% | baseline +0.0010% | +0.0010% |
| cta8 | 1813.650, 1819.321, 1819.376 | 1817.449 +/- 3.290 | -0.1127% | baseline -0.1127% | -0.1127% |
| b1_gap0 | 1818.120, 1817.379, 1780.775 | 1805.425 +/- 21.350 | -0.7736% | baseline -0.7736% | -0.7737% |
| b1_gap32 | 1818.230, 1817.096, 1818.029 | 1817.785 +/- 0.605 | -0.0943% | b1_gap0 +0.6846% | +0.6942% |
| b1_gap64 | 1820.984, 1817.495, 1813.566 | 1817.348 +/- 3.711 | -0.1183% | b1_gap0 +0.6604% | +0.6684% |
| b1_cross | 1816.897, 1817.685, 1815.970 | 1816.851 +/- 0.858 | -0.1456% | baseline -0.1456% | -0.1456% |

## Input `random-signed8`

| variant | samples | mean +/- sample SD | vs baseline | vs matched control | paired control delta |
|---|---|---:|---:|---:|---:|
| baseline | 1613.236, 1612.359, 1614.799 | 1613.465 +/- 1.236 | +0.0000% | baseline +0.0000% | +0.0000% |
| cta4 | 1614.012, 1616.244, 1613.940 | 1614.732 +/- 1.310 | +0.0785% | baseline +0.0785% | +0.0786% |
| cta8 | 1612.085, 1611.960, 1613.542 | 1612.529 +/- 0.880 | -0.0580% | baseline -0.0580% | -0.0580% |
| b1_gap0 | 1611.680, 1613.512, 1611.830 | 1612.341 +/- 1.017 | -0.0697% | baseline -0.0697% | -0.0696% |
| b1_gap32 | 1617.237, 1599.854, 1613.698 | 1610.263 +/- 9.186 | -0.1984% | b1_gap0 -0.1289% | -0.1286% |
| b1_gap64 | 1602.783, 1609.551, 1615.834 | 1609.389 +/- 6.527 | -0.2526% | b1_gap0 -0.1830% | -0.1830% |
| b1_cross | 1613.815, 1613.128, 1613.357 | 1613.433 +/- 0.350 | -0.0019% | baseline -0.0019% | -0.0019% |

## Interpretation gate

- Adopt only a mechanism that improves both distributions by at
  least 0.5% in paired measurements.
- A consistent but smaller gain is diagnostic and requires a
  pre-registered extension before any adoption claim.
- `cta4/cta8` change only the one-time CTA startup phase;
  `b1_gap0` controls for its code-generation change;
  `b1_gap32/64` change only late-B1 spacing per K64 stage;
  `b1_cross` relocates one existing W3 wait without adding work.
