# Direct N-split phase-shift ablation

Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5
process. The baseline is the direct N-split source, not a prefetch
candidate.

## Input `random`

| variant | samples | mean +/- SD | vs baseline | paired vs matched control |
|---|---|---:|---:|---:|
| `baseline` | 1805.502, 1805.153, 1810.957 | 1807.204 +/- 3.255 | +0.0000% | baseline +0.0000% |
| `cta4` | 1806.670, 1806.090, 1804.414 | 1805.725 +/- 1.172 | -0.0819% | baseline -0.0816% |
| `cta8` | 1806.235, 1807.974, 1806.546 | 1806.918 +/- 0.927 | -0.0158% | baseline -0.0156% |
| `b1_gap0` | 1811.869, 1808.669, 1808.029 | 1809.522 +/- 2.057 | +0.1283% | baseline +0.1286% |
| `b1_gap32` | 1810.799, 1811.401, 1789.351 | 1803.850 +/- 12.560 | -0.1856% | b1_gap0 -0.3137% |
| `b1_gap64` | 1809.100, 1813.241, 1811.243 | 1811.195 +/- 2.071 | +0.2208% | b1_gap0 +0.0926% |
| `pipe1_gap0` | 1742.316, 1745.485, 1743.740 | 1743.847 +/- 1.587 | -3.5058% | baseline -3.5056% |
| `pipe1_gap32` | 1715.662, 1719.670, 1722.306 | 1719.213 +/- 3.346 | -4.8689% | pipe1_gap0 -1.4127% |
| `pipe1_gap64` | 1673.314, 1670.978, 1671.500 | 1671.931 +/- 1.226 | -7.4852% | pipe1_gap0 -4.1239% |

## Input `random-signed8`

| variant | samples | mean +/- SD | vs baseline | paired vs matched control |
|---|---|---:|---:|---:|
| `baseline` | 1600.864, 1615.118, 1606.992 | 1607.658 +/- 7.150 | +0.0000% | baseline +0.0000% |
| `cta4` | 1601.935, 1602.566, 1612.091 | 1605.531 +/- 5.690 | -0.1323% | baseline -0.1310% |
| `cta8` | 1602.054, 1599.699, 1602.648 | 1601.467 +/- 1.560 | -0.3851% | baseline -0.3836% |
| `b1_gap0` | 1607.366, 1602.899, 1612.866 | 1607.710 +/- 4.992 | +0.0033% | baseline +0.0050% |
| `b1_gap32` | 1598.627, 1599.156, 1606.293 | 1601.359 +/- 4.281 | -0.3918% | b1_gap0 -0.3949% |
| `b1_gap64` | 1610.664, 1602.533, 1613.218 | 1608.805 +/- 5.580 | +0.0713% | b1_gap0 +0.0681% |
| `pipe1_gap0` | 1555.195, 1575.764, 1558.360 | 1563.106 +/- 11.076 | -2.7712% | baseline -2.7719% |
| `pipe1_gap32` | 1528.989, 1527.772, 1528.184 | 1528.315 +/- 0.619 | -4.9353% | pipe1_gap0 -2.2224% |
| `pipe1_gap64` | 1443.586, 1452.585, 1458.910 | 1451.694 +/- 7.701 | -9.7013% | pipe1_gap0 -7.1251% |

## Gate

Adopt only if both input distributions improve by at least 0.5%
against the matched control. Smaller consistent gains remain
diagnostic.
