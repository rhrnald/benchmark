# N-split weight-stationary B-collector result

Dense 16K BF16-input/FP32-output GEMM event TFLOP/s.  Each sample
is one W1/I5 process; four Latin-rotated passes were collected.

## Input `random`

| variant | samples | mean +/- sample SD | vs E7a | vs N-split | paired vs N-split | paired vs static |
|---|---|---:|---:|---:|---:|---:|
| e7a_exact | 1819.841, 1816.027, 1815.982, 1816.337 | 1817.047 +/- 1.870 | +0.0000% | +1.1942% | +1.1942% | +3.4898% |
| nsplit_exact | 1797.730, 1796.212, 1794.898, 1793.577 | 1795.604 +/- 1.779 | -1.1801% | +0.0000% | +0.0000% | +2.2686% |
| nsplit_static_control | 1759.011, 1752.813, 1755.820, 1755.455 | 1755.775 +/- 2.540 | -3.3721% | -2.2182% | -2.2181% | +0.0000% |
| nsplit_ws_b01 | 1682.946, 1681.537, 1681.402, 1683.649 | 1682.383 +/- 1.095 | -7.4111% | -6.3054% | -6.3054% | -4.1799% |

## Input `random-signed8`

| variant | samples | mean +/- sample SD | vs E7a | vs N-split | paired vs N-split | paired vs static |
|---|---|---:|---:|---:|---:|---:|
| e7a_exact | 1607.379, 1604.182, 1610.602, 1610.106 | 1608.067 +/- 2.952 | +0.0000% | +0.8795% | +0.8816% | +2.3280% |
| nsplit_exact | 1589.417, 1604.802, 1589.880, 1592.090 | 1594.047 +/- 7.264 | -0.8719% | +0.0000% | +0.0000% | +1.4361% |
| nsplit_static_control | 1573.779, 1569.931, 1571.922, 1570.306 | 1571.485 +/- 1.757 | -2.2750% | -1.4154% | -1.4136% | +0.0000% |
| nsplit_ws_b01 | 1512.075, 1511.308, 1516.844, 1512.471 | 1513.175 +/- 2.494 | -5.9010% | -5.0734% | -5.0717% | -3.7104% |

## Decision rule

- `nsplit_static_control` versus `nsplit_exact` isolates the
  pipe-specialized consumer/code-shape effect.
- Advance only if `nsplit_ws_b01` improves pass-matched
  `nsplit_static_control` by at least 0.5% for both inputs.
- Replace the performance reference only if it also beats exact
  E7a in both distributions.
