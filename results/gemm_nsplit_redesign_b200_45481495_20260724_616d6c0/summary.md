# 256x256 N-split redesign result

All values are event TFLOP/s for dense 16K BF16-input/FP32-output
GEMM.  Each sample is one process with W1/I5; three position-rotated
passes were collected per input and variant.

## Input `random`

| variant | samples | mean +/- sample SD | vs E7a | matched control | paired vs control |
|---|---|---:|---:|---:|---:|
| e7a_exact | 1818.459, 1817.363, 1815.248 | 1817.023 +/- 1.632 | +0.0000% | e7a_exact +0.0000% | +0.0000% |
| nsplit_exact | 1798.560, 1799.872, 1793.093 | 1797.175 +/- 3.595 | -1.0924% | nsplit_exact +0.0000% | +0.0000% |
| nsplit_u1_suspend_prod | 1797.398, 1797.890, 1795.023 | 1796.770 +/- 1.533 | -1.1146% | nsplit_exact -0.0225% | -0.0224% |

## Input `random-signed8`

| variant | samples | mean +/- sample SD | vs E7a | matched control | paired vs control |
|---|---|---:|---:|---:|---:|
| e7a_exact | 1615.785, 1615.067, 1607.140 | 1612.664 +/- 4.797 | +0.0000% | e7a_exact +0.0000% | +0.0000% |
| nsplit_exact | 1594.262, 1591.397, 1595.669 | 1593.776 +/- 2.177 | -1.1712% | nsplit_exact +0.0000% | +0.0000% |
| nsplit_u1_suspend_prod | 1599.962, 1597.880, 1605.772 | 1601.205 +/- 4.090 | -0.7106% | nsplit_exact +0.4661% | +0.4660% |

## Decision rule

- Advance an N-split mechanism only when its pass-matched change
  exceeds +0.5% for both input distributions.
- Replacing the performance reference also requires beating exact
  E7a on both distributions in this activation.
- A smaller consistent gain is retained only as a diagnostic.
