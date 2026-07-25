# Clean N-split A-locality scheduler sweep

Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5
process. All variants keep the same mainloop, TMA traffic, C store,
and 148 persistent CTAs; only output-tile order changes.

## Input `random`

| variant | samples | mean +/- SD | vs baseline | paired vs baseline |
|---|---|---:|---:|---:|
| `baseline` | 1754.261, 1754.171, 1754.003, 1753.556 | 1753.998 +/- 0.313 | +0.0000% | +0.0000% |
| `nfast_16x16` | 1765.995, 1765.252, 1765.825, 1766.647 | 1765.930 +/- 0.574 | +0.6803% | +0.6803% |
| `nfast_8x32` | 1718.722, 1687.639, 1719.276, 1718.975 | 1711.153 +/- 15.678 | -2.4427% | -2.4426% |
| `nfast_4x64` | 1530.175, 1499.347, 1501.834, 1514.254 | 1511.402 +/- 14.112 | -13.8310% | -13.8310% |

## Input `random-signed8`

| variant | samples | mean +/- SD | vs baseline | paired vs baseline |
|---|---|---:|---:|---:|
| `baseline` | 1515.949, 1519.168, 1518.683, 1531.556 | 1521.339 +/- 6.957 | +0.0000% | +0.0000% |
| `nfast_16x16` | 1522.662, 1526.699, 1528.001, 1520.122 | 1524.371 +/- 3.632 | +0.1993% | +0.2014% |
| `nfast_8x32` | 1430.285, 1426.968, 1424.456, 1421.994 | 1425.926 +/- 3.545 | -6.2717% | -6.2695% |
| `nfast_4x64` | 1259.560, 1265.257, 1302.311, 1269.658 | 1274.197 +/- 19.193 | -16.2451% | -16.2435% |

## Gate

Promote only if both input distributions improve by at least 0.5%
against baseline and no validation or resource gate regresses.
