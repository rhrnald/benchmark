# 16K E7a vs CUTLASS TFLOP/s

| implementation | process TFLOP/s | mean ± sample SD | mean runtime |
|---|---:|---:|---:|
| clean E7a | 1797.610 / 1800.297 / 1801.040 / 1801.054 | **1800.000 ± 1.632** | 4.886722 ms |
| CUTLASS selected 16K | 1425.240 / 1425.430 / 1420.530 / 1421.280 | **1423.120 ± 2.577** | 6.180872 ms |

Clean E7a relative to CUTLASS: **+26.483%**.
