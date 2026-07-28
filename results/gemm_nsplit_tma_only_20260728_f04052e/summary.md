# 16K dense TMA-load-only comparison

Logical A/B request bytes per launch: 68719476736 B (64 GiB).
Mean +/- sample SD across four independent W1/I5 processes.

| input | GEMM TFLOP/s | GEMM ms | TMA-only ms | TMA-only logical TB/s | TMA-only/GEMM time |
|---|---:|---:|---:|---:|---:|
| `[0,1)` | 1832.607 +/- 2.784 | 4.799780 +/- 0.007292 | 2.601502 +/- 0.006145 | 26.415 | 54.20% |
| `[-8,8)` | 1619.235 +/- 2.932 | 5.432267 +/- 0.009837 | 2.604890 +/- 0.004343 | 26.381 | 47.95% |
