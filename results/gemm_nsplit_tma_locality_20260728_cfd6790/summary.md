# 16K TMA-only address-locality ablation

All modes issue the same 64 GiB logical A/B payload. Mean +/- sample
SD across four independent W1/I5 processes; method order is fully
position-balanced.

| input | address mode | ms | logical TB/s | speedup vs dense |
|---|---|---:|---:|---:|
| `[0,1)` | `dense` | 2.611839 +/- 0.004385 | 26.311 | 1.0000x |
| `[0,1)` | `same_a` | 2.494930 +/- 0.000541 | 27.544 | 1.0469x |
| `[0,1)` | `same_b` | 2.502640 +/- 0.015106 | 27.459 | 1.0436x |
| `[0,1)` | `same` | 2.390038 +/- 0.025342 | 28.752 | 1.0928x |
| `[-8,8)` | `dense` | 2.610193 +/- 0.003892 | 26.327 | 1.0000x |
| `[-8,8)` | `same_a` | 2.493986 +/- 0.000612 | 27.554 | 1.0466x |
| `[-8,8)` | `same_b` | 2.500673 +/- 0.007291 | 27.480 | 1.0438x |
| `[-8,8)` | `same` | 2.383875 +/- 0.022282 | 28.827 | 1.0949x |
