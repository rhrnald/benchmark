# N-split transpose-epilogue B200 result

Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5
process. Eight Williams-balanced orders were collected per input,
so every variant occupied every execution position exactly once and
within those orders every ordered non-self adjacent pair occurred
exactly once.

## Input `random`

| variant | mean +/- sample SD TFLOP/s | vs scalar, paired 95% CI | vs exact, paired 95% CI | mean event ms |
|---|---:|---:|---:|---:|
| nsplit_exact | 1758.641 +/- 1.232 | -0.4453% [-0.5088,-0.3817] | +0.0000% [+0.0000,+0.0000] | 5.001645 |
| nsplit_transpose_nostore | 1839.790 +/- 1.020 | +4.1485% [+4.0799,+4.2171] | +4.6143% [+4.5620,+4.6667] | 4.781034 |
| nsplit_transpose_scalar | 1766.507 +/- 1.345 | +0.0000% [+0.0000,+0.0000] | +0.4473% [+0.3832,+0.5114] | 4.979374 |
| nsplit_transpose_vec2_x32 | 1763.815 +/- 1.235 | -0.1523% [-0.2087,-0.0960] | +0.2942% [+0.2585,+0.3300] | 4.986972 |
| nsplit_transpose_vec2_cf1_x32 | 1763.000 +/- 1.241 | -0.1984% [-0.3090,-0.0878] | +0.2480% [+0.1445,+0.3514] | 4.989278 |
| nsplit_transpose_vec2_cf_x32 | 1761.743 +/- 0.749 | -0.2696% [-0.3184,-0.2209] | +0.1764% [+0.1063,+0.2465] | 4.992837 |
| nsplit_transpose_vec2_cf_x64 | 1765.241 +/- 0.989 | -0.0716% [-0.1097,-0.0336] | +0.3753% [+0.3131,+0.4375] | 4.982943 |
| nsplit_transpose_vec4_cf_x64 | 1761.077 +/- 0.897 | -0.3073% [-0.3550,-0.2597] | +0.1386% [+0.0903,+0.1869] | 4.994723 |

| transpose epilogue | E2E - transpose no-store, mean +/- sample SD ms |
|---|---:|
| nsplit_transpose_scalar | 0.198340 +/- 0.003879 |
| nsplit_transpose_vec2_x32 | 0.205938 +/- 0.002979 |
| nsplit_transpose_vec2_cf1_x32 | 0.208244 +/- 0.004932 |
| nsplit_transpose_vec2_cf_x32 | 0.211803 +/- 0.003257 |
| nsplit_transpose_vec2_cf_x64 | 0.201909 +/- 0.003269 |
| nsplit_transpose_vec4_cf_x64 | 0.213689 +/- 0.002026 |

## Input `random-signed8`

| variant | mean +/- sample SD TFLOP/s | vs scalar, paired 95% CI | vs exact, paired 95% CI | mean event ms |
|---|---:|---:|---:|---:|
| nsplit_exact | 1564.629 +/- 1.840 | -0.5949% [-0.8586,-0.3313] | +0.0000% [+0.0000,+0.0000] | 5.621846 |
| nsplit_transpose_nostore | 1631.399 +/- 2.513 | +3.6471% [+3.3679,+3.9263] | +4.2676% [+4.1076,+4.4276] | 5.391760 |
| nsplit_transpose_scalar | 1574.004 +/- 4.138 | +0.0000% [+0.0000,+0.0000] | +0.5994% [+0.3320,+0.8667] | 5.588390 |
| nsplit_transpose_vec2_x32 | 1573.723 +/- 4.553 | -0.0172% [-0.3576,+0.3233] | +0.5812% [+0.3684,+0.7940] | 5.589394 |
| nsplit_transpose_vec2_cf1_x32 | 1570.659 +/- 3.969 | -0.2121% [-0.4306,+0.0063] | +0.3857% [+0.1077,+0.6636] | 5.600287 |
| nsplit_transpose_vec2_cf_x32 | 1572.339 +/- 4.840 | -0.1054% [-0.3696,+0.1587] | +0.4931% [+0.1613,+0.8249] | 5.594320 |
| nsplit_transpose_vec2_cf_x64 | 1573.960 +/- 2.387 | -0.0020% [-0.3063,+0.3023] | +0.5965% [+0.4663,+0.7266] | 5.588522 |
| nsplit_transpose_vec4_cf_x64 | 1568.183 +/- 3.095 | -0.3693% [-0.5965,-0.1421] | +0.2274% [-0.0079,+0.4627] | 5.609119 |

| transpose epilogue | E2E - transpose no-store, mean +/- sample SD ms |
|---|---:|
| nsplit_transpose_scalar | 0.196630 +/- 0.017829 |
| nsplit_transpose_vec2_x32 | 0.197634 +/- 0.014437 |
| nsplit_transpose_vec2_cf1_x32 | 0.208526 +/- 0.020501 |
| nsplit_transpose_vec2_cf_x32 | 0.202560 +/- 0.018001 |
| nsplit_transpose_vec2_cf_x64 | 0.196761 +/- 0.012363 |
| nsplit_transpose_vec4_cf_x64 | 0.217358 +/- 0.010327 |

## Cross-input ranking

Ranking uses the smaller of the two mean paired improvements
against the scalar-transpose control.

| rank | variant | `[0,1)` | `[-8,8)` | worst input |
|---:|---|---:|---:|---:|
| 1 | nsplit_transpose_scalar | +0.0000% | +0.0000% | +0.0000% |
| 2 | nsplit_transpose_vec2_cf_x64 | -0.0716% | -0.0020% | -0.0716% |
| 3 | nsplit_transpose_vec2_x32 | -0.1523% | -0.0172% | -0.1523% |
| 4 | nsplit_transpose_vec2_cf1_x32 | -0.1984% | -0.2121% | -0.2121% |
| 5 | nsplit_transpose_vec2_cf_x32 | -0.2696% | -0.1054% | -0.2696% |
| 6 | nsplit_transpose_vec4_cf_x64 | -0.3073% | -0.3693% | -0.3693% |

## Decision rule

- Correctness, zero spill/local memory, and the expected MMA/TMA
  instruction counts are mandatory.
- An epilogue candidate advances only when its paired 95% CI
  against `nsplit_transpose_scalar` is above zero for both inputs.
- The sweep winner requires a separate confirmatory matched A/B
  run before adoption. It replaces `nsplit_exact` only when that
  run shows at least +0.5% over exact for both inputs without a
  negative paired 95% CI.
