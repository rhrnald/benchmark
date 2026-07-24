# N-split scalar x32 TMEM-load B200 result

Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5
process. All six orders of the three variants were collected per
input; each variant occupied each position twice and every ordered
non-self adjacent pair occurred twice within those orders.

## Input `random`

| variant | mean +/- sample SD TFLOP/s | vs scalar x64, paired 95% CI | vs exact, paired 95% CI | mean event ms |
|---|---:|---:|---:|---:|
| nsplit_exact | 1758.255 +/- 1.236 | -0.4736% [-0.5343,-0.4129] | +0.0000% [+0.0000,+0.0000] | 5.002743 |
| nsplit_transpose_scalar_x64 | 1766.621 +/- 1.077 | +0.0000% [+0.0000,+0.0000] | +0.4759% [+0.4146,+0.5371] | 4.979051 |
| nsplit_transpose_scalar_x32 | 1765.736 +/- 0.856 | -0.0501% [-0.1521,+0.0520] | +0.4255% [+0.3251,+0.5260] | 4.981547 |

Scalar x32 minus scalar x64 event time:
`+0.002496 +/- 0.004844 ms`, paired 95% CI `[-0.002587,+0.007579] ms`.

## Input `random-signed8`

| variant | mean +/- sample SD TFLOP/s | vs scalar x64, paired 95% CI | vs exact, paired 95% CI | mean event ms |
|---|---:|---:|---:|---:|
| nsplit_exact | 1567.368 +/- 2.940 | -0.4734% [-0.7442,-0.2027] | +0.0000% [+0.0000,+0.0000] | 5.612032 |
| nsplit_transpose_scalar_x64 | 1574.829 +/- 3.793 | +0.0000% [+0.0000,+0.0000] | +0.4762% [+0.2033,+0.7492] | 5.585453 |
| nsplit_transpose_scalar_x32 | 1573.798 +/- 1.520 | -0.0652% [-0.2287,+0.0984] | +0.4106% [+0.1820,+0.6392] | 5.589089 |

Scalar x32 minus scalar x64 event time:
`+0.003636 +/- 0.008698 ms`, paired 95% CI `[-0.005491,+0.012764] ms`.

## Decision

The scalar x32 candidate does not pass the screening gate. The gate requires the paired 95% CI
against scalar x64 to be above zero for both input
distributions.

A passing sweep candidate still requires a separately committed
counterbalanced x64/x32 confirmation before adoption. Canonical
replacement additionally requires at least +0.5% over exact for
both inputs with no negative paired 95% interval.
