# N-split transpose-compute B200 result

Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5
process; four Latin-rotated passes were collected.

## Input `random`

| variant | TFLOP/s samples | mean +/- sample SD | paired vs mode control | mean event ms |
|---|---|---:|---:|---:|
| nsplit_exact | 1798.758, 1803.249, 1800.605, 1796.862 | 1799.869 +/- 2.723 | +0.0000% +/- 0.0000%p | 4.887083 |
| nsplit_transpose_scalar | 1810.501, 1806.864, 1808.921, 1806.413 | 1808.175 +/- 1.897 | +0.4617% +/- 0.1912%p | 4.864629 |
| nsplit_exact_nostore | 1865.246, 1865.010, 1869.109, 1865.446 | 1866.203 +/- 1.946 | +0.0000% +/- 0.0000%p | 4.713368 |
| nsplit_transpose_nostore | 1885.647, 1884.168, 1881.027, 1880.863 | 1882.926 +/- 2.367 | +0.8963% +/- 0.2065%p | 4.671507 |

| derived per-launch time | mean +/- sample SD |
|---|---:|
| exact epilogue (`E2E - no-store`) | 0.173715 +/- 0.008490 ms |
| scalar-transpose epilogue (`E2E - no-store`) | 0.193122 +/- 0.005455 ms |
| incremental transpose epilogue | +0.019407 +/- 0.013445 ms |

## Input `random-signed8`

| variant | TFLOP/s samples | mean +/- sample SD | paired vs mode control | mean event ms |
|---|---|---:|---:|---:|
| nsplit_exact | 1599.556, 1597.799, 1597.713, 1603.312 | 1599.595 +/- 2.619 | +0.0000% +/- 0.0000%p | 5.498962 |
| nsplit_transpose_scalar | 1598.761, 1600.023, 1599.305, 1595.199 | 1598.322 +/- 2.145 | -0.0792% +/- 0.2959%p | 5.503338 |
| nsplit_exact_nostore | 1662.636, 1669.409, 1657.930, 1660.721 | 1662.674 +/- 4.888 | +0.0000% +/- 0.0000%p | 5.290363 |
| nsplit_transpose_nostore | 1672.544, 1676.686, 1666.870, 1666.001 | 1670.525 +/- 5.029 | +0.4722% +/- 0.1224%p | 5.265501 |

| derived per-launch time | mean +/- sample SD |
|---|---:|
| exact epilogue (`E2E - no-store`) | 0.208598 +/- 0.019938 ms |
| scalar-transpose epilogue (`E2E - no-store`) | 0.237837 +/- 0.012121 ms |
| incremental transpose epilogue | +0.029239 +/- 0.012886 ms |

## Decision rule

- Adopt only if `nsplit_transpose_scalar` improves pass-matched
  `nsplit_exact` by at least 0.5% for both inputs.
- If no-store improves but E2E does not, retain only the
  transpose mapping as an epilogue-optimization candidate.
