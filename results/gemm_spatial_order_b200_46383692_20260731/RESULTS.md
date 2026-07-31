# 16K Morton/Hilbert spatial-order result

## Outcome

Neither Morton nor any Hilbert orientation improves the canonical static
`8x16` M-fast scheduler. All space-filling orders are rejected; `direct`
remains the default.

Each cell is six independent one-case processes with one warmup and five timed
launches. The six candidates occupy every execution position exactly once.

### BF16 uniform `[0,1)`

| order | TFLOP/s mean +/- sample SD | paired vs direct | paired vs identity table |
|---|---:|---:|---:|
| direct | **1763.302 +/- 2.195** | reference | -0.2320% |
| table identity | **1767.402 +/- 1.119** | +0.2326% | reference |
| Morton | 1733.624 +/- 2.992 | -1.6830% | -1.9112% |
| Hilbert | 1756.737 +/- 1.320 | -0.3721% | -0.6034% |
| Hilbert transpose | 1757.060 +/- 1.156 | -0.3538% | -0.5851% |
| Hilbert reverse | 1758.338 +/- 2.497 | -0.2813% | -0.5128% |

### BF16 uniform `[-8,8)`

| order | TFLOP/s mean +/- sample SD | paired vs direct | paired vs identity table |
|---|---:|---:|---:|
| direct | **1555.416 +/- 2.864** | reference | -0.3352% |
| table identity | **1560.647 +/- 1.058** | +0.3366% | reference |
| Morton | 1530.025 +/- 1.776 | -1.6322% | -1.9621% |
| Hilbert | 1542.923 +/- 3.147 | -0.8028% | -1.1356% |
| Hilbert transpose | 1546.447 +/- 3.304 | -0.5761% | -0.9097% |
| Hilbert reverse | 1545.789 +/- 2.859 | -0.6189% | -0.9520% |

Paired 95% confidence intervals against `direct` exclude zero for every
spatial order:

| order | `[0,1)` paired 95% CI | `[-8,8)` paired 95% CI |
|---|---:|---:|
| Morton | [-1.8982%, -1.4677%] | [-1.8205%, -1.4439%] |
| Hilbert | [-0.5377%, -0.2066%] | [-1.1018%, -0.5039%] |
| Hilbert transpose | [-0.5198%, -0.1879%] | [-0.9293%, -0.2229%] |
| Hilbert reverse | [-0.5435%, -0.0191%] | [-0.6916%, -0.5462%] |

## Interpretation

The canonical `8x16` macro order was already L2-compact: its mean wave
footprint is 28.786 unique A+B panels, versus 29.286 for Hilbert and 33.714
for Morton. Hilbert reduces the worst wave from 35 to 32 panels and balances
A/B, but it changes the mean unique-panel split from `10.0 A + 18.8 B` to
about `14.6 A + 14.7 B`. Losing the canonical order's stronger A reuse is
more expensive than the balanced footprint is beneficial in this kernel.

`table_identity` is slightly faster than `direct`, but it is not a pure lookup
latency measurement: adding the constant-table path changes the compiler's
performance-kernel allocation from 164 to 162 registers and changes
instruction scheduling. All table candidates use the same 162-register,
zero-spill code shape, so their comparison against `table_identity` cleanly
isolates spatial order. Every Morton/Hilbert variant loses by at least 0.51%
on `[0,1)` and 0.91% on `[-8,8)` against that matched control.

The identity-table point estimate is below the predeclared +0.5% adoption
threshold and does not justify replacing the simpler direct decode.

## Conditions and validation

- B200 Vast.ai instance `46383692`, 1000 W power limit.
- CUDA 12.9, `sm_100a`; sampled graphics clock 1965 MHz.
- GPU temperature 28 C before and 30 C after collection.
- `M=N=K=16384`, BF16 A/B, FP32 accumulation/output.
- Canonical `256x256`, `3 x K64`, B-first pipeline; 148 persistent CTAs.
- All six binaries passed full-C 512 pattern and ones validation bit-exactly.
- Performance kernels have zero stack/local/spill. Direct uses 164 registers;
  all constant-table variants use 162.

Exact generated CUDA, map metrics, build/resource logs, validation logs, raw
CSV, and every process stdout are retained in this directory.
