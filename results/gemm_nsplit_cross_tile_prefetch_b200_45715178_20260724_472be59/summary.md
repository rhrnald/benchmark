# N-split cross-tile K0 prefetch B200 result

Dense 16K BF16-input/FP32-output GEMM. Each sample is one W1/I5
process. All six orders of the three variants were collected per
input; each variant occupied each position twice and every ordered
non-self adjacent pair occurred twice within those orders.

A is the audited scalar-x64 control. B acquires the next task early
but issues its normal-stage K0 prefetch only after the current
epilogue. C moves that same issue between the first C-store group's
commit and wait inside the current epilogue.

## Input `random`

| variant | mean +/- sample SD TFLOP/s | vs non-overlap B, paired 95% CI | vs scalar-x64 A, paired 95% CI | mean event ms |
|---|---:|---:|---:|---:|
| nsplit_transpose_scalar_x64 | 1743.253 +/- 2.047 | -0.2630% [-0.4476,-0.0783] | +0.0000% [+0.0000,+0.0000] | 5.045799 |
| nsplit_prefetch_nonoverlap | 1747.851 +/- 1.290 | +0.0000% [+0.0000,+0.0000] | +0.2639% [+0.0786,+0.4492] | 5.032522 |
| nsplit_prefetch_overlap | 1748.909 +/- 0.882 | +0.0606% [-0.0120,+0.1332] | +0.3246% [+0.1911,+0.4581] | 5.029474 |

Diagnostic B versus A structural change:
`+0.2639% [+0.0786,+0.4492]`.
C overlap minus B non-overlap event time: `-0.003048 +/- 0.003478 ms`, paired 95% CI `[-0.006697,+0.000602] ms`.
C overlap minus A scalar-x64 event time: `-0.016324 +/- 0.006394 ms`, paired 95% CI `[-0.023035,-0.009614] ms`.

## Input `random-signed8`

| variant | mean +/- sample SD TFLOP/s | vs non-overlap B, paired 95% CI | vs scalar-x64 A, paired 95% CI | mean event ms |
|---|---:|---:|---:|---:|
| nsplit_transpose_scalar_x64 | 1558.528 +/- 2.539 | -0.2482% [-0.5643,+0.0678] | +0.0000% [+0.0000,+0.0000] | 5.643857 |
| nsplit_prefetch_nonoverlap | 1562.414 +/- 3.073 | +0.0000% [+0.0000,+0.0000] | +0.2496% [-0.0682,+0.5674] | 5.629829 |
| nsplit_prefetch_overlap | 1560.297 +/- 3.645 | -0.1353% [-0.4209,+0.1504] | +0.1137% [-0.1778,+0.4051] | 5.637475 |

Diagnostic B versus A structural change:
`+0.2496% [-0.0682,+0.5674]`.
C overlap minus B non-overlap event time: `+0.007646 +/- 0.015324 ms`, paired 95% CI `[-0.008435,+0.023728] ms`.
C overlap minus A scalar-x64 event time: `-0.006382 +/- 0.015633 ms`, paired 95% CI `[-0.022789,+0.010024] ms`.

## Decision

The overlap candidate does not pass the screening gate.

- Primary attribution requires the C-versus-B paired 95% CI
  lower bound to be above zero for both inputs.
- Net safety requires the C-versus-A paired 95% CI lower bound
  to be nonnegative for both inputs.

B-versus-A is diagnostic: it measures the early-task, rotating
C-buffer, and split-K0 structural cost without epilogue overlap.
A passing result advances to a separately committed same-session
confirmation. This three-way screen cannot by itself replace the
direct-exact canonical kernel; canonical replacement still
requires at least +0.5% over direct exact for both inputs with
positive paired 95% confidence-interval lower bounds.
