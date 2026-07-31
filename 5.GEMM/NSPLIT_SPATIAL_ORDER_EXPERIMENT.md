# N-split Morton/Hilbert spatial-order experiment

## Question

Can a space-filling output-tile order reduce the simultaneous A/B panel
footprint enough to improve L2 reuse over the canonical static `8x16` M-fast
order at 16K?

The arithmetic, `3 x K64` pipeline, TMA layout, B-first consumer wait,
epilogue, 148 persistent CTAs, and static grid-stride ownership remain fixed.
Only the mapping from a static task position to `(tile_m, tile_n)` changes.

## Candidates

| candidate | definition |
|---|---|
| `direct` | canonical static `8x16` M-fast, no lookup |
| `table_identity` | canonical mapping through the same constant-table lookup |
| `morton` | 64x64 Morton/Z-order |
| `hilbert` | 64x64 Hilbert order |
| `hilbert_transpose` | Hilbert with M/N coordinates swapped |
| `hilbert_reverse` | Hilbert distance traversed in reverse |

Each generated map is a compile-time `4096 x uint16_t` constant table. The
packed value is a canonical linear task index, so the kernel's existing
coordinate decode remains unchanged. `table_identity` isolates lookup cost.

## Offline gate

All five tables are exact permutations of the 4096 output tiles. A wave is a
contiguous group of at most 148 static task positions, matching the launched
CTA count.

| order | mean unique A | mean unique B | mean A+B | max A+B |
|---|---:|---:|---:|---:|
| canonical / identity | 10.000 | 18.786 | **28.786** | 35 |
| Morton | 19.500 | 14.214 | 33.714 | 40 |
| Hilbert | 14.571 | 14.714 | 29.286 | **32** |
| Hilbert transpose | 14.714 | 14.571 | 29.286 | **32** |
| Hilbert reverse | 14.571 | 14.714 | 29.286 | **32** |

The canonical macro order is already compact and has the lowest mean panel
count. Hilbert remains worth measuring because it balances A/B pressure and
reduces the worst wave footprint; Morton is retained as a locality control.

## Measurement protocol

- NVIDIA B200, CUDA 12.9, `sm_100a`.
- BF16 `A/B`, FP32 accumulation and output, `M=N=K=16384`.
- Uniform `[0,1)` and `[-8,8)` inputs.
- One case per process, one warmup, five timed launches.
- Six cyclic process passes per input: every candidate occupies each of the
  six execution positions exactly once.
- Full-C 512 validation with both pattern and ones inputs for every binary.
- Compare paired per-pass deltas against `direct`; the table lookup control
  separates mapping benefit from lookup overhead.

## Decision rule

Promote only a candidate that improves both distributions by at least 0.5%
against `direct` without correctness or code-generation regressions. Results
below that threshold are treated as noise or an insufficient engineering
tradeoff, even if their point estimate is positive.

## Outcome

All Morton/Hilbert candidates regressed and are rejected. Against the direct
canonical decode, Hilbert variants lost 0.28--0.37% on `[0,1)` and 0.58--0.80%
on `[-8,8)`. Against the matched identity-table code shape, their losses were
0.51--0.60% and 0.91--1.14%, respectively. The direct static `8x16` M-fast
order remains canonical.

Full results and raw artifacts:
[`RESULTS.md`](../results/gemm_spatial_order_b200_46383692_20260731/RESULTS.md).
