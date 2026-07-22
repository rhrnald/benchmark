# Non-multicast GEMM L2 experiment plan

Last updated: 2026-07-22

This document is the live plan and result ledger for improving the 16K dense
end-to-end GEMM without TMA multicast.  Update the status and result tables at
each definition/result commit so the paper ablation remains reproducible.

## Fixed target and measurement protocol

- GPU: NVIDIA B200, 148 SMs, one resident `tcgen05` CTA per SM
- Problem: BF16 `A/B`, FP32 accumulation and complete FP32 `C`
- Primary size: `M=N=K=16384`
- CTA tile: `256x256`; K stage: 64; SMEM stages: 3
- A per stage: one `256x64` TMA, 32 KiB
- B per stage: two independent `64x128` TMAs, 16 KiB each
- Persistent workers: 148 unless the experiment explicitly studies 144
- Input: deterministic BF16 uniform `[0,1)`
- Timing: one process per case, warmup 1, five timed launches
- Statistics: three processes in forward/reverse/rotated order; report mean
  and sample standard deviation
- Correctness: CPU pattern reference with complete C output before timing
- Vast workflow: edit while stopped -> definition commit -> start -> upload ->
  validate/measure -> download -> stop -> result commit

The historical standardized non-multicast peak is `1806.657 TFLOP/s` in the
phase-control sweep (`1806.223 TFLOP/s` in the fixed-overhead sweep); matched
remeasurements have varied with host/instance state, so all decisions use
paired results from one sweep rather than that number as an absolute gate.

## Current selected baseline

- Dynamic global atomic work queue, one output tile per allocation
- 148 persistent CTAs
- `16x16` output-tile macro at 16K
- Local M-fast: consecutive tasks vary M and share B
- Macro N-fast: consecutive macros retain the M range and reuse A
- No multicast, no TMA L2 promotion, no cross-CTA phase shift
- Split B producer warps, effective pipe TMA/MMA phase `0/0`

For task `t` on the 64x64 output-tile grid:

```text
macro_id = t / 256
local    = t % 256
macro_n  = macro_id % 4
macro_m  = macro_id / 4
tile_m   = macro_m * 16 + local % 16
tile_n   = macro_n * 16 + local / 16
```

## Completed work that must not be repeated by default

| Direction | Result / decision |
|---|---|
| Persistent versus normal | Persistent selected |
| Dynamic versus static grid-stride | Dynamic selected; static was `-0.44%` at 16K and much worse at 32K |
| Local and macro M/N order | Local M-fast plus macro N-fast selected |
| Macro shapes / worker counts | Broad sweeps completed; 16x16 and 148 workers selected at 16K |
| K stage / depth | K64/S3 selected; K32/S4/S5 slower; K128 prevents useful multistage buffering for 256x256 |
| Wide B | One `64x256` TMA was about 3% slower than split `64x128` streams |
| TMA L2 promotion | 128B/256B promotion was neutral or negative; this is not an eviction-priority test |
| B multicast | Useful traffic reduction, but did not beat peak non-multicast at 16K |
| A multicast | Slower than B multicast |
| Cluster 4 | Large regression |
| Dedicated epilogue warpgroup | Regression |
| Simple N-first/global-wave ordering | Regression |
| Cross-CTA phase shift | Explicitly excluded from the current plan |
| Nsight Compute counters | Unavailable on the rented host due to permission |

## Experiment A: current-source order baseline

Purpose: produce a paper-quality, same-run four-way order ablation on the
latest source before introducing new ownership logic.

| Variant | Local order | Macro order | Status |
|---|---|---|---|
| `order_mn` | M-fast | N-fast | pending |
| `order_mm` | M-fast | M-fast | pending |
| `order_nn` | N-fast | N-fast | pending |
| `order_nm` | N-fast | M-fast | pending |

Expected selection is `order_mn`; this experiment is primarily a controlled
reproduction, not a new broad search.

## Experiment B: macro-row snake

Purpose: remove the large N-address jump between adjacent M-macro rows without
changing the successful local M-fast grouping.  This is cheap but low
priority: only three macro-row wrap boundaries exist in the full 16K task
stream, so a large effect is unlikely.

| Variant | Mapping | Status |
|---|---|---|
| `snake_0` | Existing monotonic macro N and local N | pending |
| `snake_1` | Reverse macro-N order on odd macro-M rows | pending |
| `snake_2` | Reverse both macro N and local N on odd macro-M rows | pending |

At 16K, full snake changes the boundary from `(M15,N63)->(M16,N0)` to
`(M15,N63)->(M16,N63)`, preserving the exact B tile across that boundary.
Snake is a permutation only: it must not change task count or arithmetic.

Validation requirements:

- Existing 512 pattern validation for numerical correctness
- Host-side mapping test on a 64x64 tile grid: exactly 4096 unique in-range
  coordinates and the expected snake boundary coordinates

## Experiment C: dynamic N-strip ownership (primary new hypothesis)

Purpose: make one dynamically allocated work item contain several consecutive
N output tiles with fixed M.  This preserves the current M-fast cohort's B
sharing while letting the same persistent CTA/SM immediately reuse A.

For strip length `S`, one work unit maps to `(tile_m, strip_n)` and the owner
computes:

```text
(tile_m, strip_n*S + 0)
(tile_m, strip_n*S + 1)
...
(tile_m, strip_n*S + S-1)
```

Work units remain M-fast for each N strip, so approximately 16 contemporaneous
workers advance through the strip together and continue sharing B.

| Variant | S | Work units at 16K | Status |
|---|---:|---:|---|
| `strip_1` | 1 | 4096 | pending baseline |
| `strip_2` | 2 | 2048 | pending |
| `strip_4` | 4 | 1024 | pending |
| `strip_8` | 8 | 512 | deferred tail diagnostic |

The primary comparison is S=1/2/4.  S=8 has only 512 work items: its last wave
contains 68 of 148 workers and ideal aggregate worker fill falls to 86.5%.
Do not test S=16 initially.

Important limitation: the CTA completes all K=16384 work for one output tile
before moving to the next strip element.  It therefore re-reads the same
8-MiB logical A tile rather than retaining a 32-KiB K64 panel in SMEM.  This is
an L2/ownership diagnostic, not direct K64-stage reuse, and the partial-repeat
ceiling (which repeats one K64 address for every K stage) does not predict its
gain.

Implementation invariants:

- Preserve one TMEM allocation and persistent kernel context
- Preserve monotonically increasing tile epochs for all mbarrier phases
- Complete each tile's FP32 C store before reusing its accumulator bank unless
  an already-validated epilogue overlap path is selected
- Map every output tile exactly once, including clipped edge strips
- Keep `S=1` algebraically equivalent to the current dynamic baseline

## Experiment D: fixed-ownership non-multicast control (low priority)

The existing static explicit 16x9/12x12 wave already preserves each CTA's
local-M coordinate as the wave ID advances.  It regressed badly with
multicast, so a broad cohort sweep is redundant.  Run at most one non-
multicast 16x9 control only if strip results make multicast interaction worth
isolating.

| Candidate | Workers | Comparison required | Status |
|---|---:|---|---|
| non-multicast explicit `16x9` | 144 | dynamic 144 and dynamic 148 | conditional |

## Experiment E: K-outer two-output A reuse (major conditional change)

If S=2/4 do not improve, scheduler-only A reuse is exhausted.  The stronger
design is to keep one `256x64` A stage resident and apply it to two N-output
accumulator contexts before advancing K.  This directly halves A TMA traffic
for the paired outputs, but it is a mainloop/TMEM/epilogue redesign rather than
a tile scheduler change.  Before implementation, audit TMEM capacity: the
current 256x256 output already occupies four 128x128 accumulator regions, so
two simultaneous outputs may require a different CTA shape, staged spill, or
sequential accumulator strategy.

## Decision gates

1. Always validate before timing; discard timing from a failing binary.
2. Select a candidate only if all three process samples move in the same
   direction and the mean gain is at least approximately 0.5% relative to the
   paired baseline.
3. A marginal result must be rerun before selection.
4. Keep baseline behavior as the compile-time default until a winner passes.
5. Extend only the winning 16K candidate to 8K/32K; do not spend GPU time on
   a full cross-product at all sizes.

## Execution ledger

| Date | Definition commit | Experiment | Validation | Result | Decision |
|---|---|---|---|---|---|
| 2026-07-22 | pending | A/B/C S=1/2/4 combined first sweep | pending | pending | pending |

## Deferred / out of scope

- Cross-CTA phase shifting
- More multicast, cluster-4, or wide-B variants
- More K32/K128 or L2-promotion sweeps
- Morton/Z-order and exhaustive group-size search
- Broad fixed-wave/cohort sweep; the ownership behavior was already tested
- Stream-K/Split-K
- Decode/asymmetric GEMMs and non-multiple edge support; these require a
  broader kernel/interface generalization than the current square-16K target
