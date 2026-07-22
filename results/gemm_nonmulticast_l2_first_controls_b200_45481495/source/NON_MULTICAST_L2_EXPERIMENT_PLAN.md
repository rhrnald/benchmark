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
| Local and macro M/N order | Keep local M-fast plus macro N-fast; the opposite cross-order gained `+0.559%` in round one but only `+0.077%` in the focused confirmation and failed its gate |
| Macro shapes / worker counts | Broad sweeps completed; 16x16 and 148 workers selected at 16K |
| K stage / depth | K64/S3 selected; K32/S4/S5 was slower in the multicast path; K128 multistage infeasibility for 256x256 follows from SMEM capacity |
| Wide B | One `64x256` TMA was about 3% slower than split `64x128` streams |
| TMA L2 promotion | 128B/256B promotion was neutral or negative; this is not an eviction-priority test |
| TMA L2 eviction priority | A `evict_last` was stable but only `+0.113%`; asymmetric `evict_first` controls regressed, so keep no hint |
| B multicast | Useful traffic reduction, but did not beat peak non-multicast at 16K |
| A multicast | Slower than B multicast |
| Cluster 4 | Large regression |
| Dedicated epilogue warpgroup | Regression in the 128x256 path; 256x256 already occupies all four 128x128 TMEM regions |
| Simple N-first/global-wave ordering | Regression |
| Cross-CTA phase shift | Explicitly excluded from the current plan |
| Nsight Compute counters | Unavailable on the rented host due to permission |

## Experiment A: current-source order baseline

Purpose: produce a paper-quality, same-run four-way order ablation on the
latest source before introducing new ownership logic.

| Variant | Local order | Macro order | Status |
|---|---|---|---|
| `order_mn` | M-fast | N-fast | complete: `1769.017 +/- 1.387`, paired baseline |
| `order_mm` | M-fast | M-fast | complete: `1740.880 +/- 0.838` (`-1.590%`) |
| `order_nn` | N-fast | N-fast | complete: `1756.834 +/- 1.140` (`-0.689%`) |
| `order_nm` | N-fast | M-fast | round-one candidate: `+0.559%`; confirmation rejected promotion |

Contrary to the initial expectation, `order_nm` won all three paired passes
by `+0.400%`, `+0.747%`, and `+0.530%`.  The gain is just above the selection
threshold, so a focused AB/BA confirmation is required before changing the
default.

## Experiment B: macro-row snake

Purpose: remove the large N-address jump between adjacent M-macro rows without
changing the successful local M-fast grouping.  This is cheap but low
priority: only three macro-row wrap boundaries exist in the full 16K task
stream, so a large effect is unlikely.

| Variant | Mapping | Status |
|---|---|---|
| `snake_0` | Existing monotonic macro N and local N | complete: aliases `order_mn` |
| `snake_1` | Reverse only macro-N order; incomplete/likely-negative decomposition control | reject: `+0.085%` |
| `snake_2` | Reverse both macro N and local N on odd macro-M rows | reject: `+0.136%` |

At 16K, full snake changes the boundary from `(M15,N63)->(M16,N0)` to
`(M15,N63)->(M16,N63)`, preserving the exact B tile across that boundary.
Snake is a permutation only: it must not change task count or arithmetic.

Validation requirements:

- Existing 512 pattern validation for numerical correctness
- Host-side mapping test on a 64x64 tile grid: exactly 4096 unique in-range
  coordinates and the expected snake boundary coordinates

The 512 numerical test clips the macro to 2x2 and has only one macro group, so
it does not execute the odd-row snake branch.  Scheduler permutation/coverage
is established by the host mapping test; GPU validation establishes that the
unchanged tile math and store remain correct.

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
| `strip_1` | 1 | 4096 | complete: aliases `order_mn` |
| `strip_2` | 2 | 2048 | reject: `1764.801 +/- 1.567` (`-0.238%`) |
| `strip_4` | 4 | 1024 | reject: `1747.406 +/- 2.533` (`-1.222%`) |
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

## Experiment E: K-outer two-output A reuse (closed by capacity audit)

The selected kernel already applies one `256x64` A stage to both `256x128`
N halves of its `256x256` output.  Extending that reuse to the next scheduler
tile would require two live `256x256` FP32 outputs: eight `128x128` accumulator
regions, or 1024 TMEM columns.  The current output occupies four regions at
TMEM offsets `0/128/256/384` and allocates all 512 columns.  SM100 exposes only
512 columns x 128 lanes x 32 bits per CTA, so the literal design is impossible
without spilling.  See the [NVIDIA PTX tensor-memory
layout](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-memory).

SMEM cannot hold the additional 256-KiB FP32 tile.  A K64-by-K64 global spill
would add about 128 MiB of partial-C traffic per output pair to save only 8 MiB
of A requests.  Sequential full-output execution is the rejected N-strip
experiment and does not retain a K64 panel.  The exact multi-CTA realization,
two-CTA A multicast, already measured `1730.416 +/- 0.424 TFLOP/s` and lost
2.993% to the selected B-multicast control.

Decision: no GPU implementation for a single-CTA `256x512` FP32 accumulator.
A `128x512` K64/S2 CTA or a matched S2 shared-A/duplicate-A microbenchmark is
possible, but has only 102.4 FLOP/requested byte versus 128 for the selected
`256x256`; keep it as a paper-causality diagnostic, not a throughput candidate.

## Experiment F: focused order confirmation (complete)

Rerun only the existing default (`order_mn`, local M-fast plus macro N-fast)
and the round-one candidate (`order_nm`, local N-fast plus macro M-fast).
Use six independent one-case processes per binary in counterbalanced AB/BA order,
with warmup 1 and five timed launches in each process.  The source and kernel
work must remain identical to definition `f669b6f`; only the two scheduler
compile-time constants differ.

Promote `order_nm` only if all six matched pairs are positive, the aggregate
paired gain is at least 0.5%, both AB-first and BA-first subgroup means are
positive, and the two-sided paired 95% confidence interval excludes zero.
Otherwise retain the current default.  Treat each process-level W1/I5 result
as one sample, not its five timed launches as five independent observations.
This confirmation was required before either macro-shape retuning or the
K-outer redesign.

Result: complete.  The six paired changes were `+0.494%`, `+0.215%`,
`+0.375%`, `-1.654%`, `+0.469%`, and `+0.565%`.  Their mean was `+0.077%`
with sample SD `0.857%` and paired 95% interval `[-0.822%, +0.976%]`.
AB-first and BA-first subgroup means were `-0.230%` and `+0.385%`.  The
candidate failed the all-positive, 0.5%, both-subgroup, and confidence-interval
gates.  Keep `order_mn`; do not retune macro shapes for `order_nm`.

## Experiment G: TMA L2 eviction-priority hints (complete)

Tensor-map L2 promotion widens the DRAM-to-L2 fill granularity; it does not set
eviction priority.  PTX independently supports
`cp.async.bulk.tensor...L2::cache_hint` with an opaque policy produced by
`createpolicy`.  Add compile-time non-multicast policies only:

```text
0 = no cache hint (byte-for-byte baseline instruction)
1 = L2::evict_last
2 = L2::evict_first
```

Compare at 16K under the selected `order_mn` scheduler:

| Variant | A policy | B policy | Purpose | Status |
|---|---|---|---|---|
| `baseline` | none | none | paired control | complete: `1769.197 +/- 1.322` |
| `a_last` | evict_last | none | favor the operand with the larger partial-reuse gap | reject: `1771.192 +/- 1.539`, `+0.113%` |
| `b_last` | none | evict_last | direction control | reject: `1769.702 +/- 1.211`, `+0.029%` |
| `a_last_b_first` | evict_last | evict_first | strongest A-priority separation | reject: `1743.439 +/- 1.467`, `-1.456%` |
| `a_first_b_last` | evict_first | evict_last | symmetric B-priority control | reject: `1423.287 +/- 6.930`, `-19.552%` |

Keep tensor-map promotion disabled, logical TMA issue count/payload unchanged,
and do not apply the policy to C stores.  Use the standard three rotated W1/I5
process passes and exact 512 validation.  The policy is a hint and may be
ignored by hardware; without profiler permission, report L2/HBM bytes as
unmeasured.  Select only a three-for-three gain of at least 0.5%, then confirm
that candidate separately before changing the default.

Result: all five 512 pattern checks were bit-exact and all three `a_last`
samples were positive (`+0.128%`, `+0.090%`, and `+0.120%`), but the paired
mean was only `+0.113%`.  It fails the minimum-effect gate, so no focused
confirmation or 8K/32K extension is scheduled.  The much larger conditional
regression from adding A `evict_first` with B fixed at `evict_last` than from
adding B `evict_first` with A fixed at `evict_last` is consistent with A
residency being more important under this traversal.  These are not
unconditional single-factor effects, and L2/DRAM counters were unavailable;
record the result as a throughput-based inference, not measured traffic.

## Experiment H: isolated `evict_first` controls (next)

Experiment G's negative symmetric controls are conditional contrasts: adding
A `evict_first` while B is fixed at `evict_last` cost `19.575%`, whereas
adding B `evict_first` while A is fixed at `evict_last` cost `1.567%`.
Because both symmetric endpoints change both operands, they do not establish
the unconditional single-factor effect.  For a paper-quality ablation, run
only these three cases:

| Variant | A policy | B policy | Status |
|---|---|---|---|
| `baseline` | none | none | pending |
| `a_first` | evict_first | none | pending |
| `b_first` | none | evict_first | pending |

Keep every other compile flag and logical memory request identical to
Experiment G.  Use three cyclic Latin-order process passes so every variant
occupies each sequence position exactly once, with W1/I5 and exact 512
validation.  This is a causal diagnostic only: none of these policies can be
promoted over the already-rejected positive-hint candidates.  Do not extend it
to other sizes or a full policy cross-product.

## Decision gates

1. Always validate before timing; discard timing from a failing binary.
2. In a three-process exploratory sweep, select a candidate only if all three
   paired samples move in the same direction and the mean gain is at least
   approximately 0.5% relative to the paired baseline.
3. A marginal result must be rerun before selection using its declared focused
   confirmation gate; Experiment F uses six pairs and a paired confidence
   interval.
4. Keep baseline behavior as the compile-time default until a winner passes.
5. Extend only the winning 16K candidate to 8K/32K; do not spend GPU time on
   a full cross-product at all sizes.

Snake and strip do not change logical TMA count or requested A/B payload
(`16 MiB` per 16K output tile before cache reuse).  Without profiler
permission, report L2/HBM traffic as unmeasured and do not infer byte savings
from TFLOP/s alone.

## Execution ledger

| Date | Definition commit | Experiment | Validation | Result | Decision |
|---|---|---|---|---|---|
| 2026-07-22 | `f669b6f` | A/B/C S=1/2/4 combined first sweep | all 8 GPU checks exact; host coverage passed | `order_nm` `1778.907 +/- 2.584`, `+0.559%` | confirm `order_nm`; reject snake and strip 2/4 |
| 2026-07-22 | `ac71b14` | F: six-pair `order_mn`/`order_nm` confirmation | both GPU checks exact; expanded host coverage passed | paired `+0.077%`, 95% CI `[-0.822%, +0.976%]` | reject promotion; keep `order_mn` |
| 2026-07-22 | analysis at `2dfb872` | E: K-outer two-full-output capacity audit | source/PTX resource proof; no GPU run | 1024 TMEM columns required, 512 available | close as infeasible; no GPU spend |
| 2026-07-22 | `1bb691b` | G: A/B TMA eviction-priority sweep | all 5 GPU checks exact; host coverage passed | `a_last` `1771.192 +/- 1.539`, paired `+0.113%`; negative controls `-1.456%`/`-19.552%` | reject promotion; keep no hint |
| 2026-07-22 | pending definition commit | H: isolated A/B `evict_first` controls | pending | pending | pending |

Round-one artifacts are in
`../results/gemm_nonmulticast_l2_round1_b200_45481495/`.  The requested TMA
payload is invariant; hardware L2/DRAM traffic remains unmeasured because
performance-counter permission is unavailable.

Focused confirmation artifacts are in
`../results/gemm_nonmulticast_order_confirm_b200_45481495/`.

Eviction-priority artifacts are in
`../results/gemm_nonmulticast_l2_evict_b200_45481495/`.

## Deferred / out of scope

- Cross-CTA phase shifting
- More multicast, cluster-4, or wide-B variants
- More K32/K128 or L2-promotion sweeps
- More performance-oriented TMA eviction-policy sweeps; the best policy missed
  the effect-size gate
- Morton/Z-order and exhaustive group-size search
- Broad fixed-wave/cohort sweep; the ownership behavior was already tested
- Stream-K/Split-K
- Single-CTA `256x512` FP32 accumulation; exceeds TMEM capacity
- `128x512` and S2 duplicate/shared-A causality kernels unless needed for a
  paper-only mechanism ablation
- Decode/asymmetric GEMMs and non-multiple edge support; these require a
  broader kernel/interface generalization than the current square-16K target
