# 5.GEMM

Prototype Blackwell GEMM compute benchmark using TMA loads and `tcgen05.mma`.

Current kernel shape:

- One CTA computes one logical `256 x 256` C tile.
- The tile is split into four `128 x 128` accumulator tiles in TMEM.
- Each K stage loads:
  - A: `256 x 64` BF16 through TMA.
  - B: two independent `64 x 128` BF16 pipe tiles through TMA.
- Shared memory is partitioned as three `64 KiB` stages. The TMA C-store
  epilogue reuses this dynamic shared-memory region after the mainloop:
  - `A_stage`: `256 x 64 x 2B = 32 KiB`
  - `B_stage`: two `64 x 128 x 2B = 16 KiB` pipe buffers
  - total triple buffer: `192 KiB`
  - C TMA store staging: two `128 x 128` FP32 chunks (`2 x 64 KiB`),
    overlaid on the triple-buffer storage. The 8K-tuned C-store path uses a
    4D `SWIZZLE_128B` TMA store descriptor and swizzled shared-memory staging
    to reduce bank conflicts. The experimental
    `GEMM_CSTORE_CHUNK_N=256` path is not currently valid: its 512 reference
    test faults, so measurements and default builds keep `128`.
- The `K=64` stage is issued as four `K=16` `tcgen05.mma` slices for each
  `128 x 128` accumulator tile.
- The mainloop uses two N-direction MMA pipes:
  - warp 0 lane 0 issues shared A TMA plus B pipe 0.
  - warp 1 lane 0 issues B pipe 1.
  - warp 2 lane 0 issues MMA for C00/C10.
  - warp 3 lane 0 issues MMA for C01/C11.
  - stage reuse is fenced by each pipe's `mma_done` barrier from three K stages earlier.
- The default dense kernel launches 148 persistent CTAs. Its macroblocks are
  `16 x 16` at 8K/16K and `8 x 18` at 32K, with local M-fast and macro N-fast
  traversal. A/B TMA promotion is disabled and the effective pipe-1 TMA/MMA
  phase is `0/0` at all three target sizes. Generic phase constants printed in
  the configuration header are fallback values, not the selected dense
  template values; the per-case `pipe1_tma_phase` and `pipe1_mma_phase` fields
  report the effective setting.
- A and B global inputs are row-major packed BF16. TMA uses `SWIZZLE_128B`
  layouts matching the attention path:
  - A is loaded as one logical `256 x 64` row-major tile into `major_k`
    shared-memory layout.
  - B is loaded as two `64 x 128` row-major halves into `major_mn`
    shared-memory layout.

The default benchmark path consumes TMEM accumulators into a checksum sink.
`--store-c` stores the full FP32 C matrix with scalar global stores, and
`--store-c-tma` stages FP32 C chunks through shared memory and stores them with
TMA. The validation path compares the stored FP32 C matrix against a CPU
reference.

## Paper-facing benchmark timeline

The experiments fall into four different levels.  They must not be combined
as if they were measurements of the same workload:

1. MMA-only issue/completion microbenchmarks;
2. repeated-address TMA-to-MMA dependency microbenchmarks;
3. numerically validated GEMMs that repeatedly load one global A/B tile; and
4. dense end-to-end GEMMs that load the real A/B coordinates and store all of
   FP32 C.

"Repeat" below means that a large inner loop repeatedly operates on a tiny,
L2-resident global working set.  It is useful for isolating the compute and
pipeline ceilings, but it is not a dense square GEMM.

### 1. Long-repeat microbenchmark progression

| date | step | input and synchronization | issuer warps | result | interpretation |
|---|---|---|---:|---:|---|
| 2026-07-17 | MMA only | constant BF16 `1.0`; no TMA or per-step commit/wait | 2 | **2229.664 TFLOP/s** | low-switching upper control |
| 2026-07-17 | MMA only | BF16 uniform `[0,1)`; no TMA or per-step commit/wait | 2 | **1928.151 TFLOP/s** | PyTorch-`rand`-like data-activity control |
| 2026-07-18 | dependent TMA+MMA, depth 3 | same random `[0,1)` A/B panel; `wait A -> wait B -> MMA x8 -> commit` | 1 | **1263.276 TFLOP/s** | one-issuer saturation diagnostic |
| 2026-07-18 | dependent TMA+MMA, depth 3 | same random `[0,1)` A/B panel; the same wait/commit pipeline | 2 | **1800.166 TFLOP/s** | aggregate two-issuer dependency ceiling |
| 2026-07-18 | validated repeated-tile GEMM | random `[0,1)`, TMA wait/commit, full FP32 C store; CTA `128x256`, K stage 128 | 2 | **1797.227 TFLOP/s** | implicit tiled GEMM, all CTAs reload one A/B tile |
| 2026-07-18 | validated repeated-tile GEMM | random `[0,1)`, TMA wait/commit, full FP32 C store; CTA `256x256`, K stage 64 | 2 | **1756.468 TFLOP/s** | shape adopted by the dense kernel |

The first two rows are a controlled input-distribution comparison: 592 CTAs,
`N=32768`/`2N=65536` differential timing, warmup 5, 20 timed launches, and
three mixed-order process runs.  Constant to `[0,1)` changes throughput by
`-13.52%` even though the measured issue interval remains 127.997 cycles/MMA.
This is the cleanest evidence that operand activity changes the power/clock
ceiling.

The dependency sweep used 592 CTAs, `N=8192`/`2N=16384`, warmup 3, 10 timed
launches, three shared-memory stages, and the same global source panel for all
CTAs and stages.  Its full depth ablation was:

| pipeline depth | 1 issuer | 2 issuers |
|---:|---:|---:|
| 1 | 653.192 | 1012.511 |
| 2 | 1219.755 | 1658.831 |
| 3 | **1263.276** | **1800.166** |

All entries are differential TFLOP/s.  The issuer performs no completion wait
after commit; the producer waits for `stage_done` only before reusing that
shared-memory stage.  A final drain protects TMEM teardown.

The existing 1-versus-2-issuer rows are **not yet a publication-quality
fixed-work ablation**.  One issuer performs eight `m128n128k16` MMAs per stage,
whereas two issuers perform sixteen total MMAs into independent accumulators.
Consequently both total FLOPs and arithmetic intensity change.  A strict
paper ablation must keep sixteen MMAs, the same A/B loads, TMEM destinations,
CTA count, and stage count fixed, changing only whether one warp issues all
sixteen or two warps issue eight each.  That unified rerun is still pending.

For a publication-quality ablation, rerun a single source/binary family in
the following order.  Each row changes only the named factor relative to the
previous row:

| ablation row | invariant work | only changed factor | historical reference | publication status |
|---|---|---|---:|---|
| A0 | 16 MMA/stage, no TMA, two issuers | constant `1.0` input | 2229.664 | historical protocol |
| A1 | same as A0 | input becomes uniform `[0,1)` | 1928.151 | historical controlled pair |
| A2 | same 16 MMA and random operands | add A/B TMA readiness waits, commit, and producer-side reuse wait | 1800.166 | needs unified `1/5` rerun against A1 |
| A3 | same 16 MMA, TMA bytes, and output accumulators | one issuer issues 16 versus two issuers issuing 8+8 | unavailable | implementation and measurement required |
| A4 | same repeated-input `256x256x64` mainloop | enable full FP32 C epilogue/store | 1756.468 with store | no matching store-off run yet |
| A5 | same complete GEMM kernel | repeated A/B coordinates become real dense coordinates | size-dependent | rerun current kernel in matched repeat/dense modes |
| A6 | same dense GEMM | raster, macroblock swizzle, then persistent CTA | size-dependent | rerun each scheduler under `1/5` |

This table intentionally leaves missing cells unavailable instead of filling
them from a differently shaped kernel.  In particular, `1756.468` is the
two-issuer validated `256x256x64` repeated-tile GEMM; it is not the two-issuer
member of the old issuer-count sweep.

### 2. Transition to dense end-to-end GEMM

| date | implementation milestone | 8K | 16K | 32K | protocol/status |
|---|---|---:|---:|---:|---|
| 2026-07-19 | first dense `256x256x64`, real A/B addresses | 1205.758 | 1356.429 | 1186.012 | three process means; pre-L2 scheduling baseline |
| 2026-07-20 | grouped dense, issuer completion wait after every commit | 1447.835 | 1289.700 | 1094.726 | superseded; accidentally serialized the triple buffer |
| 2026-07-20 | first persistent implementation with the same bug | 843.689 | 778.423 | 774.284 | invalid as a persistent-design conclusion |
| 2026-07-20 | per-stage async barriers, normal grid | 1674.335 | 1757.211 | 1512.945 | full FP32 C store, six paired passes |
| 2026-07-20 | async barriers + 148 persistent CTAs | **1697.631** | **1777.128** | **1615.074** | full dense end-to-end GEMM, six paired passes |
| 2026-07-21 | dense L2 scheduler retuning, persistent | 1674.883 | 1740.092 | 1590.876 | different B200 instance; tuning by paired comparison |
| 2026-07-21 | short-window confirmation, persistent `[0,1)` | 1713.905 | 1777.455 | 1619.831 | warmup 3, 6 timed, three one-case processes |

All values in the dense table are TFLOP/s for square BF16 A/B, FP32
accumulation and FP32 C.  CUDA-event timing covers the complete kernel,
including the TMA C-store, but excludes allocation and input initialization.
Rows measured on different rented B200 instances or with different timing
windows are timeline evidence, not a strict cross-row ablation.

#### Frozen 1777 configuration

The recovered scheduler is now the default for `build-persistent` and
`run-persistent`. `gemm256_tma_tcgen05_persistent_1777` remains as an explicit
paper-baseline alias that freezes the same configuration. It
produced `1777.128 TFLOP/s` at 16K on instance `45378714`: 148 persistent
workers, `16x16` macroblocks at 8K/16K, `8x18` at 32K, local M-fast and macro
N-fast traversal, dense per-size CTA grouping, no A/B L2 promotion, and the
correct per-`[pipe][stage]` asynchronous completion barriers.  It is a pinned
configuration built from the current corrected mainloop, because the original
result directory retained CSVs and configuration metadata but not an exact
source snapshot.

Build and run it under the new measurement standard:

```bash
make build-persistent-1777 NVCC=/usr/local/cuda-12.9/bin/nvcc
make run-persistent-1777 SIZES=8192,16384,32768 WARMUP=1 ITERS=5 \
  PERSISTENT_CTAS=148
```

To reproduce the historical timing window instead, use `WARMUP=3 ITERS=6`.
Reaching exactly 1777 TFLOP/s is not guaranteed across rented B200 instances;
the recovery fixes the kernel configuration, while power, clock, temperature,
driver, and GPU variation remain external.

The recovered target was remeasured on B200 instance `45459938` with the new
one-warmup/five-timed-launch standard, one size per process.  Validation passed
exactly (`bad=0`, `max_abs=0`).

| size | recovered `1/5` | historical `3/6` | change |
|---:|---:|---:|---:|
| 8192 | **1692.415** | 1697.631 | -0.31% |
| 16384 | **1756.518** | 1777.128 | -1.16% |
| 32768 | **1615.840** | 1615.074 | +0.05% |

The instance started at 29 C and was 33 C immediately after the run, so this
short measurement did not encounter a high-temperature condition.  Raw
artifacts are in
`../results/gemm_persistent_1777_remeasure_b200_45459938/`.

#### Current default comparison (`warmup=1`, timed 5)

After making the recovered scheduler the default, custom, cuBLAS, and CUTLASS
were measured on B200 instance `45460466`.  Each table cell is the mean and
sample standard deviation of three independent processes; every process used
one warmup and five timed launches.  Method, size, and distribution order was
rotated, and pre-case temperatures remained between 28 and 34 C.

| input | size | custom persistent | cuBLAS | CUTLASS |
|---|---:|---:|---:|---:|
| `[0,1)` | 8192 | **1713.253 ± 0.420** | **1787.475 ± 2.912** | **1649.037 ± 2.098** |
| `[0,1)` | 16384 | **1783.963 ± 0.918** | **1873.248 ± 1.987** | **1433.417 ± 0.671** |
| `[0,1)` | 32768 | **1625.957 ± 2.371** | **1629.405 ± 22.572** | **1269.063 ± 6.500** |
| `[-8,8)` | 8192 | **1546.731 ± 2.572** | **1593.261 ± 0.909** | **1459.173 ± 2.580** |
| `[-8,8)` | 16384 | **1585.642 ± 2.353** | **1665.146 ± 2.631** | **1293.517 ± 5.059** |
| `[-8,8)` | 32768 | **1396.409 ± 7.926** | **1410.183 ± 19.077** | **1102.067 ± 2.386** |

The custom kernel is 2.92--4.77% below cuBLAS at 8K/16K and within 0.21--0.98%
at 32K.  It is 3.89--28.12% above the selected targeted CUTLASS kernels.  Full
per-process values, relative differences, configurations, and raw output are
in `../results/gemm_default_compare_1x5_b200_45460466/`.

The most relevant previous library comparison used the last short-window
configuration and bit-identical `[0,1)` inputs:

| size | custom persistent | cuBLAS | CUTLASS |
|---:|---:|---:|---:|
| 8192 | 1713.905 | **1787.349** | 1635.177 |
| 16384 | 1777.455 | **1876.069** | 1413.300 |
| 32768 | 1619.831 | **1624.791** | 1229.353 |

These library numbers used warmup 3 and 6 timed launches, so they are retained
only as historical references.  They must be remeasured under the standard
below before appearing in a final ablation table.

### 3. Measurement standard from 2026-07-21 onward

All new custom, cuBLAS, and CUTLASS measurements use one warmup launch followed
by exactly five timed launches in one process.  The reported value is the
arithmetic mean kernel time over those five launches converted to TFLOP/s.
Run one size, implementation, and input distribution per process unless a
specific paired sweep is being performed.

```text
custom:  --warmup 1 --iters 5
cuBLAS:  --warmup 1 --repeat 5
CUTLASS: --warmup=1 --iterations=5
```

Every result must record the input distribution (`[0,1)` or `[-8,8)`), seed,
matrix shape, CTA tile, K-stage size, stage count, issuer count, address mode
(`repeat` or `dense`), C-store type, GPU/driver/CUDA version, power limit, and
the aggregate CUDA-event time for the five launches.  Historical values above retain their original
protocols.  The current persistent kernel and library baseline under this
`1/5` standard is reported in the default-comparison section above.

Primary artifacts for the timeline are:

- `../1.mma/RESULTS_TORCH_RANDOM_B200_20260717.md` and
  `/home/chaewon/mma_uniform01_results_20260717_b200_45146807` for MMA-only
  input activity;
- `/home/chaewon/tma_mma_k128_random_results_20260718_b200_45229750` for the
  dependency and depth/issuer sweep;
- `/home/chaewon/benchmark-persistent-cta/5-1.gemm` for the two validated
  repeated-tile GEMMs and the first dense implementation; and
- `../results/gemm_dense_persistent_async_20260720_b200_45378714` plus
  `../results/gemm_short_compare_b200_45457682` for the corrected persistent
  kernel and library comparison.

## Build

```bash
make
# Equivalent explicit target:
make build-persistent
```

The default `make` and `make run` paths use the recovered persistent
configuration. `make build` remains available for the non-persistent baseline.

## Run

```bash
make run SIZES=4096,8192,16384,32768 WARMUP=1 ITERS=5 DEVICE=0
```

Or directly:

```bash
./gemm256_tma_tcgen05_persistent \
  --device 0 --sizes 4096,8192,16384,32768 \
  --warmup 1 --iters 5 --input-init random --persistent-ctas 148
```

## Validate

```bash
./gemm256_tma_tcgen05_persistent \
  --validate --validate-size 512 --validate-pattern pattern \
  --persistent-ctas 1
```

## Trace

`make plot` builds a trace-enabled binary with `-DGEMM_CLOCK_TRACE=1`, captures
`clock64()` ranges for CTA `(0,0)`, and renders a pipeline timeline SVG:

```bash
make plot TRACE_SIZE=4096 TRACE_START=56 TRACE_ITERS=8
```

Default outputs:

```text
log/gemm256_trace.csv
log/gemm256_trace.svg
```

## Same-address repeat-input ceiling

`GEMM_REPEAT_INPUT=1` keeps the normal square GEMM grid, K-stage count, MMA
work, TMA FP32 C store, and output coordinates, but fixes every A/B TMA source
coordinate to the first `256x64` A and `64x256` B stage.  This is the
L2-resident control used before introducing real-address tile scheduling.

Build and run the true-random BF16 `[0,1)` control with:

```bash
make build-repeat NVCC=/usr/local/cuda-12.9/bin/nvcc
make run-repeat SIZES=8192,16384,32768 WARMUP=1 ITERS=5

./gemm256_tma_tcgen05_repeat \
  --validate --validate-size 256 --validate-pattern pattern \
  --input-init random
```

The repeat specialization uses per-size settings:

- 8K: A/B TMA L2 promotion `256B`, `16x1` CTA group, TMA/MMA phase `96/0`,
  and a 128B-swizzled TMA C-store epilogue.
- 16K: no A/B promotion, `16x1` CTA group, phase `128/0`, and the same
  swizzled epilogue.
- 32K: no A/B promotion, `12x1` CTA group, phase `128/0`, and the same
  swizzled epilogue.  The old low-entropy-input phase `768/1536` is not used
  in repeat mode because it regresses true-random throughput.

Six AB/BA passes on Vast.ai instance `45320843`, NVIDIA B200 at a reported
1965 MHz SM clock and 1000 W power limit, produced:

| size | simple matched repeat | optimized repeat | gain |
|---:|---:|---:|---:|
| 8192 | 1282.17 TFLOP/s | **1711.18 TFLOP/s** | **+33.46%** |
| 16384 | 1477.74 TFLOP/s | **1644.43 TFLOP/s** | **+11.28%** |
| 32768 | 1591.05 TFLOP/s | **1630.91 TFLOP/s** | **+2.51%** |

The 256-size repeat-mode validation passed the TMA-store CPU reference with
zero differing values and zero maximum absolute error.  The 16K result uses
the no-promotion C-store-swizzle configuration and a separate six-pass AB/BA
comparison.

These tables are retained as historical phase-shift experiments.  They used
an issuer-side completion wait after every commit.  After replacing it with
per-stage producer-side reuse waits, the matched same-address normal and
persistent kernels reached approximately `1.88/1.87`, `1.95/1.96`, and
`1.78/1.79` PFLOP/s at 8K/16K/32K; see the corrected persistent section below.

### Double-pipeline and phase-shift ablation

A second B200 experiment compared the best single-issuer implementation, two
independent N-direction MMA/TMA pipes without a shift, and the same double
pipeline with only the second B-TMA producer delayed.  C-store, L2, input,
grid, and K-stage settings were held fixed.

| size | single | double `0/0` | double + TMA phase | double gain | phase gain over double |
|---:|---:|---:|---:|---:|---:|
| 8192 | 1289.27 | 1342.45 | **1708.62** (`96/0`) | +4.13% | **+27.28%** |
| 16384 | 1345.87 | 1369.58 | **1634.79** (`96/0`) | +1.76% | **+19.36%** |
| 32768 | 1324.78 | 1350.60 | **1629.81** (`160/0`) | +1.95% | **+20.67%** |

All values are TFLOP/s and six-pass AB/BA means.  A direct final phase
comparison selected `96/0` over `128/0` only at 8K (+0.75%).  At 16K and 32K,
the candidate changes were -0.08% and -0.11%, so the stable `128/0` setting
remains selected.  Delaying the second MMA issuer was consistently neutral or
negative; the useful shift is on the second B-TMA producer.

## Dense end-to-end GEMM: L2 swizzle and persistent CTA

`gemm256_tma_tcgen05_l2` is the complete dense-address kernel.  For every
output tile `(tile_m, tile_n)` and K stage `kt`, it loads the real matrix
coordinates

```text
A[tile_m * 256 : (tile_m + 1) * 256, kt * 64 : (kt + 1) * 64]
B[kt * 64 : (kt + 1) * 64, tile_n * 256 : (tile_n + 1) * 256]
```

and stores the full row-major FP32 `C = A @ B` tile with TMA.  Inputs are BF16
uniform `[0,1)`.  CUDA event time includes the complete mainloop, FP32
epilogue, and C store, but not allocation or input initialization.

The L2 scheduler launches grouped M rasters so CTAs that share a B tile run
near one another.  A B200 sweep selected:

| size | M group | second B-TMA phase | A/B L2 promotion |
|---:|---:|---:|---:|
| 8192 | 12 | 0 cycles | none / none |
| 16384 | 10 | 0 cycles | none / none |
| 32768 | 3 | 0 cycles | none / none |

Build, validate, and benchmark:

```bash
make build-l2 NVCC=/usr/local/cuda-12.9/bin/nvcc

./gemm256_tma_tcgen05_l2 \
  --validate --validate-size 512 --validate-pattern pattern

./gemm256_tma_tcgen05_l2 \
  --sizes 8192,16384,32768 --warmup 1 --iters 5 \
  --input-init random --csv final_l2.csv
```

The earlier result from instance `45369929` used one completion mbarrier per
MMA pipe and waited for every commit in the issuer warp.  That accidentally
serialized the three-stage pipeline and is superseded by the result below.

### Persistent CTA result

`gemm256_tma_tcgen05_persistent` launches 148 resident worker CTAs, initializes
mbarriers/shared memory once, allocates 512 TMEM columns once, and processes
multiple output tiles before deallocating the context.  Completion mbarriers
are indexed by `[pipe][shared-memory stage]`.  MMA issuers execute
`wait A -> wait B -> issue x8 -> commit` without waiting for completion on
every K stage; a producer waits only when it is about to reuse that stage.
The issuer drains the final commit before the epilogue.

Workers obtain output tiles from a global work queue in cache-local
macroblock order.  M varies fastest, so adjacent workers share a B panel;
macroblocks advance in N so the next block retains the same A region. 8K and
32K use `8x18` output-tile macroblocks, while 16K uses `16x16`.  An `8x18`
block contains 144 tiles, closely matching the 148-SM resident wave while
favoring A-panel reuse at the larger working set.

```bash
make build-persistent NVCC=/usr/local/cuda-12.9/bin/nvcc
make run-persistent SIZES=8192,16384,32768 WARMUP=1 ITERS=5 \
  PERSISTENT_CTAS=148
```

The 512 validation with one CTA dynamically processing four output tiles
passed exactly (`bad=0`, `max_abs=0`).  On Vast.ai B200 instance `45378714`,
six paired AB/BA passes produced:

| size | normal swizzled | persistent 148 CTA |
|---:|---:|---:|
| 8192 | 1674.335 | **1697.631 TFLOP/s** (+1.39%) |
| 16384 | 1757.211 | **1777.128 TFLOP/s** (+1.13%) |
| 32768 | 1512.945 | **1615.074 TFLOP/s** (+6.75%) |

The same-address control confirms that persistent context reuse no longer
limits the MMA pipeline: normal/persistent measured `1877.233/1874.692`,
`1953.929/1955.753`, and `1781.176/1785.165` TFLOP/s at 8K/16K/32K.
Raw CSVs are in
`results/gemm_dense_persistent_async_20260720_b200_45378714/`.

### Dense-address L2 retuning (2026-07-21)

After the asynchronous mainloop fix, the cache settings were retuned using
only real dense A/B addresses.  The sweep covered macroblock aspect ratio,
local M/N order, macroblock traversal order, 128B/256B TMA promotion, second
B-producer phase, and 128--148 persistent workers.

Key conclusions:

- local M-fast plus macro N-fast is required: workers first share B within a
  macroblock, then retain A while advancing to the next macroblock;
- 148 workers beats 128--144 at every size; matching a 144-tile macroblock by
  leaving four SMs idle is not worthwhile;
- dense 8K prefers `8x18`, phase 0, and no L2 promotion; the old 96-cycle,
  256B-promotion setting was specific to the same-address control;
- dense 16K keeps `16x16`, phase 0, no promotion;
- dense 32K keeps `8x18`, phase 0, no promotion. `9x16` and phase/promotion
  candidates were indistinguishable or slower in six-pass paired tests.

On Vast.ai B200 instance `45447842`, six AB/BA passes with six timed
iterations per pass measured:

| size | normal dense | persistent L2 schedule | gain |
|---:|---:|---:|---:|
| 8192 | 1648.527 | **1674.883 TFLOP/s** | +1.60% |
| 16384 | 1726.880 | **1740.092 TFLOP/s** | +0.77% |
| 32768 | 1480.699 | **1590.876 TFLOP/s** | +7.44% |

Absolute throughput varies across rented B200s, so tuning decisions use paired
same-instance comparisons.  Nsight Compute L2 counters were unavailable on
this host (`ERR_NVGPUCTRPERM`); the L2 conclusion is based on controlled
address-order, promotion, and worker-count ablations.  Raw results are in
`results/gemm_dense_l2_persistent_tuning_20260721_b200_45447842/`.

### Fixed-overhead and phase follow-up (2026-07-21)

Instance `45465499` retested the remaining fixed-cost paths using the current
one-warmup/five-timed-launch standard, three process samples, and rotated
orders. None beat the recovered default consistently:

| ablation | 8K | 16K | 32K | decision |
|---|---:|---:|---:|---|
| static persistent scheduler | -0.10% | -0.44% | -4.37% | keep dynamic queue |
| fused two-buffer C-store staging | -0.59% | -0.17% | -0.30% | keep split staging |
| remove per-tile checksum sink | +0.15% | -0.00% | -0.23% | noise; keep baseline |
| one `64x256` B TMA / one producer | -3.99% | -2.99% | -3.73% | keep two `64x128` producers |

The 32K static schedule is especially poor because `8x18` macroblocks contain
padded positions; the dynamic queue redistributes those short skips while the
grid-stride schedule leaves unequal valid-tile counts per worker.

A separate phase sweep tested B-TMA delays `32/64/96/128` and MMA delays
`64/128`. TMA changes were at most +0.05% at 8K and regressed 16K/32K. MMA
delay regressed every size by 0.62--1.17%. The current effective phase
therefore remains `0/0`. Exact means, validation logs, and process samples are
in:

- `results/gemm_fixed_overhead_ablation_b200_45465499/`
- `results/gemm_phase_resweep_b200_45465499/`
- `results/gemm_wide_b_ablation_b200_45465499/`

### Persistent TMEM epilogue overlap, CTA `128x256` (2026-07-22)

This experiment returns to the repeated-address `128x256` shape before
changing the dense GEMM scheduler. Each CTA computes one FP32 C tile from
BF16 A/B with two MMA issuer warps: warp 2 owns the left `128x128` accumulator
and warp 3 owns the right `128x128` accumulator. A K stage is 128, so each
issuer executes eight `m128n128k16` MMAs per stage. There are two shared-memory
stages. `GEMM_REPEAT_INPUT=1` makes every CTA and every persistent output task
reload the same global A/B panels; these numbers are therefore an L2-resident
pipeline/epilogue ablation, not a dense-address end-to-end GEMM result.

The overlap variant allocates all 512 TMEM columns once per persistent CTA and
uses them as two 256-column accumulator banks. While the four core warps
compute tile `i+1` into one bank, a dedicated epilogue warpgroup drains tile
`i` from the other bank directly to row-major global FP32 C. The prologue and
final tile remain exposed. `tcgen05.fence::{before,after}_thread_sync` plus
CTA barriers order MMA, `tcgen05.ld`, and bank reuse.

A single dedicated warp cannot drain a `128xN` TMEM result. TMEM access is
partitioned by warp ID within a warpgroup: warp IDs 0, 1, 2, and 3 can access
only lanes 0--31, 32--63, 64--95, and 96--127 respectively. Consequently the
initial one-warp prototype failed beginning at row 32 (`bad=193696`), and the
two-warp prototype was also incomplete. The valid implementation needs one
full four-warp epilogue warpgroup. This is a hardware access restriction, not
an occupancy consequence of shared-memory usage.

On Vast.ai B200 instance `45481495`, all valid variants passed the 512 pattern
test bit-exactly (`max_abs=0`, `max_rel=0`, `bad=0`). Each table entry is the
mean and population standard deviation of three separate processes. Every
process used random BF16 uniform `[0,1)`, 148 persistent CTAs, one warmup, and
five timed launches; variant order was rotated across passes. CUDA event time
includes the complete mainloop and FP32 C store.

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| E0: serialized SMEM staging + TMA store | **1598.581 ± 2.033** | **1712.793 ± 0.919** | **1558.827 ± 3.754** |
| E1: serialized TMEM load + direct global store | 1168.943 ± 0.747 | 1508.802 ± 0.243 | 1450.835 ± 1.026 |
| E2: TMEM ping-pong + dedicated 4-warp direct-store overlap | 1528.023 ± 0.420 | 1673.440 ± 0.449 | 1540.398 ± 8.247 |

E2 recovers `+30.72%`, `+10.91%`, and `+6.17%` over the serialized direct-store
control, confirming that persistent TMEM ping-pong hides a substantial part
of that epilogue. It nevertheless remains `-4.41%`, `-2.30%`, and `-1.18%`
below the existing TMA-store path. The next dense-address kernel should
therefore retain E0 as its baseline; the dedicated warpgroup path is useful as
an overlap proof, but is not the selected optimization in its current direct
global-store form.

Reproduce the complete build, validation, and three-pass measurement with:

```bash
./run_b200_gemm128x256_epilogue_ablation.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm128x256_epilogue_ablation
```

The script builds the exact macro configurations, refuses to benchmark if any
valid variant fails validation, and runs one size/variant per process. Raw
CSVs, validation logs, binary hashes, and pre/post `nvidia-smi` snapshots are
in `../results/gemm128x256_epilogue_b200_45481495/`. GPU temperature moved
from 34 C before the sequence to 40 C after it; the rotated order and low
process-to-process variance make thermal ordering an unlikely explanation for
the result.

### Recovery of the historical `128x256, K=128` repeated-tile ceiling

The exact 2026-07-18 source that originally measured 1797.227 TFLOP/s was
remeasured on instance `45481495` under the current standard. It used 592 CTAs,
8192/16384-step differential timing, one warmup, five timed launches, and
three independent processes:

| process | differential TFLOP/s |
|---:|---:|
| 1 | 1798.802 |
| 2 | 1807.125 |
| 3 | 1794.920 |
| mean | **1800.282 ± 5.092** |

The old source therefore remains reproducible. Its important difference from
the initial integrated 1712-series reconstruction was not phase shifting or
the C epilogue. It loaded one `128x128` B panel and reused it for both N
accumulators. Together with one `128x128` A panel, this makes a 64-KiB stage
and permits three stages. The initial reconstruction loaded distinct B panels
for the left and right halves, making a 96-KiB stage and permitting only two.

The integrated kernel now exposes this ceiling control as
`GEMM_REPEAT_B_BROADCAST`. A fixed-work ablation kept the persistent scheduler,
TMA C-store, random `[0,1)` input, CTA shape, K-stage, and measurement protocol
unchanged:

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| distinct B halves, 2 stages | 1599.919 ± 0.909 | 1700.903 ± 15.239 | 1556.285 ± 1.745 |
| shared B panel, 2 stages | 1673.163 ± 0.484 | 1824.972 ± 0.566 | 1661.418 ± 2.103 |
| shared B panel, 3 stages | **1787.751 ± 0.536** | **1913.033 ± 0.362** | **1773.949 ± 6.036** |

B reuse contributes 4.58--7.29%, and restoring the third stage contributes a
further 4.83--6.85%. The combined change improves the original integrated
path by 11.74--13.99% and recovers the targeted throughput. All three variants
passed the 512 pattern validation bit-exactly.

This is still an implicit tiled-B GEMM ceiling, not a dense GEMM optimization:
the two output halves intentionally use the same B data and are therefore
identical. A dense GEMM must load distinct B coordinates or obtain equivalent
reuse from a mathematically valid larger output tile, multicast, or cache
schedule.

Build and run the selected repeated-tile configuration with:

```bash
make build-repeat-128x256 NVCC=/usr/local/cuda/bin/nvcc
make run-repeat-128x256 SIZES=8192,16384,32768 WARMUP=1 ITERS=5 \
  PERSISTENT_CTAS=148
```

Exact source snapshots and raw artifacts are in:

- `../results/gemm128x256_historical_1797_remeasure_b200_45481495/`
- `../results/gemm128x256_broadcast_ablation_b200_45481495/`

### Input range comparison: `[0,1)` versus `[-8,8)`

`--input-init random-signed8` uses the same hash stream and seeds as `random`,
but stores `16*u-8` instead of `u`. On B200 instance `45453339`, three paired
passes measured the following medians for the persistent end-to-end kernel:

| size | `[0,1)` | `[-8,8)` | change |
|---:|---:|---:|---:|
| 8192 | 1697.083 | 1523.764 TFLOP/s | -10.21% |
| 16384 | 1622.905 | 1377.320 TFLOP/s | -15.13% |
| 32768 | 1552.944 | 1306.078 TFLOP/s | -15.90% |

The paired cuBLAS and CUTLASS comparison, exact input definition, kernel
configurations, and raw logs are in
`results/gemm_input_distribution_b200_45453339/summary.md`.

#### Custom-only, one case per process

To remove library ordering and process-lifetime effects, instance `45455808`
ran only the custom persistent kernel. Each process handled one size and one
distribution, used 10 warmups plus 30 timed iterations, and each case was
repeated in three independent processes with reversed pass order. Process-level
average TFLOP/s were averaged as follows:

| size | `[0,1)` | `[-8,8)` | change |
|---:|---:|---:|---:|
| 8192 | 1697.187 | 1522.766 | -10.28% |
| 16384 | 1576.862 | 1354.790 | -14.08% |
| 32768 | 1550.018 | 1310.877 | -15.43% |

Temperature peaked at only 51 C, while median power at 100% utilization was
981.16 W and median SM clock was 1214.5 MHz. A second run with telemetry
disabled reproduced the values, so NVML polling was not responsible for the
lower 16K result. Full per-process values and logs are in
`results/gemm_custom_only_single_case_b200_45455808/summary.md`.
