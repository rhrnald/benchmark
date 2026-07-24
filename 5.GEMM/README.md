# 5.GEMM

Prototype Blackwell GEMM compute benchmark using TMA loads and `tcgen05.mma`.

현재 canonical E2a N-split 구현, benchmark 단계별 해석, library 비교와
software version 주의사항은 먼저
[`CURRENT_STATUS.md`](CURRENT_STATUS.md)를 본다.
아래 문서는 이전 구현을 포함한 전체 실험 기록과 generic benchmark
interface를 보존한다.

과거 dual-wide topology를 기준으로 다시 설계한 CTA/B1 phase ablation은
[`E7A_PHASE_REDESIGN.md`](E7A_PHASE_REDESIGN.md)에 있으며, 측정 결과
선택 phase는 `0/0`이었다. 현재 A-shared/B-N-split 복구와 최적화 기록은
[`NSPLIT_REDESIGN.md`](NSPLIT_REDESIGN.md)와
[`NSPLIT_WS_COLLECTOR.md`](NSPLIT_WS_COLLECTOR.md)에 있다.

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
  `12 x 12` at 8K/32K and `16 x 16` at 16K, with local M-fast and macro N-fast
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

### Fresh N-split restoration and first optimization ablations

The canonical source now implements the requested ownership exactly: one
`256x64` A TMA is shared, B0/B1 each load `64x128`, and warp 2/3 each
accumulate one `256x128` N half.  Because CTA-group-1 has no M256 MMA, each
consumer issues two `m128n128k16` operations per K16, for 16 dynamic MMA
instructions per CTA and K64.

The first same-session B200 restoration measured:

| variant | `[0,1)` | `[-8,8)` |
|---|---:|---:|
| E7a exact reference | 1817.023 | 1612.664 |
| requested N-split exact | 1797.175 | 1593.776 |
| producer suspend | 1796.770 | 1601.205 |

Producer suspend failed the two-input selection gate.  A second four-pass
session isolated a weight-stationary B-collector attempt:

| variant | `[0,1)` | `[-8,8)` | controlled conclusion |
|---|---:|---:|---|
| E7a exact | 1817.047 | 1608.067 | performance reference |
| N-split exact | 1795.604 | 1594.047 | canonical working source |
| static ordinary-MMA control | 1755.775 | 1571.485 | -2.2181% / -1.4136% vs exact |
| static `mma.ws` B collector | 1682.383 | 1513.175 | -4.1799% / -3.7104% vs static |

The static and WS binaries both use 166 registers and their normalized SASS
differs in exactly 16 MMA opcode lines.  Thus the additional WS loss is not a
register-count or surrounding-code confound.  Static pipe specialization
itself duplicates the shared consumer body, growing the normalized kernel
from 1,913 to 2,197 lines, and is also rejected.  Keep the compact
runtime-pipe ordinary-MMA consumer as the optimization base.

The next candidate computes each logical N half as
`C_p^T = B_p^T A^T`. This keeps the same A/B TMA payload and two-warp
ownership but uses `m128n256k16`, reducing dynamic MMA issue from 16 to 8 per
CTA/K64. Fresh four-pass B200 results were:

| path | `[0,1)` vs exact | `[-8,8)` vs exact |
|---|---:|---:|
| no-C-store mainloop | +0.8963% | +0.4722% |
| scalar-transpose E2E | +0.4617% | -0.0792% |

Full-C pattern/ones validation was exact. The scalar transpose adds about
0.019/0.029 ms per launch relative to the existing epilogue, so the mapping
advances but the scalar epilogue does not. See
[`NSPLIT_TRANSPOSE.md`](NSPLIT_TRANSPOSE.md) for the descriptor, TMEM
mapping, and first decomposition. The bank-conflict analysis, locally gated
vec2/vec4 candidates, matched B200 protocol, and runner are in
[`NSPLIT_EPILOGUE.md`](NSPLIT_EPILOGUE.md).

The completed eight-pass Williams-balanced epilogue sweep found that scalar
transpose beat direct exact by `+0.4473%/+0.5994%`, but missed the two-input
`+0.5%` adoption gate on `[0,1)`. Every vec2/vec4 candidate was slower than
scalar; the closest CF2 x64 result was `-0.0716%/-0.0020%`. The scalar path
already emits the minimum 2,048 conflict-free 128-byte shared wavefronts for
one 256 KiB output tile. Vectorization reduces store instruction count but
does not reduce that traffic and adds shuffles, so no vector epilogue is
adopted.

The completed scalar-only ablation kept that mapping and streamed x32 rather
than x64 TMEM fragments. Local codegen reduced 174 registers/2,229 normalized
operations to 91/2,011 with no spill, local memory, or shuffle, but dynamic
TMEM warp loads doubled while shared stores and bank wavefronts remained
unchanged. In the six-permutation B200 run, x32 changed throughput by
`-0.0501%/-0.0652%` versus x64 and both paired 95% confidence intervals
crossed zero. X64 itself improved on direct exact by only
`+0.4759%/+0.4762%`, below the two-input `+0.5%` adoption gate. X32 is
rejected, x16 will not be measured, and direct exact remains canonical. See
[`NSPLIT_SCALAR_TMEM.md`](NSPLIT_SCALAR_TMEM.md).

The next one-factor candidate uses the otherwise idle 64 KiB shared-memory
stage during the current output epilogue to prefetch the next output tile's
first K64 A `256x64` and B0/B1 `64x128` panels.

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

### Consolidated repeated-tile ceiling history

The old results confirm that a full TMA-to-MMA pipeline with repeated global
tiles can exceed 1900 TFLOP/s.  The clearest observation is the corrected
asynchronous `256x256x64` kernel from 2026-07-20: with random BF16 `[0,1)`,
distinct left/right B panels, and the full FP32 TMA C-store, its 16K
same-address control measured 1953.929 TFLOP/s on the normal grid and
1955.753 TFLOP/s with 148 persistent CTAs.

The following table separates the historical ceilings by workload.  A dash
means that the experiment did not use the square-size benchmark interface.

| workload | tile / data reuse | TMA dependency and C store | non-size reference | 8K | 16K | 32K | protocol/status |
|---|---|---|---:|---:|---:|---:|---|
| MMA only, constant | `m128n128k16`, operands resident in SMEM | no per-step TMA or commit/wait; no C store | **2229.664** | -- | -- | -- | long differential timing, controlled upper bound |
| MMA only, random `[0,1)` | same MMA work | no per-step TMA or commit/wait; no C store | **1928.151** | -- | -- | -- | long differential timing, data-activity ceiling |
| dependent TMA+MMA | repeated A/B panel, depth 3, two issuers | `wait A/B -> MMA x8 -> commit`; no C store | **1800.166** | -- | -- | -- | long differential timing |
| historical repeated GEMM | `128x256x128`; one B panel reused by both N accumulators | dependent pipeline and FP32 C store | **1800.282 +/- 5.092** | -- | -- | -- | remeasured with warmup 1, timed 5, three processes |
| corrected async same-address GEMM, normal grid | valid `256x256x64`; distinct B halves; every CTA/stage reloads the same global panels | three-stage producer-side reuse wait and FP32 TMA C store | -- | **1877.233** | **1953.929** | **1781.176** | historical warmup 3, timed 5; one recorded process per case |
| corrected async same-address GEMM, persistent | same valid `256x256x64` work; 148 persistent workers | same pipeline and store | -- | **1874.692** | **1955.753** | **1785.165** | historical warmup 3, timed 5; one recorded process per case |
| integrated shared-B control | `128x256x128`; both N halves intentionally reuse one B panel | three stages and FP32 TMA C store | -- | **1787.751** | **1913.033** | **1773.949** | warmup 1, timed 5, three rotated processes; not a valid distinct-B GEMM |
| integrated distinct-B pipeline | valid `128x256x64`; distinct B halves | three stages and FP32 TMA C store | -- | **1642.320** | **1753.739** | **1600.875** | warmup 1, timed 5, three rotated processes |
| dense end-to-end persistent GEMM | valid `256x256x64`; real A/B coordinates | three stages and FP32 TMA C store | -- | **1697.631** | **1777.128** | **1615.074** | six paired historical passes |

All values are TFLOP/s.  The 2026-07-20 `1955.753` same-address result is the
right historical target for a valid `256x256` repeated-tile pipeline, but it
was originally only one recorded process per case.

It was reproduced on 2026-07-22 under the current warmup-1/timed-5,
three-process standard:

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| normal | 1807.666 +/- 0.955 | 1945.916 +/- 2.176 | 1779.193 +/- 4.920 |
| persistent, 148 CTAs | **1813.887 +/- 0.655** | **1949.590 +/- 1.080** | **1782.437 +/- 2.072** |

The central persistent 16K result is only 0.315% below the historical
1955.753 value, and 32K is within 0.153%.  Both are reproduced.  The new 8K
mean is 3.243% below the old single sample, so only the new three-process mean
should be used for that size.  Both variants passed the 512 pattern validation
bit-exactly.  Exact artifacts are in
`../results/gemm256_same_address_repro_b200_45481495/`.

The current `128x256` result is not an apples-to-apples regression from that
1956 result.  At K=64, a valid `128x256` tile performs 4,194,304 FLOP from
48 KiB of A+B payload, or 85.33 FLOP/requested byte.  A valid `256x256` tile
performs 8,388,608 FLOP from 64 KiB, or 128 FLOP/requested byte.  The larger
tile therefore moves one-third fewer requested operand bytes per FLOP.  The
invalid shared-B `128x256` control also reaches 128 FLOP/byte and recovers
1913 TFLOP/s, reinforcing that valid operand reuse, rather than pipeline
depth alone, separates the 1754 and 1900-series results.

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

#### Fresh 16K E7a versus CUTLASS throughput rerun

The clean E7a baseline and the selected CUTLASS 16K kernel were remeasured on
the same B200 session on 2026-07-23.  This was a throughput-only run; no trace
instrumentation was enabled.  Each implementation ran in four independent
processes, each using one warmup and five timed launches.  The order alternated
as AB/BA/AB/BA, and all measured-process snapshots were at 35--36 C and
1965 MHz.

| input | size | clean E7a | selected CUTLASS | E7a vs CUTLASS |
|---|---:|---:|---:|---:|
| BF16 `[0,1)` | 16384 | **1800.000 ± 1.632** | **1423.120 ± 2.577** | **+26.483%** |

Both kernels computed row-major BF16-input, FP32-output `C=A*B` from the same
deterministic input bytes.  E7a used a `256x256x64` three-stage persistent
kernel; CUTLASS used the selected `256x256x64`, static `4x1` cluster,
five-stage direct-store CLC kernel.  Per-process values, hashes, commands, and
raw artifacts are in
`../results/gemm_e7a_cutlass_16k_compare_b200_45601332_20260723/`.

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

The historical E7a diagnostic follows the `0.attention` trace semantics.  It
records physical W0--W3 lane-0 timestamps for eight steady K64 stages:
producer M0/M1 reuse waits and TMA prepare/issue, consumer A/B0/B1 ready
waits, MMA prepare/issue, commit, and the `kt+3` producer dependency pass.
Use [`PIPELINE_TRACE.md`](PIPELINE_TRACE.md) for generation, collection, and
rendering commands.

The measured B200 trace, including the interactive SVG, raw 164-event CSV,
per-stage metrics, validation, SASS, and interpretation, is archived at
[`../results/gemm_e7a_pipeline_stage_trace_b200_45481495_20260723/`](../results/gemm_e7a_pipeline_stage_trace_b200_45481495_20260723/).

The older `make plot` target below belongs to the legacy trace-enabled kernel.
It remains useful for that implementation, but it is not the historical E7a
per-K-stage trace:

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

### Matched same-address to dense-address and focused L2 sweep (2026-07-22)

The reproduced `256x256x64` same-address pipeline was converted to a dense
end-to-end GEMM by changing only `GEMM_REPEAT_INPUT=1 -> 0`.  The matched
persistent baseline retained the repeat-tuned phases and 8K promotion, while
all other pipeline, math, and FP32 C-store settings stayed fixed:

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| grouped grid | 1694.096 +/- 1.724 | 1761.240 +/- 1.007 | 1585.165 +/- 3.710 |
| persistent tile reuse | **1712.740 +/- 0.306** | **1780.463 +/- 0.734** | 1572.202 +/- 22.749 |

Persistent macroblock ordering improves the stable 8K and 16K cases by
1.10% and 1.09%.  The third 32K persistent process was an isolated 1546.066
TFLOP/s slowdown, so a paired control was carried into the focused sweep.

The focused sweep then compared the repeat-tuned control against dense phase
`0/0`, no A/B promotion, and three balanced 144-tile macroblocks.  Each entry
is the mean and sample standard deviation of three one-warmup/five-timed
processes:

| configuration | 8K | 16K | 32K |
|---|---:|---:|---:|
| paired control | 1714.039 +/- 0.363 | 1781.433 +/- 0.168 | 1576.632 +/- 9.325 |
| `8x18` | 1716.684 +/- 1.872 | 1781.543 +/- 0.792 | 1601.834 +/- 14.392 |
| `9x16` | 1713.532 +/- 0.946 | same binary as `8x18` | 1594.414 +/- 3.484 |
| `12x12` | **1717.398 +/- 1.343** | same binary as `8x18` | **1601.945 +/- 4.035** |

The selected schedule is `12x12` at 8K/32K and `16x16` at 16K, with local
M-fast, macro N-fast, phase `0/0`, no A/B promotion, a dynamic work queue, and
148 resident workers.  Relative to the paired control it gains 0.20% at 8K,
is neutral at 16K, and gains 1.61% at 32K.  `12x12` is selected over `8x18`
because their means are effectively identical at 32K while `12x12` has much
lower process variance.  All candidates passed the 512 pattern validation
bit-exactly.

These results show that software L2 ordering recovers only a small part of
the gap to the 1949.590-TFLOP/s same-address 16K ceiling.  Further material
improvement requires a stronger valid reuse mechanism such as cluster TMA
multicast, rather than additional phase or promotion sweeps.  Raw artifacts
are in:

- `../results/gemm256_dense_matched_baseline_b200_45481495/`
- `../results/gemm256_dense_l2_focused_sweep_b200_45481495/`

### Two-CTA B TMA multicast ablation

The next dense-GEMM experiment uses a fixed `cluster_shape=(2,1,1)`. It does
not switch the tensor-core instruction to `cta_group::2`: every CTA still
computes an independent `256x256` output tile with the existing
`tcgen05.mma cta_group::1` pipeline. The two clustered CTAs are assigned the
same N tile and adjacent M tiles, so only B is common between them.

Cluster rank 0 issues each of the two `64x128` BF16 B loads with TMA multicast
mask `0b11`. The payload is deposited at the same SMEM offset in both CTAs;
A remains a normal, independent `256x64` load in each CTA. Per K64 stage this
changes the two-CTA global-load payload from 128 KiB to 96 KiB, a 25% input
traffic reduction for the pair. Full FP32 C stores and all GEMM coordinates
remain mathematically dense.

The persistent launch remains 148 CTAs, now grouped as 74 two-CTA clusters.
One leader allocation gives the pair consecutive M-fast tasks. Remote
mbarrier arrivals make both B-ready barriers visible to the multicast, and a
two-consumer DSM reuse barrier prevents rank 0 from overwriting either CTA's B
stage before both local MMAs have completed. This is required for correctness
and is separate from the `tcgen05` completion barriers.

The paired experiment compares this path against the selected 1-CTA dense
control. Both use CTA `256x256`, K64, three stages, two B issuer/MMA warps,
random BF16 `[0,1)`, 148 CTA workers, the selected 12x12/16x16/12x12 tile
schedule, and full FP32 TMA C stores. Each case is one process with warmup 1
and five timed launches; 8K/16K/32K are measured in three rotated passes.

```bash
./run_b200_gemm256_tma_multicast_b.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm256_tma_multicast_b
```

Standalone build/run:

```bash
make build-multicast-b NVCC=/usr/local/cuda/bin/nvcc
make run-multicast-b SIZES=8192,16384,32768 WARMUP=1 ITERS=5 \
  PERSISTENT_CTAS=148
```

On B200 instance `45481495`, both variants passed the 512 pattern validation
bit-exactly. Event throughput is the mean and sample standard deviation of
three independent processes:

| size | 1-CTA control | 2-CTA B multicast | change |
|---:|---:|---:|---:|
| 8K | 1716.333 +/- 0.592 | 1663.976 +/- 2.356 | -3.051% |
| 16K | 1780.809 +/- 0.799 | 1771.340 +/- 0.639 | -0.532% |
| 32K | 1590.538 +/- 7.000 | **1633.041 +/- 3.877** | **+2.672%** |

Thus the 25% reduction in pair-level A+B input payload is useful only at 32K
in this first implementation. At 8K the selected M-fast schedule already
keeps B hot in L2 and the short kernel exposes cluster/pair synchronization
overhead; at 16K the two effects nearly balance. At 32K, where the moving
working set creates more L2/DRAM pressure, multicast gives a stable 2.67%
gain. A two-CTA clustered non-multicast control is still needed to split the
cluster/scheduler overhead from the traffic-saving benefit itself.

Raw CSVs, validation logs, source snapshots, hashes, and the full analysis are
in `../results/gemm256_tma_multicast_b_b200_45481495/`.

#### 16K multicast scheduler/swizzle sweep

The first multicast result above retained the pre-cluster 16x16 macroblock
and dynamic atomic scheduler. To test whether that remains optimal for a
two-CTA cluster, the 16K-only follow-up compares the full power-of-two
macroblock cross product M/N in `{8,16,32}`. Every shape is measured with both
the dynamic cluster-leader atomic scheduler and the fixed grid-stride static
scheduler. All other kernel, input, output, cluster, and timing conditions are
held fixed.

This is CUTLASS-inspired but not identical to CUTLASS. Our macroblock is a
custom two-dimensional ordering layer. CUTLASS's persistent scheduler uses a
cluster-aware raster/swizzle representation and newer Blackwell kernels may
use CLC rather than this global-atomic or simple grid-stride implementation.

```bash
./run_b200_gemm256_multicast_16k_scheduler_sweep.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm256_multicast_16k_scheduler_sweep
```

All 18 configurations passed the 512 pattern validation bit-exactly. Event
throughput is the mean and sample standard deviation of three rotated
one-process runs with warmup 1 and five timed launches:

| macro | dynamic | static | static change |
|---:|---:|---:|---:|
| 8x8 | 1752.756 +/- 1.831 | 1783.083 +/- 0.392 | +1.730% |
| 8x16 | 1753.843 +/- 1.332 | **1784.075 +/- 0.924** | +1.724% |
| 8x32 | 1753.457 +/- 0.784 | 1783.371 +/- 0.966 | +1.706% |
| 16x8 | 1771.626 +/- 1.357 | 1783.477 +/- 0.329 | +0.669% |
| 16x16 | **1772.407 +/- 0.984** | **1783.590 +/- 0.247** | +0.631% |
| 16x32 | 1771.766 +/- 0.236 | 1783.738 +/- 0.488 | +0.676% |
| 32x8 | 1722.044 +/- 0.868 | 1691.152 +/- 1.009 | -1.794% |
| 32x16 | 1722.082 +/- 0.801 | 1692.586 +/- 2.923 | -1.713% |
| 32x32 | 1721.882 +/- 0.220 | 1688.941 +/- 1.460 | -1.913% |

`16x16` is the best dynamic shape. With static scheduling, macro M=8 and 16
are effectively tied: `8x16` has the highest mean, but is only 0.027% above
`16x16`, while static `16x16` has the lowest variance in the sweep. Static
`16x16` is therefore selected as the robust 16K multicast default. Static
improves matched `16x16` by 0.631%. Macro M=32 is consistently poor and should
not be used with this 148-CTA grid-stride schedule.

Raw data and the full interpretation are in
`../results/gemm256_multicast_16k_scheduler_sweep_b200_45481495/`.

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

### Distinct-B pipeline depth ablation (`128x256`)

To isolate pipeline changes from the shared-B communication reduction above,
the left and right `128x128` output halves were returned to distinct B panels.
The total A/B bytes, MMA work, random `[0,1)` input, repeated addresses,
persistent schedule, and TMA C-store were fixed.  Only K-stage granularity and
ring depth changed.  Each entry is the mean and sample standard deviation of
three rotated, independent processes, each using warmup 1 and timed launches
5:

| variant | dynamic SMEM | 8K | 16K | 32K |
|---|---:|---:|---:|---:|
| `K128`, 2 stages | 197632 B | 1600.490 +/- 0.515 | 1711.930 +/- 0.969 | 1556.790 +/- 1.137 |
| `K64`, 3 stages | 148480 B | 1642.320 +/- 1.160 | **1753.739 +/- 0.744** | 1600.875 +/- 1.915 |
| `K64`, 4 stages | 197632 B | **1645.157 +/- 1.616** | 1751.973 +/- 0.547 | **1604.700 +/- 1.772** |

Thus finer `K=64` staging improves the 1712-series baseline by 2.44% at 16K
and 2.61--3.08% across all sizes without changing communication or math.  A
fourth stage is not consistently better than three, so the main gain is finer
producer/consumer interleaving rather than ring depth alone.  All variants
passed the 512 pattern validation bit-exactly.  This pipeline-only change
closes part, but not all, of the gap to the historical 1797/1800 shared-B
ceiling; that result transferred less B data.

Reproduce with:

```bash
./run_b200_gemm128x256_pipeline_depth_ablation.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm128x256_pipeline_depth_ablation
```

Raw artifacts and the exact source snapshot are in
`../results/gemm128x256_pipeline_depth_ablation_b200_45481495/`.

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

### 16K multicast operand, transaction width, K-stage, and epilogue sweep

The selected static 16x16 two-CTA kernel was tested along four optimization
directions at 16K: multicast A instead of B, merge the two B multicast
transactions, use K32 with four or five stages, and overlap a dense 128x256
epilogue using TMEM ping-pong. Every case used BF16 random `[0,1)`, complete
FP32 C output, 148 persistent CTAs, effective phase shift 0/0, and three
rotated processes with warmup 1 and five timed launches. All variants passed
the 512 pattern validation bit-exactly.

| 256x256 mainloop | TFLOP/s | change from selected |
|---|---:|---:|
| B multicast, split `64x128`, K64/S3 | **1783.800 +/- 1.301** | baseline |
| A multicast, `256x64`, K64/S3 | 1730.416 +/- 0.424 | -2.993% |
| B multicast, wide `64x256`, K64/S3 | 1740.478 +/- 0.338 | -2.429% |
| B multicast, split, K32/S4 | 1593.331 +/- 0.583 | -10.678% |
| B multicast, split, K32/S5 | 1664.468 +/- 1.057 | -6.690% |
| A multicast, K32/S4 | 1544.366 +/- 0.329 | -10.752% vs A K64/S3 |
| A multicast, K32/S5 | 1646.775 +/- 0.623 | -4.834% vs A K64/S3 |

For the separate dense 128x256 epilogue comparison, serialized TMA store
measured `1379.437 +/- 4.295 TFLOP/s`; four-warp TMEM ping-pong/direct-store
overlap measured `1342.131 +/- 1.264 TFLOP/s`, a 2.704% regression.

The selected kernel therefore remains split-B multicast, K64/S3, static
16x16, and phase shift 0/0. The A direction loses locality and/or favorable
cluster orientation, wide B loses useful two-producer issue parallelism, and
K32 pays twice as many stage/barrier epochs. Full conditions, interpretation,
raw CSVs, validation logs, and source snapshots are in
`../results/gemm_multicast_pipeline_epilogue_ablation_b200_45481495/`.

Reproduce with:

```bash
./run_b200_gemm_multicast_pipeline_epilogue_ablation.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_multicast_pipeline_epilogue_ablation
```

### 16K partial input reuse and wave-aligned scheduling

To identify whether the remaining gap to the same-address ceiling comes from
A or B, the selected 256x256 B-multicast kernel was rerun while independently
repeating each operand's TMA source coordinates.  Math, K-loop count,
wait/commit protocol, output addresses, and full FP32 C store remained intact.
Each number is three rotated one-process measurements using BF16 random
`[0,1)`, warmup 1, and five timed launches.

| A addresses | B addresses | TFLOP/s | versus dense |
|---|---|---:|---:|
| dense | dense | 1783.283 +/- 1.060 | baseline |
| repeated | dense | 1899.555 +/- 1.821 | +6.520% |
| dense | repeated | 1812.678 +/- 1.047 | +1.648% |
| repeated | repeated | **1956.014 +/- 1.943** | +9.686% |

Of the 172.731 TFLOP/s dense-to-full-repeat gap, A-only reuse recovers
116.272 TFLOP/s while B-only reuse recovers 29.395 TFLOP/s.  The remainder is
an interaction visible only when both repeat.  A locality is therefore the
dominant next optimization target; B traffic alone cannot explain or close
the gap.

The companion scheduler test rejected a literal 144-task wave and cluster 4:

| cluster / scheduler | CTAs | TFLOP/s |
|---|---:|---:|
| C2, existing static 16x16 macro | 148 | **1783.283 +/- 1.060** |
| C2, existing static 16x16 macro | 144 | 1749.012 +/- 0.845 |
| C2, explicit 16x9 wave | 144 | 1587.562 +/- 2.532 |
| C2, explicit 12x12 wave | 144 | 1437.077 +/- 0.808 |
| C4, explicit 16x9 wave | 144 | 933.048 +/- 0.177 |
| C4, explicit 12x12 wave | 144 | 934.301 +/- 0.944 |

At the same 144 CTA count, explicit 16x9 and 12x12 waves regress 9.23% and
17.83% versus the old macro traversal.  Cluster 4 loses another 35--41%
against its cluster-2 counterpart, so reducing B transactions does not repay
the added multicast/DSM synchronization and producer serialization.  Keep
cluster 2, 148 CTAs, and the existing static 16x16 task stream.

Reproduce with:

```bash
./run_b200_gemm_partial_reuse_wave_ablation.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_partial_reuse_wave_ablation
```

All variants passed pattern validation.  Full per-process values, exact
conditions, logs, and source snapshots are in
`../results/gemm_partial_reuse_wave_ablation_b200_45481495/`.

### Cluster-order follow-up: preserving B reuse wins

The partial-reuse result suggested prioritizing A, so the two-CTA B multicast
cluster was preserved while cluster IDs were reordered across N first.  This
makes different clusters request the same A panels together, but replaces the
existing cross-cluster B-panel locality.

| local scheduler | CTAs | TFLOP/s |
|---|---:|---:|
| existing M-fast 16x16 | 148 | **1783.871 +/- 2.085** |
| cluster-N-fast 32x8 | 148 | 1756.695 +/- 1.252 |
| cluster-N-fast 16x16 | 148 | 1754.691 +/- 0.621 |
| cluster-N-fast 8x32 | 148 | 1690.133 +/- 0.898 |
| cluster-N-fast 4x64 | 148 | 1615.893 +/- 0.311 |
| cluster-N-fast explicit 16x9 wave | 144 | 1580.579 +/- 0.705 |

A second sweep used a fixed 16x16 macro and visited only a bounded number of N
tiles before returning to the next M pair:

| N group | TFLOP/s | versus group 1 |
|---:|---:|---:|
| 1 (original M-fast) | **1786.590 +/- 0.364** | baseline |
| 2 | 1777.966 +/- 1.452 | -0.483% |
| 4 | 1772.420 +/- 1.278 | -0.793% |
| 8 | 1757.388 +/- 0.312 | -1.635% |
| 16 (full N-fast) | 1754.691 +/- 0.621 | -1.786% |

The loss is monotonic with N-group size.  The existing M-fast order is already
effective at keeping a B panel useful across many clusters; simultaneous A
requests do not replace that benefit.  Keep N-group 1.  Reaching the A-repeat
ceiling requires reducing or sharing A transactions without sacrificing the
current B traversal, rather than reversing the tile order.

Reproduce the two sweeps with:

```bash
./run_b200_gemm_cluster_nfast_ablation.sh \
  /workspace/benchmark/5.GEMM /workspace/gemm_cluster_nfast_ablation
./run_b200_gemm_cluster_ngroup_ablation.sh \
  /workspace/benchmark/5.GEMM /workspace/gemm_cluster_ngroup_ablation
```

Full artifacts are in
`../results/gemm_cluster_nfast_ablation_b200_45481495/` and
`../results/gemm_cluster_ngroup_ablation_b200_45481495/`.

### Non-multicast scheduler order, snake, and N-strip (2026-07-22)

The 16K non-multicast dynamic-persistent path was tested independently of the
cluster scheduler.  Each variant used 148 CTAs, a `256x256x64` CTA tile,
three stages, split `64x128` B loads, random BF16 `[0,1)`, complete FP32 C,
and effective phase `0/0`.  Results are three forward/reverse/rotated process
samples with warmup 1 and five timed launches.

| variant | local order | macro order / change | TFLOP/s | paired change |
|---|---|---|---:|---:|
| `order_mn` | M-fast | N-fast | 1769.017 +/- 1.387 | baseline |
| `order_mm` | M-fast | M-fast | 1740.880 +/- 0.838 | -1.590% |
| `order_nn` | N-fast | N-fast | 1756.834 +/- 1.140 | -0.689% |
| `order_nm` | N-fast | M-fast | **1778.907 +/- 2.584** | **+0.559%** |
| macro snake | M-fast | reverse macro N | 1770.520 +/- 0.366 | +0.085% |
| full snake | M-fast | reverse macro and local N | 1771.428 +/- 0.679 | +0.136% |
| N-strip 2 | M-fast | one allocation owns 2 N outputs | 1764.801 +/- 1.567 | -0.238% |
| N-strip 4 | M-fast | one allocation owns 4 N outputs | 1747.406 +/- 2.533 | -1.222% |

`order_nm` was faster in every paired pass (`+0.400%`, `+0.747%`, and
`+0.530%`).  It makes same-A N neighbors consecutive inside each macro and
keeps one B-coordinate range across adjacent M macros.  Because its mean gain
is close to the 0.5% selection gate, it remains a candidate pending a focused
AB/BA rerun rather than becoming the default immediately.

Snake affects only three 16K macro-row wrap boundaries and stayed below the
selection threshold.  N-strip ownership also failed: the CTA finishes all
256 K stages for one output before revisiting A for the next output, so this
does not retain a K64 A panel in shared memory and adds ownership/tail costs.
All eight binaries passed the 512 pattern reference bit-exactly, and the host
mapping test proved exact scheduler coverage.  Actual L2/DRAM traffic is
unmeasured because performance counters are unavailable on the rented host.
Exact raw data and source snapshots are in
`../results/gemm_nonmulticast_l2_round1_b200_45481495/`.

This result applies to the non-multicast dynamic scheduler and does not revise
the separate two-CTA multicast cluster-order conclusion.  Reproduce it with:

```bash
DEFINITION_COMMIT=f669b6f ./run_b200_gemm_nonmulticast_l2_round1.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_l2_round1
```

The required focused confirmation did not reproduce a selectable 0.5% gain.
Six adjacent, counterbalanced AB/BA pairs produced:

| pair | `order_mn` | `order_nm` | paired change |
|---:|---:|---:|---:|
| 1 | 1773.511 | 1782.280 | +0.494% |
| 2 | 1776.562 | 1780.378 | +0.215% |
| 3 | 1771.703 | 1778.346 | +0.375% |
| 4 | 1773.644 | 1744.313 | -1.654% |
| 5 | 1770.290 | 1778.597 | +0.469% |
| 6 | 1773.211 | 1783.224 | +0.565% |

The paired mean was `+0.077% +/- 0.857%`, with a 95% interval of
`[-0.822%, +0.976%]`.  Five pairs favored `order_nm`, but it failed the
predeclared all-positive and 0.5% gates; even the non-decisive five-positive
sensitivity mean was only `+0.424%`.  Keep local M-fast plus macro N-fast as
the default and do not retune macro shapes for the candidate.  Both binaries
passed the 512 reference bit-exactly.  Exact confirmation artifacts are in
`../results/gemm_nonmulticast_order_confirm_b200_45481495/`.

### Non-multicast TMA L2 eviction-priority hints (2026-07-22)

The selected 16K `order_mn` kernel was also tested with per-operand PTX TMA
L2 eviction hints.  This is distinct from the earlier tensor-map L2 promotion
sweep: promotion changes fill granularity, while this experiment supplies an
eviction-priority hint.  All variants kept identical TMA coordinates, logical
request payload, C store, scheduler, and W1/I5 x three-process protocol.

| variant | A / B policy | TFLOP/s | paired change |
|---|---|---:|---:|
| baseline | none / none | 1769.197 +/- 1.322 | baseline |
| `a_last` | evict_last / none | **1771.192 +/- 1.539** | **+0.113%** |
| `b_last` | none / evict_last | 1769.702 +/- 1.211 | +0.029% |
| `a_last_b_first` | evict_last / evict_first | 1743.439 +/- 1.467 | -1.456% |
| `a_first_b_last` | evict_first / evict_last | 1423.287 +/- 6.930 | -19.552% |

`a_last` was positive in all three paired passes, but its `+0.113%` mean was
below the predeclared `+0.5%` promotion gate.  No hint remains the default and
no focused confirmation or 8K/32K extension is planned.  The asymmetric
negative controls are consistent with A retention being more important under
this schedule.  Because the symmetric endpoints change both operands, the
single-factor controls below were added before making that claim.  This is
still only a throughput-based locality inference because L2/DRAM counters were
unavailable.  All five 512 pattern validations were bit-exact.

Exact CSVs, validation logs, compile commands, source snapshots, execution
order, telemetry, and hashes are in
`../results/gemm_nonmulticast_l2_evict_b200_45481495/`.  Reproduce with:

```bash
DEFINITION_COMMIT=1bb691b ./run_b200_gemm_nonmulticast_l2_evict.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_l2_evict
```

The follow-up isolated `evict_first` on one operand at a time.  Its three
cyclic process orders placed each case in every sequence position once while
preserving the same 16K W1/I5 conditions:

| variant | A / B policy | TFLOP/s | paired change |
|---|---|---:|---:|
| baseline | none / none | 1770.976 +/- 1.887 | baseline |
| `a_first` | evict_first / none | 1421.683 +/- 8.470 | **-19.723%** |
| `b_first` | none / evict_first | 1741.889 +/- 3.967 | **-1.642%** |

Both controls regressed in all three paired passes.  The single-factor
magnitudes agree with the preceding conditional contrasts, so throughput is
much more sensitive to the A eviction policy under the current local-M-fast /
macro-N-fast traversal.  This does not quantify cache misses or DRAM traffic.
All three 512 pattern checks were bit-exact; no hint remains the default, and
the eviction-hint direction is closed.  Artifacts are in
`../results/gemm_nonmulticast_l2_first_controls_b200_45481495/`.

```bash
DEFINITION_COMMIT=1463cea \
  ./run_b200_gemm_nonmulticast_l2_first_controls.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_nonmulticast_l2_first_controls
```
