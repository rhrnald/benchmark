# Recent direct N-split scheduler sweep

This experiment compares dynamic atomic task assignment with fixed static
grid-stride ownership in the recent direct N-split kernel.

- Sizes: square 8K, 16K, and 32K.
- Input: deterministic BF16 uniform `[-8,8)`.
- Shapes: `4x16`, `8x16`, `4x32`, `8x18`, `12x12`, `16x16`.
- Workers: 148 persistent CTAs in every variant.
- Dynamic: every CTA claims the next macro-ordered position with the global
  atomic counter.
- Static: CTA `blockIdx.x` owns positions
  `blockIdx.x + iteration * gridDim.x`; no per-tile atomic claim.
- Protocol: one case per process, one warmup, five timed launches, three
  position-rotated process samples per cell.

Both schedulers use exactly the same macro-to-`(tile_m,tile_n)` mapping. The
only changed policy is who owns each linear macro position.

## Measured result

The run used definition commit `94c816f`, one 1,000 W B200, CUDA `12.9.86`,
and driver `580.126.09`. All 36 size/scheduler/shape binaries passed the
pattern validation, and all 12 scheduler/shape policies also passed the 16K
ones validation with zero error.

### 8K

| shape | tasks/macro | dynamic | static | static vs dynamic |
|---:|---:|---:|---:|---:|
| `4x16` | 64 | 1539.508 +/- 1.661 | 1570.090 +/- 1.165 | +1.986% |
| `8x16` | 128 | 1567.626 +/- 1.345 | **1592.794 +/- 1.387** | +1.605% |
| `4x32` | 128 | 1538.616 +/- 0.884 | 1570.937 +/- 1.107 | +2.101% |
| `8x18` | 144 | 1565.102 +/- 0.655 | 1456.009 +/- 1.358 | -6.970% |
| `12x12` | 144 | 1565.216 +/- 3.224 | 1454.463 +/- 0.435 | -7.076% |
| `16x16` | 256 | 1565.671 +/- 2.331 | 1574.797 +/- 0.909 | +0.583% |

### 16K

| shape | tasks/macro | dynamic | static | static vs dynamic |
|---:|---:|---:|---:|---:|
| `4x16` | 64 | 1577.456 +/- 3.502 | 1599.767 +/- 0.805 | +1.414% |
| `8x16` | 128 | 1609.550 +/- 5.207 | **1628.954 +/- 4.274** | +1.206% |
| `4x32` | 128 | 1584.455 +/- 4.749 | 1599.996 +/- 3.789 | +0.981% |
| `8x18` | 144 | 1608.471 +/- 2.914 | 1546.724 +/- 3.804 | -3.839% |
| `12x12` | 144 | 1611.962 +/- 5.189 | 1560.519 +/- 3.462 | -3.191% |
| `16x16` | 256 | 1602.267 +/- 2.733 | 1595.130 +/- 0.874 | -0.445% |

### 32K

| shape | tasks/macro | dynamic | static | static vs dynamic |
|---:|---:|---:|---:|---:|
| `4x16` | 64 | 1392.386 +/- 3.943 | 1395.912 +/- 4.335 | +0.253% |
| `8x16` | 128 | **1400.041 +/- 5.789** | 1392.315 +/- 3.687 | -0.552% |
| `4x32` | 128 | 1395.772 +/- 3.203 | 1392.162 +/- 3.093 | -0.259% |
| `8x18` | 144 | 1393.736 +/- 9.719 | 1376.020 +/- 18.956 | -1.271% |
| `12x12` | 144 | 1373.661 +/- 7.892 | 1307.912 +/- 8.897 | -4.786% |
| `16x16` | 256 | 1298.482 +/- 31.351 | 1266.531 +/- 16.169 | -2.461% |

## Interpretation

`8x16` is the robust shape. Its 128 positions are reasonably close to the
148-worker wave, and all three tile grids divide exactly by both 8 and 16.
Static ownership then removes one global atomic claim and its shared CTA
publication barrier per output tile. This wins clearly at 8K and 16K.

The 144-position argument alone is insufficient. `8x18` pads 12.5% extra
linear positions at every size. `12x12` pads 26.6% at 8K/16K and 6.35% at
32K. Under dynamic assignment, a CTA that sees a padded position immediately
claims another task. Under static assignment, the same padding is permanently
attached to particular 148-stride workers. At 8K, valid work ranges from 4 to
8 tiles per CTA for static `8x18`, versus 6 to 7 for divisible shapes. This
explains the large 8K static regression.

Padding is not the full explanation at larger sizes: fixed ownership also
allows CTAs to drift in time, so the set of concurrently issued TMA requests
no longer follows the ideal consecutive macro wave. This is why static
`12x12` and `16x16` remain poor at 32K even where valid-tile count imbalance
is small or absent.

Selected signed8 candidates are:

- 8K: static `8x16`, `1592.794 TFLOP/s`;
- 16K: static `8x16`, `1628.954 TFLOP/s`;
- 32K: dynamic `8x16`, `1400.041 TFLOP/s`.

## Best candidate comparison

The table below combines the size-specific winners from this scheduler sweep
with the controlled cuBLAS and targeted CUTLASS measurements preserved in
`NSPLIT_SIGNED8_SIZE_COMPARE.md`.

| size | ours best | cuBLAS | ours / cuBLAS | targeted CUTLASS | ours / CUTLASS |
|---:|---:|---:|---:|---:|---:|
| 8K | **1592.794** (static `8x16`) | 1610.837 | **98.880%** | 1485.807 | **107.201%** |
| 16K | **1628.954** (static `8x16`) | 1683.090 | **96.784%** | 1309.717 | **124.375%** |
| 32K | **1400.041** (dynamic `8x16`) | 1422.276 | **98.437%** | 1138.963 | **122.922%** |

All values are TFLOP/s for deterministic BF16 uniform `[-8,8)`, BF16 A/B,
FP32 accumulation/output, one warmup, five timed launches, and three process
samples. The library columns and the new scheduler winners were measured on
the same B200 instance and software stack, but in separate activation
sessions. This is therefore a controlled reference comparison, not a
same-session interleaved comparison. CUTLASS denotes the previously selected
targeted configurations rather than an exhaustive proof of CUTLASS's global
optimum.

The 16K canonical source now uses fixed static ownership with the `8x16`
macro. A matched `[0,1)` measurement remains necessary to characterize its
input-distribution sensitivity.

## Static orientation control

Static `8x16` and its transposed `16x8` orientation were compared separately
at 16K. Both have 128 positions per macro and divide the `64x64` output-tile
grid exactly, so this isolates the A/B reuse orientation without padding or
dynamic load-balancing effects.

| static macro | process samples (TFLOP/s) | mean +/- sample SD | vs `8x16` |
|---:|---:|---:|---:|
| `8x16` | 1628.617 / 1630.365 / 1627.641 / 1625.577 | **1628.050 +/- 1.997** | baseline |
| `16x8` | 1600.183 / 1600.802 / 1596.822 / 1603.149 | **1600.239 +/- 2.612** | **-1.708%** |

This focused run used deterministic BF16 uniform `[-8,8)`, W1/I5, four
ABBA-position-balanced independent processes per variant. Both variants
passed pattern and ones full-C validation with zero error. Static `8x16`
therefore remains canonical.

## Static 8x16 local order

The canonical M-fast order was compared with an N-fast order while keeping
the static scheduler, `8x16` macro shape, macro N-fast order, arithmetic, and
all memory operations fixed.

| input | M-fast | N-fast | N-fast vs M-fast |
|---|---:|---:|---:|
| `[0,1)` | **1839.429 +/- 1.177** | 1781.671 +/- 1.489 | **-3.140%** |
| `[-8,8)` | **1630.438 +/- 1.106** | 1568.869 +/- 2.581 | **-3.776%** |

The run used W1/I5 and four ABBA-position-balanced independent processes per
cell. Both variants used 174 registers with zero stack/local/spill and passed
pattern and ones full-C validation with zero error. The older dynamic
`16x16` N-fast result does not transfer to fixed static `8x16`: here M-fast
is decisively better for both distributions and remains canonical.

Artifacts:

- full scheduler sweep:
  `../results/gemm_nsplit_scheduler_sweep_b200_45481495_20260727_94c816f/`;
- static orientation control:
  `../results/gemm_nsplit_static_orientation_20260727_21d6bda/`;
- static local-order control:
  `../results/gemm_nsplit_static_local_order_20260727_5c1b849/`.
