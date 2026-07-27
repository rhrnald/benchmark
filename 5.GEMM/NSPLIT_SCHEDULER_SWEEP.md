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

The canonical source is not changed yet. The static `8x16` candidate needs a
matched `[0,1)` confirmation before passing the existing two-distribution
adoption rule.

Artifact:
`../results/gemm_nsplit_scheduler_sweep_b200_45481495_20260727_94c816f/`.
