# Focused dense-address L2 sweep (B200)

## Conditions

- B200 instance `45481495`, 1000 W power limit
- BF16 `256x256x64`, FP32 accumulate and full FP32 TMA C store
- real dense A/B coordinates, three stages, two TMA/MMA pipes
- 148 persistent CTAs, dynamic work queue, local M-fast and macro N-fast
- one case per process, warmup 1, five timed launches, three rotated passes
- every candidate passed the 512 pattern validation bit-exactly

The paired control retains the repeat-tuned phase and 8K promotion from the
matched baseline.  Dense candidates use TMA/MMA phase `0/0`, no A/B L2
promotion, `16x16` at 16K, and the named 144-tile macroblock at 8K/32K.

## Result

| configuration | 8K | 16K | 32K |
|---|---:|---:|---:|
| paired control | 1714.039 +/- 0.363 | 1781.433 +/- 0.168 | 1576.632 +/- 9.325 |
| `8x18` | 1716.684 +/- 1.872 | 1781.543 +/- 0.792 | 1601.834 +/- 14.392 |
| `9x16` | 1713.532 +/- 0.946 | not rerun: identical 16K settings | 1594.414 +/- 3.484 |
| `12x12` | **1717.398 +/- 1.343** | not rerun: identical 16K settings | **1601.945 +/- 4.035** |

Values are mean +/- sample standard deviation in TFLOP/s.  Relative to the
paired control, the selected `12x12` configuration changes 8K by +0.196%, the
common dense 16K configuration by +0.006%, and 32K by +1.605%.

`8x18` and `12x12` have indistinguishable 32K means, but `12x12` has much
lower process variance.  The selected default is therefore `12x12` at
8K/32K and `16x16` at 16K, with phase `0/0`, no promotion, all 148 workers,
and the dynamic queue.  Prior experiments already rejected static scheduling,
128--144 workers, strong one-sided macroblocks, TMA promotion, MMA delay, and
wide single-producer B loads.

The result is an end-to-end dense GEMM.  L2 schedule tuning helps most at 32K
but does not close the 16K gap to the reproduced 1949.590-TFLOP/s same-address
ceiling.  A larger improvement will require valid cross-CTA operand reuse,
most plausibly cluster TMA multicast.

## Reproduction

```bash
./run_b200_gemm256_dense_l2_focused_sweep.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm256_dense_l2_focused_sweep
```

The experiment definition is Git commit `260cef0`.  Raw CSVs, validation
logs, exact source snapshot, binary hashes, and GPU snapshots are stored
beside this summary.
