# `256x256` same-address GEMM reproduction (B200)

This reruns the 2026-07-20 corrected asynchronous same-address control under
the current measurement standard.

## Conditions

- B200 instance `45481495`, 1000 W power limit
- BF16 uniform `[0,1)` A/B, FP32 accumulation and full FP32 TMA C store
- valid CTA tile `256x256x64` with distinct left/right B panels
- three shared-memory stages and two TMA/MMA pipes
- every CTA and K stage reloads the same global A/B panels
- normal grid and 148-worker persistent grid variants
- one case per process, warmup 1, five timed launches
- three process-level samples in rotated order
- both variants passed the 512 pattern validation bit-exactly

## Result

| variant | 8K | 16K | 32K |
|---|---:|---:|---:|
| normal | 1807.666 +/- 0.955 | 1945.916 +/- 2.176 | 1779.193 +/- 4.920 |
| persistent, 148 CTAs | **1813.887 +/- 0.655** | **1949.590 +/- 1.080** | **1782.437 +/- 2.072** |

Values are mean +/- sample standard deviation in TFLOP/s.

## Historical comparison

| variant | size | 2026-07-20 | reproduction | change |
|---|---:|---:|---:|---:|
| normal | 8K | 1877.233 | 1807.666 | -3.706% |
| persistent | 8K | 1874.692 | 1813.887 | -3.243% |
| normal | 16K | 1953.929 | 1945.916 | -0.410% |
| persistent | 16K | 1955.753 | 1949.590 | -0.315% |
| normal | 32K | 1781.176 | 1779.193 | -0.111% |
| persistent | 32K | 1785.165 | 1782.437 | -0.153% |

The central 16K result reproduces the historical 1.956-PFLOP/s observation:
the persistent mean is 1.950 PFLOP/s and only 0.315% lower.  The 32K result
also reproduces within 0.2%.  The 8K result is stable across the three new
processes but approximately 3.2--3.7% below the single historical sample, so
the old 8K value should not be used as a reproduced anchor.

This demonstrates that a mathematically valid, distinct-B, full-C-store
pipeline can exceed 1900 TFLOP/s when a `256x256` CTA repeatedly reads the
same global tiles.  The current `128x256` distinct-B result near 1754 TFLOP/s
is therefore not a global hardware ceiling.

## Reproduction

```bash
./run_b200_gemm256_same_address_repro.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm256_same_address_repro
```

The experiment definition is Git commit `0200756`.  Raw CSVs, validation
logs, exact source snapshot, binary hashes, and pre/post GPU snapshots are
stored beside this summary.
