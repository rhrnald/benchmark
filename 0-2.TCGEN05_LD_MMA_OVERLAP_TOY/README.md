# tcgen05.ld vs tcgen05.mma overlap toy

This folder contains a small reproducible benchmark for the TMEM contention claim:

> The bandwidth lost by `tcgen05.ld` when overlapped with `tcgen05.mma` is close to the
> accumulator C-read bandwidth required by peak `tcgen05.mma`.

## Build

```bash
cd /home/snu_avq1/workspace/chaewon/benchmark
make tcgen05_ld_mma_overlap_toy
```

## Reproduce the main claim

```bash
./build/0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_toy \
  --peak-only \
  --blocks 4096 \
  --repeats 8192 \
  --ld-repeats 81920 \
  --warmup 2 \
  --iters 5

python3 0-2.TCGEN05_LD_MMA_OVERLAP_TOY/analyze_ld_mma_contention.py
```

Outputs:

- `tcgen05_ld_mma_peak_overlap.csv`: raw peak-style rows.
- `tcgen05_ld_mma_contention_claim.csv`: one-row derivation of the claim.
- `tcgen05_ld_mma_contention_report.md`: human-readable summary.

## What is measured

- `ld_peak_only`: 4 consumer warps issue `tcgen05.ld.32x32b.x128` in a long loop.
- `mma4_peak_only`: 4 producer warps issue long `m128n128k16` BF16 MMA streams.
- `mma4_plus_ld_peak_overlap`: both run in the same CTA using independent TMEM columns.

The comparison uses the measured drop in logical LD bandwidth:

```text
LD drop = LD-only TB/s - overlapped LD TB/s
```

It then compares that drop to reference peak MMA C-read demand:

```text
FLOP/MMA = 2 * 128 * 128 * 16
C bytes/MMA = 128 * 128 * 4
C-read TB/s = reference_mma_TFLOP/s * C bytes/MMA / FLOP/MMA
```

The default reference MMA peak is `2207.606 TFLOP/s`, taken from
`1.mma/mma_throughput_bench.csv` for `m128n128k16 acc32`.

## Notes

- The peak-style overlap kernel intentionally avoids mid-loop waits.
- `ld-repeats` is larger than `repeats` so LD remains active for the MMA interval.
- The result is a contention argument, not a numerical correctness benchmark.
