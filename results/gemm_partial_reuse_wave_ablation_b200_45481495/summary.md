# 16K partial-reuse and persistent-wave ablation

## Conditions

- Date: 2026-07-22
- GPU: NVIDIA B200, Vast.ai instance `45481495`, 1000 W power limit
- Definition commits: `5754c89`, `ba22baf`, `3833a85`
- GEMM: BF16 hash-random `[0,1)`, M=N=K=16384, complete FP32 C output
- Mainloop: 256x256 CTA, K64, three stages, two split `64x128` B TMA
  producer streams, two-CTA B multicast, effective phase shift 0/0
- Measurement: one process per case, warmup 1, five timed launches, three
  independent processes in forward/reverse/rotated order

All two-CTA variants passed a 512 pattern validation bit-exactly.  The two
four-CTA variants passed a cluster-aligned 1024 pattern validation bit-exactly
(`max_abs=0`, `max_rel=0`, `bad=0`).  TFLOP/s is the mean and sample standard
deviation of the three process-level event measurements.

## Operand reuse diagnostic

This comparison changes only the source coordinates supplied to TMA.  Output
tiles, K-loop count, MMA work, wait/commit protocol, multicast, FP32 C store,
148 persistent CTAs, and static 16x16 scheduler are unchanged.

| A source addresses | B source addresses | TFLOP/s | versus dense |
|---|---|---:|---:|
| dense | dense | 1783.283 +/- 1.060 | baseline |
| repeated 256x64 tile | dense | 1899.555 +/- 1.821 | +6.520% |
| dense | repeated 64x256 tile | 1812.678 +/- 1.047 | +1.648% |
| repeated 256x64 tile | repeated 64x256 tile | **1956.014 +/- 1.943** | +9.686% |

The dense-to-full-repeat gap is 172.731 TFLOP/s.  Repeating A alone recovers
116.272 TFLOP/s (67.3% of that gap), whereas repeating B alone recovers only
29.395 TFLOP/s (17.0%).  The effects are not purely additive: the remaining
27.064 TFLOP/s appears only when both operands repeat.  Therefore A locality
is the first target, but the 1956 ceiling cannot be reached by an A-only
schedule.

## Explicit wave and cluster-size comparison

| cluster / scheduler | CTAs | TFLOP/s | versus 148-CTA selected |
|---|---:|---:|---:|
| cluster 2, existing static 16x16 macro | 148 | **1783.283 +/- 1.060** | baseline |
| cluster 2, existing static 16x16 macro | 144 | 1749.012 +/- 0.845 | -1.922% |
| cluster 2, explicit 16x9 wave | 144 | 1587.562 +/- 2.532 | -10.975% |
| cluster 2, explicit 12x12 wave | 144 | 1437.077 +/- 0.808 | -19.414% |
| cluster 4, explicit 16x9 wave | 144 | 933.048 +/- 0.177 | -47.678% |
| cluster 4, explicit 12x12 wave | 144 | 934.301 +/- 0.944 | -47.608% |

Reducing the current scheduler from 148 to 144 CTAs costs only 1.92%, so CTA
count alone does not explain the explicit-wave regressions.  At equal 144-CTA
occupancy, 16x9 and 12x12 lose 9.23% and 17.83% respectively versus the
existing macro schedule.  Forcing a wave to finish as a single grid-stride
epoch removes the favorable interleaving produced when 148 workers walk the
16x16 macro task stream.

Cluster 4 is also decisively rejected.  It saves additional B transactions,
but makes one rank supply and synchronize four CTA consumers; it loses 35--41%
against the corresponding cluster-2 wave.  This is consistent with the
partial-reuse result: eliminating B traffic has much less leverage than the
extra multicast/DSM synchronization and producer serialization costs.

## Reproduction

```bash
./run_b200_gemm_partial_reuse_wave_ablation.sh \
  /workspace/benchmark/5.GEMM \
  /workspace/gemm_partial_reuse_wave_ablation
```

The script compiles every variant, validates it, and runs three rotated
W1/I5 passes.  Raw CSVs, combined stdout, validation logs, exact source/script
snapshots, hashes, and pre/post `nvidia-smi` snapshots are in this directory.
GPU temperature moved only from 31 C to 34 C.
