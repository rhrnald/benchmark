# Recent direct N-split signed8 size comparison

This experiment extends the exact recent direct N-split source that measured
`1807.204 TFLOP/s` on `[0,1)` and `1607.658 TFLOP/s` on `[-8,8)` at 16K.
The canonical source and the phase-shift artifact baseline have identical
SHA-256 `cd595ba3...`.

The 8K and 32K ports change only:

- the compile-time square problem size;
- the persistent macro shape.

The mainloop remains the recent direct N-split dataflow: one `256x64` A TMA,
two independent `64x128` B TMA producers, two `256x128` MMA consumers,
three K64 stages, and full FP32 TMA C store. At 8K and 32K, `16x16`, `12x12`,
and `8x18` macros are measured in the same balanced run. The winning mean is
reported as ours for that size. The 16K cell uses the exact canonical
`16x16` source.

All performance cells use deterministic BF16 uniform `[-8,8)`, one case per
process, one warmup, five timed launches, and three process samples.
