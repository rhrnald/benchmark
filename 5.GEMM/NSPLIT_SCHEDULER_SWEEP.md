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
