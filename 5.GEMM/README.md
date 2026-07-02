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
    overlaid on the triple-buffer storage. `GEMM_CSTORE_CHUNK_N=256` is also
    supported for paired `128 x 256` TMA stores, but it is not the balanced
    default across the benchmark sizes.
- The `K=64` stage is issued as four `K=16` `tcgen05.mma` slices for each
  `128 x 128` accumulator tile.
- The mainloop uses two N-direction MMA pipes:
  - warp 0 lane 0 issues shared A TMA plus B pipe 0.
  - warp 1 lane 0 issues B pipe 1.
  - warp 2 lane 0 issues MMA for C00/C10.
  - warp 3 lane 0 issues MMA for C01/C11.
  - stage reuse is fenced by each pipe's `mma_done` barrier from three K stages earlier.
- C-store benchmark runs use an M-major rectangular CTA swizzle by default
  (`12 x 1` groups for sizes with at least 32 CTA tiles per dimension). This
  preserves B-tile locality better than the earlier square swizzle when FP32 C
  stores are enabled. Pipe 1 uses a size-specific phase shift: `96` cycles for
  8K and `1000` cycles for larger sizes.
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

## Build

```bash
make build
```

## Run

```bash
make run SIZES=4096,8192,16384,32768 WARMUP=2 ITERS=5 DEVICE=0
make run SIZES=4096,8192,16384,32768 WARMUP=5 ITERS=30 STORE_C=--store-c-tma
```

Or directly:

```bash
./gemm256_tma_tcgen05_bench --device 0 --sizes 4096,8192,16384,32768
```

## Validate

```bash
./gemm256_tma_tcgen05_bench --validate --validate-size 256 --validate-pattern pattern
./gemm256_tma_tcgen05_bench --validate --validate-size 512 --validate-pattern pattern
./gemm256_tma_tcgen05_bench --validate --validate-size 512 --validate-pattern pattern --store-c-tma
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
