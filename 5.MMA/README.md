# 5.MMA

Prototype Blackwell GEMM compute benchmark using TMA loads and `tcgen05.mma`.

Current kernel shape:

- One CTA computes one logical `256 x 256` C tile.
- The tile is split into four `128 x 128` accumulator tiles in TMEM.
- Each K stage loads:
  - A: `256 x 64` BF16 through TMA.
  - B: `64 x 256` BF16 through TMA.
- Shared memory is partitioned as three `64 KiB` stages:
  - `A_stage`: `256 x 64 x 2B = 32 KiB`
  - `B_stage`: `64 x 256 x 2B = 32 KiB`
  - total triple buffer: `192 KiB`
- The `K=64` stage is issued as four `K=16` `tcgen05.mma` slices for each
  `128 x 128` accumulator tile.
- A and B inputs are stored in an atom-packed operand layout matching the
  tcgen05 shared-memory descriptors. The B tile is packed as two `128 x 64`
  logical `B^T` operands so the kernel still computes logical GEMM
  `C = A(M,K) * B(K,N)`.

The benchmark path consumes TMEM accumulators into a checksum sink. The
validation path stores the full FP32 C matrix to global memory and compares it
against a CPU reference.

## Build

```bash
make build
```

## Run

```bash
make run SIZES=4096,8192,16384,32768 WARMUP=2 ITERS=5 DEVICE=0
```

Or directly:

```bash
./gemm256_tma_tcgen05_bench --device 0 --sizes 4096,8192,16384,32768
```

## Validate

```bash
./gemm256_tma_tcgen05_bench --validate --validate-size 256 --validate-pattern pattern
./gemm256_tma_tcgen05_bench --validate --validate-size 512 --validate-pattern pattern
```
