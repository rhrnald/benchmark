# Blackwell Benchmark Workspace

CUDA microbenchmarks and experiments for SM100/Blackwell attention, TMEM,
TCGEN05, TMA, and MMA pipeline behavior.

## Folders

- `0.attention/`: Current fused attention benchmark, validation harness, trace
  plotter, and archived attention kernel variants.
- `0-1.TCGEN05_LD_PACK_TOY/`: Small TCGEN05 load and register-pack experiment.
- `0-2.TCGEN05_LD_MMA_OVERLAP_TOY/`: TCGEN05 load versus MMA overlap and
  contention toy benchmark.
- `0-3.TMA_SMEM_STORE_OVERLAP_TOY/`: TMA and shared-memory store overlap toy
  benchmark.
- `1.mma/`: MMA throughput and scheduling microbenchmarks.
- `n.mma/`: cuBLAS GEMM throughput benchmark.
- `2.TMEM_LDST/`: TMEM load/store behavior microbenchmark.
- `3.MMA_LD_PIPELINE/`: MMA and load pipeline overlap experiments.
- `3-1.MMA_LD_PIPELINE_128KB/`: 128KB working-set variant of the MMA/load
  pipeline experiment.
- `3-2.TMA_MMA_LD_PIPELINE/`: TMA, MMA, and load pipeline overlap experiment.
- `4.TMA_MULTICAST/`: TMA multicast microbenchmark.
