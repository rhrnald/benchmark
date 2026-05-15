# Blackwell Attention Microbenchmarks

This workspace contains CUDA microbenchmarks for SM100/Blackwell primitives. The
current primary target is the fused attention pipeline in `0.attention/main.cu`.
The older exploratory kernel files are kept in `0.attention/old_cu/` for
reference, and generated CSV/SVG reports are kept locally under
`0.attention/log/`.

## Main Attention Path

Build the current attention benchmark:

```bash
make attention_custom_kernel_fastest \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'
```

Run the full-size benchmark:

```bash
./build/0.attention/attention_custom_kernel \
  --blocks 4096 --k-tiles 256 --warmup 3 --iters 10 \
  --csv /tmp/attention_main.csv
```

Expected shape on the current system is about 1595 TFLOP/s total for the
BF16-output benchmark with producer regs `120`, consumer regs `184`, and the
default dependency mask `0x00f`. `make` and `make run` use this path; the
default benchmark CSV is written under `/tmp` so generated logs/CSV/SVG do not
enter the repo.

## Validation Path

The validation build uses the same fused `qk_tma_mma_ld_kernel`, not a separate
scalar attention kernel. The kernel accumulates BF16 softmax row sums while
packing `S`, waits for both PV pipes, loads the two TMEM output accumulators,
sums them into shared memory, normalizes by the row sum, packs final
`O[128,128]` to BF16, and stores it to global memory with TMA.

```bash
make attention_custom_kernel_store_output \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'

make attention_validation_fastest \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'
```

Run end-to-end validation:

```bash
./build/0.attention/attention_validation \
  --validate --pattern rank1 --k-tiles 4 \
  --csv /tmp/attention_main_validate_rank1_k4.csv
```

The validation path compares the fused kernel's normalized BF16 output against a
CPU reference using `softmax(QK - row_max) @ V / row_sum`, then checks both the
dequantized normalized values and final BF16 output bits.

The `attention_actual_kernel` and `attention_validation_actual` targets are
aliases for this output-enabled fused kernel path:

```bash
make attention_actual_kernel \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'

make attention_validation_actual \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'
```

Run end-to-end validation:

```bash
./build/0.attention/attention_validation_actual \
  --validate --pattern rank1 --k-tiles 4 \
  --csv /tmp/attention_main_validate_rank1_k4.csv
```

The benchmark path uses `repeats == k_tiles` for actual attention semantics and
stores `O[blocks,128,128]` BF16:

```bash
./build/0.attention/attention_validation_actual \
  --blocks 4096 --k-tiles 256 --warmup 3 --iters 10 \
  --csv /tmp/attention_main_output_perf.csv
```

## Trace Plots

The trace CSV/SVG helpers in `0.attention/plot_attention_trace.py` visualize
cycle-level overlap between K TMA, QK MMA, LD/PACK/ST, V TMA, and PV MMA.
