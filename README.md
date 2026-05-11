# Blackwell Attention Microbenchmarks

This workspace contains CUDA microbenchmarks for SM100/Blackwell primitives. The
current primary target is the fused attention pipeline in `0.attention/`.

## Fast Attention Path

Build the current fastest attention benchmark without output-store code:

```bash
make attention_custom_kernel_fastest \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'
```

Run the full-size benchmark:

```bash
./build/0.attention/attention_custom_kernel \
  --blocks 4096 --repeats 8192 --k-tiles 8192 --warmup 2 --iters 3 \
  --csv /tmp/attention_fastest.csv
```

Expected shape on the current system is about 1722 TFLOP/s total with
`ATTENTION_NVCC_MANAGED_LD_REGS=1`, producer regs `128`, consumer regs `184`,
and `ATTENTION_STORE_OUTPUT=0`.

## Output Store and Validation Path

Output storage is intentionally compiled into a separate binary because even
dead output-store code can perturb the fastest benchmark schedule.

```bash
make attention_custom_kernel_store_output \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'

make attention_validation_fastest \
  NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'
```

Run row-wise max/sum production-ish validation:

```bash
./build/0.attention/attention_validation \
  --stage all --row-max --pattern rank1 --k-tiles 4 \
  --csv 0.attention/attention_validation_fastest_rowmax_fused_rank1_k4.csv
```

The validation path checks QK, row max, softmax numerator, row sum, PV,
normalized output, and fused-kernel output captured from TMEM.

## Trace Plots

The trace CSV/SVG helpers in `0.attention/plot_attention_trace.py` visualize
cycle-level overlap between K TMA, QK MMA, LD/PACK/ST, V TMA, and PV MMA.
