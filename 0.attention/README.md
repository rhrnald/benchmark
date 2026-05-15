# Attention Benchmark

This directory contains the current fused Blackwell attention benchmark.

- `main.cu`: host-side driver, CLI, benchmark, and validation harness
- `attention.cu`: core fused attention and validation CUDA kernels
- `ptx_wrappers.cuh`: low-level PTX/TMA/TCGEN05 helper wrappers
- `Makefile`: local build, run, validation, and trace plot entrypoint
- `run.py`: small wrapper for benchmark and trace commands
- `plot_attention_trace.py`: cycle timeline SVG renderer
- `old_cu/main_full.cu`: pre-cleanup full source with compile-time experiment options
- `old_cu/`: older exploratory CUDA kernels kept for reference
- `log/`: ignored local benchmark logs, CSV files, and SVG plots

## Quick Start

From this directory:

```bash
make
make run
make plot
make validation
```

`make` compiles the fastest benchmark binary only. `make run` measures the
full-size benchmark. `make plot` captures a clock trace and writes the cycle
timeline SVG. `make validation` runs the fused correctness path.

## Benchmark

Default benchmark shape:

```text
blocks=4096
k_tiles=256
warmup=3
iters=10
```

The expected result on the current system is about `1594 TFLOP/s` total. The
latest checked run was:

```text
total_TFLOP_per_s=1593.769 status=ok
```

The benchmark CSV is written to:

```text
log/best.csv
```

## Plot

`make plot` builds a trace-enabled binary with `-DATTENTION_CLOCK_TRACE=1`, runs
one timed trace pass, then renders:

```text
log/best.svg
```

The default plot window is iterations `56..63`. Override it like this:

```bash
make plot TRACE_START=24 TRACE_ITERS=8
```

## Validation

`make validation` checks the fused kernel against a CPU reference for:

```text
B=1, H=1, Sq=128, Skv=512, D=128, pattern=rank1
```

It compares the full validation output tile, `O[128,128]`. It does not compare
the full benchmark output `O[4096,128,128]`.

The validation CSV is written to:

```text
/tmp/attention_main_validate_rank1_k4.csv
```

## Useful Overrides

```bash
make run BLOCKS=4096 K_TILES=256 WARMUP=3 ITERS=10
make run RUN_CSV=log/my_run.csv
make validation VALIDATE_PATTERN=random VALIDATE_K_TILES=8
make plot TRACE_SVG=log/my_trace.svg
```

Use `NVCCFLAGS` to add ptxas output or extra compile flags:

```bash
make NVCCFLAGS='-O3 -std=c++17 -gencode=arch=compute_100a,code=sm_100a --expt-relaxed-constexpr -Xptxas=-v'
```
