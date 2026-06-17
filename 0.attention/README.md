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

## Sequence length (1k vs 32k)

The default build and `make run`/`make validation` use the 32k base schedule (~1800 TFLOPS).
To run the 1k shape you must pass an option:

```bash
make run SEQLEN=1k            # 1k on the base kernel (~818)
make run SEQLEN=1k FAST=1     # 1k on the persistent QK-peel kernel (~963)
```

`FAST=1` builds a separate persistent occupancy-1 kernel that wins at 1k (~963 vs 818) but is
~4% slower at 32k, so it is opt-in.

- TODO: (1) push 1k higher by overlapping more of the epilogue's idle tensor core; (2) cut the
persistent per-tile overhead at 32k (per-tile mbarrier re-init, the inter-tile `__syncthreads`)
so one persistent build can coexist with / match the base 32k schedule.

Validation is the same: `make validation` checks the default
base kernel; add `FAST=1` to validate the persistent kernel. (1k uses extra warmup so the short
iters reach the GPU boost clock; otherwise it under-reports ~954 instead of ~963.)

## Benchmark

Default benchmark shape:

```text
blocks=4096
k_tiles=256
warmup=3
iters=10
```

The default build uses the current fastest measured schedule. The latest
checked serial 100-run benchmark was:

```text
ok=100 fail=0 avg=1801.207 TFLOP/s min=1800.598 max=1801.617
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

## Dangerous Optimization Notes

The default path is the current fastest measured schedule:

```text
ATTENTION_SPLIT_V_TMA=1
ATTENTION_SPLIT_V_H0_WITH_K_TMA=1
ATTENTION_SPLIT_V_H0_BEFORE_K_TMA=0
ATTENTION_PIPE1_TMA_HEAD_DELAY_CYCLES=1728
ATTENTION_CROSS_PIPE_PHASE=0
ATTENTION_SKIP_V_H0_READY_WAIT=1
ATTENTION_SKIP_V_H1_READY_WAIT=1
ATTENTION_SKIP_V_TMA_EXPECT_TX=1
ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER=9
```

This schedule has two experimental dependency relaxations. First, it skips the
explicit ready waits between:

```text
V TMA h0 done -> PV h0
V TMA h1 done -> PV h1
```

Second, `ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER=9` commits `qk_done` after all
QK MMA issues and one PV h0 MMA issue, then issues the remaining PV h0 MMAs.
This is risky because it changes the meaning of `qk_done` from "QK+PV h0
fully issued" to "QK+PV h0 early-enough for the measured schedule". Values
below 9 are not currently allowed; `ATTENTION_QK_PVH0_EARLY_COMMIT_AFTER=8`
failed validation during testing.

Current validation status:

```text
rank1, B=1,H=1,Sq=128,Skv=32768,D=128, checksum-repeats=100: ok
```

Observed checksum values:

```text
rank1: 8bbfabd24db01067
```

Measured full benchmark performance for the default path is:

```text
blocks=4096, k_tiles=256, warmup=3, iters=10, serial repeats=100
ok=100 fail=0 avg=1801.207 TFLOP/s min=1800.598 max=1801.617
```

The 100-repeat rank1 validation completed with `checksum_stable=yes`. The
100-repeat serial benchmark also completed with `100 ok / 0 fail`, so no
unspecified CUDA error was observed for this default build in that run.

This is not a proven dependency removal. It is only known to pass the current
validation coverage and should be treated as schedule-sensitive until tested
against broader data patterns and timing perturbations.

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
