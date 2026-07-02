# Attention Benchmark

This directory contains the current fused Blackwell attention benchmark.

- `main.cu`: host-side driver, CLI, benchmark, and validation harness
- `attention.cu`: core fused attention and validation CUDA kernels
- `ptx_wrappers.cuh`: low-level PTX/TMA/TCGEN05 helper wrappers
- `Makefile`: local build, run, validation, and trace plot entrypoint (base/PERSIST/BEST/FAST)
- `run.py`: small wrapper for benchmark and trace commands
- `plot_attention_trace.py`: base per-iteration cycle timeline SVG renderer
- `plot_inter_qtile.py`: persistent inter-Q-tile overlap SVG renderer (two-tile trace)
- `sweep_seqlen.sh`: one-shot GPU batch (seqlen sweep + cleanup regression + persistent SVG)
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

## Sequence length and build profiles

`SEQLEN` selects the shape (`1k 2k 4k 8k 16k 32k` <-> k_tiles `8 16 32 64 128 256`). The default
build and `make run`/`make validation` use the 32k base schedule (~1800 TFLOPS at 32k, ~818 at 1k).
Three opt-in profiles pick the schedule at build time:

```bash
make run SEQLEN=1k              # base schedule (~818 at 1k, ~1800 at 32k)
make run SEQLEN=1k PERSIST=1    # persistent CONTINUOUS_FLAT "scr" schedule, the 1k max (~1071)
make run SEQLEN=8k BEST=1       # static per-seqlen best of {base, scr} (crossover from the sweep)
```

- `PERSIST=1` builds the persistent occupancy-1 CONTINUOUS_FLAT kernel; it wins at low seqlen
  (~1071 vs 818 at 1k) but loses to base at high seqlen, so it is opt-in. Its acceptance gate is
  numerical equality (`make validation` + masked-ck / autopsy diff <=1ULP), not raw bit-identity,
  because timing occasionally realizes the codebase's pre-existing benign +-1ULP rounding wobble.
- `BEST=1` compiles, for the given `SEQLEN`, whichever of base/scr the sweep measured faster.
  Measured 2026-07-02 (B200 @1965MHz, TFLOP/s) — **scr wins <=4k, base wins >=8k**:

  | seqlen | 1k | 2k | 4k | 8k | 16k | 32k |
  |---|---|---|---|---|---|---|
  | base | 818 | 1135 | 1404 | **1613** | **1734** | **1799** |
  | scr | **1072** | **1286** | **1448** | 1569 | 1640 | 1676 |

  The per-seqlen picks live in the `Makefile` (`BEST_1k`..`BEST_32k`); re-run `sweep_seqlen.sh` to refresh.
- `FAST=1` builds the persistent QK-peel kernel (~963 at 1k). It is the only trace-able persistent
  kernel (see Plot), so it is kept for `make plot PERSIST=1`.

`make validation` accepts the same `PERSIST=1`/`BEST=1`/`FAST=1` profiles. (1k..4k use extra warmup
so the short iters reach the GPU boost clock; otherwise they under-report.)

`sweep_seqlen.sh` is a one-shot GPU batch: it sweeps base vs scr over 1k..32k, runs the cleanup
regression gate (base kt64 O_CHECKSUM `094ac4da579d0383`, scr `make validation`), and renders the
persistent SVG.

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
one timed trace pass, then renders `log/best.svg`. The default window is iterations
`56..63`; override with `make plot TRACE_START=24 TRACE_ITERS=8`.

`make plot PERSIST=1` renders the persistent inter-Q-tile overlap SVG
(`log/persist_inter_qtile.svg`) via `plot_inter_qtile.py` from a two-tile clock trace
(`-DATTENTION_CLOCK_TRACE_2TILE=1`). It builds the **FAST(963)** kernel, not scr: the
`scr` (CONTINUOUS_FLAT) schedule is `#error`-incompatible with `CLOCK_TRACE` because the
clock-trace records into per-iteration slots keyed off the base kernel's outer per-tile
loop, which CONTINUOUS_FLAT deletes (each role runs its own flat cross-tile loop). Porting
the two-tile captures into the flat seam would let scr reuse this same CSV format and
renderer — that is the intended path to an scr SVG.

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
