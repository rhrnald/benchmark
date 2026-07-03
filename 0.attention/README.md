# Attention Benchmark

This directory contains the current fused Blackwell attention benchmark.

- `main.cu`: host-side driver, CLI, benchmark, and validation harness
- `attention.cu`: core fused attention and validation CUDA kernels
- `ptx_wrappers.cuh`: low-level PTX/TMA/TCGEN05 helper wrappers
- `Makefile`: local build, run, validation, and trace plot entrypoint (base/PERSIST/BEST/STABLE)
- `run.py`: small wrapper for benchmark and trace commands
- `plot_attention_trace.py`: base per-iteration cycle timeline SVG renderer
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

`SEQLEN` selects the shape (`1k 2k 4k 8k 16k 32k` <-> k_tiles `8 16 32 64 128 256`).

```bash
make run SEQLEN=8k                 # base schedule, unordered V (default; fast, ck wobbles)
make run SEQLEN=8k STABLE=1        # order the V waits -> one deterministic checksum per shape
make run SEQLEN=1k PERSIST=1       # persistent scr schedule, fastest at low seqlen
make run SEQLEN=8k BEST=1          # per-seqlen best of {base, scr}: scr <=4k, base >=8k
make run SEQLEN=8k CHECKSUM=1      # also print raw+masked O_CHECKSUM (works with any profile)
make run SEQLEN=32k CAUSAL=1       # causal mask (base profile only; see CAUSAL_HANDOFF.md)
```

Per-seqlen TFLOP/s (B200 @1965MHz):

| SEQLEN | 1k | 2k | 4k | 8k | 16k | 32k |
|---|---|---|---|---|---|---|
| base (default) | 818 | 1135 | 1404 | 1613 | 1734 | 1799 |
| base `STABLE=1` | 789 | 1110 | 1381 | 1586 | 1714 | 1780 |
| scr (`PERSIST=1`) | 1012 | 1240 | 1418 | 1552 | 1631 | 1673 |

The default matches unordered V handling (fast; the raw checksum wobbles run-to-run). `STABLE=1` orders the V waits so every shape gives one deterministic checksum, bit-identical. `BEST=1` picks scr `<=4k` / base `>=8k`.

- `PERSIST=1` (scr) targets bit-stable checksums (~1012 at 1k); if you only require `make validation`
  to pass and give up run-to-run checksum stability entirely, the same schedule reaches **~1071 at 1k**.

`make validation` accepts the same `PERSIST=1`/`BEST=1`/`STABLE=1` profiles. (1k..4k use extra warmup
so the short iters reach the GPU boost clock; otherwise they under-report.)

`CAUSAL=1` applies the causal mask at runtime (same base binary, dynamic
`<0,0,true>` dispatch). Each query tile qb of a k_tiles window walks k tiles
`[0, qb]` only (~half the work), with an LPT launch order. Causal is base-build
only: it is incompatible with `PERSIST=1`/`BEST=1` (the persistent schedule
assumes a fixed trip count per tile). `make validation-suite` runs the fused
suite including the causal cases. Design/status: `CAUSAL_HANDOFF.md`.

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

The persistent (scr) schedule is not supported by `make plot`.

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
