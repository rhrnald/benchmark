# 16K E7a versus CUTLASS throughput

This is a throughput-only comparison.  No trace-instrumented binary was used.

## Result

| implementation | independent process TFLOP/s | mean ± sample SD | mean runtime |
|---|---:|---:|---:|
| clean E7a | 1797.610 / 1800.297 / 1801.040 / 1801.054 | **1800.000 ± 1.632** | 4.886722 ms |
| selected CUTLASS 16K | 1425.240 / 1425.430 / 1420.530 / 1421.280 | **1423.120 ± 2.577** | 6.180872 ms |

In this exact comparison, clean E7a is **26.483% faster** than the selected
CUTLASS kernel.  The CUTLASS row is the selected 16K configuration described
below, not a claim about every possible CUTLASS kernel.

Each process value is computed from the aggregate CUDA-event time of five
timed launches.  The reported uncertainty is the sample standard deviation
across four process values, not the variation among the five launches inside
one process.

## Controlled conditions

- Date: 2026-07-23
- Vast.ai instance: `45601332`, destroyed after collecting the artifacts
- GPU: NVIDIA B200, UUID
  `GPU-f0c8b07f-01f9-a3f5-9d7d-cfa6e67eea92`
- Driver: `580.126.09`
- E7a build toolkit: CUDA 12.9
- GPU power limit: 1000 W
- Problem: row-major `C=A*B`, `M=N=K=16384`
- Types: BF16 A/B, FP32 accumulation, FP32 C
- Work: `2*M*N*K` floating-point operations
- Input: deterministic BF16 uniform `[0,1)`
- A seed: `20260719 ^ 0xa511e9b3 = 0xa424cedc`
- B seed: `20260719 ^ 0x63d83595 = 0x62ed12fa`
- Per process: one warmup launch followed by five timed launches
- Repetitions: four separate processes per implementation
- Order: E7a/CUTLASS, CUTLASS/E7a, E7a/CUTLASS, CUTLASS/E7a
- Pre-case temperature: 35--36 C
- Pre-case SM clock: 1965 MHz for every measured process

The alternating order prevents a fixed first- or second-run advantage from
systematically favoring either implementation.

## Implementations

### Clean E7a

- Source: `source/gemm256_bf16_16k.cu`
- Source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a`
- Measured binary SHA-256:
  `1d1a7d26b1a071bfd913c52786a58e669a148de08099a8ab827975c71b480940`
- CTA tile: `256x256x64`
- Pipeline: three stages, two B parts, unroll 1
- MMA: `m128n256k16`, split across the M dimension
- Scheduler: 148 persistent CTAs, dynamic `16x16` macroblocks, M-fast order
- Phase shift: disabled (`0/0`)
- Epilogue: FP32 C, TMA store with 128-byte swizzle
- Dynamic shared memory: 197632 bytes

### Selected CUTLASS 16K

- CUTLASS base revision:
  `e8ecfad75b44d1ad56264f5001d877e9e47fe080`
- Modified source: `source/cutlass_70_blackwell_bf16_bench.cu`
- Patch from the base revision: `source/cutlass_70_bench.patch`
- Measured binary SHA-256:
  `5c993a7d1bf77ee7c5e1bbe2568cbc0bf965f23d592a53d60ab72fe16eeaf754`
- MMA tile: `256x256x64`
- Cluster: static `4x1x1`, 2-SM MMA
- Pipeline: five stages
- Epilogue: direct FP32 store
- Scheduler: Cluster Launch Control (CLC)

The CUTLASS driver swaps column-major operands so its raw output memory is the
same row-major `C=A*B` layout as E7a.  Both drivers use the same deterministic
initializer and seeds.

## Correctness scope

Before timing, the clean E7a binary passed exact 512-square `pattern` and
`ones` validation (`bad=0`, `max_abs=0`, `max_rel=0`).  A full 16K reference
GEMM was intentionally not run inside either timing process.  The selected
CUTLASS benchmark also skips its expensive full reference GEMM, so this run
establishes matched-input throughput rather than a new independent full-size
correctness proof.

Allocation, deterministic input initialization, and benchmark setup are
outside the timed region.  The timed region includes the complete GEMM kernel,
including the FP32 output write.

## Reproduce

Build clean E7a on the B200 host:

```bash
/usr/local/cuda/bin/nvcc -std=c++17 -O3 \
  -gencode arch=compute_100a,code=sm_100a -lineinfo -Xptxas=-v \
  source/gemm256_bf16_16k.cu -o bin/gemm_e7a_clean -lcuda
```

Place the selected CUTLASS binary at
`bin/cutlass_bf16_best_clc_bench`, then run:

```bash
BIN_DIR="$PWD/bin" OUT_DIR="$PWD/results" \
  ../../run_b200_gemm_e7a_cutlass_16k_compare.sh

python3 ../../5.GEMM/summarize_e7a_cutlass_16k.py "$PWD/results"
```

The exact execution order is in `sequence.tsv`.  `aggregate.csv` and
`comparison_table.md` contain the machine-readable and compact Markdown
summaries.  The four E7a CSVs and CUTLASS logs preserve the individual process
measurements; `logs/` also contains validation, build, environment, and
per-case GPU snapshots.
