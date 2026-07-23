# E7a per-K-stage pipeline trace on B200

This is the attention-style `clock64()` trace of the current clean E7a
`256x256x64` GEMM.  It replaces the earlier output-tile aggregate view for
pipeline diagnosis: eight steady K64 stages are shown on the physical W0--W3
lanes, with three future reuse stages retained only to close the software
dependency observations.

The main visual result is
[`pipeline_trace_random.svg`](pipeline_trace_random.svg).  Hovering a bar or
marker shows the exact normalized start/end cycle.  The raw 164 records are in
[`data/pipeline_trace_random.csv`](data/pipeline_trace_random.csv), and the
compact per-stage values are in
[`pipeline_trace_random_metrics.csv`](pipeline_trace_random_metrics.csv).

## Measurement scope

| item | value |
|---|---|
| date | 2026-07-23 |
| GPU | NVIDIA B200, Vast instance `45481495` |
| driver / CUDA | `580.126.09` / CUDA `12.9` (`nvcc 12.9.86`) |
| GEMM | `M=N=K=16384`, BF16 A/B uniform `[0,1)`, FP32 C |
| random seeds | A `0xa424cedc`, B `0x62ed12fa` |
| kernel | E7a dual-wide, 4 warps, `256x256x64`, 3-stage ring, 148 persistent CTAs |
| per-stage TMA payload | A 32 KiB; B0 16 KiB; B1 16 KiB |
| dynamic shared memory | 197,632 bytes |
| sampled CTA | block 0, SM 142 |
| sampled output | ninth valid tile (`tile_iter=8`), linear tile 1225, `(tile_m,tile_n)=(25,12)` |
| core window | `kt=56..63` |
| reuse-observation context | `kt=64..66` |
| launch protocol | one untraced full launch, then one traced full launch |
| trace definition commit | `d6bdf8018019c62a1b9a04525a21721dc3f4248b` |
| clean source SHA-256 | `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a` |
| generated trace source SHA-256 | `98038f59e4a9422418d1767a0d919bfb5eb06b4c6d20ccfd48c64392ef7526a8` |
| raw trace SHA-256 | `a6049bc6c9f187dbcfbb64594a0f8df9b8fa3925465b7740f61ee259f584a2cc` |

The B200 was 36 C immediately before validation and 38 C immediately after the
trace.  The instance was stopped after the artifacts were downloaded.

## Correctness

The same generated executable passed both 512 reference checks bit-exactly on
its untraced (`pipeline_trace=nullptr`) path:

| pattern | max abs | max rel | bad |
|---|---:|---:|---:|
| pattern | 0 | 0 | 0 |
| ones | 0 | 0 | 0 |

The exact logs are
[`logs/validate_pattern.log`](logs/validate_pattern.log) and
[`logs/validate_ones.log`](logs/validate_ones.log).
The timestamp/store-active 16K launch wrote the complete trace but did not
separately compare its full C matrix, so active trace-path numerical
correctness is not claimed.

## Pipeline result

The selected CTA advances one completed K64 stage every approximately 1,064
cycles: the median cadence of the later W2/W3 commit endpoint is 1,064 cycles
over the seven core transitions.  Per consumer warp, the commit cadence
medians are 1,002 cycles for W2 and 1,025 cycles for W3.  The two warps trade
the lead rather than one remaining uniformly behind: the same-stage absolute
commit skew has a 193-cycle median and 443-cycle maximum.

The producers are strongly back-pressured by the three-stage MMA reuse
dependency:

- measured reuse-wait-call occupancy is 71.8% on W0 and 77.2% on W1;
- the first ordered dependency, M0/W2, takes a median 689 cycles on W0 and 703
  cycles on W1, with maxima of 983 and 1,013 cycles;
- the following M1/W3 wait is only 61 cycles on W0 and 74 cycles on W1 at the
  median.  This does **not** prove that M1 hardware execution takes 61--74
  cycles: M1 is checked only after the long M0 wait, so it is normally ready by
  then.

The consumer-side data waits show the complementary behavior:

| ordered wait | median / max cycles |
|---|---:|
| A ready | 88 / 704 |
| B0 ready | 86 / 247 |
| B1 ready | 77 / 81 |

These values take the slower W2/W3 wait in each stage before computing the
median.  B1 is effectively hidden by A, B0, and the first MMA half; B0 is also
normally reduced to the ready-check cost.  The visible residual stalls are
mostly A.  All three large A episodes repeat on physical ring slot 1
(`kt=56,59,62`), although which consumer waits longer varies.

The sum of all ready-wait call spans is 38.2% of the traced W2 span and 38.9%
of W3.  Using the minimum observed A/B0/B1 call durations
(`88+86+74=248` cycles/stage) only as an empirical ready-check baseline, the
excess is about 14.7% of the W2 span and 15.2% of the W3 span.  This baseline
subtraction is diagnostic, not a hardware stall counter.  Of the 2,514 excess
wait-call cycles across both warps, 2,327 (92.6%) occur in A waits.

The TMA calls finish well before the consumers reach them:

| value | A | B0 | B1 |
|---|---:|---:|---:|
| median issue-post-call to first wait start | 1,376 | 1,537 | 1,854 |
| median issue-post-call to first wait pass | 1,618 | 1,623 | 1,929 |
| median prepare/issue start to first wait pass bound | 1,697 | 1,756 | 2,005 |

B0/B1's later observation does not mean their transfers are slower; the
consumer observes them only after the ordered A wait and, for B1, after the
first MMA half.  The actual wait-call bars above are the useful residual
readiness signal.

For MMA, the median prepare-start to first producer reuse-pass bound is 1,291
cycles for W2/M0 and 1,322 cycles for W3/M1.  The median commit-post-call to
that observation is 806 and 859 cycles.  Again, W3/M1 is observed late because
the producer first performs the M0 wait, so these are software dependency-pass
bounds rather than tensor-core execution latencies.

Overall, the sampled pipeline overlaps TMA with roughly one to two K64-stage
cadences and hides B traffic well.  The visible sequence is instead consistent
with `M0 reuse back-pressure -> delayed next A issue -> intermittent A-ready
bubble`, but this is a diagnostic association rather than proof of causality.
A useful follow-up is to shift the trace window or sampled tile and test
whether the A bubbles continue to follow physical ring slot 1; a single
instrumented CTA/window is not sufficient to call that a stable slot
imbalance.

## Representative exact cycles

Cycles below are normalized to the first visible record.  This is `kt=56`;
the SVG and raw CSV contain the same start/end data for all eight core stages.

| warp | event | start--end | duration |
|---|---|---:|---:|
| W0 | M0 reuse wait for `kt=53` | 9--712 | 703 |
| W0 | M1 reuse wait for `kt=53` | 712--774 | 62 |
| W0 | A TMA prepare/expect/issue | 774--900 | 126 |
| W0 | B1 TMA prepare/expect/issue | 900--987 | 87 |
| W1 | M0 reuse wait for `kt=53` | 0--714 | 714 |
| W1 | M1 reuse wait for `kt=53` | 714--791 | 77 |
| W1 | B0 TMA prepare/expect/issue | 791--946 | 155 |
| W2 | A / B0 ready waits | 2,110--2,658 / 2,658--2,744 | 548 / 86 |
| W2 | MMA B0 / B1 prepare+issue | 2,744--2,909 / 2,983--3,132 | 165 / 149 |
| W2 | B1 wait / commit | 2,909--2,983 / 3,132--3,191 | 74 / 59 |
| W3 | A / B0 ready waits | 2,238--2,639 / 2,639--2,725 | 401 / 86 |
| W3 | MMA B0 / B1 prepare+issue | 2,725--2,890 / 2,964--3,113 | 165 / 149 |
| W3 | B1 wait / commit | 2,890--2,964 / 3,113--3,172 | 74 / 59 |

## Interpretation limits

- TMA bar end is an issuer post-call timestamp, not DMA completion.
- MMA commit bar end is a post-call timestamp, not tensor-core completion.
- Ready/reuse wait end is a software dependency-pass observation.  It is not
  the exact hardware completion edge.
- Waits are ordered: A then B0 then MMA B0 then B1, and M0 then M1.  A later
  wait can be short because earlier work already hid it.
- `clock64()` spans include any warp descheduling between the two stamps.
- The CSV `base_clock` is the host-selected first visible record, not the
  unsynchronized header timestamp written earlier by kernel thread 0.
- The trace changes compiler code generation.  The clean/trace binaries use
  172/196 registers and 1,184/4,592 bytes of `SHARED` resource according to
  `cuobjdump`; neither spills.  Static TMA instruction-site counts also
  change (`UTMALDG.2D` 7 to 3 and `.4D` 14 to 6), although the source-level
  dynamic loop work and four MMA sites are retained.  Therefore this is an
  **instrumented E7a diagnostic**, not an unperturbed timing of the clean
  performance binary.
- Shared trace-record stores run after each selected stage's observed sequence
  and can perturb the next-stage cadence.  The reported cadence and occupancy
  are therefore properties of this diagnostic binary.
- The dynamic atomic scheduler can assign block 0/tile iteration 8 a different
  output coordinate and L2 history on a rerun.  This is one launch, one CTA,
  and eight core stages; it is not a whole-kernel distribution.
- Do not derive TFLOP/s from this trace launch.

Build logs, full SASS, resource reports, telemetry, source snapshots, and
hashes are retained under [`logs/`](logs/) and [`source/`](source/); all
artifact hashes are listed in [`SHA256SUMS`](SHA256SUMS).

## Exact commands

Starting from trace-definition commit `d6bdf80`:

```bash
python3 5.GEMM/generate_gemm_e7a_pipeline_trace.py \
  --output gemm_e7a_pipeline_trace.cu

/usr/local/cuda/bin/nvcc -std=c++17 -O3 \
  -gencode arch=compute_100a,code=sm_100a \
  -lineinfo -Xptxas=-v \
  gemm_e7a_pipeline_trace.cu -o gemm_trace -lcuda

/usr/local/cuda/bin/nvcc -std=c++17 -O3 \
  -gencode arch=compute_100a,code=sm_100a \
  -lineinfo -Xptxas=-v \
  5.GEMM/baseline/gemm256_bf16_16k.cu -o gemm_clean -lcuda

./gemm_trace --validate --validate-size 512 --validate-pattern pattern
./gemm_trace --validate --validate-size 512 --validate-pattern ones
./gemm_trace --input-init random \
  --pipeline-trace-csv data/pipeline_trace_random.csv

python3 5.GEMM/plot_gemm_e7a_pipeline_trace.py \
  --trace data/pipeline_trace_random.csv \
  --svg pipeline_trace_random.svg \
  --metrics-csv pipeline_trace_random_metrics.csv \
  --title "B200 E7a GEMM per-K-stage pipeline trace"
```
