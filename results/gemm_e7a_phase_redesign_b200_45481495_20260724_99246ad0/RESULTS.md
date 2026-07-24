# E7a phase-redesign B200 result

## Outcome

No candidate improves the current E7a kernel.  The exact canonical baseline
measured `1819.500 +/- 0.641 TFLOP/s` for BF16 uniform `[0,1)` and
`1613.465 +/- 1.236 TFLOP/s` for BF16 uniform `[-8,8)` in the primary
three-process run.  Every proposed phase mechanism was below the predeclared
`+0.5%` adoption gate on at least one distribution, and none showed a
repeatable positive effect in the outlier follow-up.  The canonical source
therefore remains unchanged at phase `0/0`.

## Provenance and fixed conditions

- Definition commit:
  `99246ad06ac7ded30b3363f65fd407dbf3e2b8cb`
- Canonical source SHA-256:
  `37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a`
- Vast instance: `45481495`
- GPU: NVIDIA B200, 148 SMs, 1000 W power limit, 1965 MHz maximum SM clock
- Driver/toolkit: driver 580.126.09; CUDA 12.9; nvcc 12.9.86
- GEMM: row-major BF16 A/B, FP32 accumulate and complete FP32 C,
  `M=N=K=16384`
- Kernel: one `256x256` output per logical CTA task, K64, three shared stages,
  148 dynamic persistent CTAs, `16x16` macro schedule
- Inputs: deterministic BF16 uniform `[0,1)` and `[-8,8)`
- Timing: one case per process, one warmup, five timed launches, CUDA-event
  TFLOP/s
- Primary sampling: three position-rotated processes per variant and input
- Correctness: size-512 pattern and ones full-C validation before timing

## Primary result

The percentages are means of pass-matched changes from the exact baseline.
Mean and `+/-` sample standard deviation are computed over three independent
W1/I5 processes.  The five timed launches inside one process are not treated
as five statistical samples.

| variant | `[0,1)` TFLOP/s | paired change | `[-8,8)` TFLOP/s | paired change |
|---|---:|---:|---:|---:|
| `baseline` | 1819.500 +/- 0.641 | -- | 1613.465 +/- 1.236 | -- |
| `cta4` | 1819.517 +/- 3.568 | +0.001% | 1614.732 +/- 1.310 | +0.079% |
| `cta8` | 1817.449 +/- 3.290 | -0.113% | 1612.529 +/- 0.880 | -0.058% |
| `b1_gap0` | 1805.425 +/- 21.350 | -0.774% | 1612.341 +/- 1.017 | -0.070% |
| `b1_gap32` | 1817.785 +/- 0.605 | -0.094% | 1610.263 +/- 9.186 | -0.199% |
| `b1_gap64` | 1817.348 +/- 3.711 | -0.118% | 1609.389 +/- 6.527 | -0.253% |
| `b1_cross` | 1816.851 +/- 0.858 | -0.146% | 1613.433 +/- 0.350 | -0.002% |

The mechanisms are:

- `cta4`: one-time `(blockIdx.x % 4) * 256`-cycle CTA startup staggering
- `cta8`: one-time `(blockIdx.x % 8) * 128`-cycle CTA startup staggering
- `b1_gap0`: a zero-duration instruction at the per-stage B1 gap site,
  controlling for the changed B1-path code generation
- `b1_gap32/64`: suspend B1's producer warp for 32/64 cycles after A issue
- `b1_cross`: move W3's existing B1 wait before its first MMA half, without
  adding a wait or changing accumulation order

## Preserved outlier and extension

Primary `b1_gap0/[0,1)/pass3` measured `1780.775 TFLOP/s`; its first two
values were `1818.120` and `1817.379`.  The before-process snapshot was 35 C,
226.90 W, and 1965 MHz, matching the neighboring processes.  There is no
recorded thermal or clock event that explains the low value, so it is
retained rather than silently deleted.

After observing it, three additional rotated W1/I5 processes were collected
for the exact baseline and all three B1-gap variants under both inputs.  The
original 42 CSVs were not modified; the 24 extension CSVs are separate.

All six processes give:

| variant | `[0,1)` TFLOP/s | paired vs baseline | `[-8,8)` TFLOP/s | paired vs baseline |
|---|---:|---:|---:|---:|
| `baseline` | 1819.161 +/- 0.997 | -- | 1614.696 +/- 2.404 | -- |
| `b1_gap0` | 1810.922 +/- 14.853 | -0.453% | 1611.485 +/- 4.679 | -0.199% |
| `b1_gap32` | 1815.853 +/- 2.593 | -0.182% | 1611.968 +/- 6.256 | -0.169% |
| `b1_gap64` | 1816.715 +/- 2.626 | -0.134% | 1611.333 +/- 5.013 | -0.208% |

The six paired changes relative to the exact baseline are:

| variant | input | paired mean +/- SD | paired 95% t interval | paired median |
|---|---|---:|---:|---:|
| `b1_gap32` | `[0,1)` | -0.182% +/- 0.159% | [-0.348%, -0.016%] | -0.144% |
| `b1_gap64` | `[0,1)` | -0.134% +/- 0.134% | [-0.275%, +0.007%] | -0.152% |
| `b1_gap32` | `[-8,8)` | -0.169% +/- 0.334% | [-0.520%, +0.182%] | -0.114% |
| `b1_gap64` | `[-8,8)` | -0.208% +/- 0.327% | [-0.551%, +0.135%] | -0.092% |

Relative to the code-generation control, the six-process paired changes are
`+0.278%/+0.325%` for `b1_gap32/64` on `[0,1)`, but
`+0.031%/-0.009%` on `[-8,8)`.  The positive `[0,1)` means are dominated by
the retained `b1_gap0` low sample: their paired 95% intervals both include
zero, and neither direction repeats on signed input.  A median sensitivity
check also leaves all B1 variants 0.07--0.14% below the baseline median.
The signed measurements likewise contain isolated lows in different
variants (`b1_gap32` primary pass 2 and `b1_gap0` extension pass 5), rather
than a phase-specific pattern.  The before-process snapshots remained at
35--36 C and 1965 MHz, but they cannot exclude an unrecorded transient during
the timed kernel.

## Correctness and code generation

All 14 validation runs reported `max_abs=0`, `max_rel=0`, and `bad=0`.
The baseline snapshot is byte-identical to the canonical source.

| variant family | registers | stack/local/spill | static shared |
|---|---:|---:|---:|
| `baseline`, `cta4`, `cta8` | 172 | 0 | 1184 B |
| `b1_gap0`, `b1_gap32`, `b1_gap64` | 180 | 0 | 1184 B |
| `b1_cross` | 174 | 0 | 1184 B |

The `b1_gap0` control is essential because merely exposing the per-stage
sleep site changes register allocation by eight registers.  `b1_gap32/64`
must be compared with both that control and the exact canonical baseline.
Neither comparison supports adoption.

## Decision

Keep exact E7a:

- no one-time CTA cohort offset;
- no per-stage B1 sleep;
- keep W2/W3's current wait/MMA order;
- effective phase remains `0/0`.

Simple phase staggering is now closed for the current dual-wide topology.
Reopen it only if a later pipeline change moves B1 or cross-SM issue
contention onto a measured critical path.

## Artifact layout and reproduction

`summary.md` and `aggregate.csv` are the untouched primary three-pass
summaries.  `csv/` contains the 42 primary process records.
`csv_extension/` contains the 24 follow-up records, and
`extension_rationale.txt` records why they were added.  The immutable local
archive and `SHA256SUMS.remote` retain the generated sources, full SASS,
validation, telemetry, and original downloaded manifest.  The Git artifact
keeps the source, build/resource output, raw measurements, and concise
`logs/sass_counts.txt`, but omits executable binaries and full SASS dumps.
The extracted `sequence_extension.tsv` was normalized from literal `\t` text
to real tab separators after the immutable archive was verified; no
measurement value was changed.  `POST_DOWNLOAD_SHA256SUMS` covers that
normalized file and the two post-run analysis files.

Reproduce the predeclared primary experiment with:

```bash
./run_b200_gemm_e7a_phase_redesign.sh \
  /workspace/benchmark \
  /workspace/gemm_e7a_phase_redesign_b200
```

The full downloaded archive was independently verified against
`gemm_e7a_phase_redesign_b200_45481495_20260724_99246ad0.tar.gz.sha256`, and
all 201 entries in its internal `SHA256SUMS` passed before analysis.
