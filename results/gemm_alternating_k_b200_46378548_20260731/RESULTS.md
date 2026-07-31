# Alternating K64/K128 shared-memory stages on B200

## Experiment

- Commit: `326f141` (unsplit candidate is preserved at `acbe790`).
- GPU: NVIDIA B200, instance `46378548`.
- Shape/type: BF16 GEMM, `M=N=K=16384`, FP32 output.
- Scheduler: 148 persistent CTAs, static 8x16 macrotiles, M-fast.
- Timing: one warmup plus five timed kernel iterations per process.
- Statistics: five independent processes per variant and distribution.
- Ordering: baseline/candidates were alternated AB/BA.
- Baseline: three uniform K64 stages.
- Candidate: one K64 slot alternating with one K128 slot, same 192 KiB
  mainloop capacity.

Full-C validation at `M=N=K=512` passed bit-exactly for both the formula
pattern and all-ones input. The formula pattern was repeated ten times for the
unsplit candidate and five times for the split candidate.

## Results

Event TFLOP/s, mean ± sample standard deviation:

| input | 3xK64 baseline | K64/K128, one MMA group | relative | K64/K128, two MMA groups | relative |
|---|---:|---:|---:|---:|---:|
| uniform `[0,1)` | 1853.878 ± 1.375 | 1411.446 ± 0.562 | 76.13% | 1389.382 ± 0.819 | 74.94% |
| uniform `[-8,8)` | 1644.888 ± 3.922 | 1374.802 ± 4.662 | 83.58% | 1362.523 ± 2.647 | 82.83% |

Splitting the K128 consumer work into two eight-MMA completion groups did not
recover performance; it reduced throughput by another 1.6% for `[0,1)` and
0.9% for `[-8,8)`. Therefore a 16-MMA completion group is not the primary
cause of the regression.

## Interpretation

The experiment reduces logical barrier epochs and B TMA transactions, but it
also changes a regular three-stage pipeline into uneven K64/K128 bursts. The
measured loss is much larger than noise and shows that equal total shared
memory and equal K192 reuse distance are not sufficient: issue granularity
and regular stage cadence matter more here.

The generated alternating kernel also has substantially larger steady-loop
code because both K64 and K128 paths remain in the compiled control flow.
Across the three specialized shapes in each binary, `cuobjdump` contains 72
static `UTCHMMA` sites for the alternating binaries versus 24 for baseline.
This count is a code-generation indicator, not a dynamic instruction count,
but it identifies branch/code-size cleanup as the prerequisite for any fair
second attempt.

The canonical three-K64 kernel was not replaced.

## Artifacts

- `logs/paired_all.txt`: all interleaved process outputs.
- `logs/validation_v7.txt`: unsplit correctness repetitions.
- `logs/validation_split.txt`: split-group correctness repetitions.
- `logs/build_baseline.txt`, `logs/build_alternating_v7.txt`, and
  `logs/build_alternating_split.txt`: compiler resource reports.
- `logs/sass_*.txt`: SASS dumps.
- `source/`: exact compiled sources.
- `logs/sha256.txt`: source and binary hashes.
