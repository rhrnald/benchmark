# Two-stage K96 GEMM ablation on B200

## Configuration

- GPU: NVIDIA B200, Vast instance `46381755`.
- Shape/type: BF16 `M=N=K=16384`, FP32 output.
- Scheduler: 148 persistent CTAs, static 8x16 macrotiles, M-fast.
- Candidate: two equal K96 stages, 96 KiB each.
- A per stage: three `256x32` SW64 slabs with independent barriers.
- B per stage: one `96x128` TMA per consumer pipe.
- Timing: one warmup plus five timed iterations per process.
- Statistics: five independent AB/BA-interleaved processes per input.

## Correctness

Full-C validation passed bit-exactly for:

- K=256, pattern and ones; final K64 tail.
- K=512, pattern and ones; final K32 tail.

The initial SW128 K32-slab attempt failed because a 64-byte row did not match
the assumed packed layout. The measured candidate uses matching SW64 TMA and
MMA descriptors.

## Performance

Event TFLOP/s, mean ± sample standard deviation:

| input | canonical 3xK64 | two-stage K96 | relative |
|---|---:|---:|---:|
| uniform `[0,1)` | 1772.122 ± 1.839 | 1568.677 ± 0.858 | 88.52% |
| uniform `[-8,8)` | 1565.444 ± 5.182 | 1429.819 ± 1.058 | 91.34% |

K96 is substantially better than the earlier unequal K64/K128 candidate, but
it remains 8.7-11.5% below the regular three-stage K64 pipeline.

## Code generation

| binary | registers, 16K specialization | static UTCHMMA sites | TMA 2D sites | TMA 4D sites |
|---|---:|---:|---:|---:|
| canonical | 164 | 24 | 16 | 36 |
| K96 | 162 | 36 | 39 | 26 |

The K96 kernel does not spill. Its three A slabs reduce the intended TMA issue
savings: each K96 stage uses three A TMA instructions plus B0 and B1, versus
one A plus B0/B1 for each canonical K64 stage.

## Interpretation

Equal K96 stages remove the severe alternating cadence of K64/K128, explaining
the large recovery. However, two logical buffers still provide less
producer/consumer decoupling than three K64 buffers. K96 also issues twelve
MMA instructions per consumer completion group and requires three independent
A transfers. These costs outweigh the reduction from 256 to 171 logical
stages.

The canonical kernel was not replaced.

## Artifacts

- `logs/paired.txt`: complete interleaved measurements.
- `logs/validation_v2.txt`: passing correctness tests.
- `logs/build_baseline.txt` and `logs/build_k96_v2.txt`: compiler resources.
- `logs/sass_*.txt` and `logs/sass_counts.txt`: disassembly evidence.
- `source/`: exact measured sources.
