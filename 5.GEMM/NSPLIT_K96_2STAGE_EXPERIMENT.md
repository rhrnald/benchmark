# Two-stage K96 experiment

## Layout

Both logical shared-memory slots have the same 96 KiB capacity:

| item per slot | shape | bytes |
|---|---:|---:|
| A | `256 x 96` BF16 | 48 KiB |
| B0 | `96 x 128` BF16 | 24 KiB |
| B1 | `96 x 128` BF16 | 24 KiB |

Two slots consume 192 KiB, equal to the canonical three K64 stages. Each
consumer warp issues twelve `tcgen05.mma` instructions per stage: six K16
slices times two M128 blocks.

## TMA representation

A 192-byte row is wider than one SW128 span. A is therefore stored as three
independently SW64-swizzled `256 x 32` slabs with three ready barriers.
B0 and B1 each use one K96 TMA transaction.

## K tail

`K=16384` is not divisible by 96. The kernel launches 171 stages and relies on
TMA out-of-bounds zero fill for the final K64 tail. Validation at K=256 and
K=512 exercises K64 and K32 tails respectively.

## Protocol

- Full-C validation: sizes 256 and 512, formula pattern and ones.
- Performance: BF16 `M=N=K=16384`, FP32 output.
- Inputs: uniform `[0,1)` and `[-8,8)`.
- One warmup and five timed iterations per process.
- Five AB/BA-interleaved processes against the canonical 3xK64 kernel.
- Preserve compiler resources and SASS in the result archive.

## Outcome

The implementation passed bit-exact full-C validation. At 16K it reached
1568.677 TFLOP/s on `[0,1)` and 1429.819 TFLOP/s on `[-8,8)`, corresponding
to 88.52% and 91.34% of the canonical 3xK64 kernel measured in the same
interleaved run. The canonical kernel remains unchanged.

The measured `clock64` pipeline comparison against the canonical three-stage
K64 implementation is in
[`RESULTS.md`](../results/gemm_k64_k96_pipeline_trace_b200_46382599_20260731/RESULTS.md).
