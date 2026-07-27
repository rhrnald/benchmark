# Signed8 best-kernel comparison

This experiment remeasures the strongest source-backed generic persistent
kernel against cuBLAS and the previously selected targeted CUTLASS kernels.

- Input: deterministic BF16 uniform `[-8,8)` for both A and B.
- Operation: square row-major `C=A*B`, BF16 input, FP32 accumulation/output.
- Sizes: `M=N=K=8192,16384,32768`.
- Protocol: one case per process, one warmup, five timed launches, three
  position-balanced process samples per cell.
- Ours: `256x256x64`, three stages, split `64x128` B TMA producers, 148
  persistent CTAs; scheduler macros `16x16`, `16x16`, and `8x18` for
  8K/16K/32K. The compile explicitly enables `GEMM_PERSISTENT_CTA=1`,
  repeat/dense tuning, local M-fast, and macro N-fast; the CLI worker count
  alone does not select the persistent kernel.
- CUTLASS: selected targeted `256x256x64` kernels from the earlier controlled
  comparison: dynamic `2x1` Stream-K at 8K and static `4x1` CLC at 16K/32K.

Run:

```bash
./run_b200_gemm_signed8_best_compare.sh \
  /workspace/benchmark \
  /workspace/gemm_signed8_best_compare
```

The runner validates ours with full-C pattern and ones checks, records source
and binary hashes, rotates size/method order, summarizes results, and creates
a checksummed archive.
