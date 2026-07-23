# 6.cuBLAS

cuBLAS GEMM throughput benchmark for NVIDIA tensor core paths.

The benchmark excludes CUDA context creation, cuBLAS handle creation, allocation,
initialization, and warmup from the measured region. It reports both CUDA event
time and CPU wall time for a repeated GEMM loop.

## Build

```bash
make build
```

By default the binary is built for `sm_80` and `sm_100`.

## Run

```bash
make run M=16384 N=16384 K=16384 REPEAT=5 WARMUP=1 DEVICE=0
```

For the apples-to-apples `5.GEMM` comparison, use `bf16fp32`:

```bash
./cublas_gemm_bench --device 0 --m 16384 --n 16384 --k 16384 \
  --repeat 5 --warmup 1 --mode bf16fp32 --input-dist unit
```

In this mode A and B are initialized bit-for-bit with the same deterministic
BF16 uniform `[0,1)` generator and seeds used by
`5.GEMM/gemm256_tma_tcgen05_bench.cu`.  The call swaps A and B at the cuBLAS
column-major interface so the mathematical operation is the same row-major
`C=A*B`; `alpha=1`, `beta=0`, FP32 accumulation and FP32 C are used. Allocation,
initialization, handle creation, and warmup are outside the CUDA-event region.
Use `--input-dist signed8` to apply `16*u-8` to the identical random stream and
measure BF16 uniform `[-8,8)` instead.

The reproducible 16K environment-reference experiment runs preserved `p0`,
the current source-backed GEMM, and the exact historical cuBLAS executable in
one position-balanced session:

```bash
OUT_DIR=/workspace/gemm_env_cublas_reference \
P0_BIN=/workspace/benchmark/results/gemm_phase_resweep_b200_45465499/bin/p0 \
CUBLAS_REFERENCE_BIN=/workspace/benchmark/6.cuBLAS/cublas_gemm_bench \
../run_b200_gemm_env_cublas_reference.sh
```

The 2026-07-23 B200 result and its interpretation are recorded in
`../results/gemm_env_cublas_reference_b200_45481495_20260723/summary.md`.

The 8K/32K extension uses the same pinned cuBLAS executable and generates
size-specialized E7a sources from the macro-free 16K baseline. It measures a
direct `16x16` port and a separate size-tuned `8x18` E7a at 32K:

```bash
OUT_DIR=/workspace/gemm_env_cublas_size_extension \
P0_BIN=/workspace/benchmark/results/gemm_phase_resweep_b200_45465499/bin/p0 \
CUBLAS_REFERENCE_BIN=/workspace/benchmark/6.cuBLAS/cublas_gemm_bench \
../run_b200_gemm_env_cublas_size_extension.sh
```

Results, process samples, validation, and the 8K signed confirmation are in
`../results/gemm_env_cublas_size_extension_b200_45481495_20260723/summary.md`.

Supported modes:

- `fp16`: FP16 input/output, FP32 accumulate.
- `bf16`: BF16 input/output, FP32 accumulate.
- `bf16fp32`: BF16 input, FP32 output and accumulation; exact `5.GEMM`
  comparison input.
- `tf32`: FP32 input/output, TF32 tensor cores.
- `fp32`: FP32 input/output, CUDA core FP32 path.
