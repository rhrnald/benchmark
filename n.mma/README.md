# n.mma

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
make run M=16384 N=16384 K=16384 REPEAT=50 WARMUP=10 DEVICE=0
```

Or run one mode directly:

```bash
./cublas_gemm_bench --device 0 --m 16384 --n 16384 --k 16384 \
  --repeat 50 --warmup 10 --mode bf16
```

Supported modes:

- `fp16`: FP16 input/output, FP32 accumulate.
- `bf16`: BF16 input/output, FP32 accumulate.
- `tf32`: FP32 input/output, TF32 tensor cores.
- `fp32`: FP32 input/output, CUDA core FP32 path.
