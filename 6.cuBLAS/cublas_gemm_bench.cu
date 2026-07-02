#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t status = (call);                                                \
    if (status != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(status));                                 \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CHECK_CUBLAS(call)                                                      \
  do {                                                                          \
    cublasStatus_t status = (call);                                             \
    if (status != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,     \
                   static_cast<int>(status));                                   \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

template <typename T>
__global__ void init_kernel(T *ptr, size_t n, float scale) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = (static_cast<float>(idx % 251) - 125.0f) * scale;
    ptr[idx] = static_cast<T>(v);
  }
}

template <>
__global__ void init_kernel<__half>(__half *ptr, size_t n, float scale) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = (static_cast<float>(idx % 251) - 125.0f) * scale;
    ptr[idx] = __float2half(v);
  }
}

template <>
__global__ void init_kernel<__nv_bfloat16>(__nv_bfloat16 *ptr, size_t n,
                                           float scale) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const float v = (static_cast<float>(idx % 251) - 125.0f) * scale;
    ptr[idx] = __float2bfloat16(v);
  }
}

template <typename T>
void init_device(T *ptr, size_t n, float scale) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  init_kernel<T><<<blocks, threads>>>(ptr, n, scale);
  CHECK_CUDA(cudaGetLastError());
}

struct Options {
  int device = 0;
  int m = 16384;
  int n = 16384;
  int k = 16384;
  int repeat = 50;
  int warmup = 10;
  std::string mode = "bf16";
};

void usage(const char *argv0) {
  std::printf(
      "Usage: %s [--device N] [--m M] [--n N] [--k K] [--repeat R] "
      "[--warmup W] [--mode fp16|bf16|bf16fp32|tf32|fp32]\n",
      argv0);
}

Options parse_options(int argc, char **argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    auto need_arg = [&](const char *name) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", name);
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };

    if (std::strcmp(argv[i], "--device") == 0) {
      opt.device = std::atoi(need_arg("--device"));
    } else if (std::strcmp(argv[i], "--m") == 0) {
      opt.m = std::atoi(need_arg("--m"));
    } else if (std::strcmp(argv[i], "--n") == 0) {
      opt.n = std::atoi(need_arg("--n"));
    } else if (std::strcmp(argv[i], "--k") == 0) {
      opt.k = std::atoi(need_arg("--k"));
    } else if (std::strcmp(argv[i], "--repeat") == 0) {
      opt.repeat = std::atoi(need_arg("--repeat"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      opt.warmup = std::atoi(need_arg("--warmup"));
    } else if (std::strcmp(argv[i], "--mode") == 0) {
      opt.mode = need_arg("--mode");
    } else if (std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }

  if (opt.m <= 0 || opt.n <= 0 || opt.k <= 0 || opt.repeat <= 0 ||
      opt.warmup < 0) {
    std::fprintf(stderr, "m/n/k/repeat must be positive and warmup >= 0\n");
    std::exit(EXIT_FAILURE);
  }
  return opt;
}

template <typename T>
void *alloc_and_init(size_t elements, float scale) {
  T *ptr = nullptr;
  CHECK_CUDA(cudaMalloc(&ptr, elements * sizeof(T)));
  init_device<T>(ptr, elements, scale);
  return ptr;
}

struct GemmConfig {
  cudaDataType_t a_type;
  cudaDataType_t b_type;
  cudaDataType_t c_type;
  cublasComputeType_t compute_type;
  cublasGemmAlgo_t algo;
  size_t a_element_size;
  size_t b_element_size;
  size_t c_element_size;
  const char *label;
};

GemmConfig config_for_mode(const std::string &mode) {
  if (mode == "fp16") {
    return {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP, sizeof(__half), sizeof(__half),
            sizeof(__half), "FP16 input/output, FP32 accumulate"};
  }
  if (mode == "bf16") {
    return {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP, sizeof(__nv_bfloat16),
            sizeof(__nv_bfloat16), sizeof(__nv_bfloat16),
            "BF16 input/output, FP32 accumulate"};
  }
  if (mode == "bf16fp32") {
    return {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP, sizeof(__nv_bfloat16),
            sizeof(__nv_bfloat16), sizeof(float),
            "BF16 input, FP32 output, FP32 accumulate"};
  }
  if (mode == "tf32") {
    return {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP, sizeof(float), sizeof(float),
            sizeof(float),
            "TF32 tensor cores, FP32 input/output"};
  }
  if (mode == "fp32") {
    return {CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC,
            CUBLAS_GEMM_DEFAULT, sizeof(float), sizeof(float), sizeof(float),
            "FP32 CUDA cores, FP32 input/output"};
  }

  std::fprintf(stderr, "Unsupported mode: %s\n", mode.c_str());
  std::exit(EXIT_FAILURE);
}

void *alloc_and_init_by_type(cudaDataType_t type, size_t elements, float scale) {
  if (type == CUDA_R_16F) {
    return alloc_and_init<__half>(elements, scale);
  }
  if (type == CUDA_R_16BF) {
    return alloc_and_init<__nv_bfloat16>(elements, scale);
  }
  if (type == CUDA_R_32F) {
    return alloc_and_init<float>(elements, scale);
  }
  std::fprintf(stderr, "Unsupported allocation type: %d\n",
               static_cast<int>(type));
  std::exit(EXIT_FAILURE);
}

const char *type_name(cudaDataType_t type) {
  switch (type) {
  case CUDA_R_16F:
    return "fp16";
  case CUDA_R_16BF:
    return "bf16";
  case CUDA_R_32F:
    return "fp32";
  default:
    return "unknown";
  }
}

double ms_since(std::chrono::steady_clock::time_point start,
                std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv) {
  Options opt = parse_options(argc, argv);

  CHECK_CUDA(cudaSetDevice(opt.device));
  CHECK_CUDA(cudaFree(nullptr));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, opt.device));

  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

  GemmConfig cfg = config_for_mode(opt.mode);

  const size_t a_elems = static_cast<size_t>(opt.m) * opt.k;
  const size_t b_elems = static_cast<size_t>(opt.k) * opt.n;
  const size_t c_elems = static_cast<size_t>(opt.m) * opt.n;

  void *A = nullptr;
  void *B = nullptr;
  void *C = nullptr;
  A = alloc_and_init_by_type(cfg.a_type, a_elems, 1.0e-3f);
  B = alloc_and_init_by_type(cfg.b_type, b_elems, 1.0e-3f);
  C = alloc_and_init_by_type(cfg.c_type, c_elems, 0.0f);
  CHECK_CUDA(cudaDeviceSynchronize());

  const float alpha = 1.0f;
  const float beta = 0.0f;

  auto run_gemm = [&]() {
    CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, opt.m, opt.n,
                              opt.k, &alpha, A, cfg.a_type, opt.m, B,
                              cfg.b_type, opt.k, &beta, C, cfg.c_type,
                              opt.m, cfg.compute_type, cfg.algo));
  };

  for (int i = 0; i < opt.warmup; ++i) {
    run_gemm();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  const auto wall_start = std::chrono::steady_clock::now();
  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < opt.repeat; ++i) {
    run_gemm();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  const auto wall_stop = std::chrono::steady_clock::now();

  float event_total_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&event_total_ms, start, stop));

  const double wall_total_ms = ms_since(wall_start, wall_stop);
  const double flops_per_gemm =
      2.0 * static_cast<double>(opt.m) * opt.n * opt.k;
  const double event_avg_ms = static_cast<double>(event_total_ms) / opt.repeat;
  const double wall_avg_ms = wall_total_ms / opt.repeat;
  const double event_tflops = flops_per_gemm / (event_avg_ms * 1.0e-3) / 1.0e12;
  const double wall_tflops = flops_per_gemm / (wall_avg_ms * 1.0e-3) / 1.0e12;
  const double gib =
      static_cast<double>(a_elems * cfg.a_element_size +
                          b_elems * cfg.b_element_size +
                          c_elems * cfg.c_element_size) /
      (1024.0 * 1024.0 * 1024.0);

  std::printf("device=%d name=\"%s\" cc=%d.%d\n", opt.device, prop.name,
              prop.major, prop.minor);
  std::printf("mode=%s (%s)\n", opt.mode.c_str(), cfg.label);
  std::printf("m=%d n=%d k=%d repeat=%d warmup=%d\n", opt.m, opt.n, opt.k,
              opt.repeat, opt.warmup);
  std::printf("a_type=%s b_type=%s c_type=%s compute_type=%d memory=%.2f GiB\n",
              type_name(cfg.a_type), type_name(cfg.b_type),
              type_name(cfg.c_type), static_cast<int>(cfg.compute_type), gib);
  std::printf(
      "event_total_ms=%.3f event_avg_ms=%.6f event_TFLOPS=%.3f\n",
      event_total_ms, event_avg_ms, event_tflops);
  std::printf("wall_total_ms=%.3f wall_avg_ms=%.6f wall_TFLOPS=%.3f\n",
              wall_total_ms, wall_avg_ms, wall_tflops);
  std::printf(
      "csv,%s,%d,%d,%d,%d,%d,%.6f,%.3f,%.6f,%.3f,%s\n",
      opt.mode.c_str(), opt.m, opt.n, opt.k, opt.repeat, opt.warmup,
      event_avg_ms, event_tflops, wall_avg_ms, wall_tflops, prop.name);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  CHECK_CUDA(cudaFree(A));
  CHECK_CUDA(cudaFree(B));
  CHECK_CUDA(cudaFree(C));
  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
