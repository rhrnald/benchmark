#define main gemm256_reference_main_unused
#include "../5.GEMM/gemm256_tma_tcgen05_bench.cu"
#undef main

namespace repeated_tile_gemm {

constexpr int kRtM = 128;
constexpr int kRtN = 256;
constexpr int kRtPanelK = 128;
constexpr int kRtMmaK = 16;
constexpr int kRtSlices = kRtPanelK / kRtMmaK;
constexpr int kRtStages = 3;
constexpr int kRtIssuerWarps = 2;
constexpr int kRtWarps = 4;
constexpr int kRtThreads = kRtWarps * 32;
constexpr int kRtOperandWords = kRtM * kRtPanelK / 2;
constexpr int kRtHalfWords = kRtOperandWords / 2;
constexpr int kRtOperandBytes = kRtOperandWords * sizeof(uint32_t);
constexpr int kRtStageWords = 2 * kRtOperandWords;
constexpr int kRtStageBytes = kRtStageWords * sizeof(uint32_t);
constexpr int kRtDynamicSmemBytes = kRtStages * kRtStageBytes + 1023;
constexpr int kRtTmemTileStride = 128;
constexpr double kRtFlopsPerMma = 2.0 * kRtM * 128.0 * kRtMmaK;

struct RtArgs {
  int device = 0;
  int blocks = 592;
  int steps = 8192;
  int warmup = 3;
  int iters = 10;
  uint32_t seed = 20260718u;
  bool validate = false;
  const char* csv = "gemm128x256_repeated_tile.csv";
};

__device__ __forceinline__ uint32_t rt_mix32(uint32_t x) {
  x += 0x9e3779b9u;
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  return x ^ (x >> 16);
}

__device__ __forceinline__ uint16_t rt_uniform_bf16(uint32_t seed) {
  const uint32_t random24 = rt_mix32(seed) >> 8;
  const float value = static_cast<float>(random24) * 0x1.0p-24f;
  uint32_t bits = __float_as_uint(value);
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

__global__ void rt_fill(uint32_t* data, size_t words, uint32_t seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= words) return;
  const uint32_t index = static_cast<uint32_t>(i);
  const uint16_t lo = rt_uniform_bf16(seed ^ index ^ 0x9e3779b9u);
  const uint16_t hi = rt_uniform_bf16(seed ^ index ^ 0x243f6a88u);
  data[i] = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

void rt_encode_a(CUtensorMap* map, void* base) {
  const cuuint64_t global_dim[2] = {kRtPanelK / 2, kRtM};
  const cuuint64_t global_stride[1] = {
      static_cast<cuuint64_t>(kRtPanelK / 2) * sizeof(uint32_t)};
  const cuuint32_t box_dim[2] = {kRtPanelK / 4, kRtM};
  const cuuint32_t elem_stride[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(
                   map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, base, global_dim,
                   global_stride, box_dim, elem_stride,
                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "rt_encode_a");
}

void rt_encode_b(CUtensorMap* map, void* base) {
  constexpr cuuint64_t cols_words = 128 / 2;
  const cuuint64_t global_dim[4] = {cols_words, kRtMmaK, 2, kRtSlices};
  const cuuint64_t global_stride[3] = {
      cols_words * sizeof(uint32_t),
      static_cast<cuuint64_t>(128 / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kRtMmaK) * cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {128 / 4, kRtMmaK, 2, kRtSlices / 2};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(
                   map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, base, global_dim,
                   global_stride, box_dim, elem_stride,
                   CU_TENSOR_MAP_INTERLEAVE_NONE,
                   CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "rt_encode_b");
}

__global__ __launch_bounds__(kRtThreads, 1) void rt_gemm_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    float* __restrict__ c,
    int steps) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)c;
  (void)steps;
#else
  extern __shared__ uint32_t raw_smem[];
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw_smem) + 1023u) &
      ~static_cast<uintptr_t>(1023u);
  uint32_t* pipeline = reinterpret_cast<uint32_t*>(aligned);

  __shared__ __align__(8) uint64_t a_ready[kRtStages];
  __shared__ __align__(8) uint64_t b_ready[kRtStages];
  __shared__ __align__(8) uint64_t stage_done[kRtStages];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool lane0 = lane == 0;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int stage = 0; stage < kRtStages; ++stage) {
      mbarrier_init(&a_ready[stage], 1);
      mbarrier_init(&b_ready[stage], 1);
      mbarrier_init(&stage_done[stage], kRtIssuerWarps);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane0) tmem_base_shared = taddr;
  }
  __syncthreads();

  if (warp == 0 && lane0) {
    for (int step = 0; step < steps; ++step) {
      const int stage = step % kRtStages;
      if (step >= kRtStages) {
        const int old_step = step - kRtStages;
        mbarrier_wait(&stage_done[stage],
                      static_cast<uint32_t>((old_step / kRtStages) & 1));
      }
      uint32_t* stage_base = pipeline + stage * kRtStageWords;
      // B precedes A so the major-MN descriptor footprint never touches the
      // end of the dynamic shared-memory allocation.
      uint32_t* b_smem = stage_base;
      uint32_t* a_smem = stage_base + kRtOperandWords;

      mbarrier_expect_tx(&a_ready[stage], kRtOperandBytes);
      tma_load_2d(&a_map, smem_ptr_u32(a_smem), &a_ready[stage], 0, 0);
      tma_load_2d(&a_map, smem_ptr_u32(a_smem + kRtHalfWords),
                  &a_ready[stage], kRtPanelK / 4, 0);
      mbarrier_expect_tx(&b_ready[stage], kRtOperandBytes);
      tma_load_4d(&b_map, smem_ptr_u32(b_smem), &b_ready[stage], 0, 0, 0, 0);
      tma_load_4d(&b_map, smem_ptr_u32(b_smem + kRtHalfWords),
                  &b_ready[stage], 0, 0, 0, kRtSlices / 2);
    }
    const int drain_begin = steps > kRtStages ? steps - kRtStages : 0;
    for (int step = drain_begin; step < steps; ++step) {
      const int stage = step % kRtStages;
      mbarrier_wait(&stage_done[stage],
                    static_cast<uint32_t>((step / kRtStages) & 1));
    }
  }

  if ((warp == 1 || warp == 2) && lane0) {
    const int issuer = warp - 1;
    // A complete FP32 128x128 accumulator occupies 128 TMEM columns.
    const uint32_t d_taddr =
        tmem_base_shared + issuer * kRtTmemTileStride;
    const uint32_t idesc = make_bf16_idesc() | (1u << 16);
    for (int step = 0; step < steps; ++step) {
      const int stage = step % kRtStages;
      const uint32_t ready_phase =
          static_cast<uint32_t>((step / kRtStages) & 1);
      uint32_t* stage_base = pipeline + stage * kRtStageWords;
      uint32_t* b_smem = stage_base;
      uint32_t* a_smem = stage_base + kRtOperandWords;
      mbarrier_wait(&a_ready[stage], ready_phase);
      mbarrier_wait(&b_ready[stage], ready_phase);
      tcgen05_fence_after_thread_sync();
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint32_t* a_half = a_smem + half * kRtHalfWords;
        uint32_t* b_half = b_smem + half * kRtHalfWords;
#pragma unroll
        for (int kk = 0; kk < kRtSlices / 2; ++kk) {
          const int slice = half * (kRtSlices / 2) + kk;
          const uint64_t a_desc =
              make_sw128_major_k_smem_desc(smem_ptr_u32(a_half), kk);
          const uint64_t b_desc =
              make_sw128_major_mn_smem_desc(smem_ptr_u32(b_half), kk);
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc,
                              step != 0 || slice != 0);
        }
      }
      tcgen05_commit(&stage_done[stage]);
    }
  }
  __syncthreads();

  // Four warps cooperatively store both independent 128x128 accumulators.
  const int row_base = static_cast<int>(blockIdx.x) * kRtM;
  store_128x128_float_tile(tmem_base_shared, c, kRtN, row_base, 0);
  store_128x128_float_tile(tmem_base_shared + kRtTmemTileStride, c, kRtN,
                           row_base, 128);
  __syncthreads();

  if (warp == 0) tcgen05_dealloc_512cols(tmem_base_shared);
  __syncthreads();
  if (warp == 0) tcgen05_relinquish_alloc_permit();
#endif
}

int rt_positive(const char* text, const char* option) {
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end || value <= 0 || value > (1 << 30)) {
    std::fprintf(stderr, "invalid %s: %s\n", option, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

RtArgs rt_parse(int argc, char** argv) {
  RtArgs args;
  for (int i = 1; i < argc; ++i) {
    const char* key = argv[i];
    auto take = [&]() {
      if (++i >= argc) {
        std::fprintf(stderr, "missing value for %s\n", key);
        std::exit(EXIT_FAILURE);
      }
      return argv[i];
    };
    if (!std::strcmp(key, "--device")) args.device = std::atoi(take());
    else if (!std::strcmp(key, "--blocks")) args.blocks = rt_positive(take(), key);
    else if (!std::strcmp(key, "--steps")) args.steps = rt_positive(take(), key);
    else if (!std::strcmp(key, "--warmup")) args.warmup = rt_positive(take(), key);
    else if (!std::strcmp(key, "--iters")) args.iters = rt_positive(take(), key);
    else if (!std::strcmp(key, "--seed")) args.seed = static_cast<uint32_t>(std::strtoul(take(), nullptr, 10));
    else if (!std::strcmp(key, "--csv")) args.csv = take();
    else if (!std::strcmp(key, "--validate")) args.validate = true;
    else if (!std::strcmp(key, "--help")) {
      std::printf("Usage: %s [--blocks N] [--steps N] [--warmup N] "
                  "[--iters N] [--seed N] [--validate] [--csv PATH]\n", argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "unknown option: %s\n", key);
      std::exit(EXIT_FAILURE);
    }
  }
  return args;
}

float rt_bf16(uint16_t bits) {
  uint32_t word = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

void rt_validate(const CUtensorMap& a_map, const CUtensorMap& b_map,
                 const std::vector<uint32_t>& a, const std::vector<uint32_t>& b,
                 float* d_c) {
  constexpr int steps = 2;
  rt_gemm_kernel<<<1, kRtThreads, kRtDynamicSmemBytes>>>(a_map, b_map, d_c,
                                                         steps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> got(static_cast<size_t>(kRtM) * kRtN);
  CUDA_CHECK(cudaMemcpy(got.data(), d_c, got.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t bad = 0;
  for (int row = 0; row < kRtM; ++row) {
    for (int col = 0; col < kRtN; ++col) {
      float ref = 0.0f;
      for (int kk = 0; kk < kRtPanelK; ++kk) {
        const uint32_t aw = a[(row * kRtPanelK + kk) / 2];
        const uint32_t bw = b[(kk * 128 + (col & 127)) / 2];
        const uint16_t ab = static_cast<uint16_t>(aw >> (16 * (kk & 1)));
        const uint16_t bb =
            static_cast<uint16_t>(bw >> (16 * ((col & 127) & 1)));
        ref += rt_bf16(ab) * rt_bf16(bb);
      }
      ref *= steps;
      const float value = got[static_cast<size_t>(row) * kRtN + col];
      const double abs_error = std::abs(static_cast<double>(value) - ref);
      const double rel_error = abs_error / std::max(1.0, std::abs(static_cast<double>(ref)));
      max_abs = std::max(max_abs, abs_error);
      max_rel = std::max(max_rel, rel_error);
      if (abs_error > 2.0e-3 * std::max(1.0, std::abs(static_cast<double>(ref)))) ++bad;
    }
  }
  std::printf("validation: %s bad=%zu max_abs=%.6g max_rel=%.6g\n",
              bad == 0 ? "PASS" : "FAIL", bad, max_abs, max_rel);
  if (bad != 0) std::exit(EXIT_FAILURE);
}

}  // namespace repeated_tile_gemm

int main(int argc, char** argv) {
  using namespace repeated_tile_gemm;
  const RtArgs args = rt_parse(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
  if (prop.major < 10) return 77;

  uint32_t* d_a = nullptr;
  uint32_t* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, kRtOperandBytes));
  CUDA_CHECK(cudaMalloc(&d_b, kRtOperandBytes));
  const size_t c_elements = static_cast<size_t>(args.blocks) * kRtM * kRtN;
  CUDA_CHECK(cudaMalloc(&d_c, c_elements * sizeof(float)));
  rt_fill<<<(kRtOperandWords + 255) / 256, 256>>>(
      d_a, kRtOperandWords, args.seed ^ 0xa511e9b3u);
  rt_fill<<<(kRtOperandWords + 255) / 256, 256>>>(
      d_b, kRtOperandWords, args.seed ^ 0x63d83595u);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUtensorMap a_map{}, b_map{};
  rt_encode_a(&a_map, d_a);
  rt_encode_b(&b_map, d_b);
  CUDA_CHECK(cudaFuncSetAttribute(rt_gemm_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kRtDynamicSmemBytes));

  if (args.validate) {
    std::vector<uint32_t> a(kRtOperandWords), b(kRtOperandWords);
    CUDA_CHECK(cudaMemcpy(a.data(), d_a, kRtOperandBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_b, kRtOperandBytes, cudaMemcpyDeviceToHost));
    rt_validate(a_map, b_map, a, b, d_c);
  }

  auto run = [&](int steps) {
    for (int i = 0; i < args.warmup; ++i) {
      rt_gemm_kernel<<<args.blocks, kRtThreads, kRtDynamicSmemBytes>>>(
          a_map, b_map, d_c, steps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < args.iters; ++i) {
      rt_gemm_kernel<<<args.blocks, kRtThreads, kRtDynamicSmemBytes>>>(
          a_map, b_map, d_c, steps);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return total_ms / args.iters;
  };

  const float ms_n = run(args.steps);
  const float ms_2n = run(args.steps * 2);
  const double delta_ms = ms_2n - ms_n;
  const double added_flops = static_cast<double>(args.blocks) * args.steps *
                             kRtIssuerWarps * kRtSlices * kRtFlopsPerMma;
  const double tflops = added_flops / (delta_ms * 1.0e9);
  const double logical_bytes = static_cast<double>(args.blocks) * args.steps *
                               2.0 * kRtOperandBytes;
  const double logical_tbps = logical_bytes / (delta_ms * 1.0e9);

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) return 1;
  std::fprintf(csv, "blocks,steps_n,steps_2n,warmup,iters,seed,ms_n,ms_2n,delta_ms,differential_tflops,logical_tma_TBps,c_store_bytes\n");
  std::fprintf(csv, "%d,%d,%d,%d,%d,%u,%.9f,%.9f,%.9f,%.6f,%.6f,%zu\n",
               args.blocks, args.steps, args.steps * 2, args.warmup, args.iters,
               args.seed, ms_n, ms_2n, delta_ms, tflops, logical_tbps,
               c_elements * sizeof(float));
  std::fclose(csv);
  std::printf("Device: %s, repeated random BF16 A/B 128x128, C=%dx128x256 FP32\n",
              prop.name, args.blocks);
  std::printf("N %.6f ms, 2N %.6f ms, delta %.6f ms: %.3f TFLOP/s, logical TMA %.3f TB/s\n",
              ms_n, ms_2n, delta_ms, tflops, logical_tbps);

  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_a));
  return 0;
}
