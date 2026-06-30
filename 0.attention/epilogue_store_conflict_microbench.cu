#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CUDA_CHECK(stmt)                                                        \
  do {                                                                         \
    cudaError_t err__ = (stmt);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt, __FILE__,    \
                   __LINE__, cudaGetErrorString(err__));                       \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

static constexpr int kWarpSize = 32;
static constexpr int kPipeCount = 2;
static constexpr int kConsumerWarpsPerPipe = 4;
static constexpr int kProducerWarpCount = 4;
static constexpr int kConsumerBaseWarp = kProducerWarpCount;
static constexpr int kWarps = kProducerWarpCount + kPipeCount * kConsumerWarpsPerPipe;
static constexpr int kThreads = kWarps * kWarpSize;
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kTileWords = kTileM * kTileN / 2;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kDynamicSmemBytes = kTileBytes + 1024;
static constexpr int kChunksPerHalf = 4;
static constexpr int kStoresPerChunk = 2;
static constexpr int kEpilogueStoreWarps = kPipeCount * kConsumerWarpsPerPipe;
static constexpr int kStoresPerCtaRepeat =
    kEpilogueStoreWarps * kChunksPerHalf * kStoresPerChunk;

enum class LayoutKind : int {
  kRowMajor = 0,
  kAtomMajor = 1,
  kTmaSwizzle128B = 2,
  kTmaOnly = 3,
};

struct Args {
  int blocks = 148;
  int repeats = 256;
  int warmup = 0;
  int iters = 1;
  LayoutKind layout = LayoutKind::kRowMajor;
  bool tma = true;
  bool tma_swizzle_128b = false;
  bool tma_single = false;
  bool validate = false;
};

struct RunResult {
  float ms = 0.0f;
  cudaError_t error = cudaSuccess;
  const char* status = "ok";
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__host__ __device__ __forceinline__ int atom_major_k_word_offset(int row,
                                                                 int col_pair) {
  const int k16_atom = col_pair >> 3;
  const int pair_in_atom = col_pair & 7;
  const int row_group8 = row >> 3;
  const int row_in8 = row & 7;
  const int chunk16 = pair_in_atom >> 2;
  const int word_in_chunk = pair_in_atom & 3;
  return k16_atom * 1024 + row_group8 * 64 + chunk16 * 32 + row_in8 * 4 +
         word_in_chunk;
}

__host__ __device__ __forceinline__ int tma_sw128_half_word_offset(int row,
                                                                   int col_pair) {
  const int half = col_pair >> 5;
  const int in_half = col_pair & 31;
  return half * (kTileWords / 2) + row * (kTileN / 4) +
         (in_half ^ ((row & 7) << 2));
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

uint16_t float_to_bf16_bits_host(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

uint32_t pack_bf16_pair_host(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits_host(lo)) |
         (static_cast<uint32_t>(float_to_bf16_bits_host(hi)) << 16);
}

float pipe_accum_value_host(int pipe, int row, int col, int repeat, int block) {
  const float base = pipe == 0 ? 0.015625f : 0.0234375f;
  const float row_term = static_cast<float>((row & 31) + 1) * 0.001953125f;
  const float col_term = static_cast<float>((col & 63) + 1) * 0.000244140625f;
  const float iter_term = static_cast<float>((repeat + block) & 15) * 0.0001220703125f;
  return base + row_term + col_term + iter_term;
}

uint32_t make_epilogue_word_host(int row, int col_pair, int repeat, int block) {
  const int col0 = col_pair * 2;
  const int col1 = col0 + 1;
  const float row_bias = static_cast<float>((row & 7) + 1) * 0.00390625f;
  const float pipe0_scale = 0.75f + row_bias;
  const float pipe1_scale = 1.25f - row_bias;
  const float inv_sum = 1.0f / (2.0f + static_cast<float>((row & 3) + 1) * 0.25f);
  const float lo =
      (pipe_accum_value_host(0, row, col0, repeat, block) * pipe0_scale +
       pipe_accum_value_host(1, row, col0, repeat, block) * pipe1_scale) *
      inv_sum;
  const float hi =
      (pipe_accum_value_host(0, row, col1, repeat, block) * pipe0_scale +
       pipe_accum_value_host(1, row, col1, repeat, block) * pipe1_scale) *
      inv_sum;
  return pack_bf16_pair_host(lo, hi);
}

__device__ __forceinline__ uint32_t pack_bf16_pair_device(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits_device(lo)) |
         (static_cast<uint32_t>(float_to_bf16_bits_device(hi)) << 16);
}

__device__ __forceinline__ float pipe_accum_value(int pipe,
                                                  int row,
                                                  int col,
                                                  int repeat,
                                                  int block) {
  const float base = pipe == 0 ? 0.015625f : 0.0234375f;
  const float row_term = static_cast<float>((row & 31) + 1) * 0.001953125f;
  const float col_term = static_cast<float>((col & 63) + 1) * 0.000244140625f;
  const float iter_term = static_cast<float>((repeat + block) & 15) * 0.0001220703125f;
  return base + row_term + col_term + iter_term;
}

__device__ __forceinline__ uint32_t make_epilogue_word(int row,
                                                       int col_pair,
                                                       int repeat,
                                                       int block) {
  const int col0 = col_pair * 2;
  const int col1 = col0 + 1;
  const float row_bias = static_cast<float>((row & 7) + 1) * 0.00390625f;
  const float pipe0_scale = 0.75f + row_bias;
  const float pipe1_scale = 1.25f - row_bias;
  const float inv_sum = 1.0f / (2.0f + static_cast<float>((row & 3) + 1) * 0.25f);
  const float lo =
      (pipe_accum_value(0, row, col0, repeat, block) * pipe0_scale +
       pipe_accum_value(1, row, col0, repeat, block) * pipe1_scale) *
      inv_sum;
  const float hi =
      (pipe_accum_value(0, row, col1, repeat, block) * pipe0_scale +
       pipe_accum_value(1, row, col1, repeat, block) * pipe1_scale) *
      inv_sum;
  return pack_bf16_pair_device(lo, hi);
}

__device__ __forceinline__ uint4 make_epilogue_vec4(int row,
                                                    int col_pair,
                                                    int repeat,
                                                    int block) {
  return make_uint4(make_epilogue_word(row, col_pair + 0, repeat, block),
                    make_epilogue_word(row, col_pair + 1, repeat, block),
                    make_epilogue_word(row, col_pair + 2, repeat, block),
                    make_epilogue_word(row, col_pair + 3, repeat, block));
}

__device__ __forceinline__ void store_shared_u4(uint32_t* dst, uint4 value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(dst);
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "r"(addr), "r"(value.x), "r"(value.y), "r"(value.z),
                 "r"(value.w)
               : "memory");
#else
  reinterpret_cast<uint4*>(dst)[0] = value;
#endif
}

__device__ __forceinline__ void tma_store_fence() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_4d(const CUtensorMap* map,
                                             uint32_t src_smem,
                                             int c,
                                             int r,
                                             int d,
                                             int b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
      " [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(map), "r"(src_smem), "r"(c), "r"(r), "r"(d), "r"(b)
      : "memory");
#else
  (void)map;
  (void)src_smem;
  (void)c;
  (void)r;
  (void)d;
  (void)b;
#endif
}

__device__ __forceinline__ void tma_store_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_wait_group_read() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
}

template <LayoutKind Layout, bool kDoTma>
__global__ __launch_bounds__(kThreads, 1) void epilogue_store_conflict_kernel(
    const __grid_constant__ CUtensorMap o_map,
    uint32_t* __restrict__ sink,
    int repeats,
    int tma_swizzle_128b,
    int tma_single) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)o_map;
  (void)sink;
  (void)repeats;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* output_bf16_smem = reinterpret_cast<uint32_t*>(smem_addr);

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    if constexpr (Layout != LayoutKind::kTmaOnly) {
      const bool epilogue_warp =
          warp_id >= kConsumerBaseWarp && warp_id < kConsumerBaseWarp + kEpilogueStoreWarps;
      if (epilogue_warp) {
        const int epilogue_slot = warp_id - kConsumerBaseWarp;
        const int consumer_warp = epilogue_slot & (kConsumerWarpsPerPipe - 1);
        const int consumer_half = epilogue_slot / kConsumerWarpsPerPipe;
        const int row = consumer_warp * kWarpSize + lane;
        const int col_pair_base = consumer_half * (kTileN / 4);
#pragma unroll
        for (int chunk = 0; chunk < kChunksPerHalf; ++chunk) {
#pragma unroll
          for (int vec = 0; vec < kStoresPerChunk; ++vec) {
            const int col_pair = col_pair_base + chunk * 8 + vec * 4;
            const int word_offset =
                Layout == LayoutKind::kRowMajor
                    ? row * (kTileN / 2) + col_pair
                    : (Layout == LayoutKind::kAtomMajor
                           ? atom_major_k_word_offset(row, col_pair)
                           : tma_sw128_half_word_offset(row, col_pair));
            const uint4 packed =
                make_epilogue_vec4(row, col_pair, repeat, static_cast<int>(blockIdx.x));
            store_shared_u4(output_bf16_smem + word_offset, packed);
          }
        }
      }
    }

    if constexpr (kDoTma) {
      tma_store_fence();
      __syncthreads();
      if (warp_id == 0 && lane == 0) {
        if (tma_swizzle_128b) {
          if (tma_single) {
            tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem), 0, 0, 0,
                         static_cast<int>(blockIdx.x));
          } else {
            tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem), 0, 0,
                         static_cast<int>(blockIdx.x), 0);
            tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem + kTileWords / 2),
                         kTileN / 2, 0, static_cast<int>(blockIdx.x), 0);
          }
        } else {
          tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem), 0, 0,
                       static_cast<int>(blockIdx.x), 0);
        }
        tma_store_commit_group();
        tma_store_wait_group_read();
      }
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    uint32_t out = static_cast<uint32_t>(repeats) ^
                   static_cast<uint32_t>(kDoTma ? 0x9e3779b9u : 0x7f4a7c15u) ^
                   static_cast<uint32_t>(blockIdx.x);
    sink[blockIdx.x] = out;
  }
#endif
}

void driver_check(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* msg = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &msg);
    std::fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "unknown",
                 msg ? msg : "unknown");
    std::exit(1);
  }
}

void encode_bf16_output_tma_map(CUtensorMap* map,
                                void* base,
                                uint64_t tiles,
                                bool swizzle_128b,
                                bool tma_single) {
  if (swizzle_128b && tma_single) {
    const cuuint64_t global_dim[4] = {kTileN / 2, kTileM, 2, tiles};
    const cuuint64_t global_stride[3] = {
        kTileN * sizeof(uint16_t),
        (kTileN / 2) * sizeof(uint16_t),
        static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(uint16_t)};
    const cuuint32_t box_dim[4] = {kTileN / 2, kTileM, 2, 1};
    const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
    driver_check(cuTensorMapEncodeTiled(map,
                                        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                        4,
                                        base,
                                        global_dim,
                                        global_stride,
                                        box_dim,
                                        elem_stride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_128B,
                                        CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled(output_bf16_single_sw128)");
    return;
  }

  const cuuint64_t global_dim[4] = {kTileN, kTileM, tiles, 1};
  const cuuint64_t global_stride[3] = {
      kTileN * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * tiles * sizeof(uint16_t)};
  const cuuint32_t box_dim[4] = {
      static_cast<cuuint32_t>(swizzle_128b ? kTileN / 2 : kTileN),
      kTileM, 1, 1};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                      4,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      swizzle_128b ? CU_TENSOR_MAP_SWIZZLE_128B
                                                   : CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(output_bf16)");
}

template <LayoutKind Layout, bool kDoTma>
RunResult run_kernel(const Args& args,
                     const CUtensorMap& map,
                     uint32_t* sink,
                     int* active_ctas_per_sm) {
  RunResult result{};
  auto kernel = epilogue_store_conflict_kernel<Layout, kDoTma>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, kernel, kThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    kernel<<<args.blocks, kThreads, kDynamicSmemBytes>>>(
        map, sink, args.repeats, args.tma_swizzle_128b ? 1 : 0,
        args.tma_single ? 1 : 0);
    result.error = cudaGetLastError();
    if (result.error != cudaSuccess) {
      result.status = "warmup_launch_failed";
      return result;
    }
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "warmup_runtime_failed";
    return result;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    kernel<<<args.blocks, kThreads, kDynamicSmemBytes>>>(
        map, sink, args.repeats, args.tma_swizzle_128b ? 1 : 0,
        args.tma_single ? 1 : 0);
    result.error = cudaGetLastError();
    if (result.error != cudaSuccess) {
      result.status = "timed_launch_failed";
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return result;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  result.error = cudaEventSynchronize(stop);
  if (result.error != cudaSuccess) {
    result.status = "timed_runtime_failed";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "timed_runtime_failed";
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return result;
  }
  CUDA_CHECK(cudaEventElapsedTime(&result.ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return result;
}

RunResult dispatch_run(const Args& args,
                       const CUtensorMap& map,
                       uint32_t* sink,
                       int* active_ctas_per_sm) {
  if (args.layout == LayoutKind::kRowMajor && args.tma) {
    return run_kernel<LayoutKind::kRowMajor, true>(args, map, sink,
                                                   active_ctas_per_sm);
  }
  if (args.layout == LayoutKind::kRowMajor && !args.tma) {
    return run_kernel<LayoutKind::kRowMajor, false>(args, map, sink,
                                                    active_ctas_per_sm);
  }
  if (args.layout == LayoutKind::kAtomMajor && args.tma) {
    return run_kernel<LayoutKind::kAtomMajor, true>(args, map, sink,
                                                    active_ctas_per_sm);
  }
  if (args.layout == LayoutKind::kAtomMajor && !args.tma) {
    return run_kernel<LayoutKind::kAtomMajor, false>(args, map, sink,
                                                     active_ctas_per_sm);
  }
  if (args.layout == LayoutKind::kTmaSwizzle128B && args.tma) {
    return run_kernel<LayoutKind::kTmaSwizzle128B, true>(args, map, sink,
                                                        active_ctas_per_sm);
  }
  if (args.layout == LayoutKind::kTmaSwizzle128B && !args.tma) {
    return run_kernel<LayoutKind::kTmaSwizzle128B, false>(args, map, sink,
                                                         active_ctas_per_sm);
  }
  if (args.layout == LayoutKind::kTmaOnly) {
    return run_kernel<LayoutKind::kTmaOnly, true>(args, map, sink,
                                                  active_ctas_per_sm);
  }
  std::fprintf(stderr, "--layout tma-only requires --tma 1\n");
  std::exit(1);
}

const char* layout_name(LayoutKind layout) {
  switch (layout) {
    case LayoutKind::kRowMajor:
      return "row";
    case LayoutKind::kAtomMajor:
      return "atom";
    case LayoutKind::kTmaSwizzle128B:
      return "sw128";
    case LayoutKind::kTmaOnly:
      return "tma-only";
  }
  return "unknown";
}

bool parse_layout(const char* value, LayoutKind* layout) {
  if (!std::strcmp(value, "row") || !std::strcmp(value, "row-major")) {
    *layout = LayoutKind::kRowMajor;
    return true;
  }
  if (!std::strcmp(value, "atom") || !std::strcmp(value, "atom-major")) {
    *layout = LayoutKind::kAtomMajor;
    return true;
  }
  if (!std::strcmp(value, "sw128") || !std::strcmp(value, "tma-sw128")) {
    *layout = LayoutKind::kTmaSwizzle128B;
    return true;
  }
  if (!std::strcmp(value, "tma-only")) {
    *layout = LayoutKind::kTmaOnly;
    return true;
  }
  return false;
}

void parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      args->blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
      args->repeats = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--layout") && i + 1 < argc) {
      if (!parse_layout(argv[++i], &args->layout)) {
        std::fprintf(stderr, "unsupported --layout; use row, atom, sw128, or tma-only\n");
        std::exit(1);
      }
    } else if (!std::strcmp(argv[i], "--tma") && i + 1 < argc) {
      args->tma = std::atoi(argv[++i]) != 0;
    } else if (!std::strcmp(argv[i], "--tma-swizzle") && i + 1 < argc) {
      const char* mode = argv[++i];
      if (!std::strcmp(mode, "none") || !std::strcmp(mode, "0")) {
        args->tma_swizzle_128b = false;
      } else if (!std::strcmp(mode, "128") || !std::strcmp(mode, "128b") ||
                 !std::strcmp(mode, "sw128")) {
        args->tma_swizzle_128b = true;
      } else {
        std::fprintf(stderr, "unsupported --tma-swizzle; use none or 128\n");
        std::exit(1);
      }
    } else if (!std::strcmp(argv[i], "--validate") && i + 1 < argc) {
      args->validate = std::atoi(argv[++i]) != 0;
    } else if (!std::strcmp(argv[i], "--tma-single") && i + 1 < argc) {
      args->tma_single = std::atoi(argv[++i]) != 0;
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [--layout row|atom|sw128|tma-only] [--tma 0|1] "
          "[--tma-swizzle none|128] [--tma-single 0|1] [--validate 0|1] "
          "[--blocks N] [--repeats N] [--warmup N] [--iters N]\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
      std::exit(1);
    }
  }
  if (args->blocks < 1) args->blocks = 1;
  if (args->repeats < 1) args->repeats = 1;
  if (args->warmup < 0) args->warmup = 0;
  if (args->iters < 1) args->iters = 1;
  if (args->layout == LayoutKind::kTmaOnly) args->tma = true;
  if (!args->tma && args->tma_swizzle_128b) {
    std::fprintf(stderr, "--tma-swizzle requires --tma 1\n");
    std::exit(1);
  }
  if (args->tma_single && !args->tma_swizzle_128b) {
    std::fprintf(stderr, "--tma-single currently requires --tma-swizzle 128\n");
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, &args);

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  driver_check(cuInit(0), "cuInit");

  const size_t output_words = static_cast<size_t>(args.blocks) * kTileWords;
  void* d_output = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_output, output_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_output, 0, output_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));

  CUtensorMap o_map{};
  encode_bf16_output_tma_map(&o_map, d_output, static_cast<uint64_t>(args.blocks),
                             args.tma_swizzle_128b, args.tma_single);

  int active_ctas_per_sm = 0;
  RunResult result = dispatch_run(args, o_map, d_sink, &active_ctas_per_sm);

  uint32_t h_sink = 0;
  CUDA_CHECK(cudaMemcpy(&h_sink, d_sink, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  int validation_mismatches = 0;
  if (args.validate && args.tma && args.layout != LayoutKind::kTmaOnly) {
    std::vector<uint32_t> h_output(output_words);
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_words * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost));
    const int repeat = args.repeats - 1;
    for (int block = 0; block < args.blocks; ++block) {
      for (int row = 0; row < kTileM; ++row) {
        for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
          const size_t idx =
              static_cast<size_t>(block) * kTileWords + row * (kTileN / 2) + col_pair;
          const uint32_t expected =
              make_epilogue_word_host(row, col_pair, repeat, block);
          if (h_output[idx] != expected) {
            if (validation_mismatches < 8) {
              std::fprintf(stderr,
                           "mismatch block=%d row=%d col_pair=%d got=%08x expected=%08x\n",
                           block, row, col_pair, h_output[idx], expected);
            }
            ++validation_mismatches;
          }
        }
      }
    }
  }

  const long long store_inst_per_launch =
      args.layout == LayoutKind::kTmaOnly
          ? 0ll
          : static_cast<long long>(args.blocks) * args.repeats * kStoresPerCtaRepeat;
  const long long ideal_wavefronts_per_launch = store_inst_per_launch * 4ll;
  const long long row_major_wavefronts_per_launch =
      args.layout == LayoutKind::kRowMajor ? store_inst_per_launch * 32ll
                                           : ideal_wavefronts_per_launch;
  const long long row_major_conflicts_per_launch =
      args.layout == LayoutKind::kRowMajor ? store_inst_per_launch * 28ll : 0ll;
  const long long tma_ops_per_launch =
      args.tma ? static_cast<long long>(args.blocks) * args.repeats *
                     (args.tma_swizzle_128b && !args.tma_single ? 2ll : 1ll)
               : 0ll;

  std::printf(
      "gpu,cc,layout,tma,tma_swizzle,tma_single,blocks,repeats,warmup,iters,threads_per_cta,active_ctas_per_sm,"
      "dynamic_smem_bytes,expected_sts128_inst_per_launch,expected_ideal_wavefronts_per_launch,"
      "expected_row_major_wavefronts_per_launch,expected_row_major_conflicts_per_launch,"
      "expected_tma_store_ops_per_launch,validation_mismatches,elapsed_ms,status,error,sink\n");
  std::printf(
      "%s,sm_%d%d,%s,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%d,%.6f,%s,%s,%08x\n",
      prop.name, prop.major, prop.minor, layout_name(args.layout), args.tma ? 1 : 0,
      args.tma_swizzle_128b ? "128b" : "none", args.tma_single ? 1 : 0,
      args.blocks, args.repeats,
      args.warmup, args.iters, kThreads, active_ctas_per_sm,
      kDynamicSmemBytes, store_inst_per_launch, ideal_wavefronts_per_launch,
      row_major_wavefronts_per_launch, row_major_conflicts_per_launch,
      tma_ops_per_launch, validation_mismatches, result.ms, result.status,
      cudaGetErrorString(result.error), h_sink);

  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_sink));
  return result.error == cudaSuccess ? 0 : 1;
}
