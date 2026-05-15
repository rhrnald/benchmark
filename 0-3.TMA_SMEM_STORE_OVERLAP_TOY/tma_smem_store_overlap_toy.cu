#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
static constexpr int kWarps = 16;
static constexpr int kThreads = kWarps * kWarpSize;
static constexpr int kTileColsWords = 128;
static constexpr int kTileRows = 64;
static constexpr int kTileWords = kTileColsWords * kTileRows;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kStoreWarpBase = 4;
static constexpr int kDynamicSmemBytes = 3 * kTileBytes + 1024;

enum Mode : int {
  kModeTmaOnly = 0,
  kModeStoreOnly = 1,
  kModeOverlap = 2,
};

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int source_tiles = 8192;
  int warmup = 2;
  int iters = 5;
  int store_warps = 0;   // 0 means sweep 4,8,12.
  int store_tiles = 0;   // 0 means sweep 1,2,4,8.
  int store_vec = 0;     // 0 means sweep 1,2,4 u32 words per lane store.
  const char* csv = "0-3.TMA_SMEM_STORE_OVERLAP_TOY/tma_smem_store_overlap_summary.csv";
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

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count)
               : "memory");
#else
  (void)barrier;
  (void)count;
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* barrier, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :: "r"(addr), "r"(bytes)
               : "memory");
#else
  (void)barrier;
  (void)bytes;
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile(
      "{ .reg .pred p; "
      "L_wait_%=: "
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
      "@p bra.uni L_done_%=; "
      "bra.uni L_wait_%=; "
      "L_done_%=: }"
      :: "r"(addr), "r"(phase)
      : "memory");
#else
  (void)barrier;
  (void)phase;
#endif
}

__device__ __forceinline__ void tma_load_4d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c,
                                            int r,
                                            int d,
                                            int b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6}], [%2];"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c), "r"(r), "r"(d), "r"(b)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c;
  (void)r;
  (void)d;
  (void)b;
#endif
}

template <int StoreVecWords>
__device__ __forceinline__ void st_shared_vec(uint32_t smem_addr, uint32_t value) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (StoreVecWords == 1) {
    asm volatile("st.shared.u32 [%0], %1;" :: "r"(smem_addr), "r"(value) : "memory");
  } else if constexpr (StoreVecWords == 2) {
    asm volatile("st.shared.v2.u32 [%0], {%1, %2};"
                 :: "r"(smem_addr), "r"(value), "r"(value ^ 0x9e3779b9u)
                 : "memory");
  } else if constexpr (StoreVecWords == 4) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
                 :: "r"(smem_addr), "r"(value), "r"(value ^ 0x9e3779b9u),
                    "r"(value ^ 0x7f4a7c15u), "r"(value ^ 0x94d049bbu)
                 : "memory");
  }
#else
  (void)smem_addr;
  (void)value;
#endif
}

template <int StoreWarps, int StoreTilesPerTma, int StoreVecWords>
__global__ __launch_bounds__(kThreads, 1) void tma_smem_store_overlap_kernel(
    const __grid_constant__ CUtensorMap map,
    uint32_t* __restrict__ sink,
    int repeats,
    int source_tiles,
    int mode) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)map;
  (void)sink;
  (void)repeats;
  (void)source_tiles;
  (void)mode;
#else
  static_assert(StoreWarps == 4 || StoreWarps == 8 || StoreWarps == 12,
                "StoreWarps must be 4, 8, or 12.");
  static_assert(StoreTilesPerTma == 1 || StoreTilesPerTma == 2 ||
                    StoreTilesPerTma == 4 || StoreTilesPerTma == 8,
                "StoreTilesPerTma must be 1, 2, 4, or 8.");
  static_assert(StoreVecWords == 1 || StoreVecWords == 2 || StoreVecWords == 4,
                "StoreVecWords must be 1, 2, or 4.");
  static_assert(kTileWords % StoreVecWords == 0, "tile must be vector divisible.");

  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* tma_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* store_smem = tma_smem + 2 * kTileWords;

  __shared__ uint64_t tma_ready;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  uint32_t acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (threadIdx.x == 0) {
    mbarrier_init(&tma_ready, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if ((mode == kModeTmaOnly || mode == kModeOverlap) && warp_id == 0 && lane == 0) {
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const int tile = (static_cast<int>(blockIdx.x) * repeats + iter) % source_tiles;
      uint32_t* dst = tma_smem + (iter & 1) * kTileWords;
      mbarrier_expect_tx(&tma_ready, kTileBytes);
      tma_load_4d(&map, smem_ptr_u32(dst), &tma_ready, 0, tile * kTileRows, 0, 0);
      mbarrier_wait(&tma_ready, phase);
      acc ^= dst[(iter * 131) & (kTileWords - 1)];
    }
  }

  if (mode == kModeStoreOnly || mode == kModeOverlap) {
      const int store_warp = warp_id - kStoreWarpBase;
    if (store_warp >= 0 && store_warp < StoreWarps) {
      const int store_thread = store_warp * kWarpSize + lane;
      const int store_threads = StoreWarps * kWarpSize;
      constexpr int kTileVectors = kTileWords / StoreVecWords;
      for (int iter = 0; iter < repeats; ++iter) {
#pragma unroll
        for (int tile = 0; tile < StoreTilesPerTma; ++tile) {
          const uint32_t value =
              static_cast<uint32_t>(iter * 0x9e3779b9u + tile * 0x7f4a7c15u + threadIdx.x);
          for (int vec = store_thread; vec < kTileVectors; vec += store_threads) {
            const int idx = vec * StoreVecWords;
            st_shared_vec<StoreVecWords>(smem_ptr_u32(store_smem + idx),
                                         value ^ static_cast<uint32_t>(idx));
          }
        }
      }
      acc ^= store_smem[(store_thread * 17) & (kTileWords - 1)];
    }
  }

  if (lane == 0) warp_sinks[warp_id] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t out = static_cast<uint32_t>(mode) ^ static_cast<uint32_t>(repeats);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      out ^= warp_sinks[w] + static_cast<uint32_t>(w * 17);
    }
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

void encode_tma_map(CUtensorMap* map, void* base, uint64_t source_tiles) {
  const cuuint64_t global_dim[4] = {kTileColsWords, source_tiles * kTileRows, 1, 1};
  const cuuint64_t global_stride[3] = {
      kTileColsWords * sizeof(uint32_t),
      kTileColsWords * source_tiles * kTileRows * sizeof(uint32_t),
      kTileColsWords * source_tiles * kTileRows * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kTileColsWords, kTileRows, 1, 1};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                      4,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled");
}

double tbps_from_bytes(double bytes, double ms) {
  return ms > 0.0 ? bytes / (ms * 1.0e-3) / 1.0e12 : 0.0;
}

template <int StoreWarps, int StoreTilesPerTma, int StoreVecWords>
RunResult run_mode(const Args& args,
                   const CUtensorMap& map,
                   uint32_t* sink,
                   int mode,
                   int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(
      tma_smem_store_overlap_kernel<StoreWarps, StoreTilesPerTma, StoreVecWords>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm,
      tma_smem_store_overlap_kernel<StoreWarps, StoreTilesPerTma, StoreVecWords>,
      kThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    tma_smem_store_overlap_kernel<StoreWarps, StoreTilesPerTma, StoreVecWords>
        <<<args.blocks, kThreads, kDynamicSmemBytes>>>(map, sink, args.repeats,
                                                       args.source_tiles, mode);
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
    tma_smem_store_overlap_kernel<StoreWarps, StoreTilesPerTma, StoreVecWords>
        <<<args.blocks, kThreads, kDynamicSmemBytes>>>(map, sink, args.repeats,
                                                       args.source_tiles, mode);
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

template <int StoreWarps, int StoreTilesPerTma, int StoreVecWords>
void run_combo(const Args& args, const CUtensorMap& map, uint32_t* sink, FILE* csv) {
  int active = 0;
  RunResult tma = run_mode<StoreWarps, StoreTilesPerTma, StoreVecWords>(
      args, map, sink, kModeTmaOnly, &active);
  RunResult store = run_mode<StoreWarps, StoreTilesPerTma, StoreVecWords>(
      args, map, sink, kModeStoreOnly, &active);
  RunResult overlap = run_mode<StoreWarps, StoreTilesPerTma, StoreVecWords>(
      args, map, sink, kModeOverlap, &active);

  const double groups = static_cast<double>(args.blocks) * args.repeats * args.iters;
  const double tma_bytes = groups * kTileBytes;
  const double store_bytes = groups * StoreTilesPerTma * kTileBytes;
  const double expected_ms = tma.ms > store.ms ? tma.ms : store.ms;
  const double extra_ms = overlap.ms - expected_ms;
  const double efficiency = overlap.ms > 0.0 ? expected_ms / overlap.ms : 0.0;
  const double overlap_tma_tbps = tbps_from_bytes(tma_bytes, overlap.ms);
  const double overlap_store_tbps = tbps_from_bytes(store_bytes, overlap.ms);
  const char* status = (tma.error == cudaSuccess && store.error == cudaSuccess &&
                        overlap.error == cudaSuccess)
                           ? "ok"
                           : "error";
  const char* notes =
      extra_ms > 0.05 * expected_ms
          ? "overlap_slower_than_no_share_model_possible_smem_write_path_contention"
          : "overlap_close_to_max_of_individual_runs";

  std::fprintf(
      csv,
      "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%s,%s,%s,%s,%s\n",
      StoreWarps, StoreTilesPerTma, StoreVecWords, args.blocks, args.repeats, args.warmup, args.iters,
      kThreads, active, kDynamicSmemBytes, tma.ms, store.ms, overlap.ms,
      tbps_from_bytes(tma_bytes, tma.ms), tbps_from_bytes(store_bytes, store.ms),
      overlap_tma_tbps, overlap_store_tbps, overlap_tma_tbps + overlap_store_tbps,
      tbps_from_bytes(tma_bytes + store_bytes, overlap.ms), expected_ms, extra_ms,
      efficiency, status, cudaGetErrorString(tma.error), cudaGetErrorString(store.error),
      cudaGetErrorString(overlap.error), notes);
}

void dispatch_combo(const Args& args,
                    const CUtensorMap& map,
                    uint32_t* sink,
                    FILE* csv,
                    int store_warps,
                    int store_tiles,
                    int store_vec) {
#define DISPATCH_VEC(W, T)                                                     \
  do {                                                                         \
    if (store_warps == (W) && store_tiles == (T) && store_vec == 1)             \
      return run_combo<(W), (T), 1>(args, map, sink, csv);                     \
    if (store_warps == (W) && store_tiles == (T) && store_vec == 2)             \
      return run_combo<(W), (T), 2>(args, map, sink, csv);                     \
    if (store_warps == (W) && store_tiles == (T) && store_vec == 4)             \
      return run_combo<(W), (T), 4>(args, map, sink, csv);                     \
  } while (0)
  DISPATCH_VEC(4, 1);
  DISPATCH_VEC(4, 2);
  DISPATCH_VEC(4, 4);
  DISPATCH_VEC(4, 8);
  DISPATCH_VEC(8, 1);
  DISPATCH_VEC(8, 2);
  DISPATCH_VEC(8, 4);
  DISPATCH_VEC(8, 8);
  DISPATCH_VEC(12, 1);
  DISPATCH_VEC(12, 2);
  DISPATCH_VEC(12, 4);
  DISPATCH_VEC(12, 8);
#undef DISPATCH_VEC
  std::fprintf(stderr, "unsupported --store-warps/--store-tiles combination\n");
  std::exit(1);
}

void parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      args->blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
      args->repeats = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--source-tiles") && i + 1 < argc) {
      args->source_tiles = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--store-warps") && i + 1 < argc) {
      args->store_warps = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--store-tiles") && i + 1 < argc) {
      args->store_tiles = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--store-vec") && i + 1 < argc) {
      args->store_vec = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [--blocks N] [--repeats N] [--source-tiles N] [--warmup N] "
          "[--iters N] [--store-warps 4|8|12] [--store-tiles 1|2|4|8] "
          "[--store-vec 1|2|4] [--csv PATH]\n",
          argv[0]);
      std::exit(0);
    }
  }
  if (args->blocks < 1) args->blocks = 1;
  if (args->repeats < 1) args->repeats = 1;
  if (args->source_tiles < 1) args->source_tiles = 1;
  if (args->warmup < 0) args->warmup = 0;
  if (args->iters < 1) args->iters = 1;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, &args);

  driver_check(cuInit(0), "cuInit");
  const size_t source_words = static_cast<size_t>(args.source_tiles) * kTileWords;
  uint32_t* d_source = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_source, source_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_source, 0x3f, source_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));

  CUtensorMap map{};
  encode_tma_map(&map, d_source, args.source_tiles);

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    return 1;
  }
  std::fprintf(
      csv,
      "store_warps,store_tiles_per_tma,store_vec_words,blocks,repeats,warmup,iters,threads_per_cta,actual_ctas_per_sm,dynamic_smem_bytes,"
      "tma_only_ms,store_only_ms,overlap_ms,tma_only_TBps,store_only_TBps,overlap_tma_TBps,overlap_store_TBps,"
      "overlap_sum_TBps,total_logical_overlap_TBps,expected_no_share_ms,overlap_extra_ms,overlap_efficiency,status,"
      "tma_error,store_error,overlap_error,notes\n");

  const int store_warps_list[3] = {4, 8, 12};
  const int store_tiles_list[4] = {1, 2, 4, 8};
  const int store_vec_list[3] = {1, 2, 4};
  for (int w : store_warps_list) {
    if (args.store_warps != 0 && args.store_warps != w) continue;
    for (int t : store_tiles_list) {
      if (args.store_tiles != 0 && args.store_tiles != t) continue;
      for (int v : store_vec_list) {
        if (args.store_vec != 0 && args.store_vec != v) continue;
        dispatch_combo(args, map, d_sink, csv, w, t, v);
      }
    }
  }
  std::fclose(csv);

  CUDA_CHECK(cudaFree(d_source));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
