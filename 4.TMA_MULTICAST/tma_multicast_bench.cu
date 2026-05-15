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

static constexpr int kThreads = 128;
static constexpr int kTileColsWords = 128;
static constexpr int kMaxTileRows = 256;

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int warmup = 2;
  int iters = 5;
  int source_tiles = 8192;
  bool large_only = false;
  const char* csv = "4.TMA_MULTICAST/tma_multicast_bench.csv";
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count)
               : "memory");
#else
  (void)barrier;
  (void)count;
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t* barrier, uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
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

__device__ __forceinline__ void tma_load_4d_multicast(const CUtensorMap* map,
                                                      uint32_t dst_smem,
                                                      uint64_t* barrier,
                                                      int c,
                                                      int r,
                                                      int d,
                                                      int b,
                                                      uint16_t cta_mask) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes."
      "multicast::cluster [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c), "r"(r), "r"(d), "r"(b),
        "h"(cta_mask)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c;
  (void)r;
  (void)d;
  (void)b;
  (void)cta_mask;
#endif
}

template <int ClusterSize, int TileRows, int BufferDepth, int IssuerCount, bool UseMulticast>
__global__ __launch_bounds__(kThreads, 1) void tma_multicast_kernel(
    const __grid_constant__ CUtensorMap map,
    uint32_t* __restrict__ sink,
    int repeats,
    int source_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
  (void)map;
  (void)sink;
  (void)repeats;
  (void)source_tiles;
#else
  static_assert(ClusterSize == 1 || ClusterSize == 2 || ClusterSize == 4 || ClusterSize == 8,
                "ClusterSize must be 1, 2, 4, or 8.");
  static_assert(TileRows == 16 || TileRows == 32 || TileRows == 64 || TileRows == 128 ||
                    TileRows == 256,
                "TileRows must be 16, 32, 64, 128, or 256.");
  static_assert(BufferDepth == 1 || BufferDepth == 2, "BufferDepth must be 1 or 2.");
  static_assert(IssuerCount == 1 || IssuerCount == 2 || IssuerCount == 4,
                "IssuerCount must be 1, 2, or 4.");

  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);
  constexpr int kTileWords = kTileColsWords * TileRows;
  constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));

  __shared__ uint64_t ready[BufferDepth * IssuerCount];
  __shared__ uint32_t thread_sinks[kThreads];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < BufferDepth; ++i) {
#pragma unroll
      for (int issuer = 0; issuer < IssuerCount; ++issuer) {
        mbarrier_init(&ready[i * IssuerCount + issuer], 1);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  int rank = 0;
  if constexpr (ClusterSize > 1) {
    rank = static_cast<int>(__clusterRelativeBlockRank());
  }
  const int cluster_id = static_cast<int>(blockIdx.x) / ClusterSize;
  const uint16_t mask = static_cast<uint16_t>((1u << ClusterSize) - 1u);
  uint32_t acc = static_cast<uint32_t>(blockIdx.x * 131u + threadIdx.x);

  for (int iter = 0; iter < repeats; ++iter) {
    const int buf = iter & (BufferDepth - 1);
    const int phase = (iter / BufferDepth) & 1;

    if (threadIdx.x == 0) {
#pragma unroll
      for (int issuer = 0; issuer < IssuerCount; ++issuer) {
        mbarrier_expect_tx(&ready[buf * IssuerCount + issuer], kTileBytes);
      }
    }
    __syncthreads();

    if constexpr (ClusterSize > 1) {
      __cluster_barrier_arrive();
      __cluster_barrier_wait();
    }

    const int issuer_id = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0 && issuer_id < IssuerCount) {
      const int slot = buf * IssuerCount + issuer_id;
      uint32_t* tile = smem_words + slot * kTileWords;
      const int src_tile =
          (cluster_id * repeats * IssuerCount + iter * IssuerCount + issuer_id) % source_tiles;
      if constexpr (UseMulticast) {
        if (rank == 0) {
          tma_load_4d_multicast(&map, smem_ptr_u32(tile), &ready[slot], 0,
                                src_tile * TileRows, 0, 0, mask);
        }
      } else {
        tma_load_4d(&map, smem_ptr_u32(tile), &ready[slot], 0, src_tile * TileRows, 0, 0);
      }
    }

#pragma unroll
    for (int issuer = 0; issuer < IssuerCount; ++issuer) {
      mbarrier_wait(&ready[buf * IssuerCount + issuer], static_cast<uint32_t>(phase));
    }
    __syncthreads();

#pragma unroll
    for (int issuer = 0; issuer < IssuerCount; ++issuer) {
      uint32_t* tile = smem_words + (buf * IssuerCount + issuer) * kTileWords;
      const int idx = (threadIdx.x * 37 + iter * 17 + issuer * 11) & (kTileWords - 1);
      acc ^= tile[idx] + static_cast<uint32_t>(iter + issuer);
    }
  }

  thread_sinks[threadIdx.x] = acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < kThreads; ++i) {
      out ^= thread_sinks[i] + static_cast<uint32_t>(i);
    }
    sink[blockIdx.x] = out;
  }
#endif
}

template <int TileRows, int BufferDepth>
__global__ __launch_bounds__(kThreads, 1) void tma_rank_pair_multicast_kernel(
    const __grid_constant__ CUtensorMap map,
    uint32_t* __restrict__ sink,
    int repeats,
    int source_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 900)
  (void)map;
  (void)sink;
  (void)repeats;
  (void)source_tiles;
#else
  static_assert(TileRows == 16 || TileRows == 32 || TileRows == 64 || TileRows == 128 ||
                    TileRows == 256,
                "TileRows must be 16, 32, 64, 128, or 256.");
  static_assert(BufferDepth == 1 || BufferDepth == 2, "BufferDepth must be 1 or 2.");
  constexpr int kClusterSize = 2;
  constexpr int kIssuerCount = 2;
  constexpr int kTileWords = kTileColsWords * TileRows;
  constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));

  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);

  __shared__ uint64_t ready[BufferDepth * kIssuerCount];
  __shared__ uint32_t thread_sinks[kThreads];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int i = 0; i < BufferDepth * kIssuerCount; ++i) {
      mbarrier_init(&ready[i], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int rank = static_cast<int>(__clusterRelativeBlockRank());
  const int cluster_id = static_cast<int>(blockIdx.x) / kClusterSize;
  uint32_t acc = static_cast<uint32_t>(blockIdx.x * 257u + threadIdx.x);

  for (int iter = 0; iter < repeats; ++iter) {
    const int buf = iter & (BufferDepth - 1);
    const int phase = (iter / BufferDepth) & 1;
    if (threadIdx.x == 0) {
#pragma unroll
      for (int issuer = 0; issuer < kIssuerCount; ++issuer) {
        mbarrier_expect_tx(&ready[buf * kIssuerCount + issuer], kTileBytes);
      }
    }
    __syncthreads();

    __cluster_barrier_arrive();
    __cluster_barrier_wait();

    if (threadIdx.x == 0) {
      const int slot = buf * kIssuerCount + rank;
      uint32_t* tile = smem_words + slot * kTileWords;
      const int src_tile =
          (cluster_id * repeats * kIssuerCount + iter * kIssuerCount + rank) % source_tiles;
      tma_load_4d_multicast(&map, smem_ptr_u32(tile), &ready[slot], 0,
                            src_tile * TileRows, 0, 0, 0x3);
    }

#pragma unroll
    for (int issuer = 0; issuer < kIssuerCount; ++issuer) {
      mbarrier_wait(&ready[buf * kIssuerCount + issuer], static_cast<uint32_t>(phase));
    }
    __syncthreads();

#pragma unroll
    for (int issuer = 0; issuer < kIssuerCount; ++issuer) {
      uint32_t* tile = smem_words + (buf * kIssuerCount + issuer) * kTileWords;
      const int idx = (threadIdx.x * 37 + iter * 17 + issuer * 11) & (kTileWords - 1);
      acc ^= tile[idx] + static_cast<uint32_t>(iter + issuer);
    }
  }

  thread_sinks[threadIdx.x] = acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t out = 0;
#pragma unroll
    for (int i = 0; i < kThreads; ++i) {
      out ^= thread_sinks[i] + static_cast<uint32_t>(i);
    }
    sink[blockIdx.x] = out;
  }
#endif
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
    } else if (!std::strcmp(argv[i], "--source-tiles") && i + 1 < argc) {
      args->source_tiles = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--large-only")) {
      args->large_only = true;
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [--blocks N] [--repeats N] [--warmup N] [--iters N] "
          "[--source-tiles N] [--large-only] [--csv PATH]\n",
          argv[0]);
      std::exit(0);
    }
  }
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

void encode_tma_map(CUtensorMap* map,
                    void* base,
                    uint64_t dim0_words,
                    uint64_t dim1_rows,
                    uint32_t box0_words,
                    uint32_t box1_rows) {
  const cuuint64_t global_dim[4] = {dim0_words, dim1_rows, 1, 1};
  const cuuint64_t global_stride[3] = {
      dim0_words * sizeof(uint32_t),
      dim0_words * dim1_rows * sizeof(uint32_t),
      dim0_words * dim1_rows * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {box0_words, box1_rows, 1, 1};
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

template <int ClusterSize, int TileRows, int BufferDepth, int IssuerCount, bool UseMulticast>
cudaError_t launch_one(const Args& args,
                       const CUtensorMap& map,
                       uint32_t* sink,
                       float* elapsed_ms) {
  auto kernel =
      tma_multicast_kernel<ClusterSize, TileRows, BufferDepth, IssuerCount, UseMulticast>;
  constexpr int kTileBytes = kTileColsWords * TileRows * static_cast<int>(sizeof(uint32_t));
  constexpr int kSmemBytes = BufferDepth * IssuerCount * kTileBytes + 1024;

  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterWidth, ClusterSize);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterHeight, 1);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterDepth, 1);
  if (err != cudaSuccess) return err;

  const int launch_blocks = (args.blocks / ClusterSize) * ClusterSize;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = ClusterSize;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(launch_blocks);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = nullptr;
  config.attrs = &attr;
  config.numAttrs = 1;

  for (int i = 0; i < args.warmup; ++i) {
    err = cudaLaunchKernelEx(&config, kernel, map, sink, args.repeats, args.source_tiles);
    if (err != cudaSuccess) return err;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    err = cudaLaunchKernelEx(&config, kernel, map, sink, args.repeats, args.source_tiles);
    if (err != cudaSuccess) return err;
  }
  CUDA_CHECK(cudaEventRecord(stop));
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) return err;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  *elapsed_ms /= static_cast<float>(args.iters);
  return cudaSuccess;
}

void write_header(FILE* csv) {
  std::fprintf(csv,
               "mode,cluster_size,tile_rows,tile_kb,buffer_depth,issuer_count,blocks,launch_blocks,"
               "clusters,repeats,warmup,iters,source_tiles,elapsed_ms,tma_ops,"
               "logical_delivered_GB,estimated_physical_read_GB,logical_delivered_TBps,"
               "estimated_physical_read_TBps,logical_speedup_vs_unicast,status,notes\n");
}

void write_row(FILE* csv,
               const char* mode,
               int cluster_size,
               int tile_rows,
               int buffer_depth,
               int issuer_count,
               const Args& args,
               double elapsed_ms,
               const char* status,
               const char* notes) {
  const int launch_blocks = (args.blocks / cluster_size) * cluster_size;
  const int clusters = launch_blocks / cluster_size;
  const double tile_bytes = static_cast<double>(kTileColsWords) * tile_rows * sizeof(uint32_t);
  const bool multicast =
      std::strcmp(mode, "multicast") == 0 || std::strcmp(mode, "rank_pair_multicast") == 0;
  const double tma_ops =
      static_cast<double>(multicast ? clusters : launch_blocks) * args.repeats * issuer_count;
  const double logical_bytes =
      static_cast<double>(launch_blocks) * args.repeats * issuer_count * tile_bytes;
  const double physical_bytes =
      static_cast<double>(multicast ? clusters : launch_blocks) * args.repeats * issuer_count *
      tile_bytes;
  const double logical_tbps = tbps_from_bytes(logical_bytes, elapsed_ms);
  const double physical_tbps = tbps_from_bytes(physical_bytes, elapsed_ms);
  const double speedup = multicast ? static_cast<double>(cluster_size) : 1.0;
  std::fprintf(csv,
               "%s,%d,%d,%.1f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.0f,%.3f,%.3f,%.3f,"
               "%.3f,%.3f,%s,%s\n",
               mode, cluster_size, tile_rows, tile_bytes / 1024.0, buffer_depth, issuer_count,
               args.blocks, launch_blocks, clusters, args.repeats, args.warmup, args.iters,
               args.source_tiles, elapsed_ms, tma_ops, logical_bytes / 1.0e9,
               physical_bytes / 1.0e9, logical_tbps, physical_tbps, speedup, status, notes);
}

template <int ClusterSize, int TileRows, int BufferDepth, int IssuerCount, bool UseMulticast>
void run_case(FILE* csv, const Args& args, const CUtensorMap& map, uint32_t* sink) {
  float ms = 0.0f;
  cudaError_t err =
      launch_one<ClusterSize, TileRows, BufferDepth, IssuerCount, UseMulticast>(args, map, sink,
                                                                               &ms);
  if (err != cudaSuccess) {
    write_row(csv, UseMulticast ? "multicast" : "unicast", ClusterSize, TileRows, BufferDepth,
              IssuerCount, args, 0.0, "skipped_launch_failed", cudaGetErrorString(err));
    cudaGetLastError();
    cudaDeviceSynchronize();
    return;
  }
  write_row(csv, UseMulticast ? "multicast" : "unicast", ClusterSize, TileRows, BufferDepth,
            IssuerCount, args, ms, "ok", "independent_tiles_per_issuer");
}

template <int TileRows, int BufferDepth>
cudaError_t launch_rank_pair(const Args& args,
                             const CUtensorMap& map,
                             uint32_t* sink,
                             float* elapsed_ms) {
  auto kernel = tma_rank_pair_multicast_kernel<TileRows, BufferDepth>;
  constexpr int kClusterSize = 2;
  constexpr int kIssuerCount = 2;
  constexpr int kTileBytes = kTileColsWords * TileRows * static_cast<int>(sizeof(uint32_t));
  constexpr int kSmemBytes = BufferDepth * kIssuerCount * kTileBytes + 1024;

  cudaError_t err =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterWidth, kClusterSize);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterHeight, 1);
  if (err != cudaSuccess) return err;
  err = cudaFuncSetAttribute(kernel, cudaFuncAttributeRequiredClusterDepth, 1);
  if (err != cudaSuccess) return err;

  const int launch_blocks = (args.blocks / kClusterSize) * kClusterSize;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim.x = kClusterSize;
  attr.val.clusterDim.y = 1;
  attr.val.clusterDim.z = 1;

  cudaLaunchConfig_t config{};
  config.gridDim = dim3(launch_blocks);
  config.blockDim = dim3(kThreads);
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = nullptr;
  config.attrs = &attr;
  config.numAttrs = 1;

  for (int i = 0; i < args.warmup; ++i) {
    err = cudaLaunchKernelEx(&config, kernel, map, sink, args.repeats, args.source_tiles);
    if (err != cudaSuccess) return err;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    err = cudaLaunchKernelEx(&config, kernel, map, sink, args.repeats, args.source_tiles);
    if (err != cudaSuccess) return err;
  }
  CUDA_CHECK(cudaEventRecord(stop));
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) return err;
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) return err;
  CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  *elapsed_ms /= static_cast<float>(args.iters);
  return cudaSuccess;
}

template <int TileRows, int BufferDepth>
void run_rank_pair_case(FILE* csv, const Args& args, const CUtensorMap& map, uint32_t* sink) {
  float ms = 0.0f;
  cudaError_t err = launch_rank_pair<TileRows, BufferDepth>(args, map, sink, &ms);
  if (err != cudaSuccess) {
    write_row(csv, "rank_pair_multicast", 2, TileRows, BufferDepth, 2, args, 0.0,
              "skipped_launch_failed", cudaGetErrorString(err));
    cudaGetLastError();
    cudaDeviceSynchronize();
    return;
  }
  write_row(csv, "rank_pair_multicast", 2, TileRows, BufferDepth, 2, args, ms, "ok",
            "rank0_and_rank1_each_issue_one_tile_then_multicast");
}

template <int TileRows>
void run_rank_pair_tile(FILE* csv, const Args& args, const CUtensorMap& map, uint32_t* sink) {
  run_rank_pair_case<TileRows, 1>(csv, args, map, sink);
  std::fflush(csv);
  run_rank_pair_case<TileRows, 2>(csv, args, map, sink);
  std::fflush(csv);
}

void run_rank_pair_sweep(FILE* csv,
                         const Args& args,
                         const CUtensorMap& map16,
                         const CUtensorMap& map32,
                         const CUtensorMap& map64,
                         const CUtensorMap& map128,
                         const CUtensorMap& map256,
                         uint32_t* sink) {
  run_rank_pair_tile<16>(csv, args, map16, sink);
  run_rank_pair_tile<32>(csv, args, map32, sink);
  run_rank_pair_tile<64>(csv, args, map64, sink);
  run_rank_pair_case<128, 1>(csv, args, map128, sink);
  std::fflush(csv);
  run_rank_pair_case<256, 1>(csv, args, map256, sink);
  std::fflush(csv);
}

template <int ClusterSize, int TileRows, int BufferDepth, int IssuerCount>
void run_pair(FILE* csv, const Args& args, const CUtensorMap& map, uint32_t* sink) {
  run_case<ClusterSize, TileRows, BufferDepth, IssuerCount, false>(csv, args, map, sink);
  std::fflush(csv);
  run_case<ClusterSize, TileRows, BufferDepth, IssuerCount, true>(csv, args, map, sink);
  std::fflush(csv);
}

template <int ClusterSize, int TileRows, int IssuerCount>
void run_tile(FILE* csv, const Args& args, const CUtensorMap& map, uint32_t* sink) {
  run_pair<ClusterSize, TileRows, 1, IssuerCount>(csv, args, map, sink);
  run_pair<ClusterSize, TileRows, 2, IssuerCount>(csv, args, map, sink);
}

template <int ClusterSize>
void run_cluster(FILE* csv,
                 const Args& args,
                 const CUtensorMap& map16,
                 const CUtensorMap& map32,
                 const CUtensorMap& map64,
                 uint32_t* sink) {
  run_tile<ClusterSize, 16, 1>(csv, args, map16, sink);
  run_tile<ClusterSize, 16, 2>(csv, args, map16, sink);
  run_tile<ClusterSize, 16, 4>(csv, args, map16, sink);
  run_tile<ClusterSize, 32, 1>(csv, args, map32, sink);
  run_tile<ClusterSize, 32, 2>(csv, args, map32, sink);
  run_tile<ClusterSize, 32, 4>(csv, args, map32, sink);
  run_tile<ClusterSize, 64, 1>(csv, args, map64, sink);
  run_tile<ClusterSize, 64, 2>(csv, args, map64, sink);
  run_tile<ClusterSize, 64, 4>(csv, args, map64, sink);
}

template <int ClusterSize, int TileRows>
void run_large_tile(FILE* csv, const Args& args, const CUtensorMap& map, uint32_t* sink) {
  run_pair<ClusterSize, TileRows, 1, 1>(csv, args, map, sink);
  std::fflush(csv);
  if constexpr (TileRows <= 128) {
    run_pair<ClusterSize, TileRows, 2, 1>(csv, args, map, sink);
    std::fflush(csv);
    run_pair<ClusterSize, TileRows, 1, 2>(csv, args, map, sink);
    std::fflush(csv);
  }
}

template <int ClusterSize>
void run_large_cluster(FILE* csv,
                       const Args& args,
                       const CUtensorMap& map128,
                       const CUtensorMap& map256,
                       uint32_t* sink) {
  run_large_tile<ClusterSize, 128>(csv, args, map128, sink);
  run_large_tile<ClusterSize, 256>(csv, args, map256, sink);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, &args);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This benchmark requires SM100+; got sm_%d%d\n", prop.major, prop.minor);
    return 77;
  }
  driver_check(cuInit(0), "cuInit");

  if (args.blocks < 8) args.blocks = 8;
  if (args.source_tiles < 1) args.source_tiles = 1;

  const size_t src_words =
      static_cast<size_t>(args.source_tiles) * kMaxTileRows * kTileColsWords;
  uint32_t* d_src = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_src, src_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, args.blocks * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_src, 0x7b, src_words * sizeof(uint32_t)));

  CUtensorMap map16{}, map32{}, map64{}, map128{}, map256{};
  encode_tma_map(&map16, d_src, kTileColsWords,
                 static_cast<uint64_t>(args.source_tiles) * 16, kTileColsWords, 16);
  encode_tma_map(&map32, d_src, kTileColsWords,
                 static_cast<uint64_t>(args.source_tiles) * 32, kTileColsWords, 32);
  encode_tma_map(&map64, d_src, kTileColsWords,
                 static_cast<uint64_t>(args.source_tiles) * 64, kTileColsWords, 64);
  encode_tma_map(&map128, d_src, kTileColsWords,
                 static_cast<uint64_t>(args.source_tiles) * 128, kTileColsWords, 128);
  encode_tma_map(&map256, d_src, kTileColsWords,
                 static_cast<uint64_t>(args.source_tiles) * 256, kTileColsWords, 256);

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    return 1;
  }
  write_header(csv);

  if (!args.large_only) {
    run_cluster<1>(csv, args, map16, map32, map64, d_sink);
    run_cluster<2>(csv, args, map16, map32, map64, d_sink);
    run_rank_pair_sweep(csv, args, map16, map32, map64, map128, map256, d_sink);
    run_cluster<4>(csv, args, map16, map32, map64, d_sink);
    run_cluster<8>(csv, args, map16, map32, map64, d_sink);
  } else {
    run_large_cluster<1>(csv, args, map128, map256, d_sink);
    run_large_cluster<2>(csv, args, map128, map256, d_sink);
    run_rank_pair_case<128, 1>(csv, args, map128, d_sink);
    std::fflush(csv);
    run_rank_pair_case<256, 1>(csv, args, map256, d_sink);
    std::fflush(csv);
    run_large_cluster<4>(csv, args, map128, map256, d_sink);
    run_large_cluster<8>(csv, args, map128, map256, d_sink);
  }

  std::fclose(csv);
  cudaFree(d_src);
  cudaFree(d_sink);
  return 0;
}
