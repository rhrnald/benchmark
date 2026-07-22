#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Clean 16K dense working default.
//
// The original B0 was reconstructed from the configuration that produced the
// preserved 1806.657-TFLOP/s p0 binary.  This version also retains the measured
// E2a TMEM-address cleanup.  The exact p0 source was not archived, so p0 remains
// a performance/codegen oracle.  There are no tuning macros or experiment
// switches: no command-line -D can silently change the kernel.

void cuda_check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

void driver_check(CUresult result, const char *what) {
  if (result != CUDA_SUCCESS) {
    const char *name = nullptr;
    const char *msg = nullptr;
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &msg);
    std::fprintf(stderr, "Driver error %s: %s (%s)\n", what,
                 name ? name : "unknown", msg ? msg : "unknown");
    std::exit(EXIT_FAILURE);
  }
}

namespace {

static constexpr int kThreadsPerWarp = 32;
static constexpr int kWarps = 4;
static constexpr int kThreads = kWarps * kThreadsPerWarp;
static constexpr int kBenchmarkSize = 16384;
static constexpr int kPersistentCtas = 148;
static constexpr int kValidationCtas = 1;
static constexpr int kMaxValidationSize = 512;

static constexpr int kCtaM = 256;
static constexpr int kCtaN = 256;
static constexpr int kStageK = 64;
static constexpr int kStages = 3;
static constexpr int kMmaM = 128;
static constexpr int kMmaN = 128;
static constexpr int kMmaK = 16;
static constexpr int kPipes = 2;
static constexpr int kMBlocks = 2;
static constexpr int kBTmaN = 128;
static constexpr int kBTmaNSubtiles = kBTmaN / 64;
static constexpr int kPersistentMacroM = 16;
static constexpr int kPersistentMacroN = 16;
static constexpr int kAStageWords = kCtaM * kStageK / 2;
static constexpr int kBStageWords = kStageK * kCtaN / 2;
static constexpr int kBPipeWords = kStageK * kMmaN / 2;
static constexpr int kStageWords = kAStageWords + kBStageWords;
static constexpr int kAStageBytes =
    kAStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kBPipeBytes =
    kBPipeWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kStageBytes =
    kStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kMainloopSmemBytes = kStages * kStageBytes;
static constexpr int kCStoreChunkM = 128;
static constexpr int kCStoreChunkN = 128;
static constexpr int kCStoreWarps = kCStoreChunkM / 32;
static constexpr int kCStoreStageWords = kCStoreChunkM * kCStoreChunkN;
static constexpr int kCStoreStageBytes =
    kCStoreStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kCStoreChunksM = kCtaM / kCStoreChunkM;
static constexpr int kCStoreChunksN = kCtaN / kCStoreChunkN;
static constexpr int kCStoreChunkCount = kCStoreChunksM * kCStoreChunksN;
static constexpr int kCStoreTilesPerChunkN = kCStoreChunkN / kMmaN;
static constexpr int kCStoreBuffers = 3;
static constexpr int kCStoreTotalBytes = kCStoreBuffers * kCStoreStageBytes;
static constexpr int kDynamicSmemPayloadBytes =
    kMainloopSmemBytes > kCStoreTotalBytes ? kMainloopSmemBytes
                                           : kCStoreTotalBytes;
static constexpr int kDynamicSmemBytes = kDynamicSmemPayloadBytes + 1024;
static constexpr int kHalfTileWords = kMmaM * kStageK / 2;
static constexpr int kTmemTileStride = 128;

static_assert(kCtaM == 128 || kCtaM == 256);
static_assert(kCtaM % kMmaM == 0);
static_assert(kPipes * kBPipeWords == kBStageWords);
static_assert(kMmaM % kCStoreChunkM == 0);
static_assert(kCtaM % kCStoreChunkM == 0);
static_assert(kCStoreChunkN % kMmaN == 0);
static_assert(kCtaN % kCStoreChunkN == 0);
static_assert(kCStoreChunkN % 64 == 0);
static_assert(kCStoreWarps * (kMmaM / kCStoreChunkM) <= kWarps);
static_assert(kCStoreBuffers >= 1 && kCStoreBuffers <= kCStoreChunkCount);
static_assert(kCStoreTotalBytes <= kMainloopSmemBytes);
static_assert(kCStoreChunkN == 128 || kCStoreChunkN == 256);
static_assert(kBTmaN == 128);

enum InputInitMode : int {
  kInputInitMemset = 0,
  kInputInitFormula = 1,
  kInputInitRandom = 2,
  kInputInitRandomSigned8 = 3,
};

struct Args {
  int device = 0;
  int warmup = 1;
  int iters = 5;
  const char *csv = "gemm256_bf16_16k.csv";
  int input_init_mode = kInputInitRandom;
  bool validate = false;
  int validate_size = 512;
  const char *validate_pattern = "pattern";
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void *ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; "
               "cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__host__ __device__ __forceinline__ uint64_t
make_sw128_major_k_smem_desc(uint32_t matrix_start_addr, int mma) {
  constexpr uint64_t desc_base =
      (static_cast<uint64_t>(1u) << 16) | (static_cast<uint64_t>(64u) << 32) |
      (static_cast<uint64_t>(1u) << 46) | (static_cast<uint64_t>(2u) << 61);
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (32u >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__device__ __forceinline__ uint64_t make_stage_a_smem_desc(uint32_t *a_smem,
                                                           int mblock,
                                                           int mma) {
  uint32_t *matrix = a_smem + mblock * kHalfTileWords;
  return make_sw128_major_k_smem_desc(smem_ptr_u32(matrix), mma);
}

__host__ __device__ __forceinline__ uint64_t
make_sw128_major_mn_smem_desc(uint32_t matrix_start_addr, int mma) {
  constexpr uint64_t desc_base =
      (static_cast<uint64_t>(128u) << 16) | (static_cast<uint64_t>(64u) << 32) |
      (static_cast<uint64_t>(1u) << 46) | (static_cast<uint64_t>(2u) << 61);
  constexpr uint32_t kSliceBytes =
      static_cast<uint32_t>(kMmaK * kMmaN / 2 * sizeof(uint32_t));
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (kSliceBytes >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__host__ __device__ __forceinline__ uint32_t make_bf16_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;  // C format: F32.
  desc |= 1u << 7;  // A format: BF16.
  desc |= 1u << 10; // B format: BF16.
  desc |= static_cast<uint32_t>(kMmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kMmaM >> 4) << 24;
  return desc;
}

__device__ __forceinline__ void mbarrier_init(uint64_t *barrier,
                                              uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count)
               : "memory");
#else
  (void)barrier;
  (void)count;
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t *barrier,
                                              uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("{ .reg .pred p; "
               "L_wait_%=: "
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1; "
               "@p bra.uni L_done_%=; "
               "bra.uni L_wait_%=; "
               "L_done_%=: }" ::"r"(addr),
               "r"(phase)
               : "memory");
#else
  (void)barrier;
  (void)phase;
#endif
}

__device__ __forceinline__ void mbarrier_expect_tx(uint64_t *barrier,
                                                   uint32_t bytes) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
      "r"(bytes)
      : "memory");
#else
  (void)barrier;
  (void)bytes;
#endif
}

__device__ __forceinline__ void tma_load_2d(const CUtensorMap *map,
                                            uint32_t dst_smem,
                                            uint64_t *barrier, int c, int r) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];"
               :
               : "r"(dst_smem), "l"(map), "r"(bar), "r"(c), "r"(r)
               : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c;
  (void)r;
#endif
}

__device__ __forceinline__ void tma_load_4d(const CUtensorMap *map,
                                            uint32_t dst_smem,
                                            uint64_t *barrier, int c0, int c1,
                                            int c2, int c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5, %6}], [%2];"
               :
               : "r"(dst_smem), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2),
                 "r"(c3)
               : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
#endif
}

__device__ __forceinline__ void tma_store_fence_shared() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_4d(const CUtensorMap *map,
                                             uint32_t src_smem, int c0, int c1,
                                             int c2, int c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
               " [%0, {%2, %3, %4, %5}], [%1];"
               :
               : "l"(map), "r"(src_smem), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
               : "memory");
#else
  (void)map;
  (void)src_smem;
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
#endif
}

__device__ __forceinline__ void tma_store_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_wait_group_read_0() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_wait_group_read_2() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.wait_group.read 2;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t
tcgen05_alloc_512cols(uint32_t *smem_out_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t smem_addr = smem_ptr_u32(smem_out_taddr);
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;" ::
          "r"(smem_addr)
      : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];"
               : "=r"(taddr)
               : "r"(smem_addr)
               : "memory");
  return taddr;
#else
  (void)smem_out_taddr;
  return 0;
#endif
}

__device__ __forceinline__ void tcgen05_dealloc_512cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;" ::"r"(taddr)
      : "memory");
#else
  (void)taddr;
#endif
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                   : "memory");
#endif
}

__device__ __forceinline__ void tcgen05_commit(uint64_t *barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(addr)
               : "memory");
#else
  (void)barrier;
#endif
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x64(uint32_t (&dst)[64],
                                                      uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {"
      "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, "
      "%15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, "
      "%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, "
      "%41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, "
      "%54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
      : "=&r"(dst[0]), "=&r"(dst[1]), "=&r"(dst[2]), "=&r"(dst[3]),
        "=&r"(dst[4]), "=&r"(dst[5]), "=&r"(dst[6]), "=&r"(dst[7]),
        "=&r"(dst[8]), "=&r"(dst[9]), "=&r"(dst[10]), "=&r"(dst[11]),
        "=&r"(dst[12]), "=&r"(dst[13]), "=&r"(dst[14]), "=&r"(dst[15]),
        "=&r"(dst[16]), "=&r"(dst[17]), "=&r"(dst[18]), "=&r"(dst[19]),
        "=&r"(dst[20]), "=&r"(dst[21]), "=&r"(dst[22]), "=&r"(dst[23]),
        "=&r"(dst[24]), "=&r"(dst[25]), "=&r"(dst[26]), "=&r"(dst[27]),
        "=&r"(dst[28]), "=&r"(dst[29]), "=&r"(dst[30]), "=&r"(dst[31]),
        "=&r"(dst[32]), "=&r"(dst[33]), "=&r"(dst[34]), "=&r"(dst[35]),
        "=&r"(dst[36]), "=&r"(dst[37]), "=&r"(dst[38]), "=&r"(dst[39]),
        "=&r"(dst[40]), "=&r"(dst[41]), "=&r"(dst[42]), "=&r"(dst[43]),
        "=&r"(dst[44]), "=&r"(dst[45]), "=&r"(dst[46]), "=&r"(dst[47]),
        "=&r"(dst[48]), "=&r"(dst[49]), "=&r"(dst[50]), "=&r"(dst[51]),
        "=&r"(dst[52]), "=&r"(dst[53]), "=&r"(dst[54]), "=&r"(dst[55]),
        "=&r"(dst[56]), "=&r"(dst[57]), "=&r"(dst[58]), "=&r"(dst[59]),
        "=&r"(dst[60]), "=&r"(dst[61]), "=&r"(dst[62]), "=&r"(dst[63])
      : "r"(taddr)
      : "memory");
#else
  (void)taddr;
  for (int i = 0; i < 64; ++i)
    dst[i] = 0;
#endif
}

__device__ __forceinline__ void
tcgen05_mma_bf16_ss(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc,
                    uint32_t idesc, bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile("{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, "
               "%7, %8}, pred; }" ::"r"(d_taddr),
               "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p), "r"(mask[0]),
               "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
               : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

__host__ __device__ __forceinline__ int
cstore_sw128_float_word_offset(int row, int col) {
  const int col_block = col >> 5;
  const int in_block = col & 31;
  return col_block * (kCStoreChunkM * 32) + row * 32 +
         (in_block ^ ((row & 7) << 2));
}

__device__ __forceinline__ void store_u32x4_smem(uint32_t *smem,
                                                 int word_offset, uint32_t p0,
                                                 uint32_t p1, uint32_t p2,
                                                 uint32_t p3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  reinterpret_cast<uint4 *>(smem + word_offset)[0] = make_uint4(p0, p1, p2, p3);
#else
  smem[word_offset + 0] = p0;
  smem[word_offset + 1] = p1;
  smem[word_offset + 2] = p2;
  smem[word_offset + 3] = p3;
#endif
}

__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int selected_warp_base = 0;
  if (warp_id >= selected_warp_base &&
      warp_id < selected_warp_base + kCStoreWarps) {
    uint32_t r[64];
    const int local_warp = warp_id - selected_warp_base;
    const uint32_t row_base = static_cast<uint32_t>(local_warp * 32);
    const int local_row = local_warp * 32 + lane;
#pragma unroll
    for (int tile_n_part = 0; tile_n_part < kCStoreTilesPerChunkN;
         ++tile_n_part) {
      const int pipe = chunk_n * kCStoreTilesPerChunkN + tile_n_part;
      const int tile = chunk_m * 2 + pipe;
#pragma unroll
      for (int load = 0; load < kMmaN / 64; ++load) {
        const uint32_t col_base =
            static_cast<uint32_t>(tile_n_part * kMmaN + load * 64);
        const uint32_t row_taddr =
            tmem_base + tile * kTmemTileStride + (row_base << 16) + col_base;
        tcgen05_ld_32x32b_x64(r, row_taddr);
        tcgen05_wait_ld();
        const int col_offset = tile_n_part * kMmaN + load * 64;
#pragma unroll
        for (int i = 0; i < 64; i += 4) {
          store_u32x4_smem(
              c_smem, cstore_sw128_float_word_offset(local_row, col_offset + i),
              r[i + 0], r[i + 1], r[i + 2], r[i + 3]);
        }
      }
    }
  }
#else
  (void)tmem_base;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
#endif
}

__device__ __forceinline__ void
issue_float_c_chunk_tma(uint32_t tmem_base, const CUtensorMap *c_map,
                        uint32_t *c_smem, int chunk_m, int chunk_n,
                        int row_offset, int col_offset) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  stage_float_c_chunk(tmem_base, c_smem, chunk_m, chunk_n);
  __syncthreads();
  tma_store_fence_shared();
  __syncthreads();
  if (threadIdx.x == 0) {
    tma_store_4d(c_map, smem_ptr_u32(c_smem), 0, row_offset, col_offset / 32,
                 0);
  }
#else
  (void)tmem_base;
  (void)c_map;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
  (void)row_offset;
  (void)col_offset;
#endif
}

__device__ __forceinline__ void
store_256x256_float_tile_tma(uint32_t tmem_base,
                             const CUtensorMap *c_map, uint32_t *c_smem,
                             int row_offset, int col_offset) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#pragma unroll
  for (int chunk = 0; chunk < kCStoreChunkCount; ++chunk) {
    const int buffer = chunk % kCStoreBuffers;
    if (chunk >= kCStoreBuffers) {
      if (threadIdx.x == 0) {
        tma_store_wait_group_read_2();
      }
      __syncthreads();
    }
    uint32_t *tile_smem = c_smem + buffer * kCStoreStageWords;
    const int chunk_m = chunk / kCStoreChunksN;
    const int chunk_n = chunk - chunk_m * kCStoreChunksN;
    const int tile_row = row_offset + chunk_m * kCStoreChunkM;
    const int tile_col = col_offset + chunk_n * kCStoreChunkN;
    issue_float_c_chunk_tma(tmem_base, c_map, tile_smem, chunk_m, chunk_n,
                            tile_row, tile_col);
    if (threadIdx.x == 0) {
      tma_store_commit_group();
    }
  }
  if (threadIdx.x == 0) {
    tma_store_wait_group_read_0();
  }
  __syncthreads();
#else
  (void)tmem_base;
  (void)c_map;
  (void)c_smem;
  (void)row_offset;
  (void)col_offset;
#endif
}

__device__ __forceinline__ void issue_a_stage_tma(const CUtensorMap *a_map,
                                                  uint32_t *a_smem,
                                                  uint64_t *ready, int tile_m,
                                                  int ktile) {
  mbarrier_expect_tx(ready, kAStageBytes);
  const int a_row = tile_m * kCtaM;
  const int a_col_words = ktile * (kStageK / 2);
  tma_load_2d(a_map, smem_ptr_u32(a_smem), ready, a_col_words, a_row);
}

__device__ __forceinline__ void
issue_b_pipe_stage_tma(const CUtensorMap *b_map, uint32_t *b_smem,
                       uint64_t *ready, int tile_n, int ktile, int pipe) {
  mbarrier_expect_tx(ready, kBPipeBytes);
  const int b_col_words = tile_n * (kCtaN / 2) + pipe * (kMmaN / 2);
  const int b_k16 = ktile * (kStageK / kMmaK);
  tma_load_4d(b_map, smem_ptr_u32(b_smem), ready, b_col_words, 0, 0, b_k16);
}

__global__ __launch_bounds__(kThreads, 1) void gemm256_bf16_16k_kernel(
    const __grid_constant__ CUtensorMap a_map,
    const __grid_constant__ CUtensorMap b_map,
    const __grid_constant__ CUtensorMap c_map, uint32_t *__restrict__ sink,
    int ktiles, int mtile_count, int ntile_count) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)c_map;
  (void)sink;
  (void)ktiles;
  (void)mtile_count;
  (void)ntile_count;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr = (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) &
                              ~static_cast<uintptr_t>(1023u);
  uint32_t *smem = reinterpret_cast<uint32_t *>(smem_addr);
  uint32_t *c_store_smem = smem;

  __shared__ uint64_t a_ready[kStages];
  __shared__ uint64_t b_ready[kPipes][kStages];
  __shared__ uint64_t mma_done[kPipes][kStages];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ int persistent_task_shared;

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStages; ++s) {
      mbarrier_init(&a_ready[s], 1);
#pragma unroll
      for (int p = 0; p < kPipes; ++p) {
        mbarrier_init(&b_ready[p][s], 1);
      }
    }
#pragma unroll
    for (int p = 0; p < kPipes; ++p) {
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        mbarrier_init(&mma_done[p][s], 1);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane0)
      tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t idesc = make_bf16_idesc() | (1u << 16);

  // The dynamic counter hands out one 16x16-macroblock position at a time.
  // M varies fastest inside a macroblock so neighboring workers share B;
  // macro N varies fastest so the next macroblock retains the same M range.
  const int total_tiles = mtile_count * ntile_count;
  const int persistent_macro_m =
      mtile_count < kPersistentMacroM ? mtile_count : kPersistentMacroM;
  const int persistent_macro_n =
      ntile_count < kPersistentMacroN ? ntile_count : kPersistentMacroN;
  const int persistent_groups_m =
      (mtile_count + persistent_macro_m - 1) / persistent_macro_m;
  const int persistent_groups_n =
      (ntile_count + persistent_macro_n - 1) / persistent_macro_n;
  const int persistent_macro_tiles = persistent_macro_m * persistent_macro_n;
  const int persistent_task_count =
      persistent_groups_m * persistent_groups_n * persistent_macro_tiles;
  int tile_iter = 0;
  while (true) {
    if (threadIdx.x == 0) {
      persistent_task_shared =
          static_cast<int>(atomicAdd(sink + total_tiles, 1u));
    }
    __syncthreads();
    const int linear_tile = persistent_task_shared;
    if (linear_tile >= persistent_task_count)
      break;

    const int macro_id = linear_tile / persistent_macro_tiles;
    const int local = linear_tile - macro_id * persistent_macro_tiles;
    const int macro_n = macro_id % persistent_groups_n;
    const int macro_m = macro_id / persistent_groups_n;
    const int local_m = local % persistent_macro_m;
    const int local_n = local / persistent_macro_m;
    const int tile_m = macro_m * persistent_macro_m + local_m;
    const int tile_n = macro_n * persistent_macro_n + local_n;
    // Only edge macroblocks can contain padded tasks.  They consume no barrier
    // epochs, so the next valid tile keeps the expected parity.
    if (tile_m >= mtile_count || tile_n >= ntile_count) {
      __syncthreads();
      continue;
    }
    const int ntile = ntile_count;
    const int stage_epoch_base = tile_iter * ktiles;

    if (warp_id == 0 && lane0) {
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords;
        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
          for (int p = 0; p < kPipes; ++p) {
            mbarrier_wait(&mma_done[p][stage], reuse_phase);
          }
        }
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
      }
    }

    if (warp_id == 1 && lane0) {
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *b_smem = stage_smem + kAStageWords + kBPipeWords;
        if (stage_epoch >= kStages) {
          mbarrier_wait(
              &mma_done[1][stage],
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
        }
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords + pipe * kBPipeWords;

        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);

#pragma unroll
        for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
          const uint32_t b0 = smem_ptr_u32(b_smem);
          const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
          const bool input_d = (kt != 0) || (kk != 0);
#pragma unroll
          for (int mblock = 0; mblock < kMBlocks; ++mblock) {
            const uint64_t a_desc = make_stage_a_smem_desc(a_smem, mblock, kk);
            const int c_tile = mblock * 2 + pipe;
            tcgen05_mma_bf16_ss(tmem_base + c_tile * kTmemTileStride, a_desc,
                                b0_desc, idesc, input_d);
          }
        }
        tcgen05_commit(&mma_done[pipe][stage]);
      }
      const int last_stage_epoch = stage_epoch_base + ktiles - 1;
      const int last_stage = last_stage_epoch % kStages;
      const uint32_t last_phase =
          static_cast<uint32_t>((last_stage_epoch / kStages) & 1);
      mbarrier_wait(&mma_done[pipe][last_stage], last_phase);
    }
    __syncthreads();

    const uint32_t acc = static_cast<uint32_t>(threadIdx.x + 0x9e3779b9u);
    if (lane0)
      warp_sinks[warp_id] = acc;
    __syncthreads();

    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base,
                                 global_col_base);
    __syncthreads();

    if (threadIdx.x == 0) {
      uint32_t tile_sink = tmem_base ^ static_cast<uint32_t>(ktiles);
#pragma unroll
      for (int w = 0; w < kWarps; ++w)
        tile_sink ^= warp_sinks[w];
      sink[tile_m * ntile + tile_n] = tile_sink;
    }
    __syncthreads();

    ++tile_iter;
  } // persistent output-tile loop

  if (threadIdx.x == 0)
    tcgen05_fence_after_thread_sync();
  __syncthreads();

  if (warp_id == 0)
    tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0)
    tcgen05_relinquish_alloc_permit();
#endif
}

void encode_a_row_major_sw128_tma_map(CUtensorMap *map, void *base,
                                      uint64_t rows, uint64_t cols_bf16) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  const cuuint64_t global_dim[2] = {cols_words, rows};
  const cuuint64_t global_stride[1] = {cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[2] = {kStageK / 2, kCtaM};
  const cuuint32_t elem_stride[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(
                   map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2, base, global_dim,
                   global_stride, box_dim, elem_stride,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(a_row_major_sw128)");
}

void encode_b_row_major_sw128_k16_tma_map(CUtensorMap *map, void *base,
                                          uint64_t rows, uint64_t cols_bf16) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  const cuuint64_t global_dim[4] = {cols_words, kMmaK, kBTmaNSubtiles,
                                    rows / kMmaK};
  const cuuint64_t global_stride[3] = {
      cols_words * sizeof(uint32_t),
      static_cast<cuuint64_t>(kMmaN / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kMmaK) * cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kMmaN / 4, kMmaK, kBTmaNSubtiles,
                                 kStageK / kMmaK};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(
                   map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 4, base, global_dim,
                   global_stride, box_dim, elem_stride,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(b_row_major_sw128_k16)");
}

void encode_c_row_major_float_tma_map(CUtensorMap *map, void *base,
                                      uint64_t rows, uint64_t cols) {
  const cuuint64_t global_dim[4] = {32, rows, cols / 32, 1};
  const cuuint64_t global_stride[3] = {
      cols * sizeof(float), static_cast<cuuint64_t>(32) * sizeof(float),
      rows * cols * sizeof(float)};
  const cuuint32_t box_dim[4] = {32, kCStoreChunkM, kCStoreChunkN / 32, 1};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(
                   map, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, base, global_dim,
                   global_stride, box_dim, elem_stride,
                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(c_row_major_float_sw128)");
}

const char *input_init_mode_name(int mode) {
  switch (mode) {
  case kInputInitMemset:
    return "memset";
  case kInputInitFormula:
    return "formula";
  case kInputInitRandom:
    return "random";
  case kInputInitRandomSigned8:
    return "random-signed8";
  default:
    return "unknown";
  }
}

void usage(const char *argv0) {
  std::printf("Usage: %s [--device N] [--warmup W] [--iters I] [--csv PATH] "
              "[--input-init memset|formula|random|random-signed8] "
              "[--validate] [--validate-size N] "
              "[--validate-pattern pattern|ones]\n",
              argv0);
}

Args parse_args(int argc, char **argv) {
  Args args;
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
      args.device = std::atoi(need_arg("--device"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      args.warmup = std::atoi(need_arg("--warmup"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      args.iters = std::atoi(need_arg("--iters"));
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      args.csv = need_arg("--csv");
    } else if (std::strcmp(argv[i], "--input-init") == 0) {
      const char *mode = need_arg("--input-init");
      if (std::strcmp(mode, "memset") == 0) {
        args.input_init_mode = kInputInitMemset;
      } else if (std::strcmp(mode, "formula") == 0) {
        args.input_init_mode = kInputInitFormula;
      } else if (std::strcmp(mode, "random") == 0) {
        args.input_init_mode = kInputInitRandom;
      } else if (std::strcmp(mode, "random-signed8") == 0) {
        args.input_init_mode = kInputInitRandomSigned8;
      } else {
        std::fprintf(stderr,
                     "input init must be 'memset', 'formula', 'random', or "
                     "'random-signed8'\n");
        std::exit(EXIT_FAILURE);
      }
    } else if (std::strcmp(argv[i], "--validate") == 0) {
      args.validate = true;
    } else if (std::strcmp(argv[i], "--validate-size") == 0) {
      args.validate_size = std::atoi(need_arg("--validate-size"));
    } else if (std::strcmp(argv[i], "--validate-pattern") == 0) {
      args.validate_pattern = need_arg("--validate-pattern");
    } else if (std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (args.warmup < 0 || args.iters <= 0) {
    std::fprintf(stderr, "warmup must be >= 0 and iters > 0\n");
    std::exit(EXIT_FAILURE);
  }
  if (args.validate_size <= 0 || args.validate_size > kMaxValidationSize ||
      args.validate_size % kCtaM != 0 || args.validate_size % kCtaN != 0 ||
      args.validate_size % kStageK != 0) {
    std::fprintf(stderr,
                 "validate size must be <= %d and a positive multiple of "
                 "cta_m=%d, cta_n=%d, and stage_k=%d\n",
                 kMaxValidationSize, kCtaM, kCtaN, kStageK);
    std::exit(EXIT_FAILURE);
  }
  if (std::strcmp(args.validate_pattern, "pattern") != 0 &&
      std::strcmp(args.validate_pattern, "ones") != 0) {
    std::fprintf(stderr, "validate pattern must be 'pattern' or 'ones'\n");
    std::exit(EXIT_FAILURE);
  }
  return args;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

static constexpr float kFormulaInitScale = 1.0f / 128.0f;

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__global__ void init_formula_bf16_words(uint32_t *words, size_t word_count,
                                        uint64_t index_offset, float scale) {
  const size_t word_idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_idx >= word_count)
    return;

  const uint64_t lo_idx = index_offset + static_cast<uint64_t>(word_idx) * 2u;
  const uint64_t hi_idx = lo_idx + 1u;
  const float lo_value =
      (static_cast<float>(static_cast<int>(lo_idx % 251u)) - 125.0f) * scale;
  const float hi_value =
      (static_cast<float>(static_cast<int>(hi_idx % 251u)) - 125.0f) * scale;
  const uint32_t lo = float_to_bf16_bits_device(lo_value);
  const uint32_t hi = float_to_bf16_bits_device(hi_value);
  words[word_idx] = lo | (hi << 16);
}

__device__ __forceinline__ uint32_t random_mix32(uint32_t x) {
  x += 0x9e3779b9u;
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  return x ^ (x >> 16);
}

__global__ void init_random_bf16_words(uint32_t *words, size_t word_count,
                                       uint32_t seed, float scale, float bias) {
  const size_t word_idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_idx >= word_count)
    return;
  const uint32_t index = static_cast<uint32_t>(word_idx);
  const uint32_t lo24 = random_mix32(seed ^ index ^ 0x9e3779b9u) >> 8;
  const uint32_t hi24 = random_mix32(seed ^ index ^ 0x243f6a88u) >> 8;
  const float lo = static_cast<float>(lo24) * 0x1.0p-24f * scale + bias;
  const float hi = static_cast<float>(hi24) * 0x1.0p-24f * scale + bias;
  words[word_idx] =
      static_cast<uint32_t>(float_to_bf16_bits_device(lo)) |
      (static_cast<uint32_t>(float_to_bf16_bits_device(hi)) << 16);
}

void initialize_bf16_inputs(uint32_t *d_a, size_t a_words, uint32_t *d_b,
                            size_t b_words, int input_init_mode) {
  if (input_init_mode == kInputInitMemset) {
    cuda_check(cudaMemset(d_a, 0x3f, a_words * sizeof(uint32_t)));
    cuda_check(cudaMemset(d_b, 0x11, b_words * sizeof(uint32_t)));
    return;
  }
  if (input_init_mode == kInputInitRandom ||
      input_init_mode == kInputInitRandomSigned8) {
    constexpr int kInitThreads = 256;
    const int a_blocks =
        static_cast<int>((a_words + kInitThreads - 1) / kInitThreads);
    const int b_blocks =
        static_cast<int>((b_words + kInitThreads - 1) / kInitThreads);
    const float scale = input_init_mode == kInputInitRandom ? 1.0f : 16.0f;
    const float bias = input_init_mode == kInputInitRandom ? 0.0f : -8.0f;
    init_random_bf16_words<<<a_blocks, kInitThreads>>>(
        d_a, a_words, 20260719u ^ 0xa511e9b3u, scale, bias);
    cuda_check(cudaGetLastError());
    init_random_bf16_words<<<b_blocks, kInitThreads>>>(
        d_b, b_words, 20260719u ^ 0x63d83595u, scale, bias);
    cuda_check(cudaGetLastError());
    return;
  }
  if (input_init_mode != kInputInitFormula) {
    std::fprintf(stderr, "Unknown input init mode: %d\n", input_init_mode);
    std::exit(EXIT_FAILURE);
  }

  constexpr int kInitThreads = 256;
  const int a_blocks =
      static_cast<int>((a_words + kInitThreads - 1) / kInitThreads);
  const int b_blocks =
      static_cast<int>((b_words + kInitThreads - 1) / kInitThreads);
  init_formula_bf16_words<<<a_blocks, kInitThreads>>>(d_a, a_words, 0,
                                                      kFormulaInitScale);
  cuda_check(cudaGetLastError());
  init_formula_bf16_words<<<b_blocks, kInitThreads>>>(
      d_b, b_words, static_cast<uint64_t>(a_words) * 2u, kFormulaInitScale);
  cuda_check(cudaGetLastError());
}

void set_gemm_kernel_attribute() {
  cuda_check(cudaFuncSetAttribute(gemm256_bf16_16k_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
}

void launch_gemm_kernel(dim3 grid, const CUtensorMap &a_map,
                        const CUtensorMap &b_map, const CUtensorMap &c_map,
                        uint32_t *d_sink, int ktiles, int mtile, int ntile) {
  gemm256_bf16_16k_kernel<<<grid, kThreads, kDynamicSmemBytes>>>(
      a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);
}
struct CaseResult {
  int size = 0;
  int mtile = 0;
  int ntile = 0;
  int ktiles = 0;
  int ctas = 0;
  int launch_ctas = 0;
  int input_init_mode = kInputInitMemset;
  float event_ms = 0.0f;
  double wall_ms = 0.0;
  double event_tflops = 0.0;
  double wall_tflops = 0.0;
  uint32_t checksum = 0;
};

CaseResult run_case(int warmup, int iters, int input_init_mode) {
  constexpr int size = kBenchmarkSize;
  const int m = size;
  const int n = size;
  const int k = size;
  const int mtile = m / kCtaM;
  const int ntile = n / kCtaN;
  const int ktiles = k / kStageK;
  const int ctas = mtile * ntile;
  const size_t a_words = static_cast<size_t>(m) * k / 2;
  const size_t b_words = static_cast<size_t>(k) * n / 2;

  uint32_t *d_a = nullptr;
  uint32_t *d_b = nullptr;
  uint32_t *d_sink = nullptr;
  float *d_c = nullptr;
  cuda_check(cudaMalloc(&d_a, a_words * sizeof(uint32_t)));
  cuda_check(cudaMalloc(&d_b, b_words * sizeof(uint32_t)));
  cuda_check(
      cudaMalloc(&d_sink, (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  cuda_check(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
  cuda_check(cudaMemset(d_c, 0, static_cast<size_t>(m) * n * sizeof(float)));
  initialize_bf16_inputs(d_a, a_words, d_b, b_words, input_init_mode);
  cuda_check(cudaMemset(d_sink, 0,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  cuda_check(cudaDeviceSynchronize());

  CUtensorMap a_map{}, b_map{}, c_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);
  encode_c_row_major_float_tma_map(&c_map, d_c, m, n);

  set_gemm_kernel_attribute();

  const dim3 grid(std::min(kPersistentCtas, ctas), 1, 1);
  auto launch_gemm = [&]() {
    cuda_check(cudaMemsetAsync(d_sink + ctas, 0, sizeof(uint32_t)));
    launch_gemm_kernel(grid, a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);
  };

  for (int i = 0; i < warmup; ++i) {
    launch_gemm();
    cuda_check(cudaGetLastError());
  }
  cuda_check(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  cuda_check(cudaEventCreate(&start));
  cuda_check(cudaEventCreate(&stop));
  const auto wall_start = std::chrono::steady_clock::now();
  cuda_check(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_gemm();
    cuda_check(cudaGetLastError());
  }
  cuda_check(cudaEventRecord(stop));
  cuda_check(cudaEventSynchronize(stop));
  const auto wall_stop = std::chrono::steady_clock::now();

  float total_event_ms = 0.0f;
  cuda_check(cudaEventElapsedTime(&total_event_ms, start, stop));

  std::vector<uint32_t> h_sink(std::min(ctas, 1024));
  cuda_check(cudaMemcpy(h_sink.data(), d_sink, h_sink.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  uint32_t checksum = 0;
  for (uint32_t v : h_sink)
    checksum ^= v;

  const double avg_event_ms = static_cast<double>(total_event_ms) / iters;
  const double avg_wall_ms = elapsed_ms(wall_start, wall_stop) / iters;
  const double flops = 2.0 * static_cast<double>(m) * n * k;

  CaseResult result;
  result.size = size;
  result.mtile = mtile;
  result.ntile = ntile;
  result.ktiles = ktiles;
  result.ctas = ctas;
  result.launch_ctas = static_cast<int>(grid.x * grid.y * grid.z);
  result.input_init_mode = input_init_mode;
  result.event_ms = static_cast<float>(avg_event_ms);
  result.wall_ms = avg_wall_ms;
  result.event_tflops = flops / (avg_event_ms * 1.0e-3) / 1.0e12;
  result.wall_tflops = flops / (avg_wall_ms * 1.0e-3) / 1.0e12;
  result.checksum = checksum;

  cuda_check(cudaEventDestroy(start));
  cuda_check(cudaEventDestroy(stop));
  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_sink));
  cuda_check(cudaFree(d_c));
  return result;
}

uint16_t float_to_bf16_bits_host(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_bits_to_float_host(uint16_t bits) {
  uint32_t value = static_cast<uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &value, sizeof(out));
  return out;
}

uint32_t pack_bf16_pair_host(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

float validation_a_value(bool use_ones, int row, int col) {
  if (use_ones)
    return 1.0f;
  return (static_cast<float>((row % 17) - 8) * 0.015625f) +
         (static_cast<float>((col % 11) - 5) * 0.0078125f);
}

float validation_b_value(bool use_ones, int row, int col) {
  if (use_ones)
    return 1.0f;
  return (static_cast<float>((row % 13) - 6) * 0.01171875f) -
         (static_cast<float>((col % 19) - 9) * 0.005859375f);
}

struct ValidateResult {
  bool ok = false;
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t bad_count = 0;
  int first_bad_row = -1;
  int first_bad_col = -1;
  float first_bad_got = 0.0f;
  float first_bad_ref = 0.0f;
};

ValidateResult run_validation(int size, const char *pattern) {
  const int m = size;
  const int n = size;
  const int k = size;
  const int mtile = m / kCtaM;
  const int ntile = n / kCtaN;
  const int ktiles = k / kStageK;
  const int ctas = mtile * ntile;

  std::vector<uint32_t> h_a(static_cast<size_t>(m) * k / 2, 0);
  std::vector<uint32_t> h_b(static_cast<size_t>(k) * n / 2, 0);
  std::vector<float> a_ref(static_cast<size_t>(m) * k);
  std::vector<float> b_ref(static_cast<size_t>(k) * n);
  const bool use_ones = std::strcmp(pattern, "ones") == 0;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < k; col += 2) {
      const float lo_value = validation_a_value(use_ones, row, col);
      const float hi_value = validation_a_value(use_ones, row, col + 1);
      const uint16_t lo_bits = float_to_bf16_bits_host(lo_value);
      const uint16_t hi_bits = float_to_bf16_bits_host(hi_value);
      a_ref[static_cast<size_t>(row) * k + col] =
          bf16_bits_to_float_host(lo_bits);
      a_ref[static_cast<size_t>(row) * k + col + 1] =
          bf16_bits_to_float_host(hi_bits);
      h_a[static_cast<size_t>(row) * (k / 2) + col / 2] =
          pack_bf16_pair_host(lo_bits, hi_bits);
    }
  }
  for (int row = 0; row < k; ++row) {
    for (int col = 0; col < n; col += 2) {
      const float lo_value = validation_b_value(use_ones, row, col);
      const float hi_value = validation_b_value(use_ones, row, col + 1);
      const uint16_t lo_bits = float_to_bf16_bits_host(lo_value);
      const uint16_t hi_bits = float_to_bf16_bits_host(hi_value);
      b_ref[static_cast<size_t>(row) * n + col] =
          bf16_bits_to_float_host(lo_bits);
      b_ref[static_cast<size_t>(row) * n + col + 1] =
          bf16_bits_to_float_host(hi_bits);
      h_b[static_cast<size_t>(row) * (n / 2) + col / 2] =
          pack_bf16_pair_host(lo_bits, hi_bits);
    }
  }

  uint32_t *d_a = nullptr;
  uint32_t *d_b = nullptr;
  uint32_t *d_sink = nullptr;
  float *d_c = nullptr;
  cuda_check(cudaMalloc(&d_a, h_a.size() * sizeof(uint32_t)));
  cuda_check(cudaMalloc(&d_b, h_b.size() * sizeof(uint32_t)));
  cuda_check(
      cudaMalloc(&d_sink, (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  cuda_check(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
  cuda_check(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemset(d_sink, 0,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  cuda_check(cudaMemset(d_c, 0, static_cast<size_t>(m) * n * sizeof(float)));

  CUtensorMap a_map{}, b_map{}, c_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);
  encode_c_row_major_float_tma_map(&c_map, d_c, m, n);
  set_gemm_kernel_attribute();

  const dim3 grid(std::min(kValidationCtas, ctas), 1, 1);
  launch_gemm_kernel(grid, a_map, b_map, c_map, d_sink, ktiles, mtile, ntile);
  cuda_check(cudaGetLastError());
  cuda_check(cudaDeviceSynchronize());

  std::vector<float> got(static_cast<size_t>(m) * n);
  cuda_check(cudaMemcpy(got.data(), d_c, got.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  ValidateResult result;
  constexpr double kAbsTol = 1.0e-5;
  constexpr double kRelTol = 1.0e-5;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double ref = 0.0;
      for (int kk = 0; kk < k; ++kk) {
        ref += static_cast<double>(a_ref[static_cast<size_t>(row) * k + kk]) *
               static_cast<double>(b_ref[static_cast<size_t>(kk) * n + col]);
      }
      const double actual = got[static_cast<size_t>(row) * n + col];
      const double abs_err = std::abs(actual - ref);
      const double rel_err = abs_err / std::max(1.0e-12, std::abs(ref));
      result.max_abs = std::max(result.max_abs, abs_err);
      result.max_rel = std::max(result.max_rel, rel_err);
      if (abs_err > kAbsTol && rel_err > kRelTol) {
        if (result.bad_count == 0) {
          result.first_bad_row = row;
          result.first_bad_col = col;
          result.first_bad_got = static_cast<float>(actual);
          result.first_bad_ref = static_cast<float>(ref);
        }
        ++result.bad_count;
      }
    }
  }
  result.ok = result.bad_count == 0;

  cuda_check(cudaFree(d_a));
  cuda_check(cudaFree(d_b));
  cuda_check(cudaFree(d_sink));
  cuda_check(cudaFree(d_c));
  return result;
}

} // namespace

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  cuda_check(cudaSetDevice(args.device));
  cuda_check(cudaFree(nullptr));
  driver_check(cuInit(0), "cuInit");

  cudaDeviceProp prop{};
  cuda_check(cudaGetDeviceProperties(&prop, args.device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This benchmark requires SM100+; got sm_%d%d\n",
                 prop.major, prop.minor);
    return 77;
  }

  if (args.validate) {
    const ValidateResult r =
        run_validation(args.validate_size, args.validate_pattern);
    std::printf("validation size=%d pattern=%s status=%s "
                "max_abs=%g max_rel=%g bad=%zu\n",
                args.validate_size, args.validate_pattern, r.ok ? "ok" : "fail",
                r.max_abs, r.max_rel, r.bad_count);
    if (!r.ok) {
      std::printf("first_bad row=%d col=%d got=%g ref=%g\n", r.first_bad_row,
                  r.first_bad_col, r.first_bad_got, r.first_bad_ref);
    }
    return r.ok ? 0 : 1;
  }

  FILE *csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    return 1;
  }
  std::fprintf(
      csv, "size,m,n,k,cta_m,cta_n,stage_k,stages,mtile,ntile,ktiles,ctas,"
           "launch_ctas,warmup,iters,input_init,dynamic_smem_bytes,event_ms,"
           "wall_ms,event_TFLOPS,wall_TFLOPS,checksum,device\n");

  std::printf(
      "device=%d name=\"%s\" cc=%d.%d cta=256x256 stage_k=64 "
      "stages=3 pipes=2 persistent_ctas=%d scheduler=dynamic_16x16_mfast "
      "phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "
      "dynamic_smem=%d\n",
      args.device, prop.name, prop.major, prop.minor, kPersistentCtas,
      kDynamicSmemBytes);

  const CaseResult r = run_case(args.warmup, args.iters, args.input_init_mode);
  std::printf("size=%d mtile=%d ntile=%d ktiles=%d ctas=%d launch_ctas=%d "
              "input_init=%s event_ms=%.6f wall_ms=%.6f "
              "event_TFLOPS=%.3f wall_TFLOPS=%.3f checksum=%08x\n",
              r.size, r.mtile, r.ntile, r.ktiles, r.ctas, r.launch_ctas,
              input_init_mode_name(r.input_init_mode), r.event_ms, r.wall_ms,
              r.event_tflops, r.wall_tflops, r.checksum);
  std::fprintf(csv,
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%d,"
               "%.6f,%.6f,%.3f,%.3f,%08x,%s\n",
               r.size, r.size, r.size, r.size, kCtaM, kCtaN, kStageK, kStages,
               r.mtile, r.ntile, r.ktiles, r.ctas, r.launch_ctas, args.warmup,
               args.iters, input_init_mode_name(r.input_init_mode),
               kDynamicSmemBytes, r.event_ms, r.wall_ms, r.event_tflops,
               r.wall_tflops, r.checksum, prop.name);

  std::fclose(csv);
  return 0;
}
