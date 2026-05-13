#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef ATTENTION_PRODUCER_REGS
#define ATTENTION_PRODUCER_REGS 104
#endif
#ifndef ATTENTION_CONSUMER_REGS
#define ATTENTION_CONSUMER_REGS 176
#endif
#ifndef ATTENTION_USE_SETMAXNREG
#define ATTENTION_USE_SETMAXNREG 1
#endif
#ifndef ATTENTION_PV_PINGPONG_DEP
#define ATTENTION_PV_PINGPONG_DEP 1
#endif
#ifndef ATTENTION_PV_H0_PINGPONG_DEP
#define ATTENTION_PV_H0_PINGPONG_DEP 0
#endif
#ifndef ATTENTION_QK_PINGPONG_DEP
#define ATTENTION_QK_PINGPONG_DEP 1
#endif
#ifndef ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA
#define ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA 0
#endif
#ifndef ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
#define ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME 0
#endif
#ifndef ATTENTION_SINGLE_PIPE0
#define ATTENTION_SINGLE_PIPE0 0
#endif
#define ATTENTION_SINGLE_PIPE_MODE ATTENTION_SINGLE_PIPE0
#ifndef ATTENTION_V_TMA_AFTER_S_READY
#define ATTENTION_V_TMA_AFTER_S_READY 0
#endif
#ifndef ATTENTION_SPLIT_S_READY
#define ATTENTION_SPLIT_S_READY 1
#endif
#ifndef ATTENTION_S_SWIZZLE_MODE
#define ATTENTION_S_SWIZZLE_MODE 0
#endif

#ifndef ATTENTION_STORE_OUTPUT
#define ATTENTION_STORE_OUTPUT 1
#endif

#ifndef ATTENTION_ACCUM_ROW_SUM
#define ATTENTION_ACCUM_ROW_SUM ATTENTION_STORE_OUTPUT
#endif

#ifndef ATTENTION_MATERIALIZE_OUTPUT
#define ATTENTION_MATERIALIZE_OUTPUT ATTENTION_STORE_OUTPUT
#endif

#ifndef ATTENTION_DRAIN_OUTPUT_TMEM
#define ATTENTION_DRAIN_OUTPUT_TMEM ATTENTION_MATERIALIZE_OUTPUT
#endif

#ifndef ATTENTION_PACK_OUTPUT
#define ATTENTION_PACK_OUTPUT ATTENTION_MATERIALIZE_OUTPUT
#endif

#ifndef ATTENTION_DIRECT_STORE_OUTPUT
#define ATTENTION_DIRECT_STORE_OUTPUT ATTENTION_MATERIALIZE_OUTPUT
#endif

#ifndef ATTENTION_CLOCK_TRACE
#define ATTENTION_CLOCK_TRACE 0
#endif

#define ATTENTION_STRINGIFY_IMPL(x) #x
#define ATTENTION_STRINGIFY(x) ATTENTION_STRINGIFY_IMPL(x)

#if ATTENTION_PV_PINGPONG_DEP
#define ATTENTION_PV_DEP_NOTE "pv_pingpong_dep"
#elif ATTENTION_SPLIT_S_READY && ATTENTION_PV_H0_PINGPONG_DEP
#define ATTENTION_PV_DEP_NOTE "pv_h0_pingpong_dep"
#else
#define ATTENTION_PV_DEP_NOTE "pv_no_pingpong_dep"
#endif

#if ATTENTION_QK_PINGPONG_DEP
#define ATTENTION_QK_DEP_NOTE "qk_pingpong_dep"
#else
#define ATTENTION_QK_DEP_NOTE "qk_no_pingpong_dep"
#endif

#if ATTENTION_SPLIT_S_READY
#define ATTENTION_S_READY_NOTE "split_s_ready"
#else
#define ATTENTION_S_READY_NOTE "full_s_ready"
#endif

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
static constexpr int kPipeCount = 2;
static constexpr int kActivePipeStride = ATTENTION_SINGLE_PIPE_MODE ? 1 : kPipeCount;
static constexpr int kConsumerWarpsPerPipe = 4;
static constexpr int kTraceConsumerLanesPerPipe = kConsumerWarpsPerPipe * 2;
static constexpr int kClockTraceSyncCount = 4;
static constexpr int kMainWarps = 4 + kPipeCount * kConsumerWarpsPerPipe;
static constexpr int kMainThreads = kMainWarps * kWarpSize;
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kMmaK = 16;
static constexpr int kMmasPerTile = 8;
static constexpr int kTileBf16Elems = kTileM * kTileN;
static constexpr int kTileWords = kTileBf16Elems / 2;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kKBufferCount = 1;
static constexpr int kVBufferCount = kPipeCount;
static constexpr int kSBufferCount = kPipeCount;
static constexpr int kDynamicSmemBytes =
    (1 + kPipeCount * kKBufferCount + kVBufferCount + kSBufferCount) * kTileBytes + 1024;
static constexpr int kTmemAllocCols = 512;
static constexpr int kTmemUsedCols = 512;
static constexpr float kLog2E = 1.44269504088896340736f;
static constexpr int kQkLayoutAtom = 0;
static constexpr int kQkLayoutContiguousSingle5d = 1;
static constexpr int kQkLayoutContiguousK16Split4d = 2;
static constexpr int kQkLayoutContiguousSw128 = 3;
static constexpr int kFixedQkLayout = kQkLayoutContiguousSw128;
static constexpr double kFlopsPerMma =
    2.0 * static_cast<double>(kTileM) * static_cast<double>(kTileN) *
    static_cast<double>(kMmaK);

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int k_tiles = 8192;
  bool k_tiles_set = false;
  int warmup = 2;
  int iters = 5;
  bool store_output = true;
  bool contiguous_qk = true;
  bool contiguous_qk_single_5d = false;
  bool contiguous_qk_sw128 = true;
  bool clock_trace = false;
  int clock_trace_start = 0;
  int clock_trace_iters = 8;
  std::string stage = "benchmark";
  std::string pattern = "constant";
  const char* csv = "0.attention/attention_fused_clean.csv";

  // Real attention path.  This is the correctness-oriented implementation:
  // contiguous BF16 Q/K/V/O with shape [B, H, S, D], D fixed to 128.
  bool validation_suite = false;
  int B = 1;
  int Hq = 1;
  int Hkv = 1;
  int Sq = 128;
  int Skv = 128;
  bool skv_set = false;
  int D = 128;
  bool causal = false;
  float softmax_scale = -1.0f;  // < 0 means 1 / sqrt(D).
};

struct RunResult {
  float ms = 0.0f;
  cudaError_t error = cudaSuccess;
  const char* status = "ok";
};

struct ClockTraceRecord {
  int stage = 0;
  int iter = 0;
  int pipe = 0;
  int warp_id = 0;
  int consumer_warp = 0;
  int half = 0;
  unsigned long long start = 0;
  unsigned long long end = 0;
};

struct TraceRecord {
  unsigned long long tma_start = 0;
  unsigned long long tma_end = 0;
  unsigned long long mma_start = 0xffffffffffffffffull;
  unsigned long long mma_end = 0xffffffffffffffffull;
  unsigned long long ld_start = 0xffffffffffffffffull;
  unsigned long long ld_end = 0;
  unsigned long long ld_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long ld_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long pack_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long st_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long st_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long sum_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long sum_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_start = 0xffffffffffffffffull;
  unsigned long long pack_end = 0;
  unsigned long long st_start = 0xffffffffffffffffull;
  unsigned long long st_end = 0;
  unsigned long long v_tma_start = 0;
  unsigned long long v_tma_end = 0;
  unsigned long long pv_start = 0xffffffffffffffffull;
  unsigned long long pv_end = 0;
  unsigned long long pv_h0_start = 0xffffffffffffffffull;
  unsigned long long pv_h0_end = 0;
  unsigned long long pv_h1_start = 0xffffffffffffffffull;
  unsigned long long pv_h1_end = 0;
  unsigned long long sync_start[kClockTraceSyncCount];
  unsigned long long sync_end[kClockTraceSyncCount];
  unsigned int iter = 0;
  unsigned int pipe = 0;
  unsigned int warp_id = 0;
};

enum ClockTraceStage {
  kClockTraceQTma = 1,
  kClockTraceKTma = 2,
  kClockTraceQkMma = 3,
  kClockTraceVTma = 4,
  kClockTracePvMma = 5,
  kClockTraceLd = 6,
  kClockTracePack = 7,
  kClockTraceRowSum = 8,
  kClockTraceTailWait = 9,
  kClockTraceTmemDrain = 10,
  kClockTracePackNorm = 11,
  kClockTraceGlobalStore = 12,
  kClockTraceTailTotal = 13,
  kClockTraceStore = 14,
  kClockTracePvMmaH0 = 15,
  kClockTracePvMmaH1 = 16,
  kClockTraceSync = 17,
};

static constexpr int kClockTraceSlotsPerIter = 64;
static constexpr int kClockTraceLdBase = 8;
static constexpr int kClockTracePackStoreBase = 16;
static constexpr int kClockTraceRowSumBase = 24;
static constexpr int kClockTraceStoreBase = 32;
static constexpr int kClockTraceSyncBase = 40;
static constexpr int kClockTraceExtraSlots = 32;

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__host__ __device__ __forceinline__ uint64_t make_smem_desc(uint32_t matrix_start_addr) {
  const uint32_t matrix_start_aligned = matrix_start_addr & ~0xFu;
  constexpr uint32_t leading_dim_byte_offset = 128;
  constexpr uint32_t stride_dim_byte_offset = 256;
  constexpr uint32_t swizzle_mode = 0;
  const uint32_t lead_enc = (leading_dim_byte_offset & 0x3ffffu) >> 4;
  const uint32_t stride_enc = (stride_dim_byte_offset & 0x3ffffu) >> 4;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>(lead_enc) << 16;
  desc |= static_cast<uint64_t>(stride_enc) << 32;
  desc |= static_cast<uint64_t>(0x1u) << 46;
  desc |= static_cast<uint64_t>(0xB0u) << 53;
  desc |= static_cast<uint64_t>(swizzle_mode) << 61;
  return desc;
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_k_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  const int half = mma >> 2;
  const int in_half = mma & 3;
  const uint32_t addr =
      matrix_start_addr + static_cast<uint32_t>(half * (kTileBytes / 2) +
                                                in_half * 32);
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((addr >> 4) & 0x3fffu);
  desc |= static_cast<uint64_t>(1u) << 16;   // leading byte offset = 16B.
  desc |= static_cast<uint64_t>(64u) << 32;  // stride byte offset = 1024B.
  desc |= static_cast<uint64_t>(1u) << 46;   // Blackwell descriptor version.
  desc |= static_cast<uint64_t>(2u) << 61;   // SWIZZLE_128B.
  return desc;
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_mn_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  const uint32_t addr = matrix_start_addr + static_cast<uint32_t>(mma) * 4096u;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>((addr >> 4) & 0x3fffu);
  desc |= static_cast<uint64_t>(128u) << 16;  // leading byte offset = 2048B.
  desc |= static_cast<uint64_t>(64u) << 32;   // stride byte offset = 1024B.
  desc |= static_cast<uint64_t>(1u) << 46;    // Blackwell descriptor version.
  desc |= static_cast<uint64_t>(2u) << 61;    // SWIZZLE_128B.
  return desc;
}

__host__ __device__ __forceinline__ uint64_t make_s_smem_desc(uint32_t matrix_start_addr) {
  const uint32_t matrix_start_aligned = matrix_start_addr & ~0xFu;
  constexpr uint32_t leading_dim_byte_offset = 128;
  constexpr uint32_t stride_dim_byte_offset = 256;
  constexpr uint32_t swizzle_mode = ATTENTION_S_SWIZZLE_MODE;
  const uint32_t lead_enc = (leading_dim_byte_offset & 0x3ffffu) >> 4;
  const uint32_t stride_enc = (stride_dim_byte_offset & 0x3ffffu) >> 4;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>(lead_enc) << 16;
  desc |= static_cast<uint64_t>(stride_enc) << 32;
  desc |= static_cast<uint64_t>(0x1u) << 46;
  desc |= static_cast<uint64_t>(0xB0u) << 53;
  desc |= static_cast<uint64_t>(swizzle_mode) << 61;
  return desc;
}

__host__ __device__ __forceinline__ int atom_major_k_word_offset(int row, int col_pair) {
  const int k16_atom = col_pair >> 3;
  const int pair_in_atom = col_pair & 7;
  const int row_group8 = row >> 3;
  const int row_in8 = row & 7;
  const int chunk16 = pair_in_atom >> 2;
  const int word_in_chunk = pair_in_atom & 3;
  return k16_atom * 1024 + row_group8 * 64 + chunk16 * 32 + row_in8 * 4 +
         word_in_chunk;
}

__host__ __device__ __forceinline__ uint32_t swizzle_s_byte_offset(uint32_t byte_offset) {
#if ATTENTION_S_SWIZZLE_MODE == 6
  return byte_offset ^ ((byte_offset & (0x1u << 5)) >> 1);  // SW32: Swizzle<1,4,1>.
#elif ATTENTION_S_SWIZZLE_MODE == 4
  return byte_offset ^ ((byte_offset & (0x3u << 6)) >> 2);  // SW64: Swizzle<2,4,2>.
#elif ATTENTION_S_SWIZZLE_MODE == 2
  return byte_offset ^ ((byte_offset & (0x7u << 7)) >> 3);  // SW128: Swizzle<3,4,3>.
#elif ATTENTION_S_SWIZZLE_MODE == 1
  return byte_offset ^ ((byte_offset & (0x3u << 7)) >> 2);  // 128B_BASE32B: Swizzle<2,5,2>.
#else
  return byte_offset;
#endif
}

__host__ __device__ __forceinline__ int s_store_word_offset(int row, int col_pair) {
  const uint32_t byte_offset =
      static_cast<uint32_t>(atom_major_k_word_offset(row, col_pair)) * 4u;
  return static_cast<int>(swizzle_s_byte_offset(byte_offset) >> 2);
}

__host__ __device__ __forceinline__ uint32_t make_qk_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(kTileN >> 3) << 17;
  desc |= static_cast<uint32_t>(kTileM >> 4) << 24;
  return desc;
}

__device__ __forceinline__ void setmaxnreg_dec_producer() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && ATTENTION_USE_SETMAXNREG
  asm volatile("setmaxnreg.dec.sync.aligned.u32 " ATTENTION_STRINGIFY(ATTENTION_PRODUCER_REGS) ";"
               ::: "memory");
#endif
}

__device__ __forceinline__ void setmaxnreg_inc_consumer() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && ATTENTION_USE_SETMAXNREG
  asm volatile("setmaxnreg.inc.sync.aligned.u32 " ATTENTION_STRINGIFY(ATTENTION_CONSUMER_REGS) ";"
               ::: "memory");
#endif
}

__device__ __forceinline__ bool warp_elect_leader() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t is_leader = 0;
  asm volatile(
      "{ .reg .pred p; elect.sync _|p, 0xffffffff; selp.u32 %0, 1, 0, p; }"
      : "=r"(is_leader)
      :
      : "memory");
  return is_leader != 0;
#else
  return (threadIdx.x & 31) == 0;
#endif
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

__device__ __forceinline__ void mbarrier_arrive(uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(addr) : "memory");
#else
  (void)barrier;
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

__device__ __forceinline__ void tma_load_2d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c,
                                            int r) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
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

__device__ __forceinline__ void tma_load_5d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c0,
                                            int c1,
                                            int c2,
                                            int c3,
                                            int c4) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2),
        "r"(c3), "r"(c4)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
  (void)c4;
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

__device__ __forceinline__ uint32_t tcgen05_alloc_512cols(uint32_t* smem_out_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t smem_addr = smem_ptr_u32(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 512;"
               :: "r"(smem_addr)
               : "memory");
  __syncwarp();
  uint32_t taddr;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(taddr) : "r"(smem_addr) : "memory");
  return taddr;
#else
  (void)smem_out_taddr;
  return 0;
#endif
}

__device__ __forceinline__ void tcgen05_dealloc_512cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 512;"
               :: "r"(taddr)
               : "memory");
#else
  (void)taddr;
#endif
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_mma_bf16_ss(uint32_t d_taddr,
                                                    uint64_t a_desc,
                                                    uint64_t b_desc,
                                                    uint32_t idesc,
                                                    bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
      "{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, %6, %7, %8}, pred; }"
      :: "r"(d_taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

__device__ __forceinline__ void tcgen05_commit(uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
               :: "r"(addr)
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

#define NVCC_LD_REG_OUTPUTS_ARRAY(a)                                         \
  "=&r"(a[0]), "=&r"(a[1]), "=&r"(a[2]), "=&r"(a[3]), "=&r"(a[4]),       \
      "=&r"(a[5]), "=&r"(a[6]), "=&r"(a[7]), "=&r"(a[8]), "=&r"(a[9]),    \
      "=&r"(a[10]), "=&r"(a[11]), "=&r"(a[12]), "=&r"(a[13]),             \
      "=&r"(a[14]), "=&r"(a[15]), "=&r"(a[16]), "=&r"(a[17]),             \
      "=&r"(a[18]), "=&r"(a[19]), "=&r"(a[20]), "=&r"(a[21]),             \
      "=&r"(a[22]), "=&r"(a[23]), "=&r"(a[24]), "=&r"(a[25]),             \
      "=&r"(a[26]), "=&r"(a[27]), "=&r"(a[28]), "=&r"(a[29]),             \
      "=&r"(a[30]), "=&r"(a[31]), "=&r"(a[32]), "=&r"(a[33]),             \
      "=&r"(a[34]), "=&r"(a[35]), "=&r"(a[36]), "=&r"(a[37]),             \
      "=&r"(a[38]), "=&r"(a[39]), "=&r"(a[40]), "=&r"(a[41]),             \
      "=&r"(a[42]), "=&r"(a[43]), "=&r"(a[44]), "=&r"(a[45]),             \
      "=&r"(a[46]), "=&r"(a[47]), "=&r"(a[48]), "=&r"(a[49]),             \
      "=&r"(a[50]), "=&r"(a[51]), "=&r"(a[52]), "=&r"(a[53]),             \
      "=&r"(a[54]), "=&r"(a[55]), "=&r"(a[56]), "=&r"(a[57]),             \
      "=&r"(a[58]), "=&r"(a[59]), "=&r"(a[60]), "=&r"(a[61]),             \
      "=&r"(a[62]), "=&r"(a[63])

#define NVCC_LD_REG_OUTPUTS_0_63 NVCC_LD_REG_OUTPUTS_ARRAY(r)

#define NVCC_LD_REG_OPERANDS_0_63                                            \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "  \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "  \
  "%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "  \
  "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "  \
  "%58, %59, %60, %61, %62, %63"

#define TCGEN05_LD_X64_ISSUE_ARRAY(src_taddr, arr)                         \
  asm volatile(                                                            \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" NVCC_LD_REG_OPERANDS_0_63 \
      "}, [%64];"                                                          \
      : NVCC_LD_REG_OUTPUTS_ARRAY(arr)                                     \
      : "r"(src_taddr)                                                    \
      : "memory")

__device__ __forceinline__ uint32_t exp2_approx_bits_cpp(uint32_t x) {
  const float in = __uint_as_float(x);
  float out;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(in));
  return __float_as_uint(out);
}

__device__ __forceinline__ uint32_t exp2_approx_bits_scaled_cpp(uint32_t x,
                                                                float scale) {
  const float in = __uint_as_float(x) * scale;
  float out;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(out) : "f"(in));
  return __float_as_uint(out);
}

__device__ __forceinline__ uint32_t exp2_pack_hi16_update(uint32_t& lo_src,
                                                          uint32_t& hi_src) {
  lo_src = exp2_approx_bits_cpp(lo_src);
  hi_src = exp2_approx_bits_cpp(hi_src);
  return (lo_src >> 16) | (hi_src & 0xffff0000u);
}

__device__ __forceinline__ uint32_t exp2_pack_hi16_update_scaled(
    uint32_t& lo_src,
    uint32_t& hi_src,
    float scale) {
  lo_src = exp2_approx_bits_scaled_cpp(lo_src, scale);
  hi_src = exp2_approx_bits_scaled_cpp(hi_src, scale);
  return (lo_src >> 16) | (hi_src & 0xffff0000u);
}

__device__ __forceinline__ float bf16x2_sum_device(uint32_t packed) {
  const float lo = __uint_as_float((packed & 0x0000ffffu) << 16);
  const float hi = __uint_as_float(packed & 0xffff0000u);
  return lo + hi;
}

__device__ __forceinline__ void store_packed4_s_cpp(uint32_t* smem,
                                                    int word_offset,
                                                    uint32_t p0,
                                                    uint32_t p1,
                                                    uint32_t p2,
                                                    uint32_t p3) {
  reinterpret_cast<uint4*>(smem + word_offset)[0] = make_uint4(p0, p1, p2, p3);
}

template <bool kDoSum>
__device__ __forceinline__ float pack_store_x64_loop(uint32_t* smem_base,
                                                     uint32_t (&r)[64],
                                                     float score_to_exp2_scale) {
  float sum = 0.0f;
#pragma unroll
  for (int group = 0; group < 8; ++group) {
    alignas(16) uint32_t p[4];
    const int r_base = group * 8;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      p[i] = exp2_pack_hi16_update_scaled(r[r_base + i * 2],
                                          r[r_base + i * 2 + 1],
                                          score_to_exp2_scale);
    }
    const int word_offset = (group >> 1) * 1024 + (group & 1) * 32;
    reinterpret_cast<uint4*>(smem_base + word_offset)[0] =
        reinterpret_cast<uint4*>(p)[0];
    if constexpr (kDoSum) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        sum += bf16x2_sum_device(p[i]);
      }
    }
  }
  return sum;
}

__device__ __forceinline__ void write_clock_trace_record(ClockTraceRecord* records,
                                                         int slot,
                                                         int stage,
                                                         int iter,
                                                         int pipe,
                                                         int warp_id,
                                                         int consumer_warp,
                                                         int half,
                                                         unsigned long long start,
                                                         unsigned long long end,
                                                         unsigned long long base) {
#if ATTENTION_CLOCK_TRACE
  if (records == nullptr || end <= start) return;
  ClockTraceRecord r;
  r.stage = stage;
  r.iter = iter;
  r.pipe = pipe;
  r.warp_id = warp_id;
  r.consumer_warp = consumer_warp;
  r.half = half;
  r.start = start - base;
  r.end = end - base;
  records[slot] = r;
#else
  (void)records;
  (void)slot;
  (void)stage;
  (void)iter;
  (void)pipe;
  (void)warp_id;
  (void)consumer_warp;
  (void)half;
  (void)start;
  (void)end;
  (void)base;
#endif
}

__device__ __forceinline__ void begin_clock_trace_record(ClockTraceRecord* records,
                                                         int slot,
                                                         int stage,
                                                         int iter,
                                                         int pipe,
                                                         int warp_id,
                                                         int consumer_warp,
                                                         int half,
                                                         unsigned long long start,
                                                         unsigned long long base) {
#if ATTENTION_CLOCK_TRACE
  if (records == nullptr) return;
  ClockTraceRecord r;
  r.stage = stage;
  r.iter = iter;
  r.pipe = pipe;
  r.warp_id = warp_id;
  r.consumer_warp = consumer_warp;
  r.half = half;
  r.start = start - base;
  r.end = r.start;
  records[slot] = r;
#else
  (void)records;
  (void)slot;
  (void)stage;
  (void)iter;
  (void)pipe;
  (void)warp_id;
  (void)consumer_warp;
  (void)half;
  (void)start;
  (void)base;
#endif
}

__device__ __forceinline__ void end_clock_trace_record(ClockTraceRecord* records,
                                                       int slot,
                                                       unsigned long long end,
                                                       unsigned long long base) {
#if ATTENTION_CLOCK_TRACE
  if (records == nullptr) return;
  records[slot].end = end - base;
#else
  (void)records;
  (void)slot;
  (void)end;
  (void)base;
#endif
}

__device__ __forceinline__ void tcgen05_ld_x64_wait_pack_store_nvcc(
    uint32_t src_taddr,
    uint32_t* s_smem,
    int consumer_warp,
    int consumer_half,
    uint64_t* p_done_barrier,
    bool arrive_p_done,
    float* row_sum_pipe,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int trace_iter,
    int trace_pipe) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = consumer_warp * 32 + lane;
  const int col_pair_base = consumer_half * 32;
  uint32_t* smem_base = s_smem + s_store_word_offset(row, col_pair_base);
  uint32_t r[64];

#if ATTENTION_CLOCK_TRACE
  const int trace_idx = trace_iter - clock_trace_start;
  const bool trace_window =
      clock_trace != nullptr && blockIdx.x == 0 && trace_idx >= 0 &&
      trace_idx < clock_trace_iters;
  const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
  const bool trace_lane = trace_window && lane == 0;
  const unsigned long long ld_start = trace_lane ? clock64() : 0ull;
#endif
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" NVCC_LD_REG_OPERANDS_0_63
      "}, [%64];"
      : NVCC_LD_REG_OUTPUTS_0_63
      : "r"(src_taddr)
      : "memory");
  tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long ld_end = clock64();
    const int slot =
        trace_slot_base + kClockTraceLdBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTraceLd, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, ld_start, ld_end, clock_trace_base);
  }
#endif
  if (arrive_p_done && lane == 0) {
    mbarrier_arrive(p_done_barrier);
  }

#if ATTENTION_CLOCK_TRACE
  const unsigned long long pack_start = trace_lane ? clock64() : 0ull;
  uint32_t p[32];
  p[0] = exp2_pack_hi16_update(r[0], r[1]);
  p[1] = exp2_pack_hi16_update(r[2], r[3]);
  p[2] = exp2_pack_hi16_update(r[4], r[5]);
  p[3] = exp2_pack_hi16_update(r[6], r[7]);
  p[4] = exp2_pack_hi16_update(r[8], r[9]);
  p[5] = exp2_pack_hi16_update(r[10], r[11]);
  p[6] = exp2_pack_hi16_update(r[12], r[13]);
  p[7] = exp2_pack_hi16_update(r[14], r[15]);
  p[8] = exp2_pack_hi16_update(r[16], r[17]);
  p[9] = exp2_pack_hi16_update(r[18], r[19]);
  p[10] = exp2_pack_hi16_update(r[20], r[21]);
  p[11] = exp2_pack_hi16_update(r[22], r[23]);
  p[12] = exp2_pack_hi16_update(r[24], r[25]);
  p[13] = exp2_pack_hi16_update(r[26], r[27]);
  p[14] = exp2_pack_hi16_update(r[28], r[29]);
  p[15] = exp2_pack_hi16_update(r[30], r[31]);
  p[16] = exp2_pack_hi16_update(r[32], r[33]);
  p[17] = exp2_pack_hi16_update(r[34], r[35]);
  p[18] = exp2_pack_hi16_update(r[36], r[37]);
  p[19] = exp2_pack_hi16_update(r[38], r[39]);
  p[20] = exp2_pack_hi16_update(r[40], r[41]);
  p[21] = exp2_pack_hi16_update(r[42], r[43]);
  p[22] = exp2_pack_hi16_update(r[44], r[45]);
  p[23] = exp2_pack_hi16_update(r[46], r[47]);
  p[24] = exp2_pack_hi16_update(r[48], r[49]);
  p[25] = exp2_pack_hi16_update(r[50], r[51]);
  p[26] = exp2_pack_hi16_update(r[52], r[53]);
  p[27] = exp2_pack_hi16_update(r[54], r[55]);
  p[28] = exp2_pack_hi16_update(r[56], r[57]);
  p[29] = exp2_pack_hi16_update(r[58], r[59]);
  p[30] = exp2_pack_hi16_update(r[60], r[61]);
  p[31] = exp2_pack_hi16_update(r[62], r[63]);
  if (trace_lane) {
    const unsigned long long pack_end = clock64();
    const int slot = trace_slot_base + kClockTracePackStoreBase +
                     consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTracePack, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
  }
  const unsigned long long store_start = trace_lane ? clock64() : 0ull;
  store_packed4_s_cpp(smem_base, 0, p[0], p[1], p[2], p[3]);
  store_packed4_s_cpp(smem_base, 32, p[4], p[5], p[6], p[7]);
  store_packed4_s_cpp(smem_base, 1024, p[8], p[9], p[10], p[11]);
  store_packed4_s_cpp(smem_base, 1056, p[12], p[13], p[14], p[15]);
  store_packed4_s_cpp(smem_base, 2048, p[16], p[17], p[18], p[19]);
  store_packed4_s_cpp(smem_base, 2080, p[20], p[21], p[22], p[23]);
  store_packed4_s_cpp(smem_base, 3072, p[24], p[25], p[26], p[27]);
  store_packed4_s_cpp(smem_base, 3104, p[28], p[29], p[30], p[31]);
  if (trace_lane) {
    const unsigned long long store_end = clock64();
    const int slot =
        trace_slot_base + kClockTraceStoreBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTraceStore, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, store_start, store_end,
                             clock_trace_base);
  }
#else
  store_packed4_s_cpp(smem_base, 0, exp2_pack_hi16_update(r[0], r[1]),
                      exp2_pack_hi16_update(r[2], r[3]),
                      exp2_pack_hi16_update(r[4], r[5]),
                      exp2_pack_hi16_update(r[6], r[7]));
  store_packed4_s_cpp(smem_base, 32, exp2_pack_hi16_update(r[8], r[9]),
                      exp2_pack_hi16_update(r[10], r[11]),
                      exp2_pack_hi16_update(r[12], r[13]),
                      exp2_pack_hi16_update(r[14], r[15]));
  store_packed4_s_cpp(smem_base, 1024, exp2_pack_hi16_update(r[16], r[17]),
                      exp2_pack_hi16_update(r[18], r[19]),
                      exp2_pack_hi16_update(r[20], r[21]),
                      exp2_pack_hi16_update(r[22], r[23]));
  store_packed4_s_cpp(smem_base, 1056, exp2_pack_hi16_update(r[24], r[25]),
                      exp2_pack_hi16_update(r[26], r[27]),
                      exp2_pack_hi16_update(r[28], r[29]),
                      exp2_pack_hi16_update(r[30], r[31]));
  store_packed4_s_cpp(smem_base, 2048, exp2_pack_hi16_update(r[32], r[33]),
                      exp2_pack_hi16_update(r[34], r[35]),
                      exp2_pack_hi16_update(r[36], r[37]),
                      exp2_pack_hi16_update(r[38], r[39]));
  store_packed4_s_cpp(smem_base, 2080, exp2_pack_hi16_update(r[40], r[41]),
                      exp2_pack_hi16_update(r[42], r[43]),
                      exp2_pack_hi16_update(r[44], r[45]),
                      exp2_pack_hi16_update(r[46], r[47]));
  store_packed4_s_cpp(smem_base, 3072, exp2_pack_hi16_update(r[48], r[49]),
                      exp2_pack_hi16_update(r[50], r[51]),
                      exp2_pack_hi16_update(r[52], r[53]),
                      exp2_pack_hi16_update(r[54], r[55]));
  store_packed4_s_cpp(smem_base, 3104, exp2_pack_hi16_update(r[56], r[57]),
                      exp2_pack_hi16_update(r[58], r[59]),
                      exp2_pack_hi16_update(r[60], r[61]),
                      exp2_pack_hi16_update(r[62], r[63]));
#endif
#if ATTENTION_ACCUM_ROW_SUM
  if (row_sum_pipe != nullptr) {
#if ATTENTION_CLOCK_TRACE
    const bool trace_row_sum = trace_lane;
    const unsigned long long row_sum_start = trace_row_sum ? clock64() : 0ull;
#endif
    float half_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      half_sum += __uint_as_float(r[i] & 0xffff0000u);
    }
    row_sum_pipe[row] += half_sum;
#if ATTENTION_CLOCK_TRACE
    if (trace_row_sum) {
      const unsigned long long row_sum_end = clock64();
      const int slot =
          trace_slot_base + kClockTraceRowSumBase + consumer_warp * 2 +
          consumer_half;
      write_clock_trace_record(clock_trace, slot, kClockTraceRowSum, trace_iter,
                               trace_pipe, threadIdx.x >> 5, consumer_warp,
                               consumer_half, row_sum_start, row_sum_end,
                               clock_trace_base);
    }
#endif
  }
#else
  (void)row_sum_pipe;
#endif
#else
  (void)src_taddr;
  (void)s_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)p_done_barrier;
  (void)arrive_p_done;
  (void)row_sum_pipe;
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
  (void)clock_trace_base;
  (void)trace_iter;
  (void)trace_pipe;
#endif
}

__device__ __forceinline__ float
tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
    uint32_t src_taddr,
    uint32_t* s_smem,
    int consumer_warp,
    int consumer_half,
    uint64_t* p_done_barrier,
    bool arrive_p_done,
    float score_to_exp2_scale,
    ClockTraceRecord* clock_trace,
    int clock_trace_iters,
    int clock_trace_start,
    unsigned long long clock_trace_base,
    int trace_iter,
    int trace_pipe) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = consumer_warp * 32 + lane;
  const int col_pair_base = consumer_half * 32;
  uint32_t* smem_base = s_smem + s_store_word_offset(row, col_pair_base);
  uint32_t r[64];

#if ATTENTION_CLOCK_TRACE
  const int trace_idx = trace_iter - clock_trace_start;
  const bool trace_window =
      clock_trace != nullptr && blockIdx.x == 0 && trace_idx >= 0 &&
      trace_idx < clock_trace_iters;
  const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
  const bool trace_lane = trace_window && lane == 0;
  const unsigned long long ld_start = trace_lane ? clock64() : 0ull;
#endif
  TCGEN05_LD_X64_ISSUE_ARRAY(src_taddr, r);
  tcgen05_wait_ld();
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long ld_end = clock64();
    const int slot =
        trace_slot_base + kClockTraceLdBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, slot, kClockTraceLd, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, ld_start, ld_end, clock_trace_base);
  }
#endif
  if (arrive_p_done && lane == 0) {
    mbarrier_arrive(p_done_barrier);
  }

#if ATTENTION_CLOCK_TRACE
  const unsigned long long pack_start = trace_lane ? clock64() : 0ull;
#endif
  const float row_sum = pack_store_x64_loop<true>(smem_base, r,
                                                 score_to_exp2_scale);
#if ATTENTION_CLOCK_TRACE
  if (trace_lane) {
    const unsigned long long pack_end = clock64();
    const int pack_slot = trace_slot_base + kClockTracePackStoreBase +
                          consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, pack_slot, kClockTracePack, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int store_slot =
        trace_slot_base + kClockTraceStoreBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, store_slot, kClockTraceStore, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
    const int sum_slot =
        trace_slot_base + kClockTraceRowSumBase + consumer_warp * 2 + consumer_half;
    write_clock_trace_record(clock_trace, sum_slot, kClockTraceRowSum, trace_iter,
                             trace_pipe, threadIdx.x >> 5, consumer_warp,
                             consumer_half, pack_start, pack_end,
                             clock_trace_base);
  }
#endif
  return row_sum;
#else
  (void)src_taddr;
  (void)s_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)p_done_barrier;
  (void)arrive_p_done;
  (void)score_to_exp2_scale;
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
  (void)clock_trace_base;
  (void)trace_iter;
  (void)trace_pipe;
  return 0.0f;
#endif
}

#if ATTENTION_STORE_OUTPUT
__device__ __forceinline__ float bf16_bits_to_float_device(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ uint32_t pack_bf16_pair_device(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits_device(lo)) |
         (static_cast<uint32_t>(float_to_bf16_bits_device(hi)) << 16);
}

__device__ __noinline__ void store_tmem_x64_accum_output_smem(uint32_t src_taddr,
                                                              float* output_smem,
                                                              int consumer_warp,
                                                              int consumer_half,
                                                              bool add_to_smem) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  float* dst = output_smem + static_cast<size_t>(consumer_warp * 32 + lane) * kTileN +
               consumer_half * 64;
  uint32_t r[64];
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" NVCC_LD_REG_OPERANDS_0_63
      "}, [%64];"
      : NVCC_LD_REG_OUTPUTS_0_63
      : "r"(src_taddr)
      : "memory");
  tcgen05_wait_ld();
  if (add_to_smem) {
#pragma unroll
    for (int i = 0; i < 64; i += 4) {
      const float4 prev = reinterpret_cast<float4*>(dst + i)[0];
      reinterpret_cast<float4*>(dst + i)[0] =
          make_float4(prev.x + __uint_as_float(r[i + 0]),
                      prev.y + __uint_as_float(r[i + 1]),
                      prev.z + __uint_as_float(r[i + 2]),
                      prev.w + __uint_as_float(r[i + 3]));
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; i += 4) {
      reinterpret_cast<float4*>(dst + i)[0] =
          make_float4(__uint_as_float(r[i + 0]), __uint_as_float(r[i + 1]),
                      __uint_as_float(r[i + 2]), __uint_as_float(r[i + 3]));
    }
  }
#else
  (void)src_taddr;
  (void)output_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)add_to_smem;
#endif
}
#endif


#define TCGEN05_LD_X64_WAIT_PACK_STORE(src_taddr, s_smem, consumer_warp, consumer_half, p_done_barrier, arrive_p_done, row_sum_pipe, trace_iter, trace_pipe) \
do {                                                                        \
  tcgen05_ld_x64_wait_pack_store_nvcc(                                      \
      (src_taddr), (s_smem), (consumer_warp), (consumer_half),              \
      (p_done_barrier), (arrive_p_done), (row_sum_pipe), clock_trace,       \
      clock_trace_iters, clock_trace_start, clock_trace_base, (trace_iter), \
      (trace_pipe));                                                       \
} while (0)

__global__ void fill_packed_bf16(uint32_t* ptr, size_t words, uint32_t seed) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < words) {
    ptr[i] = 0x3f803f80u ^ ((static_cast<uint32_t>(i) + seed * 977u) & 0x000f000fu);
  }
}

static constexpr int kRealAttentionD = 128;
static constexpr int kRealAttentionThreads = kRealAttentionD;

struct RealAttentionParams {
  const uint16_t* __restrict__ q;
  const uint16_t* __restrict__ k;
  const uint16_t* __restrict__ v;
  uint16_t* __restrict__ o;
  int B;
  int Hq;
  int Hkv;
  int Sq;
  int Skv;
  int D;
  int causal;
  float softmax_scale;
};

__host__ __device__ __forceinline__ bool real_attention_key_is_valid(
    int q_idx, int k_idx, int sq, int skv, int causal) {
  if (!causal) return true;
  // Bottom-right aligned causal masking.  When Sq == Skv this is k_idx <= q_idx.
  // When Skv > Sq, the query tile is treated as the suffix of the KV sequence.
  const int causal_limit = q_idx + (skv - sq);
  return k_idx <= causal_limit;
}

__host__ __device__ __forceinline__ int real_attention_hkv_for_hq(
    int hq, int hq_count, int hkv_count) {
  if (hkv_count <= 1) return 0;
  if (hkv_count == hq_count) return hq;
  const int group = hq_count / hkv_count;
  const int mapped = group > 0 ? hq / group : 0;
  return mapped < hkv_count ? mapped : hkv_count - 1;
}

__device__ __forceinline__ float real_bf16_to_float_device(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t real_float_to_bf16_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__device__ __forceinline__ float real_block_dot_d128(float q_lane,
                                                     const uint16_t* k_row,
                                                     int tid,
                                                     float* scratch) {
  scratch[tid] = q_lane * real_bf16_to_float_device(k_row[tid]);
  __syncthreads();
#pragma unroll
  for (int stride = kRealAttentionD / 2; stride > 0; stride >>= 1) {
    if (tid < stride) scratch[tid] += scratch[tid + stride];
    __syncthreads();
  }
  return scratch[0];
}

__global__ __launch_bounds__(kRealAttentionThreads, 1)
void real_attention_bf16_d128_kernel(RealAttentionParams p) {
  if (p.D != kRealAttentionD) return;

  const int tid = threadIdx.x;
  const int q_idx = static_cast<int>(blockIdx.x);
  const int bhq = static_cast<int>(blockIdx.y);
  if (tid >= kRealAttentionD || q_idx >= p.Sq || bhq >= p.B * p.Hq) return;

  const int b = bhq / p.Hq;
  const int hq = bhq - b * p.Hq;
  const int hkv = real_attention_hkv_for_hq(hq, p.Hq, p.Hkv);

  const size_t q_base = ((static_cast<size_t>(b) * p.Hq + hq) * p.Sq + q_idx) * p.D;
  const size_t kv_base = (static_cast<size_t>(b) * p.Hkv + hkv) * p.Skv * p.D;
  const size_t o_base = ((static_cast<size_t>(b) * p.Hq + hq) * p.Sq + q_idx) * p.D;

  const float q_lane = real_bf16_to_float_device(p.q[q_base + tid]);

  __shared__ float scratch[kRealAttentionD];
  __shared__ float row_max_s;
  __shared__ float denom_s;
  __shared__ float weight_s;

  float row_max = -3.4028234663852886e+38f;
  for (int k_idx = 0; k_idx < p.Skv; ++k_idx) {
    if (!real_attention_key_is_valid(q_idx, k_idx, p.Sq, p.Skv, p.causal)) continue;
    const uint16_t* k_row = p.k + kv_base + static_cast<size_t>(k_idx) * p.D;
    const float dot = real_block_dot_d128(q_lane, k_row, tid, scratch);
    if (tid == 0) {
      const float score = dot * p.softmax_scale;
      row_max = fmaxf(row_max, score);
    }
  }
  if (tid == 0) {
    row_max_s = row_max;
    denom_s = 0.0f;
  }
  __syncthreads();

  if (row_max_s == -3.4028234663852886e+38f) {
    p.o[o_base + tid] = real_float_to_bf16_device(0.0f);
    return;
  }

  float out_acc = 0.0f;
  for (int k_idx = 0; k_idx < p.Skv; ++k_idx) {
    if (!real_attention_key_is_valid(q_idx, k_idx, p.Sq, p.Skv, p.causal)) continue;
    const uint16_t* k_row = p.k + kv_base + static_cast<size_t>(k_idx) * p.D;
    const float dot = real_block_dot_d128(q_lane, k_row, tid, scratch);
    if (tid == 0) {
      const float score = dot * p.softmax_scale;
      weight_s = expf(score - row_max_s);
      denom_s += weight_s;
    }
    __syncthreads();
    const uint16_t* v_row = p.v + kv_base + static_cast<size_t>(k_idx) * p.D;
    out_acc += weight_s * real_bf16_to_float_device(v_row[tid]);
    __syncthreads();
  }

  const float denom = denom_s;
  const float out = denom > 0.0f ? out_acc / denom : 0.0f;
  p.o[o_base + tid] = real_float_to_bf16_device(out);
}

__global__ __launch_bounds__(kMainThreads, 1)
void qk_tma_mma_ld_kernel(const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map,
                          const __grid_constant__ CUtensorMap v_map,
                          const __grid_constant__ CUtensorMap o_map,
                          int repeats,
                          int k_tiles,
                          float score_to_exp2_scale,
                          void* __restrict__ output
#if ATTENTION_CLOCK_TRACE
                          ,
                          ClockTraceRecord* __restrict__ clock_trace,
                          int clock_trace_iters,
                          int clock_trace_start
#endif
                          ) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)v_map;
  (void)o_map;
  (void)repeats;
  (void)k_tiles;
  (void)score_to_exp2_scale;
  (void)output;
#if ATTENTION_CLOCK_TRACE
  (void)clock_trace;
  (void)clock_trace_iters;
  (void)clock_trace_start;
#endif
#else
  static constexpr int qk_layout = kFixedQkLayout;
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    k_smem[p] = q_smem + (1 + p) * kTileWords;
  }
  uint32_t* v_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    v_smem[p] = q_smem + (1 + kPipeCount + p) * kTileWords;
  }
  uint32_t* s_smem[kPipeCount];
#pragma unroll
  for (int p = 0; p < kPipeCount; ++p) {
    s_smem[p] = q_smem + (1 + kPipeCount + kVBufferCount + p) * kTileWords;
  }

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[kPipeCount];
  __shared__ uint64_t qk_done[kPipeCount];
  __shared__ uint64_t p_done[kPipeCount];
  __shared__ uint64_t v_ready[kPipeCount];
#if ATTENTION_SPLIT_S_READY
  __shared__ uint64_t s_ready[kPipeCount][2];
#else
  __shared__ uint64_t s_ready[kPipeCount];
#endif
  __shared__ uint64_t pv_done[kPipeCount];
#if ATTENTION_SPLIT_S_READY && ATTENTION_PV_H0_PINGPONG_DEP && !ATTENTION_PV_PINGPONG_DEP
  __shared__ uint64_t pv_h0_done[kPipeCount];
#endif
  __shared__ volatile int pipe0_vtma_local_shared;
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
  __shared__ int pipe0_consume_phase_count[2];
#endif
#if ATTENTION_STORE_OUTPUT
  __shared__ float row_sum_partial[kPipeCount][kTileM];
#endif
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
#if ATTENTION_CLOCK_TRACE
  __shared__ unsigned long long clock_trace_base_shared;
  __shared__ unsigned long long q_tma_start_shared;
#else
  ClockTraceRecord* clock_trace = nullptr;
  const int clock_trace_iters = 0;
  const int clock_trace_start = 0;
  const unsigned long long clock_trace_base_shared = 0ull;
#endif

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;

  if (warp_id < 4) {
    setmaxnreg_dec_producer();
  } else {
    setmaxnreg_inc_consumer();
  }

  if (threadIdx.x == 0) {
    pipe0_vtma_local_shared = -1;
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
    pipe0_consume_phase_count[0] = 0;
    pipe0_consume_phase_count[1] = 0;
#endif
    mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int p = 0; p < kPipeCount; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&qk_done[p], 1);
      mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&v_ready[p], 1);
#if ATTENTION_SPLIT_S_READY
      mbarrier_init(&s_ready[p][0], kConsumerWarpsPerPipe);
      mbarrier_init(&s_ready[p][1], kConsumerWarpsPerPipe);
#else
      mbarrier_init(&s_ready[p], kConsumerWarpsPerPipe);
#endif
      mbarrier_init(&pv_done[p], 1);
#if ATTENTION_SPLIT_S_READY && ATTENTION_PV_H0_PINGPONG_DEP && !ATTENTION_PV_PINGPONG_DEP
      mbarrier_init(&pv_h0_done[p], 1);
#endif
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#if ATTENTION_STORE_OUTPUT
  for (int i = threadIdx.x; i < kPipeCount * kTileM; i += blockDim.x) {
    reinterpret_cast<float*>(row_sum_partial)[i] = 0.0f;
  }
#endif
  __syncthreads();
#if ATTENTION_CLOCK_TRACE
  if (threadIdx.x == 0) {
    clock_trace_base_shared = clock64();
  }
  __syncthreads();
#endif
  const unsigned long long clock_trace_base = clock_trace_base_shared;
#if !ATTENTION_ACCUM_ROW_SUM
  (void)score_to_exp2_scale;
#endif

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[kPipeCount] = {tmem_base, tmem_base + 128u};
  const uint32_t o_taddr[kPipeCount] = {tmem_base + 256u, tmem_base + 384u};
  const uint32_t q_tile_row = static_cast<uint32_t>(blockIdx.x * 64);
  const int q_contig_row_group = static_cast<int>(blockIdx.x) * 16;
  const int q_contig_row = static_cast<int>(blockIdx.x) * kTileM;

  if (warp_id == 0) {
    if (qk_layout == kQkLayoutContiguousSw128) {
#if ATTENTION_CLOCK_TRACE
      if (lane0) q_tma_start_shared = clock64();
#endif
      if (lane0) mbarrier_expect_tx(&q_ready, kTileBytes);
      __syncwarp();
      if (lane < 2) {
        tma_load_2d(&q_map,
                    smem_ptr_u32(q_smem + lane * (kTileWords / 2)),
                    &q_ready, lane * 32, q_contig_row);
      }
    } else if (qk_layout == kQkLayoutContiguousK16Split4d) {
#if ATTENTION_CLOCK_TRACE
      if (lane0) q_tma_start_shared = clock64();
#endif
      if (lane0) mbarrier_expect_tx(&q_ready, kTileBytes);
      __syncwarp();
      if (lane < 8) {
        tma_load_4d(&q_map, smem_ptr_u32(q_smem + lane * 1024), &q_ready,
                    lane * 8, 0, 0, q_contig_row_group);
      }
    } else if (lane0) {
#if ATTENTION_CLOCK_TRACE
      q_tma_start_shared = clock64();
#endif
      mbarrier_expect_tx(&q_ready, kTileBytes);
      if (qk_layout == kQkLayoutContiguousSingle5d) {
        tma_load_5d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0,
                    q_contig_row_group, 0);
      } else {
        tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, q_tile_row, 0);
      }
    }
  }

  if (warp_id == 0 || (!ATTENTION_SINGLE_PIPE_MODE && warp_id == 1)) {
    const int pipe = warp_id;
    const uint32_t idesc = make_qk_idesc();
    uint64_t q_desc[8];
    uint64_t k_desc[8];
    if (lane0) {
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        if (qk_layout == kQkLayoutContiguousSw128) {
          q_desc[mma] = make_sw128_major_k_smem_desc(smem_ptr_u32(q_smem), mma);
          k_desc[mma] =
              make_sw128_major_k_smem_desc(smem_ptr_u32(k_smem[pipe]), mma);
        } else {
          q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
          k_desc[mma] = make_smem_desc(smem_ptr_u32(k_smem[pipe] + mma * 1024));
        }
      }
    }
    mbarrier_wait(&q_ready, 0);
#if ATTENTION_CLOCK_TRACE
    if (blockIdx.x == 0 && lane0 && warp_id == 0 && clock_trace != nullptr) {
      const unsigned long long q_tma_end = clock64();
      const int q_tma_slot = clock_trace_iters * kClockTraceSlotsPerIter + 11;
      write_clock_trace_record(clock_trace, q_tma_slot, kClockTraceQTma, -1, -1,
                               warp_id, -1, -1, q_tma_start_shared, q_tma_end,
                               clock_trace_base);
    }
#endif
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += kActivePipeStride, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      const int k_tile = iter % k_tiles;
#if ATTENTION_CLOCK_TRACE
      const int trace_idx = iter - clock_trace_start;
      const bool trace_iter =
          clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
          trace_idx < clock_trace_iters;
      const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
      unsigned long long k_tma_start = 0ull;
#endif
      if (local > 0) {
        mbarrier_wait(&qk_done[pipe], static_cast<uint32_t>((local - 1) & 1));
#if ATTENTION_CLOCK_TRACE
        const int done_iter = iter - kActivePipeStride;
        const int done_trace_idx = done_iter - clock_trace_start;
        if (clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
            done_trace_idx >= 0 && done_trace_idx < clock_trace_iters) {
          end_clock_trace_record(clock_trace,
                                 done_trace_idx * kClockTraceSlotsPerIter + 2,
                                 clock64(), clock_trace_base);
        }
#endif
      }
      if (qk_layout == kQkLayoutContiguousSw128) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter && lane0) k_tma_start = clock64();
#endif
        if (lane0) mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        __syncwarp();
        if (lane < 2) {
          tma_load_2d(&k_map,
                      smem_ptr_u32(k_smem[pipe] + lane * (kTileWords / 2)),
                      &k_ready[pipe], lane * 32, k_tile * kTileM);
        }
      } else if (qk_layout == kQkLayoutContiguousK16Split4d) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter && lane0) k_tma_start = clock64();
#endif
        if (lane0) mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        __syncwarp();
        if (lane < 8) {
          tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe] + lane * 1024),
                      &k_ready[pipe], lane * 8, 0, 0, k_tile * 16);
        }
      } else if (lane0) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) k_tma_start = clock64();
#endif
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        if (qk_layout == kQkLayoutContiguousSingle5d) {
          tma_load_5d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0, 0,
                      k_tile * 16, 0);
        } else {
          tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                      k_tile * 64, 0);
        }
      }
      mbarrier_wait(&k_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        const unsigned long long k_tma_end = clock64();
        write_clock_trace_record(clock_trace, trace_slot_base + 1,
                                 kClockTraceKTma, iter, pipe, warp_id, -1, -1,
                                 k_tma_start, k_tma_end, clock_trace_base);
      }
#endif
      if (local > 0) {
        mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_QK_PINGPONG_DEP && !ATTENTION_SINGLE_PIPE_MODE
      if (pipe == 0) {
        if (local > 0) {
          mbarrier_wait(&qk_done[1], static_cast<uint32_t>((local - 1) & 1));
        }
      } else {
        mbarrier_wait(&qk_done[0], phase);
      }
#endif
      if (lane0) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
          begin_clock_trace_record(clock_trace, trace_slot_base + 2,
                                   kClockTraceQkMma, iter, pipe, warp_id, -1,
                                   -1, clock64(), clock_trace_base);
        }
#endif
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc, mma != 0);
        }
        tcgen05_commit(&qk_done[pipe]);
      }
    }
#if ATTENTION_CLOCK_TRACE
    if (local > 0) {
      mbarrier_wait(&qk_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      const int done_iter = iter - kActivePipeStride;
      const int done_trace_idx = done_iter - clock_trace_start;
      if (clock_trace != nullptr && blockIdx.x == 0 && lane0 &&
          done_trace_idx >= 0 && done_trace_idx < clock_trace_iters) {
        end_clock_trace_record(clock_trace,
                               done_trace_idx * kClockTraceSlotsPerIter + 2,
                               clock64(), clock_trace_base);
      }
    }
#endif
  }

  if (warp_id == 2 || (!ATTENTION_SINGLE_PIPE_MODE && warp_id == 3)) {
    const int pipe = warp_id - 2;
    const uint32_t idesc = qk_layout == kQkLayoutContiguousSw128
                                ? (make_qk_idesc() | (1u << 16))
                                : make_qk_idesc();
    uint64_t s_desc[8];
    uint64_t v_desc[8];
    if (lane0) {
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        s_desc[mma] = make_s_smem_desc(smem_ptr_u32(s_smem[pipe] + mma * 1024));
        if (qk_layout == kQkLayoutContiguousSw128) {
          v_desc[mma] =
              make_sw128_major_mn_smem_desc(smem_ptr_u32(v_smem[pipe]), mma);
        } else {
          v_desc[mma] = make_smem_desc(smem_ptr_u32(v_smem[pipe] + mma * 1024));
        }
      }
    }
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += kActivePipeStride, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
#if ATTENTION_CLOCK_TRACE
      const int trace_idx = iter - clock_trace_start;
      const bool trace_iter =
          clock_trace != nullptr && blockIdx.x == 0 && lane0 && trace_idx >= 0 &&
          trace_idx < clock_trace_iters;
      const int trace_slot_base = trace_idx * kClockTraceSlotsPerIter;
      unsigned long long v_tma_start = 0ull;
#endif
      if (local > 0) {
        mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_V_TMA_AFTER_S_READY
#if ATTENTION_SPLIT_S_READY
      mbarrier_wait(&s_ready[pipe][0], phase);
#else
      mbarrier_wait(&s_ready[pipe], phase);
#endif
#endif
      if (qk_layout == kQkLayoutContiguousSw128) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter && lane0) v_tma_start = clock64();
#endif
        if (lane0) mbarrier_expect_tx(&v_ready[pipe], kTileBytes);
        if (lane0) {
          tma_load_4d(&v_map, smem_ptr_u32(v_smem[pipe]), &v_ready[pipe], 0, 0,
                      0, (iter % k_tiles) * 8);
        }
        if (lane0 && pipe == 0) {
          pipe0_vtma_local_shared = local;
        }
      } else if (lane0) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) v_tma_start = clock64();
#endif
        mbarrier_expect_tx(&v_ready[pipe], kTileBytes);
        tma_load_4d(&v_map, smem_ptr_u32(v_smem[pipe]), &v_ready[pipe], 0, 0,
                    (iter % k_tiles) * 64, 0);
        if (pipe == 0) {
          pipe0_vtma_local_shared = local;
        }
      }
      mbarrier_wait(&v_ready[pipe], phase);
#if ATTENTION_CLOCK_TRACE
      if (trace_iter) {
        const unsigned long long v_tma_end = clock64();
        write_clock_trace_record(clock_trace, trace_slot_base + 3,
                                 kClockTraceVTma, iter, pipe, warp_id, -1, -1,
                                 v_tma_start, v_tma_end, clock_trace_base);
      }
#endif
#if !ATTENTION_V_TMA_AFTER_S_READY
#if ATTENTION_SPLIT_S_READY
      mbarrier_wait(&s_ready[pipe][0], phase);
#else
      mbarrier_wait(&s_ready[pipe], phase);
#endif
#endif
#if ATTENTION_PV_PINGPONG_DEP && !ATTENTION_SINGLE_PIPE_MODE
      if (pipe == 0) {
        if (local > 0) {
          mbarrier_wait(&pv_done[1], static_cast<uint32_t>((local - 1) & 1));
        }
      } else {
        mbarrier_wait(&pv_done[0], phase);
      }
#elif ATTENTION_SPLIT_S_READY && ATTENTION_PV_H0_PINGPONG_DEP && !ATTENTION_SINGLE_PIPE_MODE
      if (pipe == 0) {
        if (local > 0) {
          mbarrier_wait(&pv_h0_done[1],
                        static_cast<uint32_t>((local - 1) & 1));
        }
      } else {
        mbarrier_wait(&pv_h0_done[0], phase);
      }
#endif
#if ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA && !ATTENTION_SINGLE_PIPE_MODE
      if (pipe == 1 && lane0) {
        const int target_pipe0_local = local + 1;
        if (target_pipe0_local * kPipeCount < repeats) {
          while (pipe0_vtma_local_shared < target_pipe0_local) {
            asm volatile("" ::: "memory");
          }
        }
      }
#endif
      if (lane0) {
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
          begin_clock_trace_record(clock_trace, trace_slot_base + 4,
                                   kClockTracePvMma, iter, pipe, warp_id, -1,
                                   -1, clock64(), clock_trace_base);
        }
#endif
#if ATTENTION_SPLIT_S_READY
#if ATTENTION_CLOCK_TRACE
        const unsigned long long pv_h0_start =
            trace_iter ? clock64() : 0ull;
#endif
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile / 2; ++mma) {
          tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                              local != 0 || mma != 0);
        }
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
          write_clock_trace_record(clock_trace, trace_slot_base + 5,
                                   kClockTracePvMmaH0, iter, pipe, warp_id, -1,
                                   0, pv_h0_start, clock64(),
                                   clock_trace_base);
        }
#endif
#if ATTENTION_PV_H0_PINGPONG_DEP && !ATTENTION_PV_PINGPONG_DEP
        tcgen05_commit(&pv_h0_done[pipe]);
#endif
      }
      mbarrier_wait(&s_ready[pipe][1], phase);
      if (lane0) {
#if ATTENTION_CLOCK_TRACE
        const unsigned long long pv_h1_start =
            trace_iter ? clock64() : 0ull;
#endif
#pragma unroll
        for (int mma = kMmasPerTile / 2; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                              true);
        }
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
          write_clock_trace_record(clock_trace, trace_slot_base + 6,
                                   kClockTracePvMmaH1, iter, pipe, warp_id, -1,
                                   1, pv_h1_start, clock64(),
                                   clock_trace_base);
        }
#endif
#else
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                              local != 0 || mma != 0);
        }
#endif
        tcgen05_commit(&pv_done[pipe]);
#if ATTENTION_CLOCK_TRACE
        if (trace_iter) {
          mbarrier_wait(&pv_done[pipe], phase);
          end_clock_trace_record(clock_trace, trace_slot_base + 4, clock64(),
                                 clock_trace_base);
        }
#endif
      }
    }
  }

  if (warp_id >= 4 && warp_id < kMainWarps &&
      (!ATTENTION_SINGLE_PIPE_MODE || warp_id < 4 + kConsumerWarpsPerPipe)) {
    const int pipe = (warp_id - 4) / kConsumerWarpsPerPipe;
    const int consumer_slot = (warp_id - 4) - pipe * kConsumerWarpsPerPipe;
    const int consumer_warp = consumer_slot;
#if ATTENTION_ACCUM_ROW_SUM
    const bool do_row_sum = output != nullptr;
    const int row = consumer_warp * 32 + lane;
    float row_sum_reg = 0.0f;
#endif
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += kActivePipeStride, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      mbarrier_wait(&qk_done[pipe], phase);

      if (local > 0) {
        mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME && !ATTENTION_SINGLE_PIPE_MODE
      if (pipe == 1) {
        const int expected_count = ((local >> 1) + 1) * kConsumerWarpsPerPipe;
        volatile int* consume_count = pipe0_consume_phase_count;
        while (consume_count[phase] < expected_count) {
          asm volatile("" ::: "memory");
        }
      }
#endif

      const uint32_t row_taddr = p_taddr[pipe] +
                                 (static_cast<uint32_t>(consumer_warp * 32) << 16);
#if ATTENTION_ACCUM_ROW_SUM
      const float row_sum0 =
          tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
              row_taddr, s_smem[pipe], consumer_warp, 0, &p_done[pipe], false,
              score_to_exp2_scale, clock_trace, clock_trace_iters,
              clock_trace_start, clock_trace_base, iter, pipe);
#if ATTENTION_SPLIT_S_READY
      if (lane == 0) {
        mbarrier_arrive(&s_ready[pipe][0]);
      }
#endif
      const float row_sum1 =
          tcgen05_ld_x64_wait_pack_store_sum_half_nvcc(
              row_taddr + 64u, s_smem[pipe], consumer_warp, 1, &p_done[pipe],
              true, score_to_exp2_scale, clock_trace, clock_trace_iters,
              clock_trace_start, clock_trace_base, iter, pipe);
      if (do_row_sum) row_sum_reg += row_sum0 + row_sum1;
      if (lane == 0) {
#if ATTENTION_SPLIT_S_READY
        mbarrier_arrive(&s_ready[pipe][1]);
#else
        mbarrier_arrive(&s_ready[pipe]);
#endif
      }
#else
      TCGEN05_LD_X64_WAIT_PACK_STORE(row_taddr, s_smem[pipe], consumer_warp, 0,
                                     &p_done[pipe], false, nullptr, iter, pipe);
#if ATTENTION_SPLIT_S_READY
      if (lane == 0) {
        mbarrier_arrive(&s_ready[pipe][0]);
      }
#endif
      TCGEN05_LD_X64_WAIT_PACK_STORE(row_taddr + 64u, s_smem[pipe],
                                     consumer_warp, 1, &p_done[pipe], true,
                                     nullptr, iter, pipe);
      if (lane == 0) {
#if ATTENTION_SPLIT_S_READY
        mbarrier_arrive(&s_ready[pipe][1]);
#else
        mbarrier_arrive(&s_ready[pipe]);
#endif
      }
#endif
      if (lane == 0) {
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME && !ATTENTION_SINGLE_PIPE_MODE
        if (pipe == 0) {
          atomicAdd(&pipe0_consume_phase_count[phase], 1);
        }
#endif
      }
    }
#if ATTENTION_ACCUM_ROW_SUM
    if (do_row_sum) {
      row_sum_partial[pipe][row] = row_sum_reg;
    }
#endif
  }

#if ATTENTION_STORE_OUTPUT && ATTENTION_MATERIALIZE_OUTPUT
  if (output != nullptr) {
#if ATTENTION_CLOCK_TRACE
    const bool trace_cta = clock_trace != nullptr && blockIdx.x == 0;
    const int trace_extra_base = clock_trace_iters * kClockTraceSlotsPerIter;
    const unsigned long long tail_total_start =
        trace_cta && threadIdx.x == 0 ? clock64() : 0ull;
    const unsigned long long tail_wait_start =
        trace_cta && threadIdx.x == 0 ? tail_total_start : 0ull;
#else
    const bool trace_cta = false;
    const int trace_extra_base = 0;
#endif
    const int pipe0_local_count =
        ATTENTION_SINGLE_PIPE_MODE ? repeats : (repeats + 1) / 2;
    const int pipe1_local_count = ATTENTION_SINGLE_PIPE_MODE ? 0 : repeats / 2;
    if (pipe0_local_count > 0) {
      mbarrier_wait(&pv_done[0], static_cast<uint32_t>((pipe0_local_count - 1) & 1));
    }
    if (pipe1_local_count > 0) {
      mbarrier_wait(&pv_done[1], static_cast<uint32_t>((pipe1_local_count - 1) & 1));
    }
#if ATTENTION_CLOCK_TRACE
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long tail_wait_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base, kClockTraceTailWait,
                               repeats, -1, 0, -1, -1, tail_wait_start,
                               tail_wait_end, clock_trace_base);
    }
#endif
    __syncthreads();
    float* output_smem = reinterpret_cast<float*>(q_smem);
    uint32_t* output_bf16_smem =
        reinterpret_cast<uint32_t*>(output_smem + kTileBf16Elems);
#if ATTENTION_DRAIN_OUTPUT_TMEM
    const bool trace_drain =
        trace_cta && lane0 && warp_id >= 4 && warp_id < 4 + kConsumerWarpsPerPipe;
    const unsigned long long drain_start = trace_drain ? clock64() : 0ull;
    if (pipe0_local_count > 0 && pipe1_local_count > 0 && warp_id >= 4 &&
        warp_id < 4 + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - 4;
      const uint32_t row_taddr0 =
          o_taddr[0] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      const uint32_t row_taddr1 =
          o_taddr[1] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      store_tmem_x64_accum_output_smem(row_taddr0, output_smem, consumer_warp, 0,
                                       false);
      store_tmem_x64_accum_output_smem(row_taddr0 + 64u, output_smem, consumer_warp,
                                       1, false);
      store_tmem_x64_accum_output_smem(row_taddr1, output_smem, consumer_warp, 0,
                                       true);
      store_tmem_x64_accum_output_smem(row_taddr1 + 64u, output_smem, consumer_warp,
                                       1, true);
    } else if (pipe0_local_count > 0 && pipe1_local_count == 0 && warp_id >= 4 &&
               warp_id < 4 + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - 4;
      const uint32_t row_taddr =
          o_taddr[0] + (static_cast<uint32_t>(consumer_warp * 32) << 16);
      store_tmem_x64_accum_output_smem(row_taddr, output_smem, consumer_warp, 0,
                                       false);
      store_tmem_x64_accum_output_smem(row_taddr + 64u, output_smem, consumer_warp,
                                       1, false);
    }
    if (trace_drain) {
      const unsigned long long drain_end = clock64();
      const int consumer_warp = warp_id - 4;
      write_clock_trace_record(clock_trace, trace_extra_base + 1 + consumer_warp,
                               kClockTraceTmemDrain, repeats, -1, warp_id,
                               consumer_warp, -1, drain_start, drain_end,
                               clock_trace_base);
    }
#endif
    __syncthreads();
#if ATTENTION_PACK_OUTPUT
    const bool trace_pack =
        trace_cta && lane0 && warp_id >= 4 && warp_id < 4 + kConsumerWarpsPerPipe;
    const unsigned long long pack_start = trace_pack ? clock64() : 0ull;
    if (warp_id >= 4 && warp_id < 4 + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - 4;
      const int row = consumer_warp * 32 + lane;
      const float denom = row_sum_partial[0][row] +
                          (ATTENTION_SINGLE_PIPE_MODE ? 0.0f : row_sum_partial[1][row]);
      const float inv_sum = denom != 0.0f ? 1.0f / denom : 0.0f;
      const float* row_src = output_smem + static_cast<size_t>(row) * kTileN;
      uint32_t* row_dst = output_bf16_smem + static_cast<size_t>(row) * (kTileN / 2);
#pragma unroll
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int col = col_pair * 2;
        row_dst[col_pair] =
            pack_bf16_pair_device(row_src[col] * inv_sum, row_src[col + 1] * inv_sum);
      }
    }
    if (trace_pack) {
      const unsigned long long pack_end = clock64();
      const int consumer_warp = warp_id - 4;
      write_clock_trace_record(clock_trace, trace_extra_base + 5 + consumer_warp,
                               kClockTracePackNorm, repeats, -1, warp_id,
                               consumer_warp, -1, pack_start, pack_end,
                               clock_trace_base);
    }
#endif
    __syncthreads();
#if ATTENTION_DIRECT_STORE_OUTPUT
    const unsigned long long store_start =
        trace_cta && threadIdx.x == 0 ? clock64() : 0ull;
    uint32_t* output_words = reinterpret_cast<uint32_t*>(output);
    const size_t output_tile_base = static_cast<size_t>(blockIdx.x) * kTileWords;
    for (int word = threadIdx.x; word < kTileWords; word += blockDim.x) {
      output_words[output_tile_base + word] = output_bf16_smem[word];
    }
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long store_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base + 9,
                               kClockTraceGlobalStore, repeats, -1, 0, -1, -1,
                               store_start, store_end, clock_trace_base);
    }
#endif
    __syncthreads();
#if ATTENTION_CLOCK_TRACE
    if (trace_cta && threadIdx.x == 0) {
      const unsigned long long tail_total_end = clock64();
      write_clock_trace_record(clock_trace, trace_extra_base + 10,
                               kClockTraceTailTotal, repeats, -1, 0, -1, -1,
                               tail_total_start, tail_total_end,
                               clock_trace_base);
    }
#endif
  }
#endif

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
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

void encode_atom_tma_map(CUtensorMap* map, void* base, uint64_t physical_rows) {
  constexpr uint64_t kAtomWords = 32;
  constexpr uint64_t kAtomsPerPhysicalRow = 4;
  const cuuint64_t global_dim[4] = {kAtomWords, kAtomsPerPhysicalRow, physical_rows, 1};
  const cuuint64_t global_stride[3] = {
      kAtomWords * sizeof(uint32_t),
      kAtomWords * kAtomsPerPhysicalRow * sizeof(uint32_t),
      kAtomWords * kAtomsPerPhysicalRow * physical_rows * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {static_cast<cuuint32_t>(kAtomWords),
                                 static_cast<cuuint32_t>(kAtomsPerPhysicalRow), 64, 1};
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
               "cuTensorMapEncodeTiled(atom)");
}

void encode_qk_contiguous_tma_map(CUtensorMap* map, void* base, uint64_t tiles) {
  const cuuint64_t global_dim[5] = {
      4,
      8,
      2,
      static_cast<cuuint64_t>(16) * tiles,
      8};
  const cuuint64_t global_stride[4] = {
      64ull * sizeof(uint32_t),
      4ull * sizeof(uint32_t),
      512ull * sizeof(uint32_t),
      8ull * sizeof(uint32_t)};
  const cuuint32_t box_dim[5] = {4, 8, 2, 16, 8};
  const cuuint32_t elem_stride[5] = {1, 1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                      5,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(qk_contiguous)");
}

void encode_qk_contiguous_k16_split_tma_map(CUtensorMap* map, void* base,
                                            uint64_t tiles) {
  const cuuint64_t global_dim[4] = {
      kTileN / 2,
      8,
      2,
      static_cast<cuuint64_t>(16) * tiles};
  const cuuint64_t global_stride[3] = {
      64ull * sizeof(uint32_t),
      4ull * sizeof(uint32_t),
      512ull * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {4, 8, 2, 16};
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
               "cuTensorMapEncodeTiled(qk_contiguous_k16_split)");
}

void encode_qk_contiguous_sw128_tma_map(CUtensorMap* map, void* base,
                                        uint64_t tiles) {
  const cuuint64_t global_dim[2] = {
      kTileN / 2,
      static_cast<cuuint64_t>(kTileM) * tiles};
  const cuuint64_t global_stride[1] = {
      static_cast<cuuint64_t>(kTileN / 2) * sizeof(uint32_t)};
  const cuuint32_t box_dim[2] = {kTileN / 4, kTileM};
  const cuuint32_t elem_stride[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                      2,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(qk_contiguous_sw128)");
}

void encode_contiguous_sw128_k16_tma_map(CUtensorMap* map, void* base,
                                         uint64_t tiles) {
  const cuuint64_t global_dim[4] = {
      kTileN / 4,
      16,
      2,
      static_cast<cuuint64_t>(8) * tiles};
  const cuuint64_t global_stride[3] = {
      static_cast<cuuint64_t>(kTileN / 2) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kTileN / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(16 * kTileN / 2) * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kTileN / 4, 16, 2, 8};
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
                                      CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(contiguous_sw128_k16)");
}

void encode_bf16_output_tma_map(CUtensorMap* map, void* base, uint64_t tiles) {
  const cuuint64_t global_dim[4] = {kTileN, kTileM, tiles, 1};
  const cuuint64_t global_stride[3] = {
      kTileN * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * tiles * sizeof(uint16_t)};
  const cuuint32_t box_dim[4] = {kTileN, kTileM, 1, 1};
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
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(output_bf16)");
}

double tbps_from_bytes(double bytes, double ms) {
  return ms > 0.0 ? bytes / (ms * 1.0e-3) / 1.0e12 : 0.0;
}

double tflops_from_flops(double flops, double ms) {
  return ms > 0.0 ? flops / (ms * 1.0e-3) / 1.0e12 : 0.0;
}

int selected_qk_layout(const Args& args) {
  (void)args;
  return kFixedQkLayout;
}

RunResult run_kernel(const Args& args,
                     const CUtensorMap& q_map,
                     const CUtensorMap& k_map,
                     const CUtensorMap& v_map,
                     const CUtensorMap& o_map,
                     float score_to_exp2_scale,
                     void* output,
                     int* active_ctas_per_sm
#if ATTENTION_CLOCK_TRACE
                     ,
                     ClockTraceRecord* clock_trace,
                     int clock_trace_iters
#endif
                     ) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_tma_mma_ld_kernel, kMainThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    qk_tma_mma_ld_kernel<<<args.blocks, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, args.repeats, args.k_tiles,
        score_to_exp2_scale, output
#if ATTENTION_CLOCK_TRACE
        ,
        clock_trace, clock_trace_iters, args.clock_trace_start
#endif
        );
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

  if (output != nullptr) {
    CUDA_CHECK(cudaMemset(output, 0, static_cast<size_t>(args.blocks) * kTileWords * sizeof(uint32_t)));
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    qk_tma_mma_ld_kernel<<<args.blocks, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, args.repeats, args.k_tiles,
        score_to_exp2_scale, output
#if ATTENTION_CLOCK_TRACE
        ,
        clock_trace, clock_trace_iters, args.clock_trace_start
#endif
        );
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
  result.ms /= static_cast<float>(args.iters);
  return result;
}

void write_benchmark_csv(const Args& args, int active, const RunResult& result) {
  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    std::exit(1);
  }
  std::fprintf(csv,
               "mode,q_shape,k_shape,v_shape,output_shape,tile_m,tile_n,mma_k,accumulates_per_load,blocks,repeats,k_tiles,"
               "warmup,iters,threads_per_cta,actual_ctas_per_sm,dynamic_smem_bytes,tmem_columns_allocated,"
               "tmem_columns_used,p_tmem_cols,o_tmem_cols,s_storage,elapsed_ms,total_groups,total_mmas,q_tma_GB,k_tma_GB,v_tma_GB,total_tma_GB,"
               "p_read_GB,s_store_GB,q_tma_TBps,k_tma_TBps,v_tma_TBps,total_tma_TBps,p_read_TBps,"
               "s_store_TBps,qk_TFLOP_per_s,pv_TFLOP_per_s,total_TFLOP_per_s,"
               "status,cuda_error,notes\n");

  const double groups = static_cast<double>(args.blocks) * args.repeats;
  const double total_mmas = groups * kMmasPerTile * 2.0;
  const double q_tma_bytes = static_cast<double>(args.blocks) * kTileBytes;
  const double k_tma_bytes = groups * kTileBytes;
  const double v_tma_bytes = groups * kTileBytes;
  const double p_read_bytes = groups * 2.0 * kTileBytes;
  const double s_store_bytes = groups * kTileBytes;
  const double qk_flops = groups * kMmasPerTile * kFlopsPerMma;
  const double pv_flops = qk_flops;
  const int qk_layout = selected_qk_layout(args);
  const char* mode =
      qk_layout == kQkLayoutContiguousSingle5d
          ? (args.store_output ? "contiguous_qk_5d_tma_pv_2pipe_bf16_output"
                               : "contiguous_qk_5d_tma_pv_2pipe")
      : qk_layout == kQkLayoutContiguousK16Split4d
          ? (args.store_output ? "contiguous_qk_k16_split4d_tma_pv_2pipe_bf16_output"
                               : "contiguous_qk_k16_split4d_tma_pv_2pipe")
      : qk_layout == kQkLayoutContiguousSw128
          ? (args.store_output ? "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output"
                               : "contiguous_qkv_sw128_2d_tma_pv_2pipe")
          : (args.store_output ? "qk_pack_smem_pv_2pipe_bf16_output"
                               : "qk_pack_smem_pv_2pipe");
  const char* notes =
      qk_layout == kQkLayoutContiguousSingle5d
          ? "qk_contiguous_row_major_single_5d_tma_v_internal_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_bf16_output_when_enabled"
      : qk_layout == kQkLayoutContiguousK16Split4d
          ? "qk_contiguous_row_major_k16_split_4d_tma_v_internal_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_bf16_output_when_enabled"
      : qk_layout == kQkLayoutContiguousSw128
          ? "qk_contiguous_row_major_2d_sw128_tma_major_k_v_contiguous_k16_sw128_mn_major_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_bf16_output_when_enabled"
          : "active_path_p128_c184_nvcc_ld_regs_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_bf16_output_when_enabled";
  char output_shape[64];
  if (args.store_output) {
    std::snprintf(output_shape, sizeof(output_shape), "O[%d,128,128]_bf16", args.blocks);
  } else {
    std::snprintf(output_shape, sizeof(output_shape), "none");
  }

  std::fprintf(csv,
               "%s,Q[%d,128,128]_bf16,K[%d,128,128]_bf16,V[%d,128,128]_bf16,%s,%d,%d,%d,%d,%d,%d,%d,"
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s\n",
               mode, args.blocks, args.k_tiles, args.k_tiles, output_shape, kTileM, kTileN, kMmaK,
               kMmasPerTile, args.blocks, args.repeats, args.k_tiles, args.warmup, args.iters,
               kMainThreads, active, kDynamicSmemBytes, kTmemAllocCols, kTmemUsedCols, 256, 256,
               args.store_output ? "bf16_direct" : "smem", result.ms, groups, total_mmas,
               q_tma_bytes / 1.0e9, k_tma_bytes / 1.0e9, v_tma_bytes / 1.0e9,
               (q_tma_bytes + k_tma_bytes + v_tma_bytes) / 1.0e9,
               p_read_bytes / 1.0e9, s_store_bytes / 1.0e9,
               tbps_from_bytes(q_tma_bytes, result.ms), tbps_from_bytes(k_tma_bytes, result.ms),
               tbps_from_bytes(v_tma_bytes, result.ms),
               tbps_from_bytes(q_tma_bytes + k_tma_bytes + v_tma_bytes, result.ms),
               tbps_from_bytes(p_read_bytes, result.ms), tbps_from_bytes(s_store_bytes, result.ms),
               tflops_from_flops(qk_flops, result.ms), tflops_from_flops(pv_flops, result.ms),
               tflops_from_flops(qk_flops + pv_flops, result.ms),
               result.status, cudaGetErrorString(result.error), notes);
  std::fclose(csv);
}

int clock_trace_record_count(const Args& args) {
  return args.clock_trace_iters * kClockTraceSlotsPerIter + kClockTraceExtraSlots;
}

void init_trace_record(TraceRecord* r, int iter) {
  *r = TraceRecord{};
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    r->ld_warp_start[w] = 0xffffffffffffffffull;
    r->ld_warp_end[w] = 0;
    r->pack_warp_start[w] = 0xffffffffffffffffull;
    r->pack_warp_end[w] = 0;
    r->st_warp_start[w] = 0xffffffffffffffffull;
    r->st_warp_end[w] = 0;
    r->sum_warp_start[w] = 0xffffffffffffffffull;
    r->sum_warp_end[w] = 0;
  }
  for (int i = 0; i < kClockTraceSyncCount; ++i) {
    r->sync_start[i] = 0xffffffffffffffffull;
    r->sync_end[i] = 0;
  }
  r->pack_start = 0xffffffffffffffffull;
  r->st_start = 0xffffffffffffffffull;
  r->pv_start = 0xffffffffffffffffull;
  r->mma_start = 0xffffffffffffffffull;
  r->ld_start = 0xffffffffffffffffull;
  r->iter = static_cast<unsigned int>(iter);
#if ATTENTION_SINGLE_PIPE_MODE
  r->pipe = 0;
  r->warp_id = 0;
#else
  r->pipe = static_cast<unsigned int>(iter & 1);
  r->warp_id = r->pipe;
#endif
}

const char* clock_trace_stage_name(int stage) {
  switch (stage) {
    case kClockTraceQTma:
      return "q_tma";
    case kClockTraceKTma:
      return "k_tma";
    case kClockTraceQkMma:
      return "qk_mma";
    case kClockTraceVTma:
      return "v_tma";
    case kClockTracePvMma:
      return "pv_mma";
    case kClockTracePvMmaH0:
      return "pv_mma_h0";
    case kClockTracePvMmaH1:
      return "pv_mma_h1";
    case kClockTraceLd:
      return "ld_x64";
    case kClockTracePack:
      return "pack";
    case kClockTraceRowSum:
      return "row_sum";
    case kClockTraceTailWait:
      return "tail_wait_pv_done";
    case kClockTraceTmemDrain:
      return "tmem_drain";
    case kClockTracePackNorm:
      return "pack_norm";
    case kClockTraceGlobalStore:
      return "global_store";
    case kClockTraceTailTotal:
      return "tail_total";
    case kClockTraceStore:
      return "st";
    case kClockTraceSync:
      return "sync";
    default:
      return "unknown";
  }
}

void merge_trace_range(unsigned long long* start,
                       unsigned long long* end,
                       unsigned long long s,
                       unsigned long long e) {
  if (e <= s) return;
  if (*start == 0xffffffffffffffffull || s < *start) *start = s;
  if (e > *end) *end = e;
}

unsigned long long trace_cycles(unsigned long long start, unsigned long long end) {
  return end > start && start != 0xffffffffffffffffull &&
                 end != 0xffffffffffffffffull
             ? end - start
             : 0ull;
}

unsigned long long trace_start_or_zero(unsigned long long start) {
  return start == 0xffffffffffffffffull ? 0ull : start;
}

unsigned long long trace_end_or_zero(unsigned long long start, unsigned long long end) {
  return trace_cycles(start, end) > 0 ? end : 0ull;
}

void write_clock_trace_csv(const Args& args,
                           const RunResult& result,
                           const std::vector<ClockTraceRecord>& records) {
  std::vector<TraceRecord> rows(args.clock_trace_iters);
  for (int i = 0; i < args.clock_trace_iters; ++i) {
    init_trace_record(&rows[i], args.clock_trace_start + i);
  }

  for (const ClockTraceRecord& r : records) {
    if (r.stage == 0 || r.end <= r.start) continue;
    if (r.iter < args.clock_trace_start ||
        r.iter >= args.clock_trace_start + args.clock_trace_iters) {
      continue;
    }
    TraceRecord& out = rows[r.iter - args.clock_trace_start];
    switch (r.stage) {
      case kClockTraceKTma:
        out.tma_start = r.start;
        out.tma_end = r.end;
        break;
      case kClockTraceQkMma:
        out.mma_start = r.start;
        out.mma_end = r.end;
        break;
      case kClockTraceVTma:
        out.v_tma_start = r.start;
        out.v_tma_end = r.end;
        break;
      case kClockTracePvMma:
        out.pv_start = r.start;
        out.pv_end = r.end;
        break;
      case kClockTracePvMmaH0:
        out.pv_h0_start = r.start;
        out.pv_h0_end = r.end;
        break;
      case kClockTracePvMmaH1:
        out.pv_h1_start = r.start;
        out.pv_h1_end = r.end;
        break;
      case kClockTraceSync:
        if (r.half >= 0 && r.half < kClockTraceSyncCount) {
          out.sync_start[r.half] = r.start;
          out.sync_end[r.half] = r.end;
        }
        break;
      case kClockTraceLd:
      case kClockTracePack:
      case kClockTraceStore:
      case kClockTraceRowSum: {
        const int lane = r.consumer_warp * 2 + r.half;
        if (lane < 0 || lane >= kTraceConsumerLanesPerPipe) break;
        if (r.stage == kClockTraceLd) {
          out.ld_warp_start[lane] = r.start;
          out.ld_warp_end[lane] = r.end;
          merge_trace_range(&out.ld_start, &out.ld_end, r.start, r.end);
        } else if (r.stage == kClockTracePack) {
          out.pack_warp_start[lane] = r.start;
          out.pack_warp_end[lane] = r.end;
          merge_trace_range(&out.pack_start, &out.pack_end, r.start, r.end);
        } else if (r.stage == kClockTraceStore) {
          out.st_warp_start[lane] = r.start;
          out.st_warp_end[lane] = r.end;
          merge_trace_range(&out.st_start, &out.st_end, r.start, r.end);
        } else {
          out.sum_warp_start[lane] = r.start;
          out.sum_warp_end[lane] = r.end;
        }
        break;
      }
      default:
        break;
    }
  }

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    std::exit(1);
  }
  std::fprintf(csv,
               "mode,elapsed_ms,iter,pipe,warp_id,tma_start,tma_end,tma_cycles,mma_start,mma_end,"
               "mma_cycles,ld_start,ld_end,ld_cycles,pack_start,pack_end,pack_cycles,"
               "st_start,st_end,st_cycles,v_tma_start,v_tma_end,v_tma_cycles,"
               "pv_start,pv_end,pv_cycles,pv_h0_start,pv_h0_end,pv_h0_cycles,"
               "pv_h1_start,pv_h1_end,pv_h1_cycles,total_start,total_end,total_cycles");
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",ld_warp%d_start,ld_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",pack_warp%d_start,pack_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",st_warp%d_start,st_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",sum_warp%d_start,sum_warp%d_end", w, w);
  }
  for (int i = 0; i < kClockTraceSyncCount; ++i) {
    std::fprintf(csv, ",sync%d_start,sync%d_end,sync%d_cycles", i, i, i);
  }
  std::fprintf(csv, ",status,cuda_error,notes\n");

  const int qk_layout = selected_qk_layout(args);
  const char* mode =
      qk_layout == kQkLayoutContiguousSingle5d
          ? (args.store_output ? "contiguous_qk_5d_tma_pv_2pipe_bf16_output_trace"
                               : "contiguous_qk_5d_tma_pv_2pipe_trace")
      : qk_layout == kQkLayoutContiguousK16Split4d
          ? (args.store_output ? "contiguous_qk_k16_split4d_tma_pv_2pipe_bf16_output_trace"
                               : "contiguous_qk_k16_split4d_tma_pv_2pipe_trace")
      : qk_layout == kQkLayoutContiguousSw128
          ? (args.store_output ? "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_trace"
                               : "contiguous_qkv_sw128_2d_tma_pv_2pipe_trace")
          : (args.store_output ? "qk_pack_smem_pv_2pipe_bf16_output_trace"
                               : "qk_pack_smem_pv_2pipe_trace");
  const char* notes =
      qk_layout == kQkLayoutContiguousSingle5d
          ? "qk_contiguous_row_major_single_5d_tma_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_trace_schema"
      : qk_layout == kQkLayoutContiguousK16Split4d
          ? "qk_contiguous_row_major_k16_split_4d_tma_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_trace_schema"
      : qk_layout == kQkLayoutContiguousSw128
          ? "qk_contiguous_row_major_2d_sw128_tma_v_contiguous_k16_sw128_mn_major_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE "_trace_schema"
          : "existing_trace_schema_from_qk_tma_mma_ld_pack_st_" ATTENTION_QK_DEP_NOTE "_" ATTENTION_PV_DEP_NOTE "_" ATTENTION_S_READY_NOTE;
  for (const TraceRecord& r : rows) {
    const unsigned long long ld_start = trace_start_or_zero(r.ld_start);
    const unsigned long long pack_start = trace_start_or_zero(r.pack_start);
    const unsigned long long st_start = trace_start_or_zero(r.st_start);
    const unsigned long long mma_start = trace_start_or_zero(r.mma_start);
    const unsigned long long pv_start = trace_start_or_zero(r.pv_start);
    const unsigned long long pv_h0_start = trace_start_or_zero(r.pv_h0_start);
    const unsigned long long pv_h1_start = trace_start_or_zero(r.pv_h1_start);
    const unsigned long long total_start = r.tma_start;
    const unsigned long long total_end =
        std::max(std::max(r.ld_end, r.st_end),
                 std::max(trace_end_or_zero(r.pv_start, r.pv_end),
                          trace_end_or_zero(r.mma_start, r.mma_end)));
    std::fprintf(csv,
                 "%s,%.6f,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu",
                 mode, result.ms, r.iter, r.pipe, r.warp_id, r.tma_start, r.tma_end,
                 trace_cycles(r.tma_start, r.tma_end), mma_start,
                 trace_end_or_zero(r.mma_start, r.mma_end),
                 trace_cycles(r.mma_start, r.mma_end), ld_start, r.ld_end,
                 trace_cycles(r.ld_start, r.ld_end), pack_start, r.pack_end,
                 trace_cycles(r.pack_start, r.pack_end), st_start, r.st_end,
                 trace_cycles(r.st_start, r.st_end), r.v_tma_start, r.v_tma_end,
                 trace_cycles(r.v_tma_start, r.v_tma_end), pv_start, r.pv_end,
                 trace_cycles(r.pv_start, r.pv_end), pv_h0_start, r.pv_h0_end,
                 trace_cycles(r.pv_h0_start, r.pv_h0_end), pv_h1_start,
                 r.pv_h1_end, trace_cycles(r.pv_h1_start, r.pv_h1_end),
                 total_start, total_end,
                 trace_cycles(total_start, total_end));
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.ld_warp_start[w]),
                   r.ld_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.pack_warp_start[w]),
                   r.pack_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.st_warp_start[w]),
                   r.st_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.sum_warp_start[w]),
                   r.sum_warp_end[w]);
    }
    for (int i = 0; i < kClockTraceSyncCount; ++i) {
      std::fprintf(csv, ",%llu,%llu,%llu", trace_start_or_zero(r.sync_start[i]),
                   trace_end_or_zero(r.sync_start[i], r.sync_end[i]),
                   trace_cycles(r.sync_start[i], r.sync_end[i]));
    }
    std::fprintf(csv, ",%s,%s,%s\n", result.status,
                 cudaGetErrorString(result.error), notes);
  }
  std::fclose(csv);
}

uint16_t float_to_bf16_bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

float bf16_to_float(uint16_t bits) {
  uint32_t word = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &word, sizeof(value));
  return value;
}

uint32_t pack_bf16_pair(float lo, float hi) {
  return static_cast<uint32_t>(float_to_bf16_bits(lo)) |
         (static_cast<uint32_t>(float_to_bf16_bits(hi)) << 16);
}

uint32_t pack_bf16_bits_pair(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

int logical_word_offset(int row, int col_pair) {
  return atom_major_k_word_offset(row, col_pair);
}

void pack_real_row_major_words(const std::vector<uint16_t>& src,
                               int rows,
                               std::vector<uint32_t>* words) {
  words->assign(static_cast<size_t>(rows) * (kTileN / 2), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const int col = col_pair * 2;
      (*words)[static_cast<size_t>(row) * (kTileN / 2) + col_pair] =
          pack_bf16_bits_pair(src[static_cast<size_t>(row) * kTileN + col],
                              src[static_cast<size_t>(row) * kTileN + col + 1]);
    }
  }
}

void pack_real_q_internal(const std::vector<uint16_t>& q,
                          std::vector<uint32_t>* q_words) {
  q_words->assign(kTileWords, 0);
  for (int row = 0; row < kTileM; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const int col = col_pair * 2;
      (*q_words)[logical_word_offset(row, col_pair)] =
          pack_bf16_bits_pair(q[row * kTileN + col], q[row * kTileN + col + 1]);
    }
  }
}

void pack_real_k_internal(const std::vector<uint16_t>& k,
                          int k_tiles,
                          std::vector<uint32_t>* k_words) {
  k_words->assign(static_cast<size_t>(k_tiles) * kTileWords, 0);
  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      const int k_idx = tile * kTileM + row;
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int col = col_pair * 2;
        (*k_words)[static_cast<size_t>(tile) * kTileWords +
                   logical_word_offset(row, col_pair)] =
            pack_bf16_bits_pair(k[k_idx * kTileN + col],
                                k[k_idx * kTileN + col + 1]);
      }
    }
  }
}

void pack_real_v_internal(const std::vector<uint16_t>& v,
                          int k_tiles,
                          std::vector<uint32_t>* v_words) {
  v_words->assign(static_cast<size_t>(k_tiles) * kTileWords, 0);
  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int d = 0; d < kTileN; ++d) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int n = col_pair * 2;
        const int k0 = tile * kTileN + n;
        const int k1 = k0 + 1;
        (*v_words)[static_cast<size_t>(tile) * kTileWords +
                   logical_word_offset(d, col_pair)] =
            pack_bf16_bits_pair(v[k0 * kTileN + d], v[k1 * kTileN + d]);
      }
    }
  }
}

float pattern_value(const std::string& pattern, char matrix, int tile, int row, int col) {
  if (pattern == "constant") {
    if (matrix == 'v') return 0.125f;
    return 0.0625f;
  }
  if (pattern == "onehot") {
    if (matrix == 'q') return row == col ? 1.0f : 0.0f;
    if (matrix == 'k') return row == col ? 1.0f : 0.0f;
    return (row == ((col + tile) & 127)) ? 0.5f : 0.0f;
  }
  const float row_scale = static_cast<float>((row % 17) + 1) * 0.00390625f;
  const float col_scale = static_cast<float>((col % 19) + 1) * 0.00390625f;
  const float tile_scale = static_cast<float>(tile + 1) * 0.0009765625f;
  if (matrix == 'v') return row_scale + col_scale + tile_scale;
  return row_scale + col_scale;
}

uint32_t mix_hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

float centered_hash_value(uint32_t x) {
  const uint32_t h = mix_hash_u32(x);
  const float u = static_cast<float>(h & 0xffffu) * (1.0f / 65535.0f);
  return u * 2.0f - 1.0f;
}

float real_pattern_value(const std::string& pattern,
                         char matrix,
                         int b,
                         int h,
                         int s,
                         int d) {
  if (pattern == "constant") {
    return matrix == 'v' ? 0.125f : 0.0625f;
  }
  if (pattern == "onehot") {
    if (matrix == 'q') return d == (s & 127) ? 1.0f : 0.0f;
    if (matrix == 'k') return d == (s & 127) ? 1.0f : 0.0f;
    return d == ((s + h) & 127) ? 0.5f : 0.0f;
  }
  if (pattern == "rank1") {
    const float s_scale = static_cast<float>((s % 17) + 1) * 0.0078125f;
    const float d_scale = static_cast<float>((d % 19) + 1) * 0.00390625f;
    const float h_scale = static_cast<float>((h % 7) + 1) * 0.0009765625f;
    if (matrix == 'v') return 0.5f * s_scale + d_scale + h_scale;
    return s_scale + d_scale + h_scale;
  }
  const uint32_t tag = matrix == 'q' ? 0x1234u : (matrix == 'k' ? 0x5678u : 0x9abcu);
  const uint32_t x = tag ^ static_cast<uint32_t>(b * 131071 + h * 8191 + s * 257 + d);
  const float amp = matrix == 'v' ? 0.5f : 0.25f;
  return amp * centered_hash_value(x);
}

void fill_real_bf16_matrix(std::vector<uint16_t>& dst,
                           const std::string& pattern,
                           char matrix,
                           int B,
                           int H,
                           int S,
                           int D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s_idx = 0; s_idx < S; ++s_idx) {
        for (int d = 0; d < D; ++d) {
          const size_t idx = ((static_cast<size_t>(b) * H + h) * S + s_idx) * D + d;
          dst[idx] = float_to_bf16_bits(real_pattern_value(pattern, matrix, b, h, s_idx, d));
        }
      }
    }
  }
}

std::vector<float> unpack_bf16_vector_to_float(const std::vector<uint16_t>& src) {
  std::vector<float> dst(src.size(), 0.0f);
  for (size_t i = 0; i < src.size(); ++i) dst[i] = bf16_to_float(src[i]);
  return dst;
}

void build_real_attention_reference(const std::vector<uint16_t>& q,
                                    const std::vector<uint16_t>& k,
                                    const std::vector<uint16_t>& v,
                                    int B,
                                    int Hq,
                                    int Hkv,
                                    int Sq,
                                    int Skv,
                                    int D,
                                    float softmax_scale,
                                    bool causal,
                                    std::vector<float>* ref) {
  ref->assign(static_cast<size_t>(B) * Hq * Sq * D, 0.0f);
  std::vector<float> scores(static_cast<size_t>(Skv), 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < Hq; ++hq) {
      const int hkv = real_attention_hkv_for_hq(hq, Hq, Hkv);
      for (int q_idx = 0; q_idx < Sq; ++q_idx) {
        float row_max = -std::numeric_limits<float>::infinity();
        const size_t q_base = ((static_cast<size_t>(b) * Hq + hq) * Sq + q_idx) * D;
        const size_t kv_base = (static_cast<size_t>(b) * Hkv + hkv) * Skv * D;
        for (int k_idx = 0; k_idx < Skv; ++k_idx) {
          if (!real_attention_key_is_valid(q_idx, k_idx, Sq, Skv, causal ? 1 : 0)) {
            scores[k_idx] = -std::numeric_limits<float>::infinity();
            continue;
          }
          float dot = 0.0f;
          const size_t k_base = kv_base + static_cast<size_t>(k_idx) * D;
          for (int d = 0; d < D; ++d) {
            dot += bf16_to_float(q[q_base + d]) * bf16_to_float(k[k_base + d]);
          }
          const float score = dot * softmax_scale;
          scores[k_idx] = score;
          row_max = std::max(row_max, score);
        }
        if (!std::isfinite(row_max)) continue;
        float denom = 0.0f;
        for (int k_idx = 0; k_idx < Skv; ++k_idx) {
          if (!std::isfinite(scores[k_idx])) continue;
          denom += std::exp(scores[k_idx] - row_max);
        }
        const size_t o_base = ((static_cast<size_t>(b) * Hq + hq) * Sq + q_idx) * D;
        for (int d = 0; d < D; ++d) {
          float acc = 0.0f;
          for (int k_idx = 0; k_idx < Skv; ++k_idx) {
            if (!std::isfinite(scores[k_idx])) continue;
            const float w = std::exp(scores[k_idx] - row_max);
            const size_t v_idx = kv_base + static_cast<size_t>(k_idx) * D + d;
            acc += w * bf16_to_float(v[v_idx]);
          }
          (*ref)[o_base + d] = denom > 0.0f ? acc / denom : 0.0f;
        }
      }
    }
  }
}

void fill_logical_matrix(std::vector<uint32_t>& words,
                         const std::string& pattern,
                         char matrix,
                         int tiles) {
  std::fill(words.begin(), words.end(), 0);
  for (int tile = 0; tile < tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int col = col_pair * 2;
        const float lo = pattern_value(pattern, matrix, tile, row, col);
        const float hi = pattern_value(pattern, matrix, tile, row, col + 1);
        words[static_cast<size_t>(tile) * kTileWords + logical_word_offset(row, col_pair)] =
            pack_bf16_pair(lo, hi);
      }
    }
  }
}

void unpack_logical_matrix(const std::vector<uint32_t>& words,
                           int tiles,
                           std::vector<float>* values) {
  values->assign(static_cast<size_t>(tiles) * kTileBf16Elems, 0.0f);
  for (int tile = 0; tile < tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const uint32_t packed =
            words[static_cast<size_t>(tile) * kTileWords + logical_word_offset(row, col_pair)];
        const size_t elem = static_cast<size_t>(tile) * kTileBf16Elems + row * kTileN + col_pair * 2;
        (*values)[elem] = bf16_to_float(static_cast<uint16_t>(packed & 0xffffu));
        (*values)[elem + 1] = bf16_to_float(static_cast<uint16_t>(packed >> 16));
      }
    }
  }
}

void build_reference(const std::vector<uint32_t>& h_q,
                     const std::vector<uint32_t>& h_k,
                     const std::vector<uint32_t>& h_v,
                     int k_tiles,
                     std::vector<float>* norm_ref) {
  std::vector<float> q, k, v;
  unpack_logical_matrix(h_q, 1, &q);
  unpack_logical_matrix(h_k, k_tiles, &k);
  unpack_logical_matrix(h_v, k_tiles, &v);
  std::vector<float> p(static_cast<size_t>(k_tiles) * kTileBf16Elems, 0.0f);
  std::vector<float> row_max(kTileM, -std::numeric_limits<float>::infinity());
  std::vector<float> row_sum(kTileM, 0.0f);
  std::vector<float> o(kTileBf16Elems, 0.0f);
  norm_ref->assign(kTileBf16Elems, 0.0f);

  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int m = 0; m < kTileM; ++m) {
      for (int n = 0; n < kTileN; ++n) {
        float acc = 0.0f;
        for (int d = 0; d < kTileN; ++d) {
          acc += q[m * kTileN + d] *
                 k[static_cast<size_t>(tile) * kTileBf16Elems + n * kTileN + d];
        }
        const size_t idx = static_cast<size_t>(tile) * kTileBf16Elems + m * kTileN + n;
        p[idx] = acc;
        row_max[m] = std::max(row_max[m], acc);
      }
    }
  }

  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int m = 0; m < kTileM; ++m) {
      for (int n = 0; n < kTileN; ++n) {
        const size_t idx = static_cast<size_t>(tile) * kTileBf16Elems + m * kTileN + n;
        const float e = std::exp2(p[idx] - row_max[m]);
        row_sum[m] += e;
        for (int d = 0; d < kTileN; ++d) {
          o[m * kTileN + d] +=
              e * v[static_cast<size_t>(tile) * kTileBf16Elems + d * kTileN + n];
        }
      }
    }
  }

  for (int m = 0; m < kTileM; ++m) {
    for (int d = 0; d < kTileN; ++d) {
      (*norm_ref)[m * kTileN + d] = o[m * kTileN + d] / row_sum[m];
    }
  }
}

std::vector<uint32_t> pack_row_major_bf16_words(const std::vector<float>& values) {
  std::vector<uint32_t> words(kTileWords, 0);
  for (int row = 0; row < kTileM; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const int col = col_pair * 2;
      words[row * (kTileN / 2) + col_pair] =
          pack_bf16_pair(values[row * kTileN + col], values[row * kTileN + col + 1]);
    }
  }
  return words;
}

std::vector<float> unpack_row_major_bf16_words(const std::vector<uint32_t>& words) {
  std::vector<float> values(kTileBf16Elems, 0.0f);
  for (int row = 0; row < kTileM; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const uint32_t packed = words[row * (kTileN / 2) + col_pair];
      values[row * kTileN + col_pair * 2] = bf16_to_float(static_cast<uint16_t>(packed));
      values[row * kTileN + col_pair * 2 + 1] = bf16_to_float(static_cast<uint16_t>(packed >> 16));
    }
  }
  return values;
}

struct CompareResult {
  std::string stage;
  bool ok;
  float max_abs;
  float max_rel;
  size_t bad_count;
};

CompareResult compare_float(const char* stage,
                            const std::vector<float>& got,
                            const std::vector<float>& expected,
                            float abs_tol,
                            float rel_tol) {
  CompareResult r{std::string(stage), true, 0.0f, 0.0f, 0};
  for (size_t i = 0; i < got.size(); ++i) {
    const float diff = std::fabs(got[i] - expected[i]);
    const float rel = diff / std::max(std::fabs(expected[i]), 1.0e-6f);
    r.max_abs = std::max(r.max_abs, diff);
    r.max_rel = std::max(r.max_rel, rel);
    if (diff > abs_tol && rel > rel_tol) {
      r.ok = false;
      ++r.bad_count;
    }
  }
  return r;
}

CompareResult compare_bf16_words(const char* stage,
                                 const std::vector<uint32_t>& got,
                                 const std::vector<uint32_t>& expected) {
  CompareResult r{std::string(stage), true, 0.0f, 0.0f, 0};
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      r.ok = false;
      ++r.bad_count;
    }
  }
  return r;
}

void write_validation_csv(const char* path, const std::vector<CompareResult>& results) {
  FILE* csv = std::fopen(path, "w");
  if (!csv) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(csv, "stage,status,max_abs,max_rel,bad_count\n");
  for (const CompareResult& r : results) {
    std::fprintf(csv, "%s,%s,%g,%g,%zu\n", r.stage.c_str(), r.ok ? "ok" : "fail",
                 r.max_abs, r.max_rel, r.bad_count);
  }
  std::fclose(csv);
}

void parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      args->blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
      args->repeats = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--k-tiles") && i + 1 < argc) {
      args->k_tiles = std::atoi(argv[++i]);
      args->k_tiles_set = true;
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--store-output")) {
      args->store_output = true;
    } else if (!std::strcmp(argv[i], "--contiguous-qk")) {
      args->contiguous_qk = true;
      args->contiguous_qk_single_5d = false;
      args->contiguous_qk_sw128 = false;
    } else if (!std::strcmp(argv[i], "--contiguous-qk-single-5d")) {
      args->contiguous_qk = true;
      args->contiguous_qk_single_5d = true;
      args->contiguous_qk_sw128 = false;
    } else if (!std::strcmp(argv[i], "--contiguous-qk-sw128")) {
      args->contiguous_qk = true;
      args->contiguous_qk_single_5d = false;
      args->contiguous_qk_sw128 = true;
    } else if (!std::strcmp(argv[i], "--clock-trace")) {
      args->clock_trace = true;
    } else if (!std::strcmp(argv[i], "--clock-trace-start") && i + 1 < argc) {
      args->clock_trace_start = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--clock-trace-iters") && i + 1 < argc) {
      args->clock_trace_iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--validate") ||
               !std::strcmp(argv[i], "--real-validate") ||
               !std::strcmp(argv[i], "--fused-validate")) {
      args->stage = "fused_validate";
      args->store_output = true;
      args->contiguous_qk = true;
      args->contiguous_qk_sw128 = true;
    } else if (!std::strcmp(argv[i], "--scalar-validate")) {
      args->stage = "scalar_validate";
    } else if (!std::strcmp(argv[i], "--scalar-validate-suite")) {
      args->stage = "scalar_validate";
      args->validation_suite = true;
    } else if (!std::strcmp(argv[i], "--toy-validate")) {
      args->stage = "toy_validate";
      args->store_output = true;
    } else if (!std::strcmp(argv[i], "--validate-suite")) {
      args->stage = "fused_validate";
      args->validation_suite = true;
      args->store_output = true;
      args->contiguous_qk = true;
      args->contiguous_qk_sw128 = true;
    } else if (!std::strcmp(argv[i], "--pattern") && i + 1 < argc) {
      args->pattern = argv[++i];
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if ((!std::strcmp(argv[i], "--b") || !std::strcmp(argv[i], "--B")) && i + 1 < argc) {
      args->B = std::atoi(argv[++i]);
    } else if ((!std::strcmp(argv[i], "--h") || !std::strcmp(argv[i], "--H")) && i + 1 < argc) {
      args->Hq = std::atoi(argv[++i]);
      args->Hkv = args->Hq;
    } else if (!std::strcmp(argv[i], "--hq") && i + 1 < argc) {
      args->Hq = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--hkv") && i + 1 < argc) {
      args->Hkv = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--sq") && i + 1 < argc) {
      args->Sq = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--skv") && i + 1 < argc) {
      args->Skv = std::atoi(argv[++i]);
      args->skv_set = true;
    } else if ((!std::strcmp(argv[i], "--d") || !std::strcmp(argv[i], "--D")) && i + 1 < argc) {
      args->D = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--causal")) {
      args->causal = true;
    } else if (!std::strcmp(argv[i], "--no-causal")) {
      args->causal = false;
    } else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) {
      args->softmax_scale = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [benchmark args] [validation args]\n"
          "\n"
          "Benchmark/tcgen05 toy path:\n"
          "  --blocks N --repeats N --k-tiles N --warmup N --iters N --store-output\n"
          "  --contiguous-qk                read Q/K from row-major contiguous BF16-pair words using k16-split 4D TMA; V remains internal\n"
          "  --contiguous-qk-single-5d      compare against the older single 5D contiguous Q/K TMA path\n"
          "  --contiguous-qk-sw128          read Q/K from row-major contiguous BF16-pair words using 2D SW128 TMA\n"
          "  --clock-trace --clock-trace-start N --clock-trace-iters N   write output-path clock trace CSV; requires -DATTENTION_CLOCK_TRACE=1\n"
          "  --toy-validate                 validate the original internal-layout toy path; requires -DATTENTION_STORE_OUTPUT=1\n"
          "\n"
          "Fused real-attention validation path:\n"
          "  --validate                     validate qk_tma_mma_ld_kernel for B=H=1,Sq=128,D=128\n"
          "  --validate-suite               run fused validation cases for k_tiles 1/4/8\n"
          "  --scalar-validate              run the separate scalar correctness kernel\n"
          "  --b N --h N | --hq N --hkv N   batch/head counts; Hq must be divisible by Hkv\n"
          "  --sq N --skv N --d 128         sequence sizes and head dimension\n"
          "  --causal                       bottom-right aligned causal mask\n"
          "  --scale F                      softmax scale; default is 1/sqrt(D)\n"
          "  --pattern constant|rank1|onehot|random\n"
          "  --csv PATH\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (args->blocks < 1) args->blocks = 1;
  if (args->repeats < 1) args->repeats = 1;
  if (args->k_tiles < 1) args->k_tiles = 1;
  if (args->B < 1) args->B = 1;
  if (args->Hq < 1) args->Hq = 1;
  if (args->Hkv < 1) args->Hkv = 1;
  if (args->Sq < 1) args->Sq = 1;
  if (args->Skv < 1) args->Skv = 1;
  if (args->D < 1) args->D = 1;
  if (args->clock_trace_start < 0) args->clock_trace_start = 0;
  if (args->clock_trace_iters < 1) args->clock_trace_iters = 1;
  if (args->clock_trace_start >= args->k_tiles) args->clock_trace_start = args->k_tiles - 1;
  if (args->clock_trace_start + args->clock_trace_iters > args->k_tiles) {
    args->clock_trace_iters = args->k_tiles - args->clock_trace_start;
  }
}

bool validate_real_attention_args(const Args& args) {
  if (args.D != kRealAttentionD) {
    std::fprintf(stderr, "real attention path currently requires D=128; got D=%d\n", args.D);
    return false;
  }
  if (args.Hq < 1 || args.Hkv < 1 || args.Hq % args.Hkv != 0) {
    std::fprintf(stderr, "real attention path requires Hq %% Hkv == 0; got Hq=%d Hkv=%d\n",
                 args.Hq, args.Hkv);
    return false;
  }
  if (args.B < 1 || args.Sq < 1 || args.Skv < 1) {
    std::fprintf(stderr, "invalid shape: B=%d Sq=%d Skv=%d\n", args.B, args.Sq, args.Skv);
    return false;
  }
  return true;
}

float resolved_softmax_scale(const Args& args) {
  return args.softmax_scale >= 0.0f ? args.softmax_scale
                                    : 1.0f / std::sqrt(static_cast<float>(args.D));
}

float resolved_score_to_exp2_scale(const Args& args) {
  return resolved_softmax_scale(args) * kLog2E;
}

CompareResult run_real_attention_case(const Args& args, const std::string& label) {
  const float scale = resolved_softmax_scale(args);
  const size_t q_elems = static_cast<size_t>(args.B) * args.Hq * args.Sq * args.D;
  const size_t kv_elems = static_cast<size_t>(args.B) * args.Hkv * args.Skv * args.D;
  const size_t o_elems = q_elems;

  std::vector<uint16_t> h_q(q_elems);
  std::vector<uint16_t> h_k(kv_elems);
  std::vector<uint16_t> h_v(kv_elems);
  fill_real_bf16_matrix(h_q, args.pattern, 'q', args.B, args.Hq, args.Sq, args.D);
  fill_real_bf16_matrix(h_k, args.pattern, 'k', args.B, args.Hkv, args.Skv, args.D);
  fill_real_bf16_matrix(h_v, args.pattern, 'v', args.B, args.Hkv, args.Skv, args.D);

  std::vector<float> ref;
  build_real_attention_reference(h_q, h_k, h_v, args.B, args.Hq, args.Hkv, args.Sq,
                                 args.Skv, args.D, scale, args.causal, &ref);

  uint16_t* d_q = nullptr;
  uint16_t* d_k = nullptr;
  uint16_t* d_v = nullptr;
  uint16_t* d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_o, o_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_o, 0, o_elems * sizeof(uint16_t)));

  RealAttentionParams params{};
  params.q = d_q;
  params.k = d_k;
  params.v = d_v;
  params.o = d_o;
  params.B = args.B;
  params.Hq = args.Hq;
  params.Hkv = args.Hkv;
  params.Sq = args.Sq;
  params.Skv = args.Skv;
  params.D = args.D;
  params.causal = args.causal ? 1 : 0;
  params.softmax_scale = scale;

  const dim3 grid(args.Sq, args.B * args.Hq, 1);
  real_attention_bf16_d128_kernel<<<grid, kRealAttentionThreads>>>(params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint16_t> h_o(o_elems);
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  std::vector<float> got = unpack_bf16_vector_to_float(h_o);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  char stage[256];
  std::snprintf(stage, sizeof(stage),
                "%s_B%d_Hq%d_Hkv%d_Sq%d_Skv%d_D%d_%s_scale%.6g_pattern_%s",
                label.c_str(), args.B, args.Hq, args.Hkv, args.Sq, args.Skv, args.D,
                args.causal ? "causal" : "noncausal", scale, args.pattern.c_str());

  // BF16 output plus different CPU/GPU reduction orders makes bit-exact comparison
  // inappropriate.  These tolerances catch real algorithmic errors while allowing
  // expected BF16 quantization and exp/dot ordering differences.
  return compare_float(stage, got, ref, 6.0e-2f, 8.0e-2f);
}

bool prepare_fused_real_attention_args(const Args& args, Args* out) {
  *out = args;
  if (out->Sq != kTileM || out->D != kRealAttentionD) {
    std::fprintf(stderr, "fused validation V1 requires Sq=128,D=128; got Sq=%d D=%d\n",
                 out->Sq, out->D);
    return false;
  }
  if (!out->k_tiles_set) {
    if (out->skv_set) {
      if (out->Skv % kTileN != 0) {
        std::fprintf(stderr, "fused validation requires Skv %% 128 == 0; got Skv=%d\n",
                     out->Skv);
        return false;
      }
      out->k_tiles = out->Skv / kTileN;
    } else {
      out->k_tiles = 1;
      out->Skv = kTileN;
    }
  } else {
    if (out->skv_set && out->Skv != out->k_tiles * kTileN) {
      std::fprintf(stderr,
                   "fused validation requires Skv == 128 * k_tiles; got Skv=%d k_tiles=%d\n",
                   out->Skv, out->k_tiles);
      return false;
    }
    out->Skv = out->k_tiles * kTileN;
  }
  if (out->B != 1 || out->Hq != 1 || out->Hkv != 1) {
    std::fprintf(stderr,
                 "fused validation V1 requires B=1,Hq=1,Hkv=1; got B=%d Hq=%d Hkv=%d\n",
                 out->B, out->Hq, out->Hkv);
    return false;
  }
  if (out->causal) {
    std::fprintf(stderr, "fused validation V1 is non-causal only\n");
    return false;
  }
  if (out->k_tiles < 1) {
    std::fprintf(stderr, "fused validation requires k_tiles >= 1\n");
    return false;
  }
  return true;
}

CompareResult run_fused_real_attention_case(const Args& args, const std::string& label) {
#if !ATTENTION_STORE_OUTPUT
  (void)args;
  return CompareResult{label, false, 0.0f, 0.0f, 1};
#else
  const float scale = resolved_softmax_scale(args);
  const size_t q_elems = kTileBf16Elems;
  const size_t kv_elems = static_cast<size_t>(args.k_tiles) * kTileBf16Elems;

  std::vector<uint16_t> h_q(q_elems);
  std::vector<uint16_t> h_k(kv_elems);
  std::vector<uint16_t> h_v(kv_elems);
  fill_real_bf16_matrix(h_q, args.pattern, 'q', 1, 1, kTileM, kRealAttentionD);
  fill_real_bf16_matrix(h_k, args.pattern, 'k', 1, 1, args.Skv, kRealAttentionD);
  fill_real_bf16_matrix(h_v, args.pattern, 'v', 1, 1, args.Skv, kRealAttentionD);

  std::vector<float> ref;
  build_real_attention_reference(h_q, h_k, h_v, 1, 1, 1, kTileM, args.Skv,
                                 kRealAttentionD, scale, false, &ref);

  std::vector<uint32_t> h_q_contiguous;
  std::vector<uint32_t> h_k_contiguous;
  std::vector<uint32_t> h_v_words;
  const int qk_layout = selected_qk_layout(args);
  pack_real_row_major_words(h_q, kTileM, &h_q_contiguous);
  pack_real_row_major_words(h_k, args.Skv, &h_k_contiguous);
  if (qk_layout == kQkLayoutContiguousSw128) {
    pack_real_row_major_words(h_v, args.Skv, &h_v_words);
  } else {
    pack_real_v_internal(h_v, args.k_tiles, &h_v_words);
  }

  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q_contiguous.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k_contiguous.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v_words.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_o, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q_contiguous.data(), h_q_contiguous.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k_contiguous.data(), h_k_contiguous.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v_words.data(), h_v_words.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_o, 0, kTileWords * sizeof(uint32_t)));

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  if (qk_layout == kQkLayoutContiguousSingle5d) {
    encode_qk_contiguous_tma_map(&q_map, d_q, 1);
    encode_qk_contiguous_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles));
  } else if (qk_layout == kQkLayoutContiguousSw128) {
    encode_qk_contiguous_sw128_tma_map(&q_map, d_q, 1);
    encode_qk_contiguous_sw128_tma_map(&k_map, d_k,
                                       static_cast<uint64_t>(args.k_tiles));
  } else {
    encode_qk_contiguous_k16_split_tma_map(&q_map, d_q, 1);
    encode_qk_contiguous_k16_split_tma_map(&k_map, d_k,
                                           static_cast<uint64_t>(args.k_tiles));
  }
  if (qk_layout == kQkLayoutContiguousSw128) {
    encode_contiguous_sw128_k16_tma_map(&v_map, d_v,
                                        static_cast<uint64_t>(args.k_tiles));
  } else {
    encode_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  }
  encode_bf16_output_tma_map(&o_map, d_o, 1);

  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  qk_tma_mma_ld_kernel<<<1, kMainThreads, kDynamicSmemBytes>>>(
      q_map, k_map, v_map, o_map, args.k_tiles, args.k_tiles,
      resolved_score_to_exp2_scale(args), d_o
#if ATTENTION_CLOCK_TRACE
      ,
      nullptr, 0, 0
#endif
      );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> h_o(kTileWords);
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, kTileWords * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  const std::vector<float> got = unpack_row_major_bf16_words(h_o);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  char stage[256];
  std::snprintf(stage, sizeof(stage),
                "%s_fused_B1_H1_Sq128_Skv%d_D128_noncausal_scale%.6g_pattern_%s",
                label.c_str(), args.Skv, scale, args.pattern.c_str());
  return compare_float(stage, got, ref, 6.0e-2f, 8.0e-2f);
#endif
}

int run_fused_real_validation(const Args& args) {
#if !ATTENTION_STORE_OUTPUT
  std::fprintf(stderr,
               "--validate requires compiling with -DATTENTION_STORE_OUTPUT=1 for the fused kernel path.\n");
  return 2;
#else
  std::vector<CompareResult> results;
  if (args.validation_suite) {
    struct CaseSpec {
      int k_tiles;
      const char* pattern;
      const char* label;
    };
    const CaseSpec cases[] = {
        {1, "constant", "fused_real_attention_constant_k1"},
        {4, "rank1", "fused_real_attention_rank1_k4"},
        {4, "random", "fused_real_attention_random_k4"},
        {8, "random", "fused_real_attention_random_k8"},
    };
    for (const CaseSpec& c : cases) {
      Args case_args = args;
      case_args.B = 1;
      case_args.Hq = 1;
      case_args.Hkv = 1;
      case_args.Sq = kTileM;
      case_args.k_tiles = c.k_tiles;
      case_args.k_tiles_set = true;
      case_args.Skv = c.k_tiles * kTileN;
      case_args.skv_set = false;
      case_args.D = kRealAttentionD;
      case_args.causal = false;
      case_args.pattern = c.pattern;
      Args prepared;
      if (!prepare_fused_real_attention_args(case_args, &prepared)) return 2;
      results.push_back(run_fused_real_attention_case(prepared, c.label));
    }
  } else {
    Args prepared;
    if (!prepare_fused_real_attention_args(args, &prepared)) return 2;
    results.push_back(run_fused_real_attention_case(prepared, "real_attention"));
  }

  write_validation_csv(args.csv, results);
  for (const CompareResult& r : results) {
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu\n", r.stage.c_str(),
                r.ok ? "ok" : "fail", r.max_abs, r.max_rel, r.bad_count);
  }
  const bool ok = std::all_of(results.begin(), results.end(),
                              [](const CompareResult& r) { return r.ok; });
  return ok ? 0 : 1;
#endif
}

int run_scalar_real_validation(const Args& args) {
  std::vector<CompareResult> results;
  if (args.validation_suite) {
    struct CaseSpec {
      int B, Hq, Hkv, Sq, Skv;
      bool causal;
      const char* pattern;
      const char* label;
    };
    const CaseSpec cases[] = {
        {1, 1, 1, 128, 128, false, "constant", "real_attention_constant_square"},
        {1, 1, 1, 128, 256, false, "rank1", "real_attention_rank1_multi_k"},
        {1, 1, 1, 128, 384, true, "rank1", "real_attention_causal_k_longer"},
        {1, 1, 1, 77, 193, false, "random", "real_attention_random_tail"},
        {2, 4, 4, 128, 256, false, "random", "real_attention_b2_h4"},
        {1, 8, 2, 64, 128, false, "random", "real_attention_gqa"},
    };
    for (const CaseSpec& c : cases) {
      Args case_args = args;
      case_args.B = c.B;
      case_args.Hq = c.Hq;
      case_args.Hkv = c.Hkv;
      case_args.Sq = c.Sq;
      case_args.Skv = c.Skv;
      case_args.D = kRealAttentionD;
      case_args.causal = c.causal;
      case_args.pattern = c.pattern;
      if (!validate_real_attention_args(case_args)) return 2;
      results.push_back(run_real_attention_case(case_args, c.label));
    }
  } else {
    if (!validate_real_attention_args(args)) return 2;
    results.push_back(run_real_attention_case(args, "real_attention"));
  }

  write_validation_csv(args.csv, results);
  for (const CompareResult& r : results) {
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu\n", r.stage.c_str(),
                r.ok ? "ok" : "fail", r.max_abs, r.max_rel, r.bad_count);
  }
  const bool ok = std::all_of(results.begin(), results.end(),
                              [](const CompareResult& r) { return r.ok; });
  return ok ? 0 : 1;
}

int run_validation(const Args& args) {
#if !ATTENTION_STORE_OUTPUT
  std::fprintf(stderr,
               "--toy-validate requires compiling with -DATTENTION_STORE_OUTPUT=1. "
               "Use --validate for the real attention validation path.\n");
  return 2;
#endif
  std::vector<uint32_t> h_q(kTileWords);
  std::vector<uint32_t> h_k(static_cast<size_t>(args.k_tiles) * kTileWords);
  std::vector<uint32_t> h_v(static_cast<size_t>(args.k_tiles) * kTileWords);
  fill_logical_matrix(h_q, args.pattern, 'q', 1);
  fill_logical_matrix(h_k, args.pattern, 'k', args.k_tiles);
  fill_logical_matrix(h_v, args.pattern, 'v', args.k_tiles);

  std::vector<float> norm_ref;
  build_reference(h_q, h_k, h_v, args.k_tiles, &norm_ref);
  const std::vector<uint32_t> expected_bf16 = pack_row_major_bf16_words(norm_ref);

  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_o, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), kTileWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_o, 0, kTileWords * sizeof(uint32_t)));

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  encode_atom_tma_map(&q_map, d_q, 64);
  encode_atom_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_bf16_output_tma_map(&o_map, d_o, 1);

  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  qk_tma_mma_ld_kernel<<<1, kMainThreads, kDynamicSmemBytes>>>(
      q_map, k_map, v_map, o_map, args.k_tiles, args.k_tiles, 1.0f, d_o
#if ATTENTION_CLOCK_TRACE
      ,
      nullptr, 0, 0
#endif
      );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> h_o(kTileWords);
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, kTileWords * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  const std::vector<float> got_norm = unpack_row_major_bf16_words(h_o);
  std::vector<CompareResult> results;
  results.push_back(compare_float("fused_norm", got_norm, norm_ref, 1.0e-3f, 3.0e-2f));
  results.push_back(compare_bf16_words("final_o_bf16", h_o, expected_bf16));
  write_validation_csv(args.csv, results);
  for (const CompareResult& r : results) {
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu\n", r.stage.c_str(),
                r.ok ? "ok" : "fail", r.max_abs, r.max_rel, r.bad_count);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  return std::all_of(results.begin(), results.end(), [](const CompareResult& r) { return r.ok; }) ? 0 : 1;
}

int run_benchmark(const Args& args_in) {
  Args args = args_in;
#if !ATTENTION_CLOCK_TRACE
  if (args.clock_trace) {
    std::fprintf(stderr,
                 "--clock-trace requires compiling with -DATTENTION_CLOCK_TRACE=1.\n");
    return 2;
  }
#endif
#if !ATTENTION_STORE_OUTPUT
  if (args.store_output) {
    std::fprintf(stderr,
                 "--store-output requires compiling with -DATTENTION_STORE_OUTPUT=1.\n");
    return 2;
  }
#endif
#if ATTENTION_STORE_OUTPUT
  if (args.store_output) args.repeats = args.k_tiles;
#endif
  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_o = nullptr;
  ClockTraceRecord* d_clock_trace = nullptr;
  const size_t q_words = static_cast<size_t>(args.blocks) * kTileWords;
  const size_t k_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  const size_t v_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  CUDA_CHECK(cudaMalloc(&d_q, q_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, k_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, v_words * sizeof(uint32_t)));
#if ATTENTION_STORE_OUTPUT
  if (args.store_output) {
    CUDA_CHECK(cudaMalloc(&d_o, static_cast<size_t>(args.blocks) * kTileWords * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_o, 0, static_cast<size_t>(args.blocks) * kTileWords * sizeof(uint32_t)));
  }
#endif

  const int fill_threads = 256;
  fill_packed_bf16<<<static_cast<int>((q_words + fill_threads - 1) / fill_threads), fill_threads>>>(d_q, q_words, 3);
  CUDA_CHECK(cudaGetLastError());
  fill_packed_bf16<<<static_cast<int>((k_words + fill_threads - 1) / fill_threads), fill_threads>>>(d_k, k_words, 11);
  CUDA_CHECK(cudaGetLastError());
  fill_packed_bf16<<<static_cast<int>((v_words + fill_threads - 1) / fill_threads), fill_threads>>>(d_v, v_words, 17);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  const int qk_layout = selected_qk_layout(args);
  if (qk_layout == kQkLayoutContiguousSingle5d) {
    encode_qk_contiguous_tma_map(&q_map, d_q, static_cast<uint64_t>(args.blocks));
    encode_qk_contiguous_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles));
  } else if (qk_layout == kQkLayoutContiguousSw128) {
    encode_qk_contiguous_sw128_tma_map(&q_map, d_q,
                                       static_cast<uint64_t>(args.blocks));
    encode_qk_contiguous_sw128_tma_map(&k_map, d_k,
                                       static_cast<uint64_t>(args.k_tiles));
  } else if (qk_layout == kQkLayoutContiguousK16Split4d) {
    encode_qk_contiguous_k16_split_tma_map(&q_map, d_q,
                                           static_cast<uint64_t>(args.blocks));
    encode_qk_contiguous_k16_split_tma_map(&k_map, d_k,
                                           static_cast<uint64_t>(args.k_tiles));
  } else {
    encode_atom_tma_map(&q_map, d_q, static_cast<uint64_t>(args.blocks) * 64);
    encode_atom_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles) * 64);
  }
  if (qk_layout == kQkLayoutContiguousSw128) {
    encode_contiguous_sw128_k16_tma_map(&v_map, d_v,
                                        static_cast<uint64_t>(args.k_tiles));
  } else {
    encode_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  }
  if (d_o) encode_bf16_output_tma_map(&o_map, d_o, static_cast<uint64_t>(args.blocks));

#if ATTENTION_CLOCK_TRACE
  const int clock_record_count = clock_trace_record_count(args);
  if (args.clock_trace) {
    CUDA_CHECK(cudaMalloc(&d_clock_trace,
                          static_cast<size_t>(clock_record_count) *
                              sizeof(ClockTraceRecord)));
    CUDA_CHECK(cudaMemset(d_clock_trace, 0,
                          static_cast<size_t>(clock_record_count) *
                              sizeof(ClockTraceRecord)));
  }
#endif

  int active = 0;
  const float score_to_exp2_scale =
      args.store_output ? resolved_score_to_exp2_scale(args) : 1.0f;
  RunResult result = run_kernel(args, q_map, k_map, v_map, o_map,
                                score_to_exp2_scale, d_o, &active
#if ATTENTION_CLOCK_TRACE
                                ,
                                d_clock_trace, args.clock_trace_iters
#endif
                                );
#if ATTENTION_CLOCK_TRACE
  if (args.clock_trace && result.error == cudaSuccess) {
    std::vector<ClockTraceRecord> h_clock_trace(clock_record_count);
    CUDA_CHECK(cudaMemcpy(h_clock_trace.data(), d_clock_trace,
                          static_cast<size_t>(clock_record_count) *
                              sizeof(ClockTraceRecord),
                          cudaMemcpyDeviceToHost));
    write_clock_trace_csv(args, result, h_clock_trace);
  } else
#endif
  {
    write_benchmark_csv(args, active, result);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  if (d_o) CUDA_CHECK(cudaFree(d_o));
  if (d_clock_trace) CUDA_CHECK(cudaFree(d_clock_trace));
  return result.error == cudaSuccess ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, &args);
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (args.stage != "scalar_validate" && prop.major < 10) {
    std::fprintf(stderr, "The tcgen05 benchmark/toy path requires SM100+; got sm_%d%d\n",
                 prop.major, prop.minor);
    return 77;
  }
  driver_check(cuInit(0), "cuInit");
  if (args.stage == "fused_validate") return run_fused_real_validation(args);
  if (args.stage == "scalar_validate") return run_scalar_real_validation(args);
  if (args.stage == "toy_validate") return run_validation(args);
  return run_benchmark(args);
}
