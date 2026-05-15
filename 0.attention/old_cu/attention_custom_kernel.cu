#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef ATTENTION_PRODUCER_REGS
#define ATTENTION_PRODUCER_REGS 88
#endif

#ifndef ATTENTION_CONSUMER_REGS
#define ATTENTION_CONSUMER_REGS 96
#endif

#ifndef ATTENTION_USE_SETMAXNREG
#define ATTENTION_USE_SETMAXNREG 1
#endif

#ifndef ATTENTION_PV_PINGPONG_DEP
#define ATTENTION_PV_PINGPONG_DEP 1
#endif

#ifndef ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA
#define ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA 0
#endif

#ifndef ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
#define ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME 0
#endif

#ifndef ATTENTION_V_TMA_AFTER_S_READY
#define ATTENTION_V_TMA_AFTER_S_READY 0
#endif

#ifndef ATTENTION_S_SWIZZLE_MODE
#define ATTENTION_S_SWIZZLE_MODE 0
#endif

#ifndef ATTENTION_USE_V4_S_STORE
#define ATTENTION_USE_V4_S_STORE 0
#endif

#ifndef ATTENTION_NVCC_MANAGED_LD_REGS
#define ATTENTION_NVCC_MANAGED_LD_REGS 0
#endif

#ifndef ATTENTION_STORE_OUTPUT
#define ATTENTION_STORE_OUTPUT 0
#endif

#if ATTENTION_STORE_OUTPUT && !ATTENTION_NVCC_MANAGED_LD_REGS
#error ATTENTION_STORE_OUTPUT requires ATTENTION_NVCC_MANAGED_LD_REGS=1
#endif

#define ATTENTION_STRINGIFY_IMPL(x) #x
#define ATTENTION_STRINGIFY(x) ATTENTION_STRINGIFY_IMPL(x)

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
static constexpr int kConsumerWarpsPerPipe = 4;
static constexpr int kTraceConsumerLanesPerPipe = kConsumerWarpsPerPipe * 2;
static constexpr int kMainWarps = 4 + kPipeCount * kConsumerWarpsPerPipe;
static constexpr int kMainThreads = kMainWarps * kWarpSize;
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kMmaK = 16;
static constexpr int kMmasPerTile = 8;
static constexpr int kTileBf16Elems = kTileM * kTileN;
static constexpr int kTileWords = kTileBf16Elems / 2;  // packed BF16x2 words.
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kKBufferCount = 1;
static constexpr int kVBufferCount = kPipeCount;
static constexpr int kSBufferCount = kPipeCount;
static constexpr int kDynamicSmemBytes =
    (1 + kPipeCount * kKBufferCount + kVBufferCount + kSBufferCount) * kTileBytes + 1024;
static constexpr int kTmemAllocCols = 512;
static constexpr int kTmemUsedCols = 512;
static constexpr double kFlopsPerMma =
    2.0 * static_cast<double>(kTileM) * static_cast<double>(kTileN) *
    static_cast<double>(kMmaK);
struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int k_tiles = 8192;
  int warmup = 2;
  int iters = 5;
  bool cycle_probe = false;
  bool cycle_trace = false;
  bool strict_cycle_trace = false;
  bool ws_trace = false;
  bool tma_prefetch_trace = false;
  bool fused_producer_trace = false;
  bool fused3_producer_trace = false;
  bool overlap_consumer_trace = false;
  bool store_output = false;
  int trace_warps = 2;
  int trace_launch_blocks = 1;
  const char* csv = "0.attention/attention_custom_kernel.csv";
};

struct RunResult {
  float ms = 0.0f;
  cudaError_t error = cudaSuccess;
  const char* status = "ok";
};

struct CycleTotals {
  unsigned long long q_tma_cycles;
  unsigned long long k_tma_cycles;
  unsigned long long mma_cycles;
  unsigned long long ld_cycles;
  unsigned long long total_group_cycles;
  unsigned long long q_samples;
  unsigned long long group_samples;
  unsigned long long ld_samples;
  unsigned int sink;
};

struct TraceRecord {
  unsigned long long tma_start;
  unsigned long long tma_end;
  unsigned long long mma_start;
  unsigned long long mma_end;
  unsigned long long ld_start;
  unsigned long long ld_end;
  unsigned long long ld_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long ld_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long pack_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long st_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long st_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_start;
  unsigned long long pack_end;
  unsigned long long st_start;
  unsigned long long st_end;
  unsigned long long v_tma_start;
  unsigned long long v_tma_end;
  unsigned long long pv_start;
  unsigned long long pv_end;
  unsigned int iter;
  unsigned int pipe;
  unsigned int warp_id;
  unsigned int sink;
};

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

__device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x64_acc(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t acc;
  asm volatile(
      "{ .reg .b32 r<64>; .reg .b32 acc; "
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, "
      "r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, "
      "r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, "
      "r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63}, [%1]; "
      "xor.b32 acc, r0, r15; "
      "xor.b32 acc, acc, r31; "
      "xor.b32 acc, acc, r47; "
      "xor.b32 %0, acc, r63; }"
      : "=r"(acc)
      : "r"(taddr)
      : "memory");
  return acc;
#else
  (void)taddr;
  return 0;
#endif
}

#define TMEM_REGS_0_63                                                        \
  "r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, "    \
  "r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, "    \
  "r30, r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, "    \
  "r44, r45, r46, r47, r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, "    \
  "r58, r59, r60, r61, r62, r63"

#define TMEM_REGS_64_127                                                      \
  "r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, "     \
  "r78, r79, r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, "     \
  "r92, r93, r94, r95, r96, r97, r98, r99, r100, r101, r102, r103, r104, "     \
  "r105, r106, r107, r108, r109, r110, r111, r112, r113, r114, r115, r116, "   \
  "r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127"

#define TMEM_REGS_0_127 TMEM_REGS_0_63 ", " TMEM_REGS_64_127

#define EXP2_REGS_0_63                                                       \
  "ex2.approx.ftz.f32 r0, r0; "                                             \
  "ex2.approx.ftz.f32 r1, r1; "                                             \
  "ex2.approx.ftz.f32 r2, r2; "                                             \
  "ex2.approx.ftz.f32 r3, r3; "                                             \
  "ex2.approx.ftz.f32 r4, r4; "                                             \
  "ex2.approx.ftz.f32 r5, r5; "                                             \
  "ex2.approx.ftz.f32 r6, r6; "                                             \
  "ex2.approx.ftz.f32 r7, r7; "                                             \
  "ex2.approx.ftz.f32 r8, r8; "                                             \
  "ex2.approx.ftz.f32 r9, r9; "                                             \
  "ex2.approx.ftz.f32 r10, r10; "                                           \
  "ex2.approx.ftz.f32 r11, r11; "                                           \
  "ex2.approx.ftz.f32 r12, r12; "                                           \
  "ex2.approx.ftz.f32 r13, r13; "                                           \
  "ex2.approx.ftz.f32 r14, r14; "                                           \
  "ex2.approx.ftz.f32 r15, r15; "                                           \
  "ex2.approx.ftz.f32 r16, r16; "                                           \
  "ex2.approx.ftz.f32 r17, r17; "                                           \
  "ex2.approx.ftz.f32 r18, r18; "                                           \
  "ex2.approx.ftz.f32 r19, r19; "                                           \
  "ex2.approx.ftz.f32 r20, r20; "                                           \
  "ex2.approx.ftz.f32 r21, r21; "                                           \
  "ex2.approx.ftz.f32 r22, r22; "                                           \
  "ex2.approx.ftz.f32 r23, r23; "                                           \
  "ex2.approx.ftz.f32 r24, r24; "                                           \
  "ex2.approx.ftz.f32 r25, r25; "                                           \
  "ex2.approx.ftz.f32 r26, r26; "                                           \
  "ex2.approx.ftz.f32 r27, r27; "                                           \
  "ex2.approx.ftz.f32 r28, r28; "                                           \
  "ex2.approx.ftz.f32 r29, r29; "                                           \
  "ex2.approx.ftz.f32 r30, r30; "                                           \
  "ex2.approx.ftz.f32 r31, r31; "                                           \
  "ex2.approx.ftz.f32 r32, r32; "                                           \
  "ex2.approx.ftz.f32 r33, r33; "                                           \
  "ex2.approx.ftz.f32 r34, r34; "                                           \
  "ex2.approx.ftz.f32 r35, r35; "                                           \
  "ex2.approx.ftz.f32 r36, r36; "                                           \
  "ex2.approx.ftz.f32 r37, r37; "                                           \
  "ex2.approx.ftz.f32 r38, r38; "                                           \
  "ex2.approx.ftz.f32 r39, r39; "                                           \
  "ex2.approx.ftz.f32 r40, r40; "                                           \
  "ex2.approx.ftz.f32 r41, r41; "                                           \
  "ex2.approx.ftz.f32 r42, r42; "                                           \
  "ex2.approx.ftz.f32 r43, r43; "                                           \
  "ex2.approx.ftz.f32 r44, r44; "                                           \
  "ex2.approx.ftz.f32 r45, r45; "                                           \
  "ex2.approx.ftz.f32 r46, r46; "                                           \
  "ex2.approx.ftz.f32 r47, r47; "                                           \
  "ex2.approx.ftz.f32 r48, r48; "                                           \
  "ex2.approx.ftz.f32 r49, r49; "                                           \
  "ex2.approx.ftz.f32 r50, r50; "                                           \
  "ex2.approx.ftz.f32 r51, r51; "                                           \
  "ex2.approx.ftz.f32 r52, r52; "                                           \
  "ex2.approx.ftz.f32 r53, r53; "                                           \
  "ex2.approx.ftz.f32 r54, r54; "                                           \
  "ex2.approx.ftz.f32 r55, r55; "                                           \
  "ex2.approx.ftz.f32 r56, r56; "                                           \
  "ex2.approx.ftz.f32 r57, r57; "                                           \
  "ex2.approx.ftz.f32 r58, r58; "                                           \
  "ex2.approx.ftz.f32 r59, r59; "                                           \
  "ex2.approx.ftz.f32 r60, r60; "                                           \
  "ex2.approx.ftz.f32 r61, r61; "                                           \
  "ex2.approx.ftz.f32 r62, r62; "                                           \
  "ex2.approx.ftz.f32 r63, r63; "

#if ATTENTION_USE_V4_S_STORE
#define PACK4_STORE_S(r0_, r1_, r2_, r3_, r4_, r5_, r6_, r7_, offset_)       \
  "shr.u32 lo, " #r0_ ", 16; and.b32 hi, " #r1_                             \
  ", 0xffff0000; or.b32 p0, lo, hi; "                                       \
  "shr.u32 lo, " #r2_ ", 16; and.b32 hi, " #r3_                             \
  ", 0xffff0000; or.b32 p1, lo, hi; "                                       \
  "shr.u32 lo, " #r4_ ", 16; and.b32 hi, " #r5_                             \
  ", 0xffff0000; or.b32 p2, lo, hi; "                                       \
  "shr.u32 lo, " #r6_ ", 16; and.b32 hi, " #r7_                             \
  ", 0xffff0000; or.b32 p3, lo, hi; "                                       \
  "st.shared.v4.u32 [addr + " #offset_ "], {p0, p1, p2, p3}; "

#define STORE_PACKED4_S(p0_, p1_, p2_, p3_, offset_)                         \
  "st.shared.v4.u32 [addr + " #offset_ "], {" #p0_ ", " #p1_ ", " #p2_     \
  ", " #p3_ "}; "
#else
#define PACK4_STORE_S(r0_, r1_, r2_, r3_, r4_, r5_, r6_, r7_, offset_)       \
  "shr.u32 lo, " #r0_ ", 16; and.b32 hi, " #r1_                             \
  ", 0xffff0000; or.b32 p0, lo, hi; "                                       \
  "shr.u32 lo, " #r2_ ", 16; and.b32 hi, " #r3_                             \
  ", 0xffff0000; or.b32 p1, lo, hi; "                                       \
  "shr.u32 lo, " #r4_ ", 16; and.b32 hi, " #r5_                             \
  ", 0xffff0000; or.b32 p2, lo, hi; "                                       \
  "shr.u32 lo, " #r6_ ", 16; and.b32 hi, " #r7_                             \
  ", 0xffff0000; or.b32 p3, lo, hi; "                                       \
  "st.shared.u32 [addr + " #offset_ "], p0; "                               \
  "st.shared.u32 [addr + " #offset_ " + 4], p1; "                           \
  "st.shared.u32 [addr + " #offset_ " + 8], p2; "                           \
  "st.shared.u32 [addr + " #offset_ " + 12], p3; "

#define STORE_PACKED4_S(p0_, p1_, p2_, p3_, offset_)                         \
  "st.shared.u32 [addr + " #offset_ "], " #p0_ "; "                         \
  "st.shared.u32 [addr + " #offset_ " + 4], " #p1_ "; "                     \
  "st.shared.u32 [addr + " #offset_ " + 8], " #p2_ "; "                     \
  "st.shared.u32 [addr + " #offset_ " + 12], " #p3_ "; "
#endif

#if ATTENTION_NVCC_MANAGED_LD_REGS
// Experimental path: keep the tcgen05 load in PTX, but expose the 64 loaded
// values to nvcc so it owns the exp2/pack/store live ranges.
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

__device__ __forceinline__ uint32_t exp2_approx_bits_cpp(uint32_t x) {
  const float in = __uint_as_float(x);
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

__device__ __forceinline__ void store_packed4_s_cpp(uint32_t* smem,
                                                    int word_offset,
                                                    uint32_t p0,
                                                    uint32_t p1,
                                                    uint32_t p2,
                                                    uint32_t p3) {
  reinterpret_cast<uint4*>(smem + word_offset)[0] = make_uint4(p0, p1, p2, p3);
}

__device__ __forceinline__ uint32_t tcgen05_ld_x64_wait_pack_store_nvcc(
    uint32_t src_taddr,
    uint32_t* s_smem,
    int consumer_warp,
    int consumer_half,
    uint64_t* p_done_barrier,
    bool arrive_p_done,
    float* row_sum_pipe) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = consumer_warp * 32 + lane;
  const int col_pair_base = consumer_half * 32;
  uint32_t* smem_base = s_smem + s_store_word_offset(row, col_pair_base);
  uint32_t r[64];

  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" NVCC_LD_REG_OPERANDS_0_63
      "}, [%64];"
      : NVCC_LD_REG_OUTPUTS_0_63
      : "r"(src_taddr)
      : "memory");
  tcgen05_wait_ld();
  if (arrive_p_done && lane == 0) {
    mbarrier_arrive(p_done_barrier);
  }

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
  if (row_sum_pipe != nullptr) {
    float half_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      half_sum += __uint_as_float(r[i] & 0xffff0000u);
    }
    row_sum_pipe[row] += half_sum;
  }
  return r[0] ^ r[15] ^ r[31] ^ r[47] ^ r[63];
#else
  (void)src_taddr;
  (void)s_smem;
  (void)consumer_warp;
  (void)consumer_half;
  (void)p_done_barrier;
  (void)arrive_p_done;
  (void)row_sum_pipe;
  return 0;
#endif
}
#endif

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

[[maybe_unused]] __device__ __forceinline__ uint32_t tcgen05_ld_x64x2_acc(uint32_t src_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t acc;
  asm volatile(
      "{ .reg .b32 r<64>; .reg .b32 h<64>; .reg .b32 acc; "
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TMEM_REGS_0_63 "}, [%1]; "
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
      "{h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, h14, h15, "
      "h16, h17, h18, h19, h20, h21, h22, h23, h24, h25, h26, h27, h28, h29, "
      "h30, h31, h32, h33, h34, h35, h36, h37, h38, h39, h40, h41, h42, h43, "
      "h44, h45, h46, h47, h48, h49, h50, h51, h52, h53, h54, h55, h56, h57, "
      "h58, h59, h60, h61, h62, h63}, [%2]; "
      "tcgen05.wait::ld.sync.aligned; "
      "xor.b32 r0, r0, h0; xor.b32 r15, r15, h15; xor.b32 r31, r31, h31; "
      "xor.b32 r47, r47, h47; xor.b32 r63, r63, h63; "
      "xor.b32 acc, r0, r15; xor.b32 acc, acc, r31; xor.b32 acc, acc, r47; "
      "xor.b32 %0, acc, r63; }"
      : "=r"(acc)
      : "r"(src_taddr), "r"(src_taddr + 64u)
      : "memory");
  return acc;
#else
  (void)src_taddr;
  return 0;
#endif
}

#define PENDING_REGS_DECL \
  uint32_t p0 = 0; \
  uint32_t p1 = 0; \
  uint32_t p2 = 0; \
  uint32_t p3 = 0; \
  uint32_t p4 = 0; \
  uint32_t p5 = 0; \
  uint32_t p6 = 0; \
  uint32_t p7 = 0; \
  uint32_t p8 = 0; \
  uint32_t p9 = 0; \
  uint32_t p10 = 0; \
  uint32_t p11 = 0; \
  uint32_t p12 = 0; \
  uint32_t p13 = 0; \
  uint32_t p14 = 0; \
  uint32_t p15 = 0; \
  uint32_t p16 = 0; \
  uint32_t p17 = 0; \
  uint32_t p18 = 0; \
  uint32_t p19 = 0; \
  uint32_t p20 = 0; \
  uint32_t p21 = 0; \
  uint32_t p22 = 0; \
  uint32_t p23 = 0; \
  uint32_t p24 = 0; \
  uint32_t p25 = 0; \
  uint32_t p26 = 0; \
  uint32_t p27 = 0; \
  uint32_t p28 = 0; \
  uint32_t p29 = 0; \
  uint32_t p30 = 0; \
  uint32_t p31 = 0;

#define PENDING_REGS_RW_OPERANDS \
  "+r"(p0), \
  "+r"(p1), \
  "+r"(p2), \
  "+r"(p3), \
  "+r"(p4), \
  "+r"(p5), \
  "+r"(p6), \
  "+r"(p7), \
  "+r"(p8), \
  "+r"(p9), \
  "+r"(p10), \
  "+r"(p11), \
  "+r"(p12), \
  "+r"(p13), \
  "+r"(p14), \
  "+r"(p15), \
  "+r"(p16), \
  "+r"(p17), \
  "+r"(p18), \
  "+r"(p19), \
  "+r"(p20), \
  "+r"(p21), \
  "+r"(p22), \
  "+r"(p23), \
  "+r"(p24), \
  "+r"(p25), \
  "+r"(p26), \
  "+r"(p27), \
  "+r"(p28), \
  "+r"(p29), \
  "+r"(p30), \
  "+r"(p31)

#define PENDING_REGS_R_OPERANDS \
  "r"(p0), \
  "r"(p1), \
  "r"(p2), \
  "r"(p3), \
  "r"(p4), \
  "r"(p5), \
  "r"(p6), \
  "r"(p7), \
  "r"(p8), \
  "r"(p9), \
  "r"(p10), \
  "r"(p11), \
  "r"(p12), \
  "r"(p13), \
  "r"(p14), \
  "r"(p15), \
  "r"(p16), \
  "r"(p17), \
  "r"(p18), \
  "r"(p19), \
  "r"(p20), \
  "r"(p21), \
  "r"(p22), \
  "r"(p23), \
  "r"(p24), \
  "r"(p25), \
  "r"(p26), \
  "r"(p27), \
  "r"(p28), \
  "r"(p29), \
  "r"(p30), \
  "r"(p31)

#define TCGEN05_LD_X64_ISSUE_STORE_PENDING_WAIT_PACK(                  \
    src_taddr, s_smem, consumer_warp, consumer_half, s_ready_barrier, has_pending, acc_out) \
do {                                                                        \
  const int lane__ = threadIdx.x & 31;                                      \
  const int row__ = (consumer_warp) * 32 + lane__;                          \
  const int col_pair_base__ = (consumer_half) * 32;                         \
  const uint32_t smem_base__ =                                             \
      smem_ptr_u32((s_smem) + s_store_word_offset(row__, col_pair_base__)); \
  const uint32_t s_ready_addr__ = smem_ptr_u32(s_ready_barrier);           \
  asm volatile(                                                             \
      "{ .reg .b32 r<64>; .reg .u32 addr, bar; "                                           \
      ".reg .pred p; .reg .b32 acc, lo, hi; "                                           \
      "mov.u32 addr, %34; "                                           \
      "mov.u32 bar, %36; "                                           \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 "                                           \
      "{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, "                                           \
      "r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, "                                           \
      "r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, "                                           \
      "r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63}, [%33]; "                                           \
      "setp.eq.u32 p, %35, 0; "                                           \
      "@p bra SKIP_STORE_%=; "                                           \
      "st.shared.u32 [addr + 0], %0; "                                           \
      "st.shared.u32 [addr + 128], %1; "                                           \
      "st.shared.u32 [addr + 256], %2; "                                           \
      "st.shared.u32 [addr + 384], %3; "                                           \
      "st.shared.u32 [addr + 512], %4; "                                           \
      "st.shared.u32 [addr + 640], %5; "                                           \
      "st.shared.u32 [addr + 768], %6; "                                           \
      "st.shared.u32 [addr + 896], %7; "                                           \
      "st.shared.u32 [addr + 1024], %8; "                                           \
      "st.shared.u32 [addr + 1152], %9; "                                           \
      "st.shared.u32 [addr + 1280], %10; "                                           \
      "st.shared.u32 [addr + 1408], %11; "                                           \
      "st.shared.u32 [addr + 1536], %12; "                                           \
      "st.shared.u32 [addr + 1664], %13; "                                           \
      "st.shared.u32 [addr + 1792], %14; "                                           \
      "st.shared.u32 [addr + 1920], %15; "                                           \
      "st.shared.u32 [addr + 2048], %16; "                                           \
      "st.shared.u32 [addr + 2176], %17; "                                           \
      "st.shared.u32 [addr + 2304], %18; "                                           \
      "st.shared.u32 [addr + 2432], %19; "                                           \
      "st.shared.u32 [addr + 2560], %20; "                                           \
      "st.shared.u32 [addr + 2688], %21; "                                           \
      "st.shared.u32 [addr + 2816], %22; "                                           \
      "st.shared.u32 [addr + 2944], %23; "                                           \
      "st.shared.u32 [addr + 3072], %24; "                                           \
      "st.shared.u32 [addr + 3200], %25; "                                           \
      "st.shared.u32 [addr + 3328], %26; "                                           \
      "st.shared.u32 [addr + 3456], %27; "                                           \
      "st.shared.u32 [addr + 3584], %28; "                                           \
      "st.shared.u32 [addr + 3712], %29; "                                           \
      "st.shared.u32 [addr + 3840], %30; "                                           \
      "st.shared.u32 [addr + 3968], %31; "                                           \
      "setp.ne.u32 p, %37, 0; "                                           \
      "@p bra SKIP_ARRIVE_%=; "                                           \
      "mbarrier.arrive.shared::cta.b64 _, [bar]; "                                           \
      "SKIP_ARRIVE_%=: "                                           \
      "SKIP_STORE_%=: "                                           \
      "tcgen05.wait::ld.sync.aligned; "                                           \
      "and.b32 lo, r0, 0xffff0000; shr.u32 hi, r32, 16; or.b32 %0, lo, hi; "                                           \
      "and.b32 lo, r1, 0xffff0000; shr.u32 hi, r33, 16; or.b32 %1, lo, hi; "                                           \
      "and.b32 lo, r2, 0xffff0000; shr.u32 hi, r34, 16; or.b32 %2, lo, hi; "                                           \
      "and.b32 lo, r3, 0xffff0000; shr.u32 hi, r35, 16; or.b32 %3, lo, hi; "                                           \
      "and.b32 lo, r4, 0xffff0000; shr.u32 hi, r36, 16; or.b32 %4, lo, hi; "                                           \
      "and.b32 lo, r5, 0xffff0000; shr.u32 hi, r37, 16; or.b32 %5, lo, hi; "                                           \
      "and.b32 lo, r6, 0xffff0000; shr.u32 hi, r38, 16; or.b32 %6, lo, hi; "                                           \
      "and.b32 lo, r7, 0xffff0000; shr.u32 hi, r39, 16; or.b32 %7, lo, hi; "                                           \
      "and.b32 lo, r8, 0xffff0000; shr.u32 hi, r40, 16; or.b32 %8, lo, hi; "                                           \
      "and.b32 lo, r9, 0xffff0000; shr.u32 hi, r41, 16; or.b32 %9, lo, hi; "                                           \
      "and.b32 lo, r10, 0xffff0000; shr.u32 hi, r42, 16; or.b32 %10, lo, hi; "                                           \
      "and.b32 lo, r11, 0xffff0000; shr.u32 hi, r43, 16; or.b32 %11, lo, hi; "                                           \
      "and.b32 lo, r12, 0xffff0000; shr.u32 hi, r44, 16; or.b32 %12, lo, hi; "                                           \
      "and.b32 lo, r13, 0xffff0000; shr.u32 hi, r45, 16; or.b32 %13, lo, hi; "                                           \
      "and.b32 lo, r14, 0xffff0000; shr.u32 hi, r46, 16; or.b32 %14, lo, hi; "                                           \
      "and.b32 lo, r15, 0xffff0000; shr.u32 hi, r47, 16; or.b32 %15, lo, hi; "                                           \
      "and.b32 lo, r16, 0xffff0000; shr.u32 hi, r48, 16; or.b32 %16, lo, hi; "                                           \
      "and.b32 lo, r17, 0xffff0000; shr.u32 hi, r49, 16; or.b32 %17, lo, hi; "                                           \
      "and.b32 lo, r18, 0xffff0000; shr.u32 hi, r50, 16; or.b32 %18, lo, hi; "                                           \
      "and.b32 lo, r19, 0xffff0000; shr.u32 hi, r51, 16; or.b32 %19, lo, hi; "                                           \
      "and.b32 lo, r20, 0xffff0000; shr.u32 hi, r52, 16; or.b32 %20, lo, hi; "                                           \
      "and.b32 lo, r21, 0xffff0000; shr.u32 hi, r53, 16; or.b32 %21, lo, hi; "                                           \
      "and.b32 lo, r22, 0xffff0000; shr.u32 hi, r54, 16; or.b32 %22, lo, hi; "                                           \
      "and.b32 lo, r23, 0xffff0000; shr.u32 hi, r55, 16; or.b32 %23, lo, hi; "                                           \
      "and.b32 lo, r24, 0xffff0000; shr.u32 hi, r56, 16; or.b32 %24, lo, hi; "                                           \
      "and.b32 lo, r25, 0xffff0000; shr.u32 hi, r57, 16; or.b32 %25, lo, hi; "                                           \
      "and.b32 lo, r26, 0xffff0000; shr.u32 hi, r58, 16; or.b32 %26, lo, hi; "                                           \
      "and.b32 lo, r27, 0xffff0000; shr.u32 hi, r59, 16; or.b32 %27, lo, hi; "                                           \
      "and.b32 lo, r28, 0xffff0000; shr.u32 hi, r60, 16; or.b32 %28, lo, hi; "                                           \
      "and.b32 lo, r29, 0xffff0000; shr.u32 hi, r61, 16; or.b32 %29, lo, hi; "                                           \
      "and.b32 lo, r30, 0xffff0000; shr.u32 hi, r62, 16; or.b32 %30, lo, hi; "                                           \
      "and.b32 lo, r31, 0xffff0000; shr.u32 hi, r63, 16; or.b32 %31, lo, hi; "                                           \
      "xor.b32 acc, r0, r15; "                                           \
      "xor.b32 acc, acc, r31; "                                           \
      "xor.b32 acc, acc, r47; "                                           \
      "xor.b32 %32, acc, r63; }"                                           \
      : PENDING_REGS_RW_OPERANDS, "=r"(acc_out)                            \
      : "r"(src_taddr), "r"(smem_base__),                                  \
        "r"(static_cast<uint32_t>(has_pending)), "r"(s_ready_addr__), "r"(lane__) \
      : "memory");                                                         \
} while (0)

#define STORE_PENDING_REGS_TO_SMEM(s_smem, consumer_warp, consumer_half)    \
do {                                                                        \
  const int lane__ = threadIdx.x & 31;                                      \
  const int row__ = (consumer_warp) * 32 + lane__;                          \
  const int col_pair_base__ = (consumer_half) * 32;                         \
  const uint32_t smem_base__ =                                             \
      smem_ptr_u32((s_smem) + s_store_word_offset(row__, col_pair_base__)); \
  asm volatile(                                                             \
      "{ .reg .u32 addr; mov.u32 addr, %32; "                              \
      "st.shared.u32 [addr + 0], %0; "                                           \
      "st.shared.u32 [addr + 128], %1; "                                           \
      "st.shared.u32 [addr + 256], %2; "                                           \
      "st.shared.u32 [addr + 384], %3; "                                           \
      "st.shared.u32 [addr + 512], %4; "                                           \
      "st.shared.u32 [addr + 640], %5; "                                           \
      "st.shared.u32 [addr + 768], %6; "                                           \
      "st.shared.u32 [addr + 896], %7; "                                           \
      "st.shared.u32 [addr + 1024], %8; "                                           \
      "st.shared.u32 [addr + 1152], %9; "                                           \
      "st.shared.u32 [addr + 1280], %10; "                                           \
      "st.shared.u32 [addr + 1408], %11; "                                           \
      "st.shared.u32 [addr + 1536], %12; "                                           \
      "st.shared.u32 [addr + 1664], %13; "                                           \
      "st.shared.u32 [addr + 1792], %14; "                                           \
      "st.shared.u32 [addr + 1920], %15; "                                           \
      "st.shared.u32 [addr + 2048], %16; "                                           \
      "st.shared.u32 [addr + 2176], %17; "                                           \
      "st.shared.u32 [addr + 2304], %18; "                                           \
      "st.shared.u32 [addr + 2432], %19; "                                           \
      "st.shared.u32 [addr + 2560], %20; "                                           \
      "st.shared.u32 [addr + 2688], %21; "                                           \
      "st.shared.u32 [addr + 2816], %22; "                                           \
      "st.shared.u32 [addr + 2944], %23; "                                           \
      "st.shared.u32 [addr + 3072], %24; "                                           \
      "st.shared.u32 [addr + 3200], %25; "                                           \
      "st.shared.u32 [addr + 3328], %26; "                                           \
      "st.shared.u32 [addr + 3456], %27; "                                           \
      "st.shared.u32 [addr + 3584], %28; "                                           \
      "st.shared.u32 [addr + 3712], %29; "                                           \
      "st.shared.u32 [addr + 3840], %30; "                                           \
      "st.shared.u32 [addr + 3968], %31; "                                           \
      "}"                                                                       \
      :: PENDING_REGS_R_OPERANDS, "r"(smem_base__)                         \
      : "memory");                                                         \
} while (0)

#if ATTENTION_NVCC_MANAGED_LD_REGS
#define TCGEN05_LD_X64_WAIT_PACK_STORE(src_taddr, s_smem, consumer_warp, consumer_half, p_done_barrier, arrive_p_done, row_sum_pipe, acc_out) \
do {                                                                        \
  (acc_out) = tcgen05_ld_x64_wait_pack_store_nvcc(                          \
      (src_taddr), (s_smem), (consumer_warp), (consumer_half),              \
      (p_done_barrier), (arrive_p_done), (row_sum_pipe));                   \
} while (0)
#else
#define TCGEN05_LD_X64_WAIT_PACK_STORE(src_taddr, s_smem, consumer_warp, consumer_half, p_done_barrier, arrive_p_done, row_sum_pipe, acc_out) \
do {                                                                        \
  (void)(row_sum_pipe);                                                     \
  const int lane__ = threadIdx.x & 31;                                      \
  const int row__ = (consumer_warp) * 32 + lane__;                          \
  const int col_pair_base__ = (consumer_half) * 32;                         \
  const uint32_t smem_base__ =                                             \
      smem_ptr_u32((s_smem) + s_store_word_offset(row__, col_pair_base__)); \
  const uint32_t p_done_addr__ = smem_ptr_u32(p_done_barrier);             \
  const uint32_t arrive_p_done__ = static_cast<uint32_t>(arrive_p_done);   \
  asm volatile(                                                             \
      "{ .reg .b32 r<64>; .reg .u32 addr, bar; .reg .pred pred; "          \
      ".reg .b32 acc, lo, hi, p0, p1, p2, p3; "                            \
      "mov.u32 addr, %2; "                                                 \
      "mov.u32 bar, %3; "                                                  \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TMEM_REGS_0_63 "}, [%1]; " \
      "tcgen05.wait::ld.sync.aligned; "                                    \
      "setp.eq.u32 pred, %4, 0; @pred bra SKIP_P_DONE_%=; "                \
      "setp.ne.u32 pred, %5, 0; @pred bra SKIP_P_DONE_%=; "                \
      "mbarrier.arrive.shared::cta.b64 _, [bar]; "                         \
      "SKIP_P_DONE_%=: "                                                   \
      EXP2_REGS_0_63                                                        \
      PACK4_STORE_S(r0, r1, r2, r3, r4, r5, r6, r7, 0)                   \
      PACK4_STORE_S(r8, r9, r10, r11, r12, r13, r14, r15, 128)            \
      PACK4_STORE_S(r16, r17, r18, r19, r20, r21, r22, r23, 4096)         \
      PACK4_STORE_S(r24, r25, r26, r27, r28, r29, r30, r31, 4224)         \
      PACK4_STORE_S(r32, r33, r34, r35, r36, r37, r38, r39, 8192)         \
      PACK4_STORE_S(r40, r41, r42, r43, r44, r45, r46, r47, 8320)         \
      PACK4_STORE_S(r48, r49, r50, r51, r52, r53, r54, r55, 12288)        \
      PACK4_STORE_S(r56, r57, r58, r59, r60, r61, r62, r63, 12416)        \
      "xor.b32 acc, r0, r15; "                                             \
      "xor.b32 acc, acc, r31; "                                            \
      "xor.b32 acc, acc, r47; "                                            \
      "xor.b32 %0, acc, r63; }"                                            \
      : "=r"(acc_out)                                                      \
      : "r"(src_taddr), "r"(smem_base__), "r"(p_done_addr__),            \
        "r"(arrive_p_done__), "r"(static_cast<uint32_t>(lane__))          \
      : "memory");                                                         \
} while (0)
#endif

#define TCGEN05_LD_X64_WAIT_PACK_STORE_TRACE(                              \
    src_taddr, s_smem, consumer_warp, consumer_half, p_done_barrier, arrive_p_done, acc_out, \
    ld_start_out, ld_end_out, pack_start_out, pack_end_out, st_start_out, st_end_out) \
do {                                                                        \
  const int lane__ = threadIdx.x & 31;                                      \
  const int row__ = (consumer_warp) * 32 + lane__;                          \
  const int col_pair_base__ = (consumer_half) * 32;                         \
  const uint32_t smem_base__ =                                             \
      smem_ptr_u32((s_smem) + s_store_word_offset(row__, col_pair_base__)); \
  const uint32_t p_done_addr__ = smem_ptr_u32(p_done_barrier);             \
  const uint32_t arrive_p_done__ = static_cast<uint32_t>(arrive_p_done);   \
  asm volatile(                                                             \
      "{ .reg .b32 r<64>; .reg .b32 p<32>; .reg .u32 addr, bar; .reg .pred pred; " \
      ".reg .b32 acc, lo, hi; "                                            \
      "mov.u32 addr, %8; "                                                 \
      "mov.u32 bar, %9; "                                                  \
      "mov.u64 %1, %%clock64; "                                            \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TMEM_REGS_0_63 "}, [%7]; " \
      "tcgen05.wait::ld.sync.aligned; "                                    \
      "mov.u64 %2, %%clock64; "                                            \
      "setp.eq.u32 pred, %10, 0; @pred bra SKIP_P_DONE_TRACE_%=; "         \
      "setp.ne.u32 pred, %11, 0; @pred bra SKIP_P_DONE_TRACE_%=; "         \
      "mbarrier.arrive.shared::cta.b64 _, [bar]; "                         \
      "SKIP_P_DONE_TRACE_%=: "                                             \
      "mov.u64 %3, %%clock64; "                                            \
      EXP2_REGS_0_63                                                        \
      "shr.u32 lo, r0, 16; and.b32 hi, r1, 0xffff0000; or.b32 p0, lo, hi; " \
      "shr.u32 lo, r2, 16; and.b32 hi, r3, 0xffff0000; or.b32 p1, lo, hi; " \
      "shr.u32 lo, r4, 16; and.b32 hi, r5, 0xffff0000; or.b32 p2, lo, hi; " \
      "shr.u32 lo, r6, 16; and.b32 hi, r7, 0xffff0000; or.b32 p3, lo, hi; " \
      "shr.u32 lo, r8, 16; and.b32 hi, r9, 0xffff0000; or.b32 p4, lo, hi; " \
      "shr.u32 lo, r10, 16; and.b32 hi, r11, 0xffff0000; or.b32 p5, lo, hi; " \
      "shr.u32 lo, r12, 16; and.b32 hi, r13, 0xffff0000; or.b32 p6, lo, hi; " \
      "shr.u32 lo, r14, 16; and.b32 hi, r15, 0xffff0000; or.b32 p7, lo, hi; " \
      "shr.u32 lo, r16, 16; and.b32 hi, r17, 0xffff0000; or.b32 p8, lo, hi; " \
      "shr.u32 lo, r18, 16; and.b32 hi, r19, 0xffff0000; or.b32 p9, lo, hi; " \
      "shr.u32 lo, r20, 16; and.b32 hi, r21, 0xffff0000; or.b32 p10, lo, hi; " \
      "shr.u32 lo, r22, 16; and.b32 hi, r23, 0xffff0000; or.b32 p11, lo, hi; " \
      "shr.u32 lo, r24, 16; and.b32 hi, r25, 0xffff0000; or.b32 p12, lo, hi; " \
      "shr.u32 lo, r26, 16; and.b32 hi, r27, 0xffff0000; or.b32 p13, lo, hi; " \
      "shr.u32 lo, r28, 16; and.b32 hi, r29, 0xffff0000; or.b32 p14, lo, hi; " \
      "shr.u32 lo, r30, 16; and.b32 hi, r31, 0xffff0000; or.b32 p15, lo, hi; " \
      "shr.u32 lo, r32, 16; and.b32 hi, r33, 0xffff0000; or.b32 p16, lo, hi; " \
      "shr.u32 lo, r34, 16; and.b32 hi, r35, 0xffff0000; or.b32 p17, lo, hi; " \
      "shr.u32 lo, r36, 16; and.b32 hi, r37, 0xffff0000; or.b32 p18, lo, hi; " \
      "shr.u32 lo, r38, 16; and.b32 hi, r39, 0xffff0000; or.b32 p19, lo, hi; " \
      "shr.u32 lo, r40, 16; and.b32 hi, r41, 0xffff0000; or.b32 p20, lo, hi; " \
      "shr.u32 lo, r42, 16; and.b32 hi, r43, 0xffff0000; or.b32 p21, lo, hi; " \
      "shr.u32 lo, r44, 16; and.b32 hi, r45, 0xffff0000; or.b32 p22, lo, hi; " \
      "shr.u32 lo, r46, 16; and.b32 hi, r47, 0xffff0000; or.b32 p23, lo, hi; " \
      "shr.u32 lo, r48, 16; and.b32 hi, r49, 0xffff0000; or.b32 p24, lo, hi; " \
      "shr.u32 lo, r50, 16; and.b32 hi, r51, 0xffff0000; or.b32 p25, lo, hi; " \
      "shr.u32 lo, r52, 16; and.b32 hi, r53, 0xffff0000; or.b32 p26, lo, hi; " \
      "shr.u32 lo, r54, 16; and.b32 hi, r55, 0xffff0000; or.b32 p27, lo, hi; " \
      "shr.u32 lo, r56, 16; and.b32 hi, r57, 0xffff0000; or.b32 p28, lo, hi; " \
      "shr.u32 lo, r58, 16; and.b32 hi, r59, 0xffff0000; or.b32 p29, lo, hi; " \
      "shr.u32 lo, r60, 16; and.b32 hi, r61, 0xffff0000; or.b32 p30, lo, hi; " \
      "shr.u32 lo, r62, 16; and.b32 hi, r63, 0xffff0000; or.b32 p31, lo, hi; " \
      "mov.u64 %4, %%clock64; "                                            \
      "mov.u64 %5, %%clock64; "                                            \
      STORE_PACKED4_S(p0, p1, p2, p3, 0)                                  \
      STORE_PACKED4_S(p4, p5, p6, p7, 128)                                \
      STORE_PACKED4_S(p8, p9, p10, p11, 4096)                             \
      STORE_PACKED4_S(p12, p13, p14, p15, 4224)                           \
      STORE_PACKED4_S(p16, p17, p18, p19, 8192)                           \
      STORE_PACKED4_S(p20, p21, p22, p23, 8320)                           \
      STORE_PACKED4_S(p24, p25, p26, p27, 12288)                          \
      STORE_PACKED4_S(p28, p29, p30, p31, 12416)                          \
      "mov.u64 %6, %%clock64; "                                            \
      "xor.b32 acc, r0, r15; "                                             \
      "xor.b32 acc, acc, r31; "                                            \
      "xor.b32 acc, acc, r47; "                                            \
      "xor.b32 %0, acc, r63; }"                                            \
      : "=r"(acc_out), "=l"(ld_start_out), "=l"(ld_end_out),              \
        "=l"(pack_start_out), "=l"(pack_end_out),                         \
        "=l"(st_start_out), "=l"(st_end_out)                              \
      : "r"(src_taddr), "r"(smem_base__), "r"(p_done_addr__),            \
        "r"(arrive_p_done__), "r"(static_cast<uint32_t>(lane__))          \
      : "memory");                                                         \
} while (0)
__global__ void fill_packed_bf16(uint32_t* ptr, size_t words, uint32_t seed) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < words) {
    ptr[i] = 0x3f803f80u ^ ((static_cast<uint32_t>(i) + seed * 977u) & 0x000f000fu);
  }
}

__global__ __launch_bounds__(kMainThreads, 1)
void qk_tma_mma_ld_kernel(const __grid_constant__ CUtensorMap q_map,
                          const __grid_constant__ CUtensorMap k_map,
                          const __grid_constant__ CUtensorMap v_map,
                          const __grid_constant__ CUtensorMap o_map,
                          uint32_t* __restrict__ sink,
                          int repeats,
                          int k_tiles,
                          void* __restrict__ output) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)v_map;
  (void)o_map;
  (void)sink;
  (void)repeats;
  (void)k_tiles;
  (void)output;
#else
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
  __shared__ uint64_t s_ready[kPipeCount];
  __shared__ uint64_t pv_done[kPipeCount];
  __shared__ volatile int pipe0_vtma_local_shared;
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
  __shared__ int pipe0_consume_phase_count[2];
#endif
#if ATTENTION_STORE_OUTPUT
  __shared__ float row_sum_partial[kPipeCount][kTileM];
#endif
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kMainWarps];

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
      mbarrier_init(&s_ready[p], kConsumerWarpsPerPipe);
      mbarrier_init(&pv_done[p], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
#if ATTENTION_STORE_OUTPUT
  for (int i = threadIdx.x; i < kPipeCount * kTileM; i += blockDim.x) {
    reinterpret_cast<float*>(row_sum_partial)[i] = 0.0f;
  }
#endif
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[kPipeCount] = {tmem_base, tmem_base + 128u};
  const uint32_t o_taddr[kPipeCount] = {tmem_base + 256u, tmem_base + 384u};
  const uint32_t q_tile_row = static_cast<uint32_t>(blockIdx.x * 64);

  if (warp_id == 0) {
    if (lane0) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, q_tile_row, 0);
    }
  }

  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (warp_id == 0 || warp_id == 1) {
    const int pipe = warp_id;
    const uint32_t idesc = make_qk_idesc();
    uint64_t q_desc[8];
    uint64_t k_desc[8];
    if (lane0) {
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
        k_desc[mma] = make_smem_desc(smem_ptr_u32(k_smem[pipe] + mma * 1024));
      }
    }
    mbarrier_wait(&q_ready, 0);
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += kPipeCount, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      const int k_tile = iter % k_tiles;
      if (local > 0) {
        mbarrier_wait(&qk_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
      if (lane0) {
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                    k_tile * 64, 0);
      }
      mbarrier_wait(&k_ready[pipe], phase);
      if (local > 0) {
        mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
      if (lane0) {
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc, mma != 0);
        }
        tcgen05_commit(&qk_done[pipe]);
      }
    }
  }

  if (warp_id == 2 || warp_id == 3) {
    const int pipe = warp_id - 2;
    const uint32_t idesc = make_qk_idesc();
    uint64_t s_desc[8];
    uint64_t v_desc[8];
    if (lane0) {
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        s_desc[mma] = make_s_smem_desc(smem_ptr_u32(s_smem[pipe] + mma * 1024));
        v_desc[mma] = make_smem_desc(smem_ptr_u32(v_smem[pipe] + mma * 1024));
      }
    }
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += kPipeCount, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      if (local > 0) {
        mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_V_TMA_AFTER_S_READY
      mbarrier_wait(&s_ready[pipe], phase);
#endif
      if (lane0) {
        mbarrier_expect_tx(&v_ready[pipe], kTileBytes);
        tma_load_4d(&v_map, smem_ptr_u32(v_smem[pipe]), &v_ready[pipe], 0, 0,
                    (iter % k_tiles) * 64, 0);
        if (pipe == 0) {
          pipe0_vtma_local_shared = local;
        }
      }
      mbarrier_wait(&v_ready[pipe], phase);
#if !ATTENTION_V_TMA_AFTER_S_READY
      mbarrier_wait(&s_ready[pipe], phase);
#endif
#if ATTENTION_PV_PINGPONG_DEP
      if (pipe == 0) {
        if (local > 0) {
          mbarrier_wait(&pv_done[1], static_cast<uint32_t>((local - 1) & 1));
        }
      } else {
        mbarrier_wait(&pv_done[0], phase);
      }
#endif
#if ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA
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
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                              local != 0 || mma != 0);
        }
        tcgen05_commit(&pv_done[pipe]);
      }
    }
  }

  if (warp_id >= 4 && warp_id < kMainWarps) {
    const int pipe = (warp_id - 4) / kConsumerWarpsPerPipe;
    const int consumer_slot = (warp_id - 4) - pipe * kConsumerWarpsPerPipe;
    const int consumer_warp = consumer_slot;
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += kPipeCount, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      mbarrier_wait(&qk_done[pipe], phase);

      if (local > 0) {
        mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
      if (pipe == 1) {
        const int expected_count = ((local >> 1) + 1) * kConsumerWarpsPerPipe;
        volatile int* consume_count = pipe0_consume_phase_count;
        while (consume_count[phase] < expected_count) {
          asm volatile("" ::: "memory");
        }
      }
#endif

      uint32_t acc0;
      uint32_t acc1;
      const uint32_t row_taddr = p_taddr[pipe] +
                                 (static_cast<uint32_t>(consumer_warp * 32) << 16);
#if ATTENTION_STORE_OUTPUT
      float* row_sum_pipe = output != nullptr ? row_sum_partial[pipe] : nullptr;
#else
      float* row_sum_pipe = nullptr;
#endif
      TCGEN05_LD_X64_WAIT_PACK_STORE(row_taddr, s_smem[pipe], consumer_warp, 0,
                                     &p_done[pipe], false, row_sum_pipe, acc0);
      TCGEN05_LD_X64_WAIT_PACK_STORE(row_taddr + 64u, s_smem[pipe], consumer_warp, 1,
                                     &p_done[pipe], true, row_sum_pipe, acc1);
      read_acc ^= (acc0 ^ acc1) + static_cast<uint32_t>(iter * 17 + warp_id);
      if (lane == 0) {
        mbarrier_arrive(&s_ready[pipe]);
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
        if (pipe == 0) {
          atomicAdd(&pipe0_consume_phase_count[phase], 1);
        }
#endif
      }
    }
  }

#if ATTENTION_STORE_OUTPUT
  if (output != nullptr) {
    const int pipe0_local_count = (repeats + 1) / 2;
    const int pipe1_local_count = repeats / 2;
    if (pipe0_local_count > 0) {
      mbarrier_wait(&pv_done[0], static_cast<uint32_t>((pipe0_local_count - 1) & 1));
    }
    if (pipe1_local_count > 0) {
      mbarrier_wait(&pv_done[1], static_cast<uint32_t>((pipe1_local_count - 1) & 1));
    }
    __syncthreads();
    float* output_smem = reinterpret_cast<float*>(q_smem);
    uint32_t* output_bf16_smem =
        reinterpret_cast<uint32_t*>(output_smem + kTileBf16Elems);
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
    __syncthreads();
    if (warp_id >= 4 && warp_id < 4 + kConsumerWarpsPerPipe) {
      const int consumer_warp = warp_id - 4;
      const int row = consumer_warp * 32 + lane;
      const float denom = row_sum_partial[0][row] + row_sum_partial[1][row];
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
    tma_store_fence();
    __syncthreads();
    if (lane0 && warp_id == 0) {
      tma_store_4d(&o_map, smem_ptr_u32(output_bf16_smem), 0, 0,
                   static_cast<int>(blockIdx.x), 0);
      tma_store_commit_group();
      tma_store_wait_group_read();
    }
    __syncthreads();
  }
#endif

  if (lane == 0) {
    warp_sinks[warp_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats);
#pragma unroll
    for (int w = 0; w < kMainWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    sink[blockIdx.x] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void qk_sequential_cycle_kernel(const __grid_constant__ CUtensorMap q_map,
                                const __grid_constant__ CUtensorMap k_map,
                                CycleTotals* __restrict__ totals,
                                int repeats,
                                int k_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)totals;
  (void)repeats;
  (void)k_tiles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem = q_smem + kTileWords;

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready;
  __shared__ uint64_t p_ready;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long block_q_cycles;
  __shared__ unsigned long long block_k_cycles;
  __shared__ unsigned long long block_mma_cycles;
  __shared__ unsigned long long block_total_cycles;
  __shared__ unsigned long long block_ld_cycles;
  __shared__ unsigned long long step_start;
  __shared__ unsigned long long iter_start;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int wg = warp_id >> 2;
  if (wg == 0 || wg == 2) {
    setmaxnreg_dec_producer();
  } else {
    setmaxnreg_inc_consumer();
  }

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
    mbarrier_init(&k_ready, 1);
    mbarrier_init(&p_ready, 1);
    block_q_cycles = 0;
    block_k_cycles = 0;
    block_mma_cycles = 0;
    block_total_cycles = 0;
    block_ld_cycles = 0;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr = tmem_base;
  const uint32_t q_tile_row = static_cast<uint32_t>(blockIdx.x * 64);
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  uint64_t q_desc[8];
  uint64_t k_desc[8];
  if (warp_id == 2 && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[mma] = make_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
    }
  }

  if (threadIdx.x == 0) step_start = clock64();
  __syncthreads();
  if (warp_id == 0) {
    const bool leader = warp_elect_leader();
    if (leader) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, q_tile_row, 0);
    }
  }
  __syncthreads();
  if (warp_id == 2 && lane == 0) {
    mbarrier_wait(&q_ready, 0);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    block_q_cycles = clock64() - step_start;
  }
  __syncthreads();

  for (int iter = 0; iter < repeats; ++iter) {
    if (threadIdx.x == 0) iter_start = clock64();
    __syncthreads();

    if (threadIdx.x == 0) step_start = clock64();
    __syncthreads();
    if (warp_id == 1) {
      const int k_tile = iter % k_tiles;
      const bool leader = warp_elect_leader();
      if (leader) {
        mbarrier_expect_tx(&k_ready, kTileBytes);
        tma_load_4d(&k_map, smem_ptr_u32(k_smem), &k_ready, 0, 0, k_tile * 64, 0);
      }
    }
    __syncthreads();
    if (warp_id == 2 && lane == 0) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      mbarrier_wait(&k_ready, phase);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      block_k_cycles += clock64() - step_start;
    }
    __syncthreads();

    if (threadIdx.x == 0) step_start = clock64();
    __syncthreads();
    if (warp_id == 2 && lane == 0) {
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(p_taddr, q_desc[mma], k_desc[mma], idesc, mma != 0);
      }
      tcgen05_commit(&p_ready);
    }
    __syncthreads();
    if (warp_id == 3 && lane == 0) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      mbarrier_wait(&p_ready, phase);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      block_mma_cycles += clock64() - step_start;
    }
    __syncthreads();

    if (threadIdx.x == 0) step_start = clock64();
    __syncthreads();
    if (warp_id >= 4 && warp_id <= 7) {
      uint32_t acc = tcgen05_ld_32x32b_x64_acc(p_taddr);
      acc ^= tcgen05_ld_32x32b_x64_acc(p_taddr + 64);
      tcgen05_wait_ld();
      if (lane == 0) {
        read_acc ^= acc + static_cast<uint32_t>(iter * 17 + warp_id);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      block_ld_cycles += clock64() - step_start;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      block_total_cycles += clock64() - iter_start;
    }
    __syncthreads();
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    atomicAdd(&totals->q_tma_cycles, block_q_cycles);
    atomicAdd(&totals->k_tma_cycles, block_k_cycles);
    atomicAdd(&totals->mma_cycles, block_mma_cycles);
    atomicAdd(&totals->ld_cycles, block_ld_cycles);
    atomicAdd(&totals->total_group_cycles, block_total_cycles);
    atomicAdd(&totals->q_samples, 1ull);
    atomicAdd(&totals->group_samples, static_cast<unsigned long long>(repeats));
    atomicAdd(&totals->ld_samples, static_cast<unsigned long long>(repeats));
    atomicAdd(&totals->sink, out);
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void qk_two_warp_trace_kernel(const __grid_constant__ CUtensorMap q_map,
                              const __grid_constant__ CUtensorMap k_map,
                              TraceRecord* __restrict__ records,
                              int repeats,
                              int k_tiles,
                              int trace_warps,
                              bool strict_sequential) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)records;
  (void)repeats;
  (void)k_tiles;
  (void)strict_sequential;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k0_smem = q_smem + kTileWords;
  uint32_t* k1_smem = q_smem + 2 * kTileWords;

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[2];
  __shared__ uint64_t p_ready[2];
  __shared__ uint64_t p_done[2];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long base_clock;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool trace_cta = blockIdx.x == 0;
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
    mbarrier_init(&k_ready[0], 1);
    mbarrier_init(&k_ready[1], 1);
    mbarrier_init(&p_ready[0], 1);
    mbarrier_init(&p_ready[1], 1);
    mbarrier_init(&p_done[0], 4);
    mbarrier_init(&p_done[1], 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[2] = {tmem_base, tmem_base + 128u};
  uint32_t* k_smem[2] = {k0_smem, k1_smem};
  uint64_t q_desc[8];
  uint64_t k_desc[2][8];
  if ((warp_id == 0 || (trace_warps >= 2 && warp_id == 1)) && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[warp_id][mma] = make_smem_desc(smem_ptr_u32(k_smem[warp_id] + mma * 1024));
    }
  }

  if (warp_id == 15) {
    const bool leader = warp_elect_leader();
    if (leader) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    }
  }
  __syncthreads();
  if (warp_id == 15) {
    mbarrier_wait(&q_ready, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    base_clock = clock64();
  }
  __syncthreads();

  if (warp_id == 0 || (trace_warps >= 2 && warp_id == 1)) {
    const int pipe = warp_id;
    if (lane == 0) mbarrier_wait(&q_ready, 0);
    __syncwarp();
    for (int iter = 0; iter < repeats; ++iter) {
      const int k_tile = (iter * 2 + pipe) % k_tiles;
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * trace_warps + pipe);

      if (strict_sequential && lane == 0 && iter > 0) {
        mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((iter - 1) & 1));
      }
      __syncwarp();
      if (trace_cta && lane == 0) {
        records[out_idx].tma_start = 0;
        records[out_idx].tma_end = 0;
        records[out_idx].mma_start = 0xffffffffffffffffull;
        records[out_idx].mma_end = 0xffffffffffffffffull;
        records[out_idx].ld_start = 0xffffffffffffffffull;
        records[out_idx].ld_end = 0;
        records[out_idx].iter = static_cast<unsigned int>(iter);
        records[out_idx].pipe = static_cast<unsigned int>(pipe);
        records[out_idx].warp_id = static_cast<unsigned int>(warp_id);
        records[out_idx].sink = 0;
      }
      const bool leader = warp_elect_leader();
      if (leader) {
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        if (trace_cta) records[out_idx].tma_start = clock64() - base_clock;
        tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                    k_tile * 64, 0);
      }
      __syncwarp();
      mbarrier_wait(&k_ready[pipe], phase);
      __syncwarp();
      if (lane == 0) {
        if (trace_cta) records[out_idx].tma_end = clock64() - base_clock;
        if (!strict_sequential && iter > 0) {
          mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((iter - 1) & 1));
        }
      }

      __syncwarp();
      if (lane == 0) {
        if (trace_cta) records[out_idx].mma_start = clock64() - base_clock;
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[pipe][mma], idesc,
                              mma != 0);
        }
        tcgen05_commit(&p_ready[pipe]);
      }
    }
  }

  if ((warp_id >= 4 && warp_id <= 7) || (trace_warps >= 2 && warp_id >= 8 && warp_id <= 11)) {
    const int pipe = warp_id < 8 ? 0 : 1;
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * trace_warps + pipe);
      mbarrier_wait(&p_ready[pipe], phase);
      __syncwarp();
      if (trace_cta && lane == 0) {
        const unsigned long long ready_clock = clock64() - base_clock;
        atomicMin(&records[out_idx].mma_end, ready_clock);
        atomicMin(&records[out_idx].ld_start, ready_clock);
      }
      uint32_t acc = tcgen05_ld_32x32b_x64_acc(p_taddr[pipe]);
      acc ^= tcgen05_ld_32x32b_x64_acc(p_taddr[pipe] + 64);
      tcgen05_wait_ld();
      read_acc ^= acc + static_cast<uint32_t>(iter * 17 + warp_id);
      __syncwarp();
      unsigned long long ld_done_clock = 0;
      if (trace_cta && lane == 0) {
        ld_done_clock = clock64() - base_clock;
      }
      if (lane == 0) {
        mbarrier_arrive(&p_done[pipe]);
      }
      if (trace_cta && lane == 0) {
        atomicMax(&records[out_idx].ld_end, ld_done_clock);
      }
    }
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    if (trace_cta) records[0].sink ^= out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void qk_overlap_consumer_trace_kernel(const __grid_constant__ CUtensorMap q_map,
                                      const __grid_constant__ CUtensorMap k_map,
                                      TraceRecord* __restrict__ records,
                                      int repeats,
                                      int k_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)records;
  (void)repeats;
  (void)k_tiles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem = q_smem + kTileWords;

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready;
  __shared__ uint64_t p_ready;
  __shared__ uint64_t p_done;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long base_clock;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool trace_cta = blockIdx.x == 0;
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
    mbarrier_init(&k_ready, 1);
    mbarrier_init(&p_ready, 1);
    mbarrier_init(&p_done, 4);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr = tmem_base;
  uint64_t q_desc[8];
  uint64_t k_desc[8];
  if (warp_id == 0 && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[mma] = make_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
    }
  }

  if (warp_id == 15) {
    const bool leader = warp_elect_leader();
    if (leader) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    }
  }
  __syncthreads();
  if (warp_id == 15) {
    mbarrier_wait(&q_ready, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    base_clock = clock64();
  }
  __syncthreads();

  if (warp_id >= 0 && warp_id <= 3) {
    if (warp_id == 0 && lane == 0) mbarrier_wait(&q_ready, 0);
    __syncwarp();
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const int k_tile = iter % k_tiles;
      const unsigned int out_idx = static_cast<unsigned int>(iter);

      if (warp_id == 0) {
        if (lane == 0 && iter > 0) {
          mbarrier_wait(&p_done, static_cast<uint32_t>((iter - 1) & 1));
        }
        __syncwarp();
        if (trace_cta && lane == 0) {
          records[out_idx].tma_start = 0;
          records[out_idx].tma_end = 0;
          records[out_idx].mma_start = 0xffffffffffffffffull;
          records[out_idx].mma_end = 0xffffffffffffffffull;
          records[out_idx].ld_start = 0xffffffffffffffffull;
          records[out_idx].ld_end = 0;
          records[out_idx].iter = static_cast<unsigned int>(iter);
          records[out_idx].pipe = 0;
          records[out_idx].warp_id = 0;
          records[out_idx].sink = 0;
        }
        const bool leader = warp_elect_leader();
        if (leader) {
          mbarrier_expect_tx(&k_ready, kTileBytes);
          if (trace_cta) records[out_idx].tma_start = clock64() - base_clock;
          tma_load_4d(&k_map, smem_ptr_u32(k_smem), &k_ready, 0, 0,
                      k_tile * 64, 0);
        }
        __syncwarp();
        mbarrier_wait(&k_ready, phase);
        __syncwarp();
        if (lane == 0) {
          if (trace_cta) {
            records[out_idx].tma_end = clock64() - base_clock;
            records[out_idx].mma_start = clock64() - base_clock;
          }
#pragma unroll
          for (int mma = 0; mma < kMmasPerTile; ++mma) {
            tcgen05_mma_bf16_ss(p_taddr, q_desc[mma], k_desc[mma], idesc, mma != 0);
          }
          tcgen05_commit(&p_ready);
        }
      }

      mbarrier_wait(&p_ready, phase);
      __syncwarp();
      if (trace_cta && lane == 0) {
        const unsigned long long ready_clock = clock64() - base_clock;
        atomicMin(&records[out_idx].mma_end, ready_clock);
        atomicMin(&records[out_idx].ld_start, ready_clock);
      }
      uint32_t acc = tcgen05_ld_32x32b_x64_acc(p_taddr);
      acc ^= tcgen05_ld_32x32b_x64_acc(p_taddr + 64);
      tcgen05_wait_ld();
      read_acc ^= acc + static_cast<uint32_t>(iter * 17 + warp_id);
      __syncwarp();
      unsigned long long ld_done_clock = 0;
      if (trace_cta && lane == 0) {
        ld_done_clock = clock64() - base_clock;
      }
      if (lane == 0) {
        mbarrier_arrive(&p_done);
      }
      if (trace_cta && lane == 0) {
        atomicMax(&records[out_idx].ld_end, ld_done_clock);
      }
    }
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    if (trace_cta) records[0].sink ^= out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void qk_warp_specialized_trace_kernel(const __grid_constant__ CUtensorMap q_map,
                                      const __grid_constant__ CUtensorMap k_map,
                                      TraceRecord* __restrict__ records,
                                      int repeats,
                                      int k_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)records;
  (void)repeats;
  (void)k_tiles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[2] = {q_smem + kTileWords, q_smem + 2 * kTileWords};

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[2];
  __shared__ uint64_t p_ready[2];
  __shared__ uint64_t p_done[2];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long base_clock;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&p_ready[p], 1);
      mbarrier_init(&p_done[p], 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[2] = {tmem_base, tmem_base + 128u};

  uint64_t q_desc[8];
  uint64_t k_desc[2][8];
  if ((warp_id == 2 || warp_id == 3) && lane == 0) {
    const int pipe = warp_id - 2;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[pipe][mma] = make_smem_desc(smem_ptr_u32(k_smem[pipe] + mma * 1024));
    }
  }

  if (warp_id == 15) {
    const bool leader = warp_elect_leader();
    if (leader) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    }
  }
  __syncthreads();
  if (warp_id == 15) {
    mbarrier_wait(&q_ready, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    base_clock = clock64();
  }
  __syncthreads();

  if (warp_id == 0 || warp_id == 1) {
    const int pipe = warp_id;
    for (int iter = 0; iter < repeats; ++iter) {
      const unsigned int out_idx = static_cast<unsigned int>(iter * 2 + pipe);
      if (lane == 0 && iter > 0) {
        mbarrier_wait(&p_ready[pipe], static_cast<uint32_t>((iter - 1) & 1));
      }
      if (lane == 0) {
        records[out_idx].tma_start = 0;
        records[out_idx].tma_end = 0;
        records[out_idx].mma_start = 0;
        records[out_idx].mma_end = 0xffffffffffffffffull;
        records[out_idx].ld_start = 0xffffffffffffffffull;
        records[out_idx].ld_end = 0;
        records[out_idx].iter = static_cast<unsigned int>(iter);
        records[out_idx].pipe = static_cast<unsigned int>(pipe);
        records[out_idx].warp_id = static_cast<unsigned int>(warp_id);
        records[out_idx].sink = 0;
      }
      __syncwarp();

      const int k_tile = (iter * 2 + pipe) % k_tiles;
      const bool leader = warp_elect_leader();
      if (leader) {
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        records[out_idx].tma_start = clock64() - base_clock;
        tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                    k_tile * 64, 0);
      }
    }
  }

  if ((warp_id == 2 || warp_id == 3) && lane == 0) {
    const int pipe = warp_id - 2;
    mbarrier_wait(&q_ready, 0);
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * 2 + pipe);
      mbarrier_wait(&k_ready[pipe], phase);
      records[out_idx].tma_end = clock64() - base_clock;
      if (iter > 0) {
        mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((iter - 1) & 1));
      }
      records[out_idx].mma_start = clock64() - base_clock;
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[pipe][mma], idesc,
                            mma != 0);
      }
      tcgen05_commit(&p_ready[pipe]);
    }
  }

  if ((warp_id >= 4 && warp_id <= 7) || (warp_id >= 8 && warp_id <= 11)) {
    const int pipe = warp_id < 8 ? 0 : 1;
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * 2 + pipe);
      mbarrier_wait(&p_ready[pipe], phase);
      __syncwarp();
      if (lane == 0) {
        const unsigned long long ready_clock = clock64() - base_clock;
        atomicMin(&records[out_idx].mma_end, ready_clock);
        atomicMin(&records[out_idx].ld_start, ready_clock);
      }
      uint32_t acc = tcgen05_ld_32x32b_x64_acc(p_taddr[pipe]);
      acc ^= tcgen05_ld_32x32b_x64_acc(p_taddr[pipe] + 64);
      tcgen05_wait_ld();
      read_acc ^= acc + static_cast<uint32_t>(iter * 17 + warp_id);
      __syncwarp();
      if (lane == 0) {
        const unsigned long long ld_done_clock = clock64() - base_clock;
        mbarrier_arrive(&p_done[pipe]);
        atomicMax(&records[out_idx].ld_end, ld_done_clock);
      }
    }
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    records[0].sink ^= out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kMainThreads, 1)
void qk_tma_prefetch_trace_kernel(const __grid_constant__ CUtensorMap q_map,
                                  const __grid_constant__ CUtensorMap k_map,
                                  const __grid_constant__ CUtensorMap v_map,
                                  TraceRecord* __restrict__ records,
                                  int repeats,
                                  int k_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)v_map;
  (void)records;
  (void)repeats;
  (void)k_tiles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[2] = {q_smem + kTileWords, q_smem + 2 * kTileWords};
  uint32_t* v_smem[2] = {q_smem + 3 * kTileWords, q_smem + 4 * kTileWords};
  uint32_t* s_smem[2] = {q_smem + 5 * kTileWords, q_smem + 6 * kTileWords};

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[2];
  __shared__ uint64_t qk_done[2];
  __shared__ uint64_t p_done[2];
  __shared__ uint64_t v_ready[2];
  __shared__ uint64_t s_ready[2];
  __shared__ uint64_t pv_done[2];
  __shared__ volatile int pipe0_vtma_local_shared;
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
  __shared__ int pipe0_consume_phase_count[2];
#endif
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long base_clock;
  __shared__ uint32_t warp_sinks[kMainWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

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
    for (int p = 0; p < 2; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&qk_done[p], 1);
      mbarrier_init(&p_done[p], kConsumerWarpsPerPipe);
      mbarrier_init(&v_ready[p], 1);
      mbarrier_init(&s_ready[p], kConsumerWarpsPerPipe);
      mbarrier_init(&pv_done[p], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[2] = {tmem_base, tmem_base + 128u};
  const uint32_t o_taddr[2] = {tmem_base + 256u, tmem_base + 384u};

  uint64_t q_desc[8];
  uint64_t k_desc[8];
  if ((warp_id == 0 || warp_id == 1) && lane == 0) {
    const int pipe = warp_id;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[mma] = make_smem_desc(smem_ptr_u32(k_smem[pipe] + mma * 1024));
    }
  }

  if (warp_id == 0) {
    if (lane0) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    }
  }
  __syncthreads();
  if (warp_id == 0) mbarrier_wait(&q_ready, 0);
  __syncthreads();

  if (threadIdx.x == 0) base_clock = clock64();
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int g = 0; g < repeats; ++g) {
      const int pipe = g & 1;
      records[g].tma_start = 0;
      records[g].tma_end = 0;
      records[g].mma_start = 0xffffffffffffffffull;
      records[g].mma_end = 0xffffffffffffffffull;
      records[g].ld_start = 0xffffffffffffffffull;
      records[g].ld_end = 0;
      for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
        records[g].ld_warp_start[w] = 0xffffffffffffffffull;
        records[g].ld_warp_end[w] = 0;
        records[g].pack_warp_start[w] = 0xffffffffffffffffull;
        records[g].pack_warp_end[w] = 0;
        records[g].st_warp_start[w] = 0xffffffffffffffffull;
        records[g].st_warp_end[w] = 0;
      }
      records[g].pack_start = 0xffffffffffffffffull;
      records[g].pack_end = 0;
      records[g].st_start = 0xffffffffffffffffull;
      records[g].st_end = 0;
      records[g].v_tma_start = 0;
      records[g].v_tma_end = 0;
      records[g].pv_start = 0xffffffffffffffffull;
      records[g].pv_end = 0;
      records[g].iter = static_cast<unsigned int>(g);
      records[g].pipe = static_cast<unsigned int>(pipe);
      records[g].warp_id = static_cast<unsigned int>(pipe);
      records[g].sink = 0;
    }
  }
  __syncthreads();

  if (warp_id == 0 || warp_id == 1) {
    const int pipe = warp_id;
    mbarrier_wait(&q_ready, 0);
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += 2, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      const int k_tile = iter % k_tiles;
      if (local > 0) {
        mbarrier_wait(&qk_done[pipe], static_cast<uint32_t>((local - 1) & 1));
        if (lane0) {
          records[iter - 2].mma_end = clock64() - base_clock;
        }
      }
      if (lane0) {
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        records[iter].tma_start = clock64() - base_clock;
        tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                    k_tile * 64, 0);
      }
      mbarrier_wait(&k_ready[pipe], phase);
      if (lane0) records[iter].tma_end = clock64() - base_clock;
      if (local > 0) {
        mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
      if (lane0) {
        records[iter].mma_start = clock64() - base_clock;
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[mma], idesc,
                              mma != 0);
        }
        tcgen05_commit(&qk_done[pipe]);
      }
    }
    if (local > 0) {
      mbarrier_wait(&qk_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      if (lane0) {
        records[iter - 2].mma_end = clock64() - base_clock;
      }
    }
  }

  if (warp_id == 2 || warp_id == 3) {
    const int pipe = warp_id - 2;
    uint64_t s_desc[8];
    uint64_t v_desc[8];
    if (lane0) {
#pragma unroll
      for (int mma = 0; mma < kMmasPerTile; ++mma) {
        s_desc[mma] = make_s_smem_desc(smem_ptr_u32(s_smem[pipe] + mma * 1024));
        v_desc[mma] = make_smem_desc(smem_ptr_u32(v_smem[pipe] + mma * 1024));
      }
    }
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += 2, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      if (local > 0) {
        mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_V_TMA_AFTER_S_READY
      mbarrier_wait(&s_ready[pipe], phase);
#endif
      if (lane0) {
        mbarrier_expect_tx(&v_ready[pipe], kTileBytes);
        records[iter].v_tma_start = clock64() - base_clock;
        tma_load_4d(&v_map, smem_ptr_u32(v_smem[pipe]), &v_ready[pipe], 0, 0,
                    (iter % k_tiles) * 64, 0);
        if (pipe == 0) {
          pipe0_vtma_local_shared = local;
        }
      }
      mbarrier_wait(&v_ready[pipe], phase);
      if (lane0) records[iter].v_tma_end = clock64() - base_clock;
#if !ATTENTION_V_TMA_AFTER_S_READY
      mbarrier_wait(&s_ready[pipe], phase);
#endif
#if ATTENTION_PV_PINGPONG_DEP
      if (pipe == 0) {
        if (local > 0) {
          mbarrier_wait(&pv_done[1], static_cast<uint32_t>((local - 1) & 1));
        }
      } else {
        mbarrier_wait(&pv_done[0], phase);
      }
#endif
#if ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA
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
        records[iter].pv_start = clock64() - base_clock;
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(o_taddr[pipe], s_desc[mma], v_desc[mma], idesc,
                              local != 0 || mma != 0);
        }
        tcgen05_commit(&pv_done[pipe]);
        mbarrier_wait(&pv_done[pipe], phase);
        records[iter].pv_end = clock64() - base_clock;
      }
    }
  }

  if (warp_id >= 4 && warp_id < kMainWarps) {
    const int pipe = (warp_id - 4) / kConsumerWarpsPerPipe;
    const int consumer_slot = (warp_id - 4) - pipe * kConsumerWarpsPerPipe;
    const int consumer_warp = consumer_slot;
    int iter = pipe;
    int local = 0;
    for (; iter < repeats; iter += 2, ++local) {
      const uint32_t phase = static_cast<uint32_t>(local & 1);
      mbarrier_wait(&qk_done[pipe], phase);
      if (local > 0) {
        mbarrier_wait(&pv_done[pipe], static_cast<uint32_t>((local - 1) & 1));
      }
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
      if (pipe == 1) {
        const int expected_count = ((local >> 1) + 1) * kConsumerWarpsPerPipe;
        volatile int* consume_count = pipe0_consume_phase_count;
        while (consume_count[phase] < expected_count) {
          asm volatile("" ::: "memory");
        }
      }
#endif
      uint32_t acc0;
      uint32_t acc1;
      unsigned long long ld_s0;
      unsigned long long ld_e0;
      unsigned long long pack_s0;
      unsigned long long pack_e0;
      unsigned long long st_s0;
      unsigned long long st_e0;
      unsigned long long ld_s1;
      unsigned long long ld_e1;
      unsigned long long pack_s1;
      unsigned long long pack_e1;
      unsigned long long st_s1;
      unsigned long long st_e1;
      const uint32_t row_taddr = p_taddr[pipe] +
                                 (static_cast<uint32_t>(consumer_warp * 32) << 16);
      TCGEN05_LD_X64_WAIT_PACK_STORE_TRACE(row_taddr, s_smem[pipe], consumer_warp, 0,
                                           &p_done[pipe], false, acc0, ld_s0, ld_e0,
                                           pack_s0, pack_e0, st_s0, st_e0);
      TCGEN05_LD_X64_WAIT_PACK_STORE_TRACE(row_taddr + 64u, s_smem[pipe], consumer_warp,
                                           1, &p_done[pipe], true, acc1, ld_s1, ld_e1,
                                           pack_s1, pack_e1, st_s1, st_e1);
      read_acc ^= (acc0 ^ acc1) + static_cast<uint32_t>(iter * 17 + warp_id);
      if (lane == 0) {
        const int trace_lane0 = consumer_warp * 2;
        const int trace_lane1 = trace_lane0 + 1;
        records[iter].ld_warp_start[trace_lane0] = ld_s0 - base_clock;
        records[iter].ld_warp_end[trace_lane0] = ld_e0 - base_clock;
        records[iter].pack_warp_start[trace_lane0] = pack_s0 - base_clock;
        records[iter].pack_warp_end[trace_lane0] = pack_e0 - base_clock;
        records[iter].st_warp_start[trace_lane0] = st_s0 - base_clock;
        records[iter].st_warp_end[trace_lane0] = st_e0 - base_clock;
        records[iter].ld_warp_start[trace_lane1] = ld_s1 - base_clock;
        records[iter].ld_warp_end[trace_lane1] = ld_e1 - base_clock;
        records[iter].pack_warp_start[trace_lane1] = pack_s1 - base_clock;
        records[iter].pack_warp_end[trace_lane1] = pack_e1 - base_clock;
        records[iter].st_warp_start[trace_lane1] = st_s1 - base_clock;
        records[iter].st_warp_end[trace_lane1] = st_e1 - base_clock;
        mbarrier_arrive(&s_ready[pipe]);
#if ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
        if (pipe == 0) {
          atomicAdd(&pipe0_consume_phase_count[phase], 1);
        }
#endif
        atomicMin(&records[iter].ld_start,
                  static_cast<unsigned long long>(ld_s0 - base_clock));
        atomicMax(&records[iter].ld_end,
                  static_cast<unsigned long long>(ld_e1 - base_clock));
        atomicMin(&records[iter].pack_start,
                  static_cast<unsigned long long>(pack_s0 - base_clock));
        atomicMax(&records[iter].pack_end,
                  static_cast<unsigned long long>(pack_e1 - base_clock));
        atomicMin(&records[iter].st_start,
                  static_cast<unsigned long long>(st_s0 - base_clock));
        atomicMax(&records[iter].st_end,
                  static_cast<unsigned long long>(st_e1 - base_clock));
      }
    }
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base;
#pragma unroll
    for (int w = 0; w < kMainWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    out ^= o_taddr[0] ^ o_taddr[1];
    records[0].sink ^= out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void qk_fused_producer_trace_kernel(const __grid_constant__ CUtensorMap q_map,
                                    const __grid_constant__ CUtensorMap k_map,
                                    TraceRecord* __restrict__ records,
                                    int repeats,
                                    int k_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)records;
  (void)repeats;
  (void)k_tiles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[2] = {q_smem + kTileWords, q_smem + 2 * kTileWords};

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[2];
  __shared__ uint64_t p_ready[2];
  __shared__ uint64_t p_done[2];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long base_clock;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&p_ready[p], 1);
      mbarrier_init(&p_done[p], 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[2] = {tmem_base, tmem_base + 128u};

  uint64_t q_desc[8];
  uint64_t k_desc[2][8];
  if ((warp_id == 0 || warp_id == 1) && lane == 0) {
    const int pipe = warp_id;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[pipe][mma] = make_smem_desc(smem_ptr_u32(k_smem[pipe] + mma * 1024));
    }
  }

  if (warp_id == 15) {
    const bool leader = warp_elect_leader();
    if (leader) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    }
  }
  __syncthreads();
  if (warp_id == 15) {
    mbarrier_wait(&q_ready, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    base_clock = clock64();
  }
  __syncthreads();

  if (warp_id == 0 || warp_id == 1) {
    const int pipe = warp_id;
    if (lane == 0) mbarrier_wait(&q_ready, 0);
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * 2 + pipe);
      if (lane == 0) {
        records[out_idx].tma_start = 0;
        records[out_idx].tma_end = 0;
        records[out_idx].mma_start = 0;
        records[out_idx].mma_end = 0xffffffffffffffffull;
        records[out_idx].ld_start = 0xffffffffffffffffull;
        records[out_idx].ld_end = 0;
        records[out_idx].iter = static_cast<unsigned int>(iter);
        records[out_idx].pipe = static_cast<unsigned int>(pipe);
        records[out_idx].warp_id = static_cast<unsigned int>(warp_id);
        records[out_idx].sink = 0;
      }
      __syncwarp();

      const int k_tile = (iter * 2 + pipe) % k_tiles;
      const bool leader = warp_elect_leader();
      if (leader) {
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        records[out_idx].tma_start = clock64() - base_clock;
        tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                    k_tile * 64, 0);
      }
      if (lane == 0) {
        mbarrier_wait(&k_ready[pipe], phase);
        records[out_idx].tma_end = clock64() - base_clock;

        if (iter > 0) {
          mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((iter - 1) & 1));
        }
        records[out_idx].mma_start = clock64() - base_clock;
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[pipe][mma], idesc,
                              mma != 0);
        }
        tcgen05_commit(&p_ready[pipe]);
      }
    }
  }

  if ((warp_id >= 4 && warp_id <= 7) || (warp_id >= 8 && warp_id <= 11)) {
    const int pipe = warp_id < 8 ? 0 : 1;
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * 2 + pipe);
      mbarrier_wait(&p_ready[pipe], phase);
      __syncwarp();
      if (lane == 0) {
        const unsigned long long ready_clock = clock64() - base_clock;
        atomicMin(&records[out_idx].mma_end, ready_clock);
        atomicMin(&records[out_idx].ld_start, ready_clock);
      }
      uint32_t acc = tcgen05_ld_32x32b_x64_acc(p_taddr[pipe]);
      acc ^= tcgen05_ld_32x32b_x64_acc(p_taddr[pipe] + 64);
      tcgen05_wait_ld();
      read_acc ^= acc + static_cast<uint32_t>(iter * 17 + warp_id);
      __syncwarp();
      if (lane == 0) {
        const unsigned long long ld_done_clock = clock64() - base_clock;
        mbarrier_arrive(&p_done[pipe]);
        atomicMax(&records[out_idx].ld_end, ld_done_clock);
      }
    }
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    records[0].sink ^= out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreads, 1)
void qk_fused3_producer_trace_kernel(const __grid_constant__ CUtensorMap q_map,
                                     const __grid_constant__ CUtensorMap k_map,
                                     TraceRecord* __restrict__ records,
                                     int repeats,
                                     int k_tiles) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)q_map;
  (void)k_map;
  (void)records;
  (void)repeats;
  (void)k_tiles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* q_smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* k_smem[3] = {q_smem + kTileWords, q_smem + 2 * kTileWords,
                         q_smem + 3 * kTileWords};

  __shared__ uint64_t q_ready;
  __shared__ uint64_t k_ready[3];
  __shared__ uint64_t p_ready[3];
  __shared__ uint64_t p_done[3];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ unsigned long long base_clock;
  __shared__ uint32_t warp_sinks[kWarps];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t idesc = make_qk_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (threadIdx.x == 0) {
    mbarrier_init(&q_ready, 1);
#pragma unroll
    for (int p = 0; p < 3; ++p) {
      mbarrier_init(&k_ready[p], 1);
      mbarrier_init(&p_ready[p], 1);
      mbarrier_init(&p_done[p], 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t p_taddr[3] = {tmem_base, tmem_base + 128u, tmem_base + 256u};

  uint64_t q_desc[8];
  uint64_t k_desc[3][8];
  if (warp_id <= 2 && lane == 0) {
    const int pipe = warp_id;
#pragma unroll
    for (int mma = 0; mma < kMmasPerTile; ++mma) {
      q_desc[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc[pipe][mma] = make_smem_desc(smem_ptr_u32(k_smem[pipe] + mma * 1024));
    }
  }

  if (warp_id == 3) {
    const bool leader = warp_elect_leader();
    if (leader) {
      mbarrier_expect_tx(&q_ready, kTileBytes);
      tma_load_4d(&q_map, smem_ptr_u32(q_smem), &q_ready, 0, 0, 0, 0);
    }
  }
  __syncthreads();
  if (warp_id == 3) {
    mbarrier_wait(&q_ready, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    base_clock = clock64();
  }
  __syncthreads();

  if (warp_id <= 2) {
    const int pipe = warp_id;
    if (lane == 0) mbarrier_wait(&q_ready, 0);
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * 3 + pipe);
      if (lane == 0) {
        records[out_idx].tma_start = 0;
        records[out_idx].tma_end = 0;
        records[out_idx].mma_start = 0;
        records[out_idx].mma_end = 0xffffffffffffffffull;
        records[out_idx].ld_start = 0xffffffffffffffffull;
        records[out_idx].ld_end = 0;
        records[out_idx].iter = static_cast<unsigned int>(iter);
        records[out_idx].pipe = static_cast<unsigned int>(pipe);
        records[out_idx].warp_id = static_cast<unsigned int>(warp_id);
        records[out_idx].sink = 0;
      }
      __syncwarp();

      const int k_tile = (iter * 3 + pipe) % k_tiles;
      const bool leader = warp_elect_leader();
      if (leader) {
        mbarrier_expect_tx(&k_ready[pipe], kTileBytes);
        records[out_idx].tma_start = clock64() - base_clock;
        tma_load_4d(&k_map, smem_ptr_u32(k_smem[pipe]), &k_ready[pipe], 0, 0,
                    k_tile * 64, 0);
      }
      if (lane == 0) {
        mbarrier_wait(&k_ready[pipe], phase);
        records[out_idx].tma_end = clock64() - base_clock;

        if (iter > 0) {
          mbarrier_wait(&p_done[pipe], static_cast<uint32_t>((iter - 1) & 1));
        }
        records[out_idx].mma_start = clock64() - base_clock;
#pragma unroll
        for (int mma = 0; mma < kMmasPerTile; ++mma) {
          tcgen05_mma_bf16_ss(p_taddr[pipe], q_desc[mma], k_desc[pipe][mma], idesc,
                              mma != 0);
        }
        tcgen05_commit(&p_ready[pipe]);
      }
    }
  }

  if (warp_id >= 4 && warp_id <= 15) {
    const int pipe = (warp_id - 4) >> 2;
    for (int iter = 0; iter < repeats; ++iter) {
      const uint32_t phase = static_cast<uint32_t>(iter & 1);
      const unsigned int out_idx = static_cast<unsigned int>(iter * 3 + pipe);
      mbarrier_wait(&p_ready[pipe], phase);
      __syncwarp();
      if (lane == 0) {
        const unsigned long long ready_clock = clock64() - base_clock;
        atomicMin(&records[out_idx].mma_end, ready_clock);
        atomicMin(&records[out_idx].ld_start, ready_clock);
      }
      uint32_t acc = tcgen05_ld_32x32b_x64_acc(p_taddr[pipe]);
      acc ^= tcgen05_ld_32x32b_x64_acc(p_taddr[pipe] + 64);
      tcgen05_wait_ld();
      read_acc ^= acc + static_cast<uint32_t>(iter * 17 + warp_id);
      __syncwarp();
      if (lane == 0) {
        const unsigned long long ld_done_clock = clock64() - base_clock;
        mbarrier_arrive(&p_done[pipe]);
        atomicMax(&records[out_idx].ld_end, ld_done_clock);
      }
    }
  }

  if (lane == 0) warp_sinks[warp_id] = read_acc;
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) out ^= warp_sinks[w] + static_cast<uint32_t>(w);
    records[0].sink ^= out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

void parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      args->blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
      args->repeats = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--k-tiles") && i + 1 < argc) {
      args->k_tiles = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--cycle-probe")) {
      args->cycle_probe = true;
    } else if (!std::strcmp(argv[i], "--cycle-trace")) {
      args->cycle_trace = true;
    } else if (!std::strcmp(argv[i], "--strict-cycle-trace")) {
      args->strict_cycle_trace = true;
    } else if (!std::strcmp(argv[i], "--ws-trace")) {
      args->ws_trace = true;
    } else if (!std::strcmp(argv[i], "--tma-prefetch-trace")) {
      args->tma_prefetch_trace = true;
    } else if (!std::strcmp(argv[i], "--fused-producer-trace")) {
      args->fused_producer_trace = true;
    } else if (!std::strcmp(argv[i], "--fused3-producer-trace")) {
      args->fused3_producer_trace = true;
    } else if (!std::strcmp(argv[i], "--overlap-consumer-trace")) {
      args->overlap_consumer_trace = true;
    } else if (!std::strcmp(argv[i], "--store-output")) {
      args->store_output = true;
    } else if (!std::strcmp(argv[i], "--trace-warps") && i + 1 < argc) {
      args->trace_warps = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--trace-launch-blocks") && i + 1 < argc) {
      args->trace_launch_blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [--blocks N] [--repeats N] [--k-tiles N] [--warmup N] "
          "[--iters N] [--cycle-probe] [--cycle-trace] "
          "[--ws-trace] [--store-output] [--csv PATH]\n",
          argv[0]);
      std::exit(0);
    }
  }
  if (args->blocks < 1) args->blocks = 1;
  if (args->repeats < 1) args->repeats = 1;
  if (args->k_tiles < 1) args->k_tiles = 1;
  if (args->trace_warps < 1) args->trace_warps = 1;
  if (args->trace_warps > 2) args->trace_warps = 2;
  if (args->trace_launch_blocks < 1) args->trace_launch_blocks = 1;
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

void encode_output_tma_map(CUtensorMap* map, void* base, uint64_t tiles) {
  const cuuint64_t global_dim[4] = {kTileN, kTileM, tiles, 1};
  const cuuint64_t global_stride[3] = {
      kTileN * sizeof(float),
      static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(float),
      static_cast<cuuint64_t>(kTileN) * kTileM * tiles * sizeof(float)};
  const cuuint32_t box_dim[4] = {kTileN, kTileM, 1, 1};
  const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
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
               "cuTensorMapEncodeTiled(output)");
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

RunResult run_kernel(const Args& args,
                     const CUtensorMap& q_map,
                     const CUtensorMap& k_map,
                     const CUtensorMap& v_map,
                     const CUtensorMap& o_map,
                     uint32_t* sink,
                     void* output,
                     int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_tma_mma_ld_kernel, kMainThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    qk_tma_mma_ld_kernel<<<args.blocks, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, sink, args.repeats, args.k_tiles, output);
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
    qk_tma_mma_ld_kernel<<<args.blocks, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, sink, args.repeats, args.k_tiles, output);
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

RunResult run_cycle_probe(const Args& args,
                          const CUtensorMap& q_map,
                          const CUtensorMap& k_map,
                          CycleTotals* d_totals,
                          int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_sequential_cycle_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_sequential_cycle_kernel, kThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    qk_sequential_cycle_kernel<<<args.blocks, kThreads, kDynamicSmemBytes>>>(
        q_map, k_map, d_totals, args.repeats, args.k_tiles);
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
  CUDA_CHECK(cudaMemset(d_totals, 0, sizeof(CycleTotals)));

  for (int i = 0; i < args.iters; ++i) {
    qk_sequential_cycle_kernel<<<args.blocks, kThreads, kDynamicSmemBytes>>>(
        q_map, k_map, d_totals, args.repeats, args.k_tiles);
    result.error = cudaGetLastError();
    if (result.error != cudaSuccess) {
      result.status = "timed_launch_failed";
      return result;
    }
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "timed_runtime_failed";
    return result;
  }
  return result;
}

RunResult run_cycle_trace(const Args& args,
                          const CUtensorMap& q_map,
                          const CUtensorMap& k_map,
                          TraceRecord* d_records,
                          int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_two_warp_trace_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_two_warp_trace_kernel, kThreads, kDynamicSmemBytes));

  qk_two_warp_trace_kernel<<<args.trace_launch_blocks, kThreads, kDynamicSmemBytes>>>(
      q_map, k_map, d_records, args.repeats, args.k_tiles, args.trace_warps,
      args.strict_cycle_trace);
  result.error = cudaGetLastError();
  if (result.error != cudaSuccess) {
    result.status = "trace_launch_failed";
    return result;
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "trace_runtime_failed";
    return result;
  }
  return result;
}

RunResult run_overlap_consumer_trace(const Args& args,
                                     const CUtensorMap& q_map,
                                     const CUtensorMap& k_map,
                                     TraceRecord* d_records,
                                     int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_overlap_consumer_trace_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_overlap_consumer_trace_kernel, kThreads, kDynamicSmemBytes));

  qk_overlap_consumer_trace_kernel<<<args.trace_launch_blocks, kThreads, kDynamicSmemBytes>>>(
      q_map, k_map, d_records, args.repeats, args.k_tiles);
  result.error = cudaGetLastError();
  if (result.error != cudaSuccess) {
    result.status = "overlap_consumer_trace_launch_failed";
    return result;
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "overlap_consumer_trace_runtime_failed";
    return result;
  }
  return result;
}

RunResult run_ws_trace(const Args& args,
                       const CUtensorMap& q_map,
                       const CUtensorMap& k_map,
                       TraceRecord* d_records,
                       int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_warp_specialized_trace_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_warp_specialized_trace_kernel, kThreads, kDynamicSmemBytes));

  qk_warp_specialized_trace_kernel<<<1, kThreads, kDynamicSmemBytes>>>(
      q_map, k_map, d_records, args.repeats, args.k_tiles);
  result.error = cudaGetLastError();
  if (result.error != cudaSuccess) {
    result.status = "ws_trace_launch_failed";
    return result;
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "ws_trace_runtime_failed";
    return result;
  }
  return result;
}

RunResult run_tma_prefetch_trace(const Args& args,
                                 const CUtensorMap& q_map,
                                 const CUtensorMap& k_map,
                                 const CUtensorMap& v_map,
                                 TraceRecord* d_records,
                                 int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_prefetch_trace_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_tma_prefetch_trace_kernel, kMainThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    qk_tma_prefetch_trace_kernel<<<1, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, d_records, args.repeats, args.k_tiles);
    result.error = cudaGetLastError();
    if (result.error != cudaSuccess) {
      result.status = "tma_prefetch_trace_warmup_launch_failed";
      return result;
    }
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "tma_prefetch_trace_warmup_runtime_failed";
    return result;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    qk_tma_prefetch_trace_kernel<<<1, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, d_records, args.repeats, args.k_tiles);
    result.error = cudaGetLastError();
    if (result.error != cudaSuccess) {
      result.status = "tma_prefetch_trace_launch_failed";
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return result;
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  result.error = cudaEventSynchronize(stop);
  if (result.error != cudaSuccess) {
    result.status = "tma_prefetch_trace_runtime_failed";
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

RunResult run_fused_producer_trace(const Args& args,
                                   const CUtensorMap& q_map,
                                   const CUtensorMap& k_map,
                                   TraceRecord* d_records,
                                   int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_fused_producer_trace_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_fused_producer_trace_kernel, kThreads, kDynamicSmemBytes));

  qk_fused_producer_trace_kernel<<<1, kThreads, kDynamicSmemBytes>>>(
      q_map, k_map, d_records, args.repeats, args.k_tiles);
  result.error = cudaGetLastError();
  if (result.error != cudaSuccess) {
    result.status = "fused_trace_launch_failed";
    return result;
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "fused_trace_runtime_failed";
    return result;
  }
  return result;
}

RunResult run_fused3_producer_trace(const Args& args,
                                    const CUtensorMap& q_map,
                                    const CUtensorMap& k_map,
                                    TraceRecord* d_records,
                                    int* active_ctas_per_sm) {
  RunResult result{};
  CUDA_CHECK(cudaFuncSetAttribute(qk_fused3_producer_trace_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, qk_fused3_producer_trace_kernel, kThreads, kDynamicSmemBytes));

  qk_fused3_producer_trace_kernel<<<1, kThreads, kDynamicSmemBytes>>>(
      q_map, k_map, d_records, args.repeats, args.k_tiles);
  result.error = cudaGetLastError();
  if (result.error != cudaSuccess) {
    result.status = "fused3_trace_launch_failed";
    return result;
  }
  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    result.status = "fused3_trace_runtime_failed";
    return result;
  }
  return result;
}

uint64_t checksum_sink(uint32_t* d_sink, int blocks) {
  uint32_t* h_sink = static_cast<uint32_t*>(std::malloc(static_cast<size_t>(blocks) *
                                                        sizeof(uint32_t)));
  if (!h_sink) {
    std::fprintf(stderr, "malloc failed for sink checksum\n");
    std::exit(1);
  }
  CUDA_CHECK(cudaMemcpy(h_sink, d_sink, static_cast<size_t>(blocks) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  uint64_t checksum = 1469598103934665603ull;
  for (int i = 0; i < blocks; ++i) {
    checksum ^= static_cast<uint64_t>(h_sink[i]) + (static_cast<uint64_t>(i) << 32);
    checksum *= 1099511628211ull;
  }
  std::free(h_sink);
  return checksum;
}

void write_csv(const Args& args,
               int active,
               const RunResult& result,
               uint64_t sink_checksum) {
  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    std::exit(1);
  }
  std::fprintf(
      csv,
      "mode,q_shape,k_shape,v_shape,output_shape,tile_m,tile_n,mma_k,accumulates_per_load,blocks,repeats,k_tiles,"
      "warmup,iters,threads_per_cta,actual_ctas_per_sm,dynamic_smem_bytes,tmem_columns_allocated,"
      "tmem_columns_used,p_tmem_cols,o_tmem_cols,s_storage,elapsed_ms,total_groups,total_mmas,q_tma_GB,k_tma_GB,v_tma_GB,total_tma_GB,"
      "p_read_GB,s_store_GB,q_tma_TBps,k_tma_TBps,v_tma_TBps,total_tma_TBps,p_read_TBps,"
      "s_store_TBps,qk_TFLOP_per_s,pv_TFLOP_per_s,total_TFLOP_per_s,"
      "sink_checksum,status,cuda_error,notes\n");

  const double groups = static_cast<double>(args.blocks) * args.repeats;
  const double total_mmas = groups * kMmasPerTile * 2.0;
  const double q_tma_bytes = static_cast<double>(args.blocks) * kTileBytes;
  const double k_tma_bytes = groups * kTileBytes;
  const double v_tma_bytes = groups * kTileBytes;
  const double p_read_bytes = groups * 2.0 * kTileBytes;  // FP32 128x128 P.
  const double s_store_bytes = groups * kTileBytes;       // packed BF16 128x128 S.
  const double qk_flops = groups * kMmasPerTile * kFlopsPerMma;
  const double pv_flops = qk_flops;
  const char* mode = args.store_output ? "qk_pack_smem_pv_2pipe_bf16_output"
                                       : "qk_pack_smem_pv_2pipe";
  const int threads_per_cta = kMainThreads;
  const int dynamic_smem_bytes = kDynamicSmemBytes;
#if ATTENTION_PV_PINGPONG_DEP
  const char* notes =
      "smem_q_k2_v2_s2_tmem_p2_o2_warp0_1_qk_warp2_3_pv_pipe_consumers_x64_exp2_pack_store_x2_p_done_after_ld_pv_pingpong_dep_bf16_output_when_enabled";
#elif ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA
  const char* notes =
      "smem_q_k2_v2_s2_tmem_p2_o2_warp0_1_qk_warp2_3_pv_pipe_consumers_x64_exp2_pack_store_x2_p_done_after_ld_pipe1_pv_wait_pipe0_next_vtma";
#elif ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
  const char* notes =
      "smem_q_k2_v2_s2_tmem_p2_o2_warp0_1_qk_warp2_3_pv_pipe_consumers_x64_exp2_pack_store_x2_pipe1_ld_wait_pipe0_consume";
#elif ATTENTION_V_TMA_AFTER_S_READY
  const char* notes =
      "smem_q_k2_v2_s2_tmem_p2_o2_warp0_1_qk_warp2_3_pv_pipe_consumers_x64_exp2_pack_store_x2_v_tma_after_s_ready";
#else
  const char* notes =
      "smem_q_k2_v2_s2_tmem_p2_o2_warp0_1_qk_warp2_3_pv_pipe_consumers_x64_exp2_pack_store_x2_p_done_after_ld";
#endif
  std::fprintf(csv,
               "%s,Q[%d,128,128]_bf16,K[%d,128,128]_bf16,V[%d,128,128]_bf16,sink[%d]_u32_checksum,%d,%d,%d,%d,%d,%d,%d,"
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%s,%s,%s\n",
               mode, args.blocks, args.k_tiles, args.k_tiles, args.blocks, kTileM, kTileN, kMmaK, kMmasPerTile, args.blocks,
               args.repeats, args.k_tiles, args.warmup, args.iters, threads_per_cta, active,
               dynamic_smem_bytes, kTmemAllocCols, kTmemUsedCols, 256, 256,
               args.store_output ? "bf16_tma" : "smem", result.ms, groups, total_mmas,
               q_tma_bytes / 1.0e9, k_tma_bytes / 1.0e9, v_tma_bytes / 1.0e9,
               (q_tma_bytes + k_tma_bytes + v_tma_bytes) / 1.0e9,
               p_read_bytes / 1.0e9, s_store_bytes / 1.0e9,
               tbps_from_bytes(q_tma_bytes, result.ms), tbps_from_bytes(k_tma_bytes, result.ms),
               tbps_from_bytes(v_tma_bytes, result.ms),
               tbps_from_bytes(q_tma_bytes + k_tma_bytes + v_tma_bytes, result.ms),
               tbps_from_bytes(p_read_bytes, result.ms), tbps_from_bytes(s_store_bytes, result.ms),
               tflops_from_flops(qk_flops, result.ms), tflops_from_flops(pv_flops, result.ms),
               tflops_from_flops(qk_flops + pv_flops, result.ms),
               static_cast<unsigned long long>(sink_checksum), result.status,
               cudaGetErrorString(result.error), notes);
  std::fclose(csv);
}

void write_cycle_csv(const Args& args,
                     int active,
                     const RunResult& result,
                     const CycleTotals& totals) {
  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    std::exit(1);
  }
  const double q_samples = static_cast<double>(totals.q_samples);
  const double group_samples = static_cast<double>(totals.group_samples);
  const double ld_samples = static_cast<double>(totals.ld_samples);
  const double q_cycles = q_samples > 0.0 ? totals.q_tma_cycles / q_samples : 0.0;
  const double k_cycles = group_samples > 0.0 ? totals.k_tma_cycles / group_samples : 0.0;
  const double mma_cycles = group_samples > 0.0 ? totals.mma_cycles / group_samples : 0.0;
  const double ld_cycles = ld_samples > 0.0 ? totals.ld_cycles / ld_samples : 0.0;
  const double total_cycles =
      group_samples > 0.0 ? totals.total_group_cycles / group_samples : 0.0;
  const double step_sum = k_cycles + mma_cycles + ld_cycles;
  std::fprintf(
      csv,
      "mode,q_shape,k_shape,tile_m,tile_n,mma_k,accumulates_per_load,blocks,repeats,k_tiles,"
      "warmup,iters,threads_per_cta,actual_ctas_per_sm,dynamic_smem_bytes,q_tma_cycles,"
      "k_tma_cycles,mma8_cycles,ld_group_cycles,total_group_cycles,step_sum_cycles,"
      "q_samples,group_samples,ld_samples,status,cuda_error,notes\n");
  std::fprintf(csv,
               "qk_sequential_cycle_probe,Q[%d,128,128]_bf16,K[%d,128,128]_bf16,%d,%d,%d,%d,"
               "%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%llu,%llu,%s,%s,%s\n",
               args.blocks, args.k_tiles, kTileM, kTileN, kMmaK, kMmasPerTile, args.blocks,
               args.repeats, args.k_tiles, args.warmup, args.iters, kThreads, active,
               kDynamicSmemBytes, q_cycles, k_cycles, mma_cycles, ld_cycles, total_cycles,
               step_sum, totals.q_samples, totals.group_samples, totals.ld_samples, result.status,
               cudaGetErrorString(result.error),
               "single_pipeline_sequential_q_tma_once_then_k_tma_mma8_tmem_ldx64x2_cycles");
  std::fclose(csv);
}

void write_trace_csv(const Args& args, const RunResult& result, const TraceRecord* records) {
  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    std::exit(1);
  }
  std::fprintf(csv,
               "mode,elapsed_ms,iter,pipe,warp_id,tma_start,tma_end,tma_cycles,mma_start,mma_end,"
               "mma_cycles,ld_start,ld_end,ld_cycles,pack_start,pack_end,pack_cycles,"
               "st_start,st_end,st_cycles,v_tma_start,v_tma_end,v_tma_cycles,"
               "pv_start,pv_end,pv_cycles,total_start,total_end,total_cycles");
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",ld_warp%d_start,ld_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",pack_warp%d_start,pack_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",st_warp%d_start,st_warp%d_end", w, w);
  }
  std::fprintf(csv, ",status,cuda_error,notes\n");
  const int rows_per_iter = args.fused3_producer_trace
                                ? 3
                                : (args.overlap_consumer_trace
                                       ? 1
                                       : (args.tma_prefetch_trace
                                              ? 1
                                              : ((args.ws_trace || args.fused_producer_trace)
                                                     ? 2
                                                     : args.trace_warps)));
  const char* mode = args.fused3_producer_trace
                         ? "qk_fused3_producer_trace"
                         : (args.overlap_consumer_trace
                                ? "qk_overlap_consumer_trace"
                         : (args.fused_producer_trace
                                ? "qk_fused_producer_trace"
                                : (args.tma_prefetch_trace
                                       ? "qk_pack_smem_pv_2pipe_trace"
                                       : (args.ws_trace
                                              ? "qk_warp_specialized_trace"
                                              : (args.strict_cycle_trace
                                                     ? "qk_strict_sequential_trace"
                                                     : "qk_two_warp_prefetch_trace")))));
  const char* notes =
      args.fused3_producer_trace
          ? "warp0_1_2_tma_plus_mma_warp4_7_pipe0_ld_warp8_11_pipe1_ld_warp12_15_pipe2_ld"
          : (args.overlap_consumer_trace
          ? "warp0_tma_plus_mma_then_warp0_3_ld_same_group_strict_next_tma_after_ld"
                         : (args.fused_producer_trace
          ? "warp0_1_tma_plus_mma_warp4_7_pipe0_ld_warp8_11_pipe1_ld_group_times"
          : (args.tma_prefetch_trace
	                         ? "warp0_1_k_tma_qk_mma_warp2_3_v_tma_pv_mma_pipe_consumers_x64_exp2_pack_store_x2_p_done_after_ld"
#if ATTENTION_PV_PINGPONG_DEP
                           "_pv_pingpong_dep"
#elif ATTENTION_PIPE1_PV_WAIT_PIPE0_VTMA
                           "_pipe1_pv_wait_pipe0_next_vtma"
#elif ATTENTION_PIPE1_LD_WAIT_PIPE0_CONSUME
                           "_pipe1_ld_wait_pipe0_consume"
#elif ATTENTION_V_TMA_AFTER_S_READY
                           "_v_tma_after_s_ready"
#endif
                 : (args.ws_trace
                 ? "warp0_1_tma_warp2_3_mma_warp4_7_pipe0_ld_warp8_11_pipe1_ld_group_times"
	                 : (args.strict_cycle_trace
	                        ? "strict_tma_mma_then_4warp_ld_next_tma_waits_p_done"
	                        : "producer_prefetches_next_k_tma_before_4warp_ld_done")))));
  for (int iter = 0; iter < args.repeats; ++iter) {
    for (int pipe = 0; pipe < rows_per_iter; ++pipe) {
      const TraceRecord& r = records[iter * rows_per_iter + pipe];
      const unsigned long long total_start = r.tma_start;
      const unsigned long long total_end = r.pv_end > r.ld_end ? r.pv_end : r.ld_end;
      const bool has_pack = r.pack_start != 0xffffffffffffffffull && r.pack_end > r.pack_start;
      const bool has_st = r.st_start != 0xffffffffffffffffull && r.st_end > r.st_start;
      const unsigned long long pack_start = has_pack ? r.pack_start : 0;
      const unsigned long long pack_end = has_pack ? r.pack_end : 0;
      const unsigned long long st_start = has_st ? r.st_start : 0;
      const unsigned long long st_end = has_st ? r.st_end : 0;
      const unsigned long long pack_cycles = has_pack ? r.pack_end - r.pack_start : 0;
      const unsigned long long st_cycles = has_st ? r.st_end - r.st_start : 0;
      const unsigned long long v_cycles =
          r.v_tma_end > r.v_tma_start ? r.v_tma_end - r.v_tma_start : 0;
      const unsigned long long pv_cycles =
          r.pv_end > r.pv_start ? r.pv_end - r.pv_start : 0;
      std::fprintf(csv,
                   "%s,%.6f,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,"
                   "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                   "%llu,%llu,%llu,%llu,%llu,%llu",
                   mode, result.ms, r.iter, r.pipe, r.warp_id, r.tma_start, r.tma_end,
                   r.tma_end - r.tma_start, r.mma_start, r.mma_end,
                   r.mma_end - r.mma_start, r.ld_start, r.ld_end, r.ld_end - r.ld_start,
                   pack_start, pack_end, pack_cycles, st_start, st_end, st_cycles,
                   r.v_tma_start, r.v_tma_end, v_cycles, r.pv_start, r.pv_end, pv_cycles,
                   total_start, total_end, total_end - total_start);
      for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
        std::fprintf(csv, ",%llu,%llu", r.ld_warp_start[w], r.ld_warp_end[w]);
      }
      for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
        std::fprintf(csv, ",%llu,%llu", r.pack_warp_start[w], r.pack_warp_end[w]);
      }
      for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
        std::fprintf(csv, ",%llu,%llu", r.st_warp_start[w], r.st_warp_end[w]);
      }
      std::fprintf(csv, ",%s,%s,%s\n", result.status, cudaGetErrorString(result.error), notes);
    }
  }
  std::fclose(csv);
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, &args);
#if ATTENTION_STORE_OUTPUT
  if (args.store_output) {
    args.repeats = args.k_tiles;
  }
#endif

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This benchmark requires SM100+; got sm_%d%d\n", prop.major, prop.minor);
    return 77;
  }
  driver_check(cuInit(0), "cuInit");

  const size_t q_words = static_cast<size_t>(args.blocks) * kTileWords;
  const size_t k_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  const size_t v_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  const int trace_rows_per_iter = args.fused3_producer_trace
                                      ? 3
                                      : (args.overlap_consumer_trace
                                             ? 1
                                             : (args.tma_prefetch_trace
                                                    ? 1
                                                    : ((args.ws_trace || args.fused_producer_trace)
                                                           ? 2
                                                           : args.trace_warps)));
  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_sink = nullptr;
  uint32_t* d_output = nullptr;
  CycleTotals* d_cycle_totals = nullptr;
  TraceRecord* d_trace_records = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, k_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, v_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, args.blocks * sizeof(uint32_t)));
  if (args.store_output) {
#if ATTENTION_STORE_OUTPUT
    const size_t output_words = static_cast<size_t>(args.blocks) * kTileWords;
    CUDA_CHECK(cudaMalloc(&d_output, output_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_output, 0, output_words * sizeof(uint32_t)));
#else
    std::fprintf(stderr,
                 "--store-output requires compiling with -DATTENTION_STORE_OUTPUT=1\n");
    return 2;
#endif
  }
  CUDA_CHECK(cudaMalloc(&d_cycle_totals, sizeof(CycleTotals)));
  CUDA_CHECK(cudaMalloc(&d_trace_records, static_cast<size_t>(args.repeats) *
                                             trace_rows_per_iter *
                                             sizeof(TraceRecord)));

  const int fill_threads = 256;
  fill_packed_bf16<<<static_cast<int>((q_words + fill_threads - 1) / fill_threads), fill_threads>>>(
      d_q, q_words, 3);
  CUDA_CHECK(cudaGetLastError());
  fill_packed_bf16<<<static_cast<int>((k_words + fill_threads - 1) / fill_threads), fill_threads>>>(
      d_k, k_words, 11);
  CUDA_CHECK(cudaGetLastError());
  fill_packed_bf16<<<static_cast<int>((v_words + fill_threads - 1) / fill_threads), fill_threads>>>(
      d_v, v_words, 17);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  encode_atom_tma_map(&q_map, d_q, static_cast<uint64_t>(args.blocks) * 64);
  encode_atom_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  if (d_output) {
    encode_bf16_output_tma_map(&o_map, d_output, static_cast<uint64_t>(args.blocks));
  }

  int active = 0;
  RunResult result{};
  if (args.fused3_producer_trace || args.fused_producer_trace ||
      args.overlap_consumer_trace || args.tma_prefetch_trace ||
      args.ws_trace || args.cycle_trace || args.strict_cycle_trace) {
    CUDA_CHECK(cudaMemset(d_trace_records, 0,
                          static_cast<size_t>(args.repeats) * trace_rows_per_iter *
                              sizeof(TraceRecord)));
    if (args.fused3_producer_trace) {
      result = run_fused3_producer_trace(args, q_map, k_map, d_trace_records, &active);
    } else if (args.overlap_consumer_trace) {
      result = run_overlap_consumer_trace(args, q_map, k_map, d_trace_records, &active);
    } else if (args.fused_producer_trace) {
      result = run_fused_producer_trace(args, q_map, k_map, d_trace_records, &active);
    } else if (args.tma_prefetch_trace) {
      result = run_tma_prefetch_trace(args, q_map, k_map, v_map, d_trace_records, &active);
    } else if (args.ws_trace) {
      result = run_ws_trace(args, q_map, k_map, d_trace_records, &active);
    } else {
      result = run_cycle_trace(args, q_map, k_map, d_trace_records, &active);
    }
    TraceRecord* h_records =
        static_cast<TraceRecord*>(std::malloc(static_cast<size_t>(args.repeats) *
                                              trace_rows_per_iter * sizeof(TraceRecord)));
    if (!h_records) {
      std::fprintf(stderr, "malloc failed for trace records\n");
      return 1;
    }
    CUDA_CHECK(cudaMemcpy(h_records, d_trace_records,
                          static_cast<size_t>(args.repeats) * trace_rows_per_iter *
                              sizeof(TraceRecord),
                          cudaMemcpyDeviceToHost));
    write_trace_csv(args, result, h_records);
    std::free(h_records);
  } else if (args.cycle_probe) {
    CUDA_CHECK(cudaMemset(d_cycle_totals, 0, sizeof(CycleTotals)));
    result = run_cycle_probe(args, q_map, k_map, d_cycle_totals, &active);
    CycleTotals h_totals{};
    CUDA_CHECK(cudaMemcpy(&h_totals, d_cycle_totals, sizeof(CycleTotals),
                          cudaMemcpyDeviceToHost));
    write_cycle_csv(args, active, result, h_totals);
  } else {
    result = run_kernel(args, q_map, k_map, v_map, o_map, d_sink, d_output, &active);
    const uint64_t sink_checksum =
        result.error == cudaSuccess ? checksum_sink(d_sink, args.blocks) : 0ull;
    write_csv(args, active, result, sink_checksum);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_sink));
  if (d_output) CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_cycle_totals));
  CUDA_CHECK(cudaFree(d_trace_records));
  return result.error == cudaSuccess ? 0 : 1;
}
