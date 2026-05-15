#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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

static constexpr int kThreadsPerWarp = 32;
#ifndef MAX_ISSUER_WARPS
#define MAX_ISSUER_WARPS 8
#endif
static constexpr int kMaxIssuerWarps = MAX_ISSUER_WARPS;
#ifndef BLOCK_WARPS
#define BLOCK_WARPS 8
#endif
static constexpr int kBlockWarps = BLOCK_WARPS;
static constexpr int kThreads = kBlockWarps * kThreadsPerWarp;
static constexpr int kAllReadWarps = std::min(kBlockWarps, kMaxIssuerWarps);
static constexpr int kFullConsumeGroups = std::min(4, kBlockWarps / 4);
#ifndef PAIR_BARRIER_GROUPS
#define PAIR_BARRIER_GROUPS 1
#endif
static constexpr int kPairBarrierGroups = std::min(PAIR_BARRIER_GROUPS, kBlockWarps / 4);
#ifndef ISSUE_GROUP_MMAS
#define ISSUE_GROUP_MMAS 0
#endif
static constexpr int kIssueGroupMmas = ISSUE_GROUP_MMAS;
#ifndef READ_GROUP_SYNC_MODE
#define READ_GROUP_SYNC_MODE 0
#endif
static constexpr int kReadGroupSyncMode = READ_GROUP_SYNC_MODE;
#ifndef ALL_READ_ALLOC_COLS
#define ALL_READ_ALLOC_COLS 32
#endif
static constexpr int kAllReadAllocCols = ALL_READ_ALLOC_COLS;
#ifndef FULL_CONSUME_LD_WIDTH
#define FULL_CONSUME_LD_WIDTH 128
#endif
static constexpr int kFullConsumeLdWidth = FULL_CONSUME_LD_WIDTH;
#ifndef FOUR_BUFFER_SYNC_AFTER_ISSUE
#define FOUR_BUFFER_SYNC_AFTER_ISSUE 0
#endif
static constexpr int kFourBufferSyncAfterIssue = FOUR_BUFFER_SYNC_AFTER_ISSUE;
#ifndef FOUR_BUFFER_PRODUCER_WARP
#define FOUR_BUFFER_PRODUCER_WARP 8
#endif
static constexpr int kFourBufferProducerWarp = FOUR_BUFFER_PRODUCER_WARP;
#ifndef FOUR_BUFFER_ISSUE_MODE
#define FOUR_BUFFER_ISSUE_MODE 1
#endif
static constexpr int kFourBufferIssueMode = FOUR_BUFFER_ISSUE_MODE;
#ifndef FOUR_BUFFER_READS_PER_GROUP
#define FOUR_BUFFER_READS_PER_GROUP 1
#endif
static constexpr int kFourBufferReadsPerGroup = FOUR_BUFFER_READS_PER_GROUP;
#ifndef FOUR_BUFFER_FIRST_LOCAL
#define FOUR_BUFFER_FIRST_LOCAL 0
#endif
static constexpr int kFourBufferFirstLocal = FOUR_BUFFER_FIRST_LOCAL;
#ifndef MMA_M
#define MMA_M 128
#endif
static constexpr int kMmaM = MMA_M;
#ifndef MMA_N
#define MMA_N 128
#endif
static constexpr int kMmaN = MMA_N;
static constexpr int kMmaK = 16;
static constexpr int kTmemSlotStrideCols = std::max(8, kMmaN / 4);
static constexpr size_t kStaticSmemBudgetBytes = 20 * 1024;
static constexpr double kFlopsPerMma = 2.0 * kMmaM * kMmaN * kMmaK;

struct Args {
  int blocks = 512;
  int repeats = 4096;
  int group_mmas = 8;
  int warmup = 5;
  int iters = 20;
  const char* csv = "mma_throughput_bench.csv";
  bool issue_only = false;
  bool occupancy_only = false;
  bool try_acc16 = false;
  bool a_tmem_only = false;
  bool rotate_accum_only = false;
  bool read_every8_only = false;
  bool full_consume_every8_only = false;
  bool full_consume_two_groups_only = false;
  bool full_consume_four_buffers_only = false;
  bool full_consume_four_warpgroups_only = false;
  bool full_consume_quad512_only = false;
  bool full_consume_double_buffer_only = false;
  bool full_consume_two_db_only = false;
  bool full_consume_pair_barrier_only = false;
  bool split_read_every8_only = false;
  bool all8_read_every8_only = false;
  bool single_simple_only = false;
  bool pure_issue_only = false;
  bool latency_only = false;
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__host__ __device__ __forceinline__ uint64_t make_smem_desc(
    uint32_t matrix_start_addr,
    uint32_t leading_dim_byte_offset,
    uint32_t stride_dim_byte_offset,
    uint32_t swizzle_mode = 2) {
  const uint32_t matrix_start_aligned = matrix_start_addr & ~0xFu;
  const uint32_t lead_enc = (leading_dim_byte_offset & 0x3ffffu) >> 4;
  const uint32_t stride_enc = (stride_dim_byte_offset & 0x3ffffu) >> 4;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>(lead_enc) << 16;
  desc |= static_cast<uint64_t>(stride_enc) << 32;
  desc |= static_cast<uint64_t>(0x1u) << 46;
  desc |= static_cast<uint64_t>(0xB0u) << 53;
  desc |= static_cast<uint64_t>(swizzle_mode & 0x7u) << 61;
  return desc;
}

__host__ __device__ __forceinline__ uint32_t make_bf16_idesc(bool acc32) {
  uint32_t desc = 0;
  desc |= static_cast<uint32_t>(acc32 ? 1u : 0u) << 4;  // C format: 0=f16/acc16, 1=f32.
  desc |= 1u << 7;                                      // A format: BF16.
  desc |= 1u << 10;                                     // B format: BF16.
  desc |= static_cast<uint32_t>(kMmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kMmaM >> 4) << 24;
  return desc;
}

template <int MmaM, int MmaN>
__host__ __device__ __forceinline__ uint32_t make_bf16_idesc_shape(bool acc32) {
  uint32_t desc = 0;
  desc |= static_cast<uint32_t>(acc32 ? 1u : 0u) << 4;  // C format: 0=f16/acc16, 1=f32.
  desc |= 1u << 7;                                      // A format: BF16.
  desc |= 1u << 10;                                     // B format: BF16.
  desc |= static_cast<uint32_t>(MmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(MmaM >> 4) << 24;
  return desc;
}

template <int MmaN>
__host__ __device__ __forceinline__ constexpr int tmem_slot_stride_cols_for() {
  return MmaN / 4 > 8 ? MmaN / 4 : 8;
}

template <int MmaM, int MmaN>
__host__ __device__ __forceinline__ constexpr double flops_per_mma_for() {
  return 2.0 * static_cast<double>(MmaM) * static_cast<double>(MmaN) *
         static_cast<double>(kMmaK);
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
#else
  (void)barrier; (void)count;
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
  (void)barrier; (void)phase;
#endif
}

__device__ __forceinline__ uint32_t tcgen05_alloc_128cols(uint32_t* smem_out_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t smem_addr = smem_ptr_u32(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;"
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

__device__ __forceinline__ uint32_t tcgen05_alloc_32cols(uint32_t* smem_out_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t smem_addr = smem_ptr_u32(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
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

__device__ __forceinline__ void tcgen05_dealloc_128cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;"
               :: "r"(taddr)
               : "memory");
#else
  (void)taddr;
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

__device__ __forceinline__ void tcgen05_dealloc_32cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
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
  (void)d_taddr; (void)a_desc; (void)b_desc; (void)idesc; (void)input_d;
#endif
}

__device__ __forceinline__ void tcgen05_mma_bf16_ts(uint32_t d_taddr,
                                                    uint32_t a_taddr,
                                                    uint64_t b_desc,
                                                    uint32_t idesc,
                                                    bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
      "{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, %6, %7, %8}, pred; }"
      :: "r"(d_taddr), "r"(a_taddr), "l"(b_desc), "r"(idesc), "r"(p),
         "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
#else
  (void)d_taddr; (void)a_taddr; (void)b_desc; (void)idesc; (void)input_d;
#endif
}

__device__ __forceinline__ void tcgen05_ld_32x32b_x32(uint32_t (&dst)[32], uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
               "{%0, %1, %2, %3, %4, %5, %6, %7, "
               "%8, %9, %10, %11, %12, %13, %14, %15, "
               "%16, %17, %18, %19, %20, %21, %22, %23, "
               "%24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
               : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                 "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7]),
                 "=r"(dst[8]), "=r"(dst[9]), "=r"(dst[10]), "=r"(dst[11]),
                 "=r"(dst[12]), "=r"(dst[13]), "=r"(dst[14]), "=r"(dst[15]),
                 "=r"(dst[16]), "=r"(dst[17]), "=r"(dst[18]), "=r"(dst[19]),
                 "=r"(dst[20]), "=r"(dst[21]), "=r"(dst[22]), "=r"(dst[23]),
                 "=r"(dst[24]), "=r"(dst[25]), "=r"(dst[26]), "=r"(dst[27]),
                 "=r"(dst[28]), "=r"(dst[29]), "=r"(dst[30]), "=r"(dst[31])
               : "r"(taddr)
               : "memory");
#else
  (void)taddr;
  for (int i = 0; i < 32; ++i) dst[i] = 0;
#endif
}

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x128_acc(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t acc;
  asm volatile(
      "{ .reg .b32 r<128>; .reg .b32 acc; "
      "tcgen05.ld.sync.aligned.32x32b.x128.b32 "
      "{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, "
      "r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31, "
      "r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45, r46, r47, "
      "r48, r49, r50, r51, r52, r53, r54, r55, r56, r57, r58, r59, r60, r61, r62, r63, "
      "r64, r65, r66, r67, r68, r69, r70, r71, r72, r73, r74, r75, r76, r77, r78, r79, "
      "r80, r81, r82, r83, r84, r85, r86, r87, r88, r89, r90, r91, r92, r93, r94, r95, "
      "r96, r97, r98, r99, r100, r101, r102, r103, r104, r105, r106, r107, r108, r109, r110, r111, "
      "r112, r113, r114, r115, r116, r117, r118, r119, r120, r121, r122, r123, r124, r125, r126, r127}, [%1]; "
      "xor.b32 acc, r0, r31; "
      "xor.b32 acc, acc, r63; "
      "xor.b32 acc, acc, r95; "
      "xor.b32 %0, acc, r127; }"
      : "=r"(acc)
      : "r"(taddr)
      : "memory");
  return acc;
#else
  (void)taddr;
  return 0;
#endif
}

__device__ __forceinline__ void init_bf16_smem(uint32_t* smem_words, int words) {
  for (int i = threadIdx.x; i < words; i += blockDim.x) {
    const uint32_t x = 0x3f803f80u ^ static_cast<uint32_t>((i + 17 * blockIdx.x) & 0x000f000fu);
    smem_words[i] = x;
  }
}

template <bool Acc32, bool MultiIssuer, bool Use32Cols>
__global__ __launch_bounds__(kThreads, 8) void mma_issue_kernel(uint32_t* __restrict__ sink,
                                                                int repeats,
                                                                int issuer_warps) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats; (void)issuer_warps;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, kPairBarrierGroups);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr =
        Use32Cols ? tcgen05_alloc_32cols(&tmem_smem) : tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);

  if constexpr (MultiIssuer) {
    if (lane == 0 && warp_id < issuer_warps) {
      const uint32_t d_taddr = tmem_base + static_cast<uint32_t>(warp_id * kTmemSlotStrideCols);
#pragma unroll 1
      for (int i = 0; i < repeats; ++i) {
        tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, i != 0);
      }
    }
  } else {
    if (threadIdx.x == 0) {
#pragma unroll 1
      for (int i = 0; i < repeats * issuer_warps; ++i) {
        const uint32_t d_taddr =
            tmem_base + static_cast<uint32_t>((i % issuer_warps) * kTmemSlotStrideCols);
        tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, i >= issuer_warps);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ static_cast<uint32_t>(repeats * issuer_warps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    if constexpr (Use32Cols) {
      tcgen05_dealloc_32cols(tmem_base);
    } else {
      tcgen05_dealloc_128cols(tmem_base);
    }
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32, bool Accumulate>
__global__ __launch_bounds__(kThreads, 8) void mma_issue_single_simple_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);

  if (threadIdx.x == 0 && repeats > 0) {
    tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, false);
#pragma unroll 1
    for (int i = 1; i < repeats; ++i) {
      tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, Accumulate);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ static_cast<uint32_t>(repeats);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_128cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32, bool MultiIssuer, int PredMode>
__global__ __launch_bounds__(kThreads, 8) void mma_issue_clock_kernel(
    uint32_t* __restrict__ sink,
    unsigned long long* __restrict__ issue_cycles,
    int repeats,
    int issuer_warps) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)issue_cycles; (void)repeats; (void)issuer_warps;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);

  const bool participates =
      MultiIssuer ? (lane == 0 && warp_id < issuer_warps) : (threadIdx.x == 0);
  const int logical_issuer = MultiIssuer ? warp_id : 0;
  if (participates) {
    const uint32_t d_taddr =
        tmem_base + static_cast<uint32_t>(logical_issuer * kTmemSlotStrideCols);
    const unsigned long long start = clock64();
    if constexpr (PredMode == 2) {
      if (repeats > 0) {
        tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, false);
#pragma unroll 1
        for (int i = 1; i < repeats; ++i) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, true);
        }
      }
    } else {
      constexpr bool kInputD = PredMode == 1;
#pragma unroll 1
      for (int i = 0; i < repeats; ++i) {
        tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, kInputD);
      }
    }
    const unsigned long long stop = clock64();
    issue_cycles[static_cast<size_t>(blockIdx.x) * kMaxIssuerWarps + logical_issuer] =
        stop - start;
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ static_cast<uint32_t>(repeats * issuer_warps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_128cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32, bool DoMma, bool InputD>
__global__ __launch_bounds__(kThreads, 8) void mma_latency_clock_kernel(
    uint32_t* __restrict__ sink,
    unsigned long long* __restrict__ latency_cycles,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)latency_cycles; (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);

  uint32_t phase = 0;
  if (threadIdx.x == 0) {
    if constexpr (DoMma && InputD) {
      tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, false);
      tcgen05_commit(&mma_barrier);
      mbarrier_wait(&mma_barrier, phase);
      phase ^= 1u;
    }

    const unsigned long long start = clock64();
#pragma unroll 1
    for (int i = 0; i < repeats; ++i) {
      if constexpr (DoMma) {
        tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, InputD);
      }
      tcgen05_commit(&mma_barrier);
      mbarrier_wait(&mma_barrier, phase);
      phase ^= 1u;
    }
    const unsigned long long stop = clock64();
    latency_cycles[blockIdx.x] = stop - start;
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ static_cast<uint32_t>(repeats);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_128cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32, bool Use32Cols>
__global__ __launch_bounds__(kThreads, 8) void mma_issue_a_tmem_kernel(uint32_t* __restrict__ sink,
                                                                       int repeats,
                                                                       int issuer_warps) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats; (void)issuer_warps;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_a_smem;
  __shared__ uint32_t tmem_c_smem;
  __shared__ uint32_t tmem_a_shared;
  __shared__ uint32_t tmem_c_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t a_taddr =
        Use32Cols ? tcgen05_alloc_32cols(&tmem_a_smem) : tcgen05_alloc_128cols(&tmem_a_smem);
    const uint32_t c_taddr =
        Use32Cols ? tcgen05_alloc_32cols(&tmem_c_smem) : tcgen05_alloc_128cols(&tmem_c_smem);
    if ((threadIdx.x & 31) == 0) {
      tmem_a_shared = a_taddr;
      tmem_c_shared = c_taddr;
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t a_taddr = tmem_a_shared;
  const uint32_t c_taddr = tmem_c_shared;
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);

  if (lane == 0 && warp_id < issuer_warps) {
    const uint32_t d_taddr = c_taddr + static_cast<uint32_t>(warp_id * kTmemSlotStrideCols);
#pragma unroll 1
    for (int i = 0; i < repeats; ++i) {
      tcgen05_mma_bf16_ts(d_taddr, a_taddr, b_desc, idesc, i != 0);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = a_taddr ^ c_taddr ^ static_cast<uint32_t>(repeats * issuer_warps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    if constexpr (Use32Cols) {
      tcgen05_dealloc_32cols(a_taddr);
      tcgen05_dealloc_32cols(c_taddr);
    } else {
      tcgen05_dealloc_128cols(a_taddr);
      tcgen05_dealloc_128cols(c_taddr);
    }
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 8) void mma_issue_rotate_accum_kernel(
    uint32_t* __restrict__ sink,
    int repeats,
    int issuer_warps) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats; (void)issuer_warps;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem[kMaxIssuerWarps];
  __shared__ uint32_t tmem_base_shared[kMaxIssuerWarps];

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int slot = 0; slot < kAllReadWarps; ++slot) {
      if (slot < issuer_warps) {
        const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem[slot]);
        if ((threadIdx.x & 31) == 0) tmem_base_shared[slot] = taddr;
      }
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  const int slots_per_warp = std::max(1, 128 / kTmemSlotStrideCols);

  if (lane == 0 && warp_id < issuer_warps) {
    const uint32_t warp_tmem_base = tmem_base_shared[warp_id];
#pragma unroll 1
    for (int i = 0; i < repeats; ++i) {
      const int slot = (i / 8) % slots_per_warp;
      const uint32_t d_taddr = warp_tmem_base + static_cast<uint32_t>(slot * kTmemSlotStrideCols);
      tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, (i & 7) != 0);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base_shared[0] ^ static_cast<uint32_t>(repeats * issuer_warps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int slot = 0; slot < kMaxIssuerWarps; ++slot) {
      if (slot < issuer_warps) {
        tcgen05_dealloc_128cols(tmem_base_shared[slot]);
      }
    }
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_read_every8_kernel(
    uint32_t* __restrict__ sink,
    int repeats,
    int issuer_warps,
    int group_mmas) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats; (void)issuer_warps; (void)group_mmas;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : group_mmas;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (lane == 0 && warp_id < issuer_warps) {
      const uint32_t d_taddr =
          tmem_base + static_cast<uint32_t>(warp_id * kTmemSlotStrideCols);
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll 1
        for (int k = 0; k < group_mmas; ++k) {
          const int i = group * group_mmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 3) {
      mbarrier_wait(&mma_barrier, group & 1);
    } else {
      if (lane == 0) mbarrier_wait(&mma_barrier, group & 1);
      __syncwarp();
    }
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 1) {
      __syncthreads();
    }

    if (warp_id < issuer_warps) {
      const uint32_t d_taddr =
          tmem_base + static_cast<uint32_t>(warp_id * kTmemSlotStrideCols);
      uint32_t r[32];
      tcgen05_ld_32x32b_x32(r, d_taddr);
      read_acc ^= r[(group + lane) & 31];
      tcgen05_wait_ld();
    }
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 1) {
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^ static_cast<uint32_t>(repeats * issuer_warps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_128cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_every8_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (threadIdx.x == 0) {
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = group * 8 + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
    mbarrier_wait(&mma_barrier, group & 1);
    __syncthreads();

    if (warp_id < 4) {
      if constexpr (kFullConsumeLdWidth == 128) {
#pragma unroll 1
        for (int col = 0; col < 128; col += 4) {
          read_acc ^= tcgen05_ld_32x32b_x128_acc(tmem_base + static_cast<uint32_t>(col));
          tcgen05_wait_ld();
        }
      } else {
        uint32_t r[32];
#pragma unroll 1
        for (int col = 0; col < 128; ++col) {
          tcgen05_ld_32x32b_x32(r, tmem_base + static_cast<uint32_t>(col));
          read_acc ^= r[(col + group + lane) & 31];
          tcgen05_wait_ld();
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^ static_cast<uint32_t>(repeats);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_128cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_two_groups_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem[2];
  __shared__ uint32_t tmem_base_shared[2];

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr0 = tcgen05_alloc_128cols(&tmem_smem[0]);
    const uint32_t taddr1 = tcgen05_alloc_128cols(&tmem_smem[1]);
    if ((threadIdx.x & 31) == 0) {
      tmem_base_shared[0] = taddr0;
      tmem_base_shared[1] = taddr1;
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group_id = warp_id >> 2;
  const uint32_t tmem_base = group_id < 2 ? tmem_base_shared[group_id] : 0;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (lane == 0 && (warp_id == 0 || warp_id == 4)) {
      const uint32_t issue_tmem_base = tmem_base_shared[warp_id >> 2];
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = group * 8 + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
    mbarrier_wait(&mma_barrier, group & 1);
    __syncthreads();

    if (warp_id < 8) {
#pragma unroll 1
      for (int col = 0; col < 128; col += 4) {
        read_acc ^= tcgen05_ld_32x32b_x128_acc(tmem_base + static_cast<uint32_t>(col));
        tcgen05_wait_ld();
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base_shared[0] ^ tmem_base_shared[1] ^ read_acc ^
                       static_cast<uint32_t>(repeats * 2);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    tcgen05_dealloc_128cols(tmem_base_shared[0]);
    tcgen05_dealloc_128cols(tmem_base_shared[1]);
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_four_buffers_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier[4];
  __shared__ uint32_t tmem_smem[4];
  __shared__ uint32_t tmem_base_shared[4];

  if (threadIdx.x == 0) {
    if constexpr (kFourBufferIssueMode == 1) {
      mbarrier_init(&mma_barrier[0], 4);
      for (int b = 1; b < 4; ++b) {
        mbarrier_init(&mma_barrier[b], 1);
      }
    } else {
      for (int b = 0; b < 4; ++b) {
        mbarrier_init(&mma_barrier[b], 1);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem[b]);
      if ((threadIdx.x & 31) == 0) tmem_base_shared[b] = taddr;
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if constexpr (kFourBufferIssueMode == 1) {
      if (lane == 0 && warp_id < 4) {
        const int b = warp_id;
        const uint32_t tmem_base = tmem_base_shared[b];
        if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
          for (int k = 0; k < kIssueGroupMmas; ++k) {
            const int i = group * kIssueGroupMmas + k;
            if (i < repeats) {
              tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
            }
          }
        } else {
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int i = group * 8 + k;
            if (i < repeats) {
              tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
            }
          }
        }
        tcgen05_commit(&mma_barrier[0]);
      }
    } else if (lane == 0 && warp_id == kFourBufferProducerWarp) {
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t tmem_base = tmem_base_shared[b];
        if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
          for (int k = 0; k < kIssueGroupMmas; ++k) {
            const int i = group * kIssueGroupMmas + k;
            if (i < repeats) {
              tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
            }
          }
        } else {
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int i = group * 8 + k;
            if (i < repeats) {
              tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
            }
          }
        }
        tcgen05_commit(&mma_barrier[b]);
      }
    }

    if constexpr (kFourBufferIssueMode == 1) {
      __syncthreads();
      mbarrier_wait(&mma_barrier[0], group & 1);
      __syncthreads();
    } else if constexpr (kFourBufferSyncAfterIssue) {
      __syncthreads();
    }

    if (warp_id < 8) {
      const int consumer_group = warp_id >> 2;
      const int first_buffer = consumer_group * 2;
#pragma unroll
      for (int local = kFourBufferFirstLocal;
           local < kFourBufferFirstLocal + kFourBufferReadsPerGroup; ++local) {
        const int buffer = first_buffer + local;
        if constexpr (kFourBufferIssueMode != 1) {
          mbarrier_wait(&mma_barrier[buffer], group & 1);
        }
        const uint32_t tmem_base = tmem_base_shared[buffer];
#pragma unroll 1
        for (int col = 0; col < 128; col += 4) {
          read_acc ^= tcgen05_ld_32x32b_x128_acc(tmem_base + static_cast<uint32_t>(col));
          tcgen05_wait_ld();
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base_shared[0] ^ tmem_base_shared[1] ^
                       tmem_base_shared[2] ^ tmem_base_shared[3] ^ read_acc ^
                       static_cast<uint32_t>(repeats * 4);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      tcgen05_dealloc_128cols(tmem_base_shared[b]);
    }
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_four_warpgroups_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier[4];
  __shared__ uint32_t tmem_smem[4];
  __shared__ uint32_t tmem_base_shared[4];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      mbarrier_init(&mma_barrier[g], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem[g]);
      if ((threadIdx.x & 31) == 0) tmem_base_shared[g] = taddr;
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warpgroup = warp_id >> 2;
  const int warp_in_group = warp_id & 3;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (warpgroup < 4 && warp_in_group == 0 && lane == 0) {
      const uint32_t tmem_base = tmem_base_shared[warpgroup];
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = group * 8 + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
      tcgen05_commit(&mma_barrier[warpgroup]);
    }

    __syncthreads();
    if (warpgroup < 4) {
      mbarrier_wait(&mma_barrier[warpgroup], group & 1);
      const uint32_t tmem_base = tmem_base_shared[warpgroup];
      read_acc ^= tcgen05_ld_32x32b_x128_acc(tmem_base);
      tcgen05_wait_ld();
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base_shared[0] ^ tmem_base_shared[1] ^
                       tmem_base_shared[2] ^ tmem_base_shared[3] ^ read_acc ^
                       static_cast<uint32_t>(repeats * 4);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      tcgen05_dealloc_128cols(tmem_base_shared[g]);
    }
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_quad512_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, kFullConsumeGroups);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int group_id = warp_id >> 2;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (lane == 0 && (warp_id % 4 == 0) && ((warp_id >> 2) < kFullConsumeGroups)) {
      const uint32_t issue_tmem_base =
          tmem_base + static_cast<uint32_t>((warp_id >> 2) * 128);
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = group * 8 + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
      tcgen05_commit(&mma_barrier);
    }

    __syncthreads();
    mbarrier_wait(&mma_barrier, group & 1);
    __syncthreads();

    if (warp_id < kFullConsumeGroups * 4) {
      const uint32_t read_tmem_base = tmem_base + static_cast<uint32_t>(group_id * 128);
#pragma unroll 1
      for (int col = 0; col < 128; col += 4) {
        read_acc ^= tcgen05_ld_32x32b_x128_acc(read_tmem_base + static_cast<uint32_t>(col));
        tcgen05_wait_ld();
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^ static_cast<uint32_t>(repeats * 4);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_double_buffer_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier[2];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier[0], 1);
    mbarrier_init(&mma_barrier[1], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  const int groups = (repeats + active_group_mmas - 1) / active_group_mmas;

  auto issue_group = [&](int group) {
    const int buffer = group & 1;
    const uint32_t issue_tmem_base = tmem_base + static_cast<uint32_t>(buffer * 128);
    if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
      for (int k = 0; k < kIssueGroupMmas; ++k) {
        const int i = group * kIssueGroupMmas + k;
        if (i < repeats) {
          tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
        }
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int i = group * 8 + k;
        if (i < repeats) {
          tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
        }
      }
    }
    tcgen05_commit(&mma_barrier[buffer]);
  };

  if (lane == 0 && warp_id == 4 && groups > 0) {
    issue_group(0);
  }
  __syncthreads();

  for (int group = 0; group < groups; ++group) {
    if (lane == 0 && warp_id == 4 && group + 1 < groups) {
      issue_group(group + 1);
    }

    if (warp_id < 4) {
      const int buffer = group & 1;
      const uint32_t read_tmem_base = tmem_base + static_cast<uint32_t>(buffer * 128);
      mbarrier_wait(&mma_barrier[buffer], (group >> 1) & 1);
      __syncwarp();
#pragma unroll 1
      for (int col = 0; col < 128; col += 4) {
        read_acc ^= tcgen05_ld_32x32b_x128_acc(read_tmem_base + static_cast<uint32_t>(col));
        tcgen05_wait_ld();
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^ static_cast<uint32_t>(repeats);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_pair_barrier_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  constexpr int kPairGroups = 2;
  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  const int groups = (repeats + active_group_mmas - 1) / active_group_mmas;
  const int pairs = (groups + kPairGroups - 1) / kPairGroups;

  for (int pair = 0; pair < pairs; ++pair) {
    if ((threadIdx.x & 31) == 0 && (warp_id % 4 == 0) &&
        ((warp_id >> 2) < kPairBarrierGroups)) {
      const int issue_group_id = warp_id >> 2;
#pragma unroll
      for (int local_group = 0; local_group < kPairGroups; ++local_group) {
        const int group = pair * kPairGroups + local_group;
        if (group < groups) {
          const uint32_t issue_tmem_base =
              tmem_base + static_cast<uint32_t>(issue_group_id * 256 + local_group * 128);
          if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
            for (int k = 0; k < kIssueGroupMmas; ++k) {
              const int i = group * kIssueGroupMmas + k;
              if (i < repeats) {
                tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
              }
            }
          } else {
#pragma unroll
            for (int k = 0; k < 8; ++k) {
              const int i = group * 8 + k;
              if (i < repeats) {
                tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
              }
            }
          }
        }
      }
      tcgen05_commit(&mma_barrier);
    }

    __syncthreads();
    mbarrier_wait(&mma_barrier, pair & 1);
    __syncthreads();

    if (warp_id < kPairBarrierGroups * 4) {
      const int read_group_id = warp_id >> 2;
#pragma unroll
      for (int local_group = 0; local_group < kPairGroups; ++local_group) {
        const int group = pair * kPairGroups + local_group;
        if (group < groups) {
          const uint32_t read_tmem_base =
              tmem_base + static_cast<uint32_t>(read_group_id * 256 + local_group * 128);
#pragma unroll 1
          for (int col = 0; col < 128; col += 4) {
            read_acc ^= tcgen05_ld_32x32b_x128_acc(read_tmem_base + static_cast<uint32_t>(col));
            tcgen05_wait_ld();
          }
        }
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^
                       static_cast<uint32_t>(repeats * kPairBarrierGroups);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_full_consume_two_db_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier[4];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    for (int i = 0; i < 4; ++i) mbarrier_init(&mma_barrier[i], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  const int groups = (repeats + active_group_mmas - 1) / active_group_mmas;

  auto issue_group = [&](int consumer_group, int group) {
    const int buffer = group & 1;
    const uint32_t issue_tmem_base =
        tmem_base + static_cast<uint32_t>(consumer_group * 256 + buffer * 128);
    if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
      for (int k = 0; k < kIssueGroupMmas; ++k) {
        const int i = group * kIssueGroupMmas + k;
        if (i < repeats) tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int i = group * 8 + k;
        if (i < repeats) tcgen05_mma_bf16_ss(issue_tmem_base, a_desc, b_desc, idesc, k != 0);
      }
    }
    tcgen05_commit(&mma_barrier[consumer_group * 2 + buffer]);
  };

  if (lane == 0 && groups > 0) {
    if (warp_id == 4) issue_group(0, 0);
    if (warp_id == 9) issue_group(1, 0);
  }
  __syncthreads();

  for (int group = 0; group < groups; ++group) {
    if (lane == 0 && group + 1 < groups) {
      if (warp_id == 4) issue_group(0, group + 1);
      if (warp_id == 9) issue_group(1, group + 1);
    }

    const int consumer_group = warp_id < 4 ? 0 : (warp_id >= 5 && warp_id < 9 ? 1 : -1);
    if (consumer_group >= 0) {
      const int buffer = group & 1;
      const uint32_t read_tmem_base =
          tmem_base + static_cast<uint32_t>(consumer_group * 256 + buffer * 128);
      mbarrier_wait(&mma_barrier[consumer_group * 2 + buffer], (group >> 1) & 1);
      __syncwarp();
#pragma unroll 1
      for (int col = 0; col < 128; col += 4) {
        read_acc ^= tcgen05_ld_32x32b_x128_acc(read_tmem_base + static_cast<uint32_t>(col));
        tcgen05_wait_ld();
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^ static_cast<uint32_t>(repeats * 2);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_split_read_every8_kernel(
    uint32_t* __restrict__ sink,
    int repeats,
    int issuer_warps,
    int group_mmas) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats; (void)issuer_warps; (void)group_mmas;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_128cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : group_mmas;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (lane == 0 && warp_id < issuer_warps) {
      const uint32_t d_taddr =
          tmem_base + static_cast<uint32_t>(warp_id * kTmemSlotStrideCols);
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll 1
        for (int k = 0; k < group_mmas; ++k) {
          const int i = group * group_mmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 3) {
      mbarrier_wait(&mma_barrier, group & 1);
    } else {
      if (lane == 0) mbarrier_wait(&mma_barrier, group & 1);
      __syncwarp();
    }
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 1) {
      __syncthreads();
    }

    const int reader_id = warp_id - issuer_warps;
    if (reader_id >= 0 && reader_id < issuer_warps) {
      const uint32_t d_taddr =
          tmem_base + static_cast<uint32_t>(reader_id * kTmemSlotStrideCols);
      uint32_t r[32];
      tcgen05_ld_32x32b_x32(r, d_taddr);
      read_acc ^= r[(group + lane) & 31];
      tcgen05_wait_ld();
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ read_acc ^ static_cast<uint32_t>(repeats * issuer_warps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_128cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32>
__global__ __launch_bounds__(kThreads, 1) void mma_issue_all8_read_every8_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem[kMaxIssuerWarps];
  __shared__ uint32_t tmem_base_shared[kMaxIssuerWarps];

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int slot = 0; slot < kAllReadWarps; ++slot) {
      const uint32_t taddr = kAllReadAllocCols == 128
          ? tcgen05_alloc_128cols(&tmem_smem[slot])
          : tcgen05_alloc_32cols(&tmem_smem[slot]);
      if ((threadIdx.x & 31) == 0) tmem_base_shared[slot] = taddr;
    }
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const uint32_t d_taddr = tmem_base_shared[warp_id];
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc(Acc32);
  uint32_t read_acc = 0;

  const int active_group_mmas = kIssueGroupMmas > 0 ? kIssueGroupMmas : 8;
  for (int group = 0; group * active_group_mmas < repeats; ++group) {
    if (lane == 0) {
      if constexpr (kIssueGroupMmas > 0) {
#pragma unroll
        for (int k = 0; k < kIssueGroupMmas; ++k) {
          const int i = group * kIssueGroupMmas + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int i = group * 8 + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
      }
    }

    __syncthreads();
    if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 3) {
      mbarrier_wait(&mma_barrier, group & 1);
    } else {
      if (lane == 0) mbarrier_wait(&mma_barrier, group & 1);
      __syncwarp();
    }
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 1) {
      __syncthreads();
    }

    uint32_t r[32];
    tcgen05_ld_32x32b_x32(r, d_taddr);
    read_acc ^= r[(group + lane) & 31];
    tcgen05_wait_ld();
    if constexpr (kReadGroupSyncMode == 0 || kReadGroupSyncMode == 1) {
      __syncthreads();
    }
  }

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base_shared[0] ^ read_acc ^
                       static_cast<uint32_t>(repeats * kAllReadWarps);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
#pragma unroll
    for (int slot = 0; slot < kAllReadWarps; ++slot) {
      if constexpr (kAllReadAllocCols == 128) {
        tcgen05_dealloc_128cols(tmem_base_shared[slot]);
      } else {
        tcgen05_dealloc_32cols(tmem_base_shared[slot]);
      }
    }
  }
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

void parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      args->blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--repeats") && i + 1 < argc) {
      args->repeats = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--group-mmas") && i + 1 < argc) {
      args->group_mmas = std::max(1, std::atoi(argv[++i]));
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if (!std::strcmp(argv[i], "--issue-only")) {
      args->issue_only = true;
    } else if (!std::strcmp(argv[i], "--occupancy-only")) {
      args->occupancy_only = true;
    } else if (!std::strcmp(argv[i], "--try-acc16")) {
      args->try_acc16 = true;
    } else if (!std::strcmp(argv[i], "--a-tmem-only")) {
      args->a_tmem_only = true;
    } else if (!std::strcmp(argv[i], "--rotate-accum-only")) {
      args->rotate_accum_only = true;
    } else if (!std::strcmp(argv[i], "--read-every8-only")) {
      args->read_every8_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-every8-only")) {
      args->full_consume_every8_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-two-groups-only")) {
      args->full_consume_two_groups_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-four-buffers-only")) {
      args->full_consume_four_buffers_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-four-warpgroups-only")) {
      args->full_consume_four_warpgroups_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-quad512-only")) {
      args->full_consume_quad512_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-double-buffer-only")) {
      args->full_consume_double_buffer_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-two-db-only")) {
      args->full_consume_two_db_only = true;
    } else if (!std::strcmp(argv[i], "--full-consume-pair-barrier-only")) {
      args->full_consume_pair_barrier_only = true;
    } else if (!std::strcmp(argv[i], "--split-read-every8-only")) {
      args->split_read_every8_only = true;
    } else if (!std::strcmp(argv[i], "--all8-read-every8-only")) {
      args->all8_read_every8_only = true;
    } else if (!std::strcmp(argv[i], "--single-simple-only")) {
      args->single_simple_only = true;
    } else if (!std::strcmp(argv[i], "--pure-issue-only")) {
      args->pure_issue_only = true;
    } else if (!std::strcmp(argv[i], "--latency-only")) {
      args->latency_only = true;
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf("Usage: %s [--blocks N] [--repeats N] [--warmup N] [--iters N] "
                  "[--group-mmas N] "
                  "[--csv PATH] [--issue-only] [--occupancy-only] [--try-acc16] "
                  "[--a-tmem-only] [--rotate-accum-only] [--read-every8-only] "
                  "[--full-consume-every8-only] [--split-read-every8-only] "
                  "[--full-consume-two-groups-only] [--full-consume-four-buffers-only] "
                  "[--full-consume-four-warpgroups-only] "
                  "[--full-consume-quad512-only] [--full-consume-double-buffer-only] "
                  "[--full-consume-two-db-only] [--full-consume-pair-barrier-only] "
                  "[--all8-read-every8-only] [--single-simple-only] [--pure-issue-only] "
                  "[--latency-only]\n",
                  argv[0]);
      std::exit(0);
    }
  }
}

template <bool Acc32, bool MultiIssuer, bool Use32Cols>
void configure_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_kernel<Acc32, MultiIssuer, Use32Cols>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32, bool MultiIssuer, bool Use32Cols>
int occupancy_for(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_kernel<Acc32, MultiIssuer, Use32Cols>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32, bool MultiIssuer, bool Use32Cols>
float run_case(uint32_t* sink,
               const Args& args,
               int blocks,
               int repeats,
               int issuer_warps,
               size_t dynamic_smem) {
  configure_kernel_smem<Acc32, MultiIssuer, Use32Cols>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_kernel<Acc32, MultiIssuer, Use32Cols><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_kernel<Acc32, MultiIssuer, Use32Cols><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32, bool Accumulate>
void configure_single_simple_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_single_simple_kernel<Acc32, Accumulate>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32, bool Accumulate>
int occupancy_for_single_simple(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_single_simple_kernel<Acc32, Accumulate>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32, bool Accumulate>
float run_single_simple_case(uint32_t* sink,
                             const Args& args,
                             int blocks,
                             int repeats,
                             size_t dynamic_smem) {
  configure_single_simple_kernel_smem<Acc32, Accumulate>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_single_simple_kernel<Acc32, Accumulate><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_single_simple_kernel<Acc32, Accumulate><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32, bool MultiIssuer, int PredMode>
int occupancy_for_pure_issue_clock(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_clock_kernel<Acc32, MultiIssuer, PredMode>, kThreads,
      dynamic_smem));
  return active;
}

template <bool Acc32, bool MultiIssuer, int PredMode>
double run_pure_issue_clock_case(uint32_t* sink,
                                 const Args& args,
                                 int blocks,
                                 int repeats,
                                 int issuer_warps) {
  unsigned long long* device_cycles = nullptr;
  const size_t entries = static_cast<size_t>(blocks) * kMaxIssuerWarps;
  CUDA_CHECK(cudaMalloc(&device_cycles, entries * sizeof(unsigned long long)));
  std::vector<unsigned long long> host_cycles(entries);

  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    CUDA_CHECK(cudaMemset(device_cycles, 0, entries * sizeof(unsigned long long)));
    mma_issue_clock_kernel<Acc32, MultiIssuer, PredMode><<<grid, block>>>(
        sink, device_cycles, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double cycle_sum = 0.0;
  size_t cycle_count = 0;
  for (int iter = 0; iter < args.iters; ++iter) {
    CUDA_CHECK(cudaMemset(device_cycles, 0, entries * sizeof(unsigned long long)));
    mma_issue_clock_kernel<Acc32, MultiIssuer, PredMode><<<grid, block>>>(
        sink, device_cycles, repeats, issuer_warps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_cycles.data(), device_cycles,
                          entries * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    const int active_issuers = MultiIssuer ? issuer_warps : 1;
    for (int b = 0; b < blocks; ++b) {
      for (int issuer = 0; issuer < active_issuers; ++issuer) {
        const unsigned long long cycles =
            host_cycles[static_cast<size_t>(b) * kMaxIssuerWarps + issuer];
        if (cycles != 0) {
          cycle_sum += static_cast<double>(cycles);
          ++cycle_count;
        }
      }
    }
  }
  CUDA_CHECK(cudaFree(device_cycles));
  return cycle_count == 0 ? 0.0 : cycle_sum / static_cast<double>(cycle_count);
}

template <bool Acc32, bool DoMma, bool InputD>
int occupancy_for_latency_clock(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_latency_clock_kernel<Acc32, DoMma, InputD>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32, bool DoMma, bool InputD>
double run_latency_clock_case(uint32_t* sink,
                              const Args& args,
                              int blocks,
                              int repeats) {
  unsigned long long* device_cycles = nullptr;
  CUDA_CHECK(cudaMalloc(&device_cycles,
                        static_cast<size_t>(blocks) * sizeof(unsigned long long)));
  std::vector<unsigned long long> host_cycles(static_cast<size_t>(blocks));

  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    CUDA_CHECK(cudaMemset(device_cycles, 0,
                          static_cast<size_t>(blocks) * sizeof(unsigned long long)));
    mma_latency_clock_kernel<Acc32, DoMma, InputD><<<grid, block>>>(
        sink, device_cycles, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  double cycle_sum = 0.0;
  size_t cycle_count = 0;
  for (int iter = 0; iter < args.iters; ++iter) {
    CUDA_CHECK(cudaMemset(device_cycles, 0,
                          static_cast<size_t>(blocks) * sizeof(unsigned long long)));
    mma_latency_clock_kernel<Acc32, DoMma, InputD><<<grid, block>>>(
        sink, device_cycles, repeats);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_cycles.data(), device_cycles,
                          static_cast<size_t>(blocks) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    for (int b = 0; b < blocks; ++b) {
      const unsigned long long cycles = host_cycles[static_cast<size_t>(b)];
      if (cycles != 0) {
        cycle_sum += static_cast<double>(cycles);
        ++cycle_count;
      }
    }
  }
  CUDA_CHECK(cudaFree(device_cycles));
  return cycle_count == 0 ? 0.0 : cycle_sum / static_cast<double>(cycle_count);
}

template <bool Acc32, bool Use32Cols>
void configure_a_tmem_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_a_tmem_kernel<Acc32, Use32Cols>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32, bool Use32Cols>
int occupancy_for_a_tmem(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_a_tmem_kernel<Acc32, Use32Cols>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32, bool Use32Cols>
float run_a_tmem_case(uint32_t* sink,
                      const Args& args,
                      int blocks,
                      int repeats,
                      int issuer_warps,
                      size_t dynamic_smem) {
  configure_a_tmem_kernel_smem<Acc32, Use32Cols>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_a_tmem_kernel<Acc32, Use32Cols><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_a_tmem_kernel<Acc32, Use32Cols><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_rotate_accum_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_rotate_accum_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_rotate_accum(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_rotate_accum_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_rotate_accum_case(uint32_t* sink,
                            const Args& args,
                            int blocks,
                            int repeats,
                            int issuer_warps,
                            size_t dynamic_smem) {
  configure_rotate_accum_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_rotate_accum_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_rotate_accum_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_read_every8_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_read_every8_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_read_every8(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_read_every8_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_read_every8_case(uint32_t* sink,
                           const Args& args,
                           int blocks,
                           int repeats,
                           int issuer_warps,
                           size_t dynamic_smem) {
  configure_read_every8_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_read_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps, args.group_mmas);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_read_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps, args.group_mmas);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_every8_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_every8_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_every8(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_every8_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_every8_case(uint32_t* sink,
                                   const Args& args,
                                   int blocks,
                                   int repeats,
                                   size_t dynamic_smem) {
  configure_full_consume_every8_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_two_groups_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_two_groups_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_two_groups(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_two_groups_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_two_groups_case(uint32_t* sink,
                                       const Args& args,
                                       int blocks,
                                       int repeats,
                                       size_t dynamic_smem) {
  configure_full_consume_two_groups_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_two_groups_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_two_groups_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_four_buffers_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_four_buffers_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_four_buffers(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_four_buffers_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_four_buffers_case(uint32_t* sink,
                                         const Args& args,
                                         int blocks,
                                         int repeats,
                                         size_t dynamic_smem) {
  configure_full_consume_four_buffers_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_four_buffers_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_four_buffers_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_four_warpgroups_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_four_warpgroups_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_four_warpgroups(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_four_warpgroups_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_four_warpgroups_case(uint32_t* sink,
                                            const Args& args,
                                            int blocks,
                                            int repeats,
                                            size_t dynamic_smem) {
  configure_full_consume_four_warpgroups_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_four_warpgroups_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_four_warpgroups_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_quad512_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_quad512_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_quad512(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_quad512_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_quad512_case(uint32_t* sink,
                                    const Args& args,
                                    int blocks,
                                    int repeats,
                                    size_t dynamic_smem) {
  configure_full_consume_quad512_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_quad512_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_quad512_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_double_buffer_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_double_buffer_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_double_buffer(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_double_buffer_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_double_buffer_case(uint32_t* sink,
                                          const Args& args,
                                          int blocks,
                                          int repeats,
                                          size_t dynamic_smem) {
  configure_full_consume_double_buffer_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_double_buffer_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_double_buffer_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_pair_barrier_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_pair_barrier_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_pair_barrier(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_pair_barrier_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_pair_barrier_case(uint32_t* sink,
                                         const Args& args,
                                         int blocks,
                                         int repeats,
                                         size_t dynamic_smem) {
  configure_full_consume_pair_barrier_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_pair_barrier_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_pair_barrier_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_full_consume_two_db_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_full_consume_two_db_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_full_consume_two_db(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_full_consume_two_db_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_full_consume_two_db_case(uint32_t* sink,
                                   const Args& args,
                                   int blocks,
                                   int repeats,
                                   size_t dynamic_smem) {
  configure_full_consume_two_db_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_full_consume_two_db_kernel<Acc32><<<grid, block, dynamic_smem>>>(sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_full_consume_two_db_kernel<Acc32><<<grid, block, dynamic_smem>>>(sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_split_read_every8_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_split_read_every8_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_split_read_every8(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_split_read_every8_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_split_read_every8_case(uint32_t* sink,
                                 const Args& args,
                                 int blocks,
                                 int repeats,
                                 int issuer_warps,
                                 size_t dynamic_smem) {
  configure_split_read_every8_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_split_read_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps, args.group_mmas);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_split_read_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats, issuer_warps, args.group_mmas);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <bool Acc32>
void configure_all8_read_every8_kernel_smem(size_t dynamic_smem) {
  CUDA_CHECK(cudaFuncSetAttribute(mma_issue_all8_read_every8_kernel<Acc32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(dynamic_smem)));
}

template <bool Acc32>
int occupancy_for_all8_read_every8(size_t dynamic_smem) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_issue_all8_read_every8_kernel<Acc32>, kThreads, dynamic_smem));
  return active;
}

template <bool Acc32>
float run_all8_read_every8_case(uint32_t* sink,
                                const Args& args,
                                int blocks,
                                int repeats,
                                size_t dynamic_smem) {
  configure_all8_read_every8_kernel_smem<Acc32>(dynamic_smem);
  dim3 grid(blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    mma_issue_all8_read_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_issue_all8_read_every8_kernel<Acc32><<<grid, block, dynamic_smem>>>(
        sink, repeats);
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

double tflops(double flops, float ms) {
  return ms > 0.0f ? flops / (static_cast<double>(ms) * 1.0e9) : 0.0;
}

double mma_per_s(double mmas, float ms) {
  return ms > 0.0f ? mmas / (static_cast<double>(ms) * 1.0e-3) : 0.0;
}

void write_row(FILE* csv,
               const char* benchmark,
               const char* mode,
               const char* accum,
               int issuer_warps,
               int requested_ctas_per_sm,
               int actual_ctas_per_sm,
               int blocks,
               int repeats,
               size_t dynamic_smem,
               float elapsed_ms,
               const char* notes) {
  const double total_mmas =
      static_cast<double>(blocks) * static_cast<double>(repeats) *
      static_cast<double>(issuer_warps);
  const double total_flops = total_mmas * kFlopsPerMma;
  std::fprintf(csv,
               "%s,%s,%s,%d,%d,%d,%d,%d,%d,%zu,%.6f,%.0f,%.3f,%.3f,%s\n",
               benchmark, mode, accum, issuer_warps, requested_ctas_per_sm,
               actual_ctas_per_sm, blocks, kThreads, repeats, dynamic_smem,
               elapsed_ms, total_mmas, mma_per_s(total_mmas, elapsed_ms),
               tflops(total_flops, elapsed_ms), notes);
  std::printf("%s,%s,%s,%d,%d,%d,%d,%d,%d,%zu,%.6f,%.0f,%.3f,%.3f,%s\n",
              benchmark, mode, accum, issuer_warps, requested_ctas_per_sm,
              actual_ctas_per_sm, blocks, kThreads, repeats, dynamic_smem,
              elapsed_ms, total_mmas, mma_per_s(total_mmas, elapsed_ms),
              tflops(total_flops, elapsed_ms), notes);
}

void write_pure_issue_row(FILE* csv,
                          const cudaDeviceProp& prop,
                          const char* mode,
                          const char* accum,
                          int issuer_warps,
                          int actual_ctas_per_sm,
                          int blocks,
                          int repeats,
                          double avg_issue_loop_cycles) {
  int device = 0;
  int clock_khz = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  const double clock_hz = static_cast<double>(clock_khz) * 1000.0;
  const double total_mmas =
      static_cast<double>(blocks) * static_cast<double>(repeats) *
      static_cast<double>(issuer_warps);
  const double waves = static_cast<double>(blocks) /
                       static_cast<double>(prop.multiProcessorCount);
  const float elapsed_ms = static_cast<float>(
      clock_hz > 0.0 ? waves * avg_issue_loop_cycles / clock_hz * 1.0e3 : 0.0);
  const double cycles_per_mma_per_issuer =
      repeats > 0 ? avg_issue_loop_cycles / static_cast<double>(repeats) : 0.0;
  const double aggregate_cycles_per_mma =
      issuer_warps > 0 ? cycles_per_mma_per_issuer / static_cast<double>(issuer_warps) : 0.0;
  char notes[192];
  std::snprintf(notes, sizeof(notes),
                "clock64_issue_loop_avg_cycles_%.1f_cycles_per_mma_per_issuer_%.3f_aggregate_%.3f",
                avg_issue_loop_cycles, cycles_per_mma_per_issuer,
                aggregate_cycles_per_mma);
  write_row(csv, "pure_issue_clock", mode, accum, issuer_warps, 0, actual_ctas_per_sm,
            blocks, repeats, 0, elapsed_ms, notes);
}

void write_latency_row(FILE* csv,
                       const cudaDeviceProp& prop,
                       const char* mode,
                       const char* accum,
                       int actual_ctas_per_sm,
                       int blocks,
                       int repeats,
                       double avg_latency_loop_cycles,
                       double baseline_cycles_per_iter) {
  int device = 0;
  int clock_khz = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, device));
  const double clock_hz = static_cast<double>(clock_khz) * 1000.0;
  const double waves = static_cast<double>(blocks) /
                       static_cast<double>(prop.multiProcessorCount);
  const float elapsed_ms = static_cast<float>(
      clock_hz > 0.0 ? waves * avg_latency_loop_cycles / clock_hz * 1.0e3 : 0.0);
  const double cycles_per_iter =
      repeats > 0 ? avg_latency_loop_cycles / static_cast<double>(repeats) : 0.0;
  const double mma_minus_barrier_cycles = cycles_per_iter - baseline_cycles_per_iter;
  char notes[224];
  std::snprintf(notes, sizeof(notes),
                "clock64_commit_wait_loop_avg_cycles_%.1f_cycles_per_iter_%.3f_minus_barrier_%.3f_barrier_baseline_%.3f",
                avg_latency_loop_cycles, cycles_per_iter, mma_minus_barrier_cycles,
                baseline_cycles_per_iter);
  write_row(csv, "mma_latency_clock", mode, accum, 1, 0, actual_ctas_per_sm, blocks,
            repeats, 0, elapsed_ms, notes);
}

size_t dynamic_smem_for_target(const cudaDeviceProp& prop, int requested_ctas_per_sm) {
  const size_t max_optin = static_cast<size_t>(prop.sharedMemPerBlockOptin);
  if (requested_ctas_per_sm <= 1) {
    return max_optin > kStaticSmemBudgetBytes ? max_optin - kStaticSmemBudgetBytes : 0;
  }
  const size_t per_sm = static_cast<size_t>(prop.sharedMemPerMultiprocessor);
  const size_t target = per_sm / static_cast<size_t>(requested_ctas_per_sm);
  if (target == 0) return 0;
  const size_t dyn_target =
      target > kStaticSmemBudgetBytes ? target - kStaticSmemBudgetBytes : 0;
  return std::min(max_optin, dyn_target);
}

template <bool Acc32>
void run_single_simple_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_accum = occupancy_for_single_simple<Acc32, true>(0);
  const float accum_ms =
      run_single_simple_case<Acc32, true>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "single_thread_simplified", "same_buffer_accum", accum, 1, 0,
            active_accum, args.blocks, args.repeats, 0, accum_ms,
            "no_mod_no_round_robin_constant_predicate");

  const int active_no_c = occupancy_for_single_simple<Acc32, false>(0);
  const float no_c_ms =
      run_single_simple_case<Acc32, false>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "single_thread_simplified", "same_buffer_no_c", accum, 1, 0,
            active_no_c, args.blocks, args.repeats, 0, no_c_ms,
            "no_mod_no_round_robin_input_c_false");
}

template <bool Acc32>
void run_pure_issue_clock_sweep(FILE* csv,
                                uint32_t* sink,
                                const Args& args,
                                const cudaDeviceProp& prop) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int max_safe_issuer_warps =
      std::max(1, std::min(kMaxIssuerWarps, 128 / kTmemSlotStrideCols));

  const int active_single_false = occupancy_for_pure_issue_clock<Acc32, false, 0>(0);
  const double single_false_cycles =
      run_pure_issue_clock_case<Acc32, false, 0>(sink, args, args.blocks, args.repeats, 1);
  write_pure_issue_row(csv, prop, "single_pred_false", accum, 1, active_single_false,
                       args.blocks, args.repeats, single_false_cycles);

  const int active_single_true = occupancy_for_pure_issue_clock<Acc32, false, 1>(0);
  const double single_true_cycles =
      run_pure_issue_clock_case<Acc32, false, 1>(sink, args, args.blocks, args.repeats, 1);
  write_pure_issue_row(csv, prop, "single_pred_true", accum, 1, active_single_true,
                       args.blocks, args.repeats, single_true_cycles);

  const int active_single_accum = occupancy_for_pure_issue_clock<Acc32, false, 2>(0);
  const double single_accum_cycles =
      run_pure_issue_clock_case<Acc32, false, 2>(sink, args, args.blocks, args.repeats, 1);
  write_pure_issue_row(csv, prop, "single_first_false_then_true", accum, 1,
                       active_single_accum, args.blocks, args.repeats,
                       single_accum_cycles);

  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    const int active_multi = occupancy_for_pure_issue_clock<Acc32, true, 2>(0);
    const double multi_cycles =
        run_pure_issue_clock_case<Acc32, true, 2>(sink, args, args.blocks,
                                                 args.repeats, issuer_warps);
    write_pure_issue_row(csv, prop, "multi_first_false_then_true", accum, issuer_warps,
                         active_multi, args.blocks, args.repeats, multi_cycles);
  }
}

template <bool Acc32>
void run_latency_clock_sweep(FILE* csv,
                             uint32_t* sink,
                             const Args& args,
                             const cudaDeviceProp& prop) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_barrier = occupancy_for_latency_clock<Acc32, false, false>(0);
  const double barrier_cycles =
      run_latency_clock_case<Acc32, false, false>(sink, args, args.blocks, args.repeats);
  const double barrier_cycles_per_iter =
      args.repeats > 0 ? barrier_cycles / static_cast<double>(args.repeats) : 0.0;
  write_latency_row(csv, prop, "mbarrier_commit_wait_only", accum, active_barrier,
                    args.blocks, args.repeats, barrier_cycles, 0.0);

  const int active_no_c = occupancy_for_latency_clock<Acc32, true, false>(0);
  const double no_c_cycles =
      run_latency_clock_case<Acc32, true, false>(sink, args, args.blocks, args.repeats);
  write_latency_row(csv, prop, "mma_input_c_false_commit_wait", accum, active_no_c,
                    args.blocks, args.repeats, no_c_cycles, barrier_cycles_per_iter);

  const int active_accum = occupancy_for_latency_clock<Acc32, true, true>(0);
  const double accum_cycles =
      run_latency_clock_case<Acc32, true, true>(sink, args, args.blocks, args.repeats);
  write_latency_row(csv, prop, "mma_input_c_true_commit_wait", accum, active_accum,
                    args.blocks, args.repeats, accum_cycles, barrier_cycles_per_iter);
}

template <bool Acc32>
void run_issue_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int max_safe_issuer_warps = std::max(1, std::min(kMaxIssuerWarps, 128 / kTmemSlotStrideCols));
  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    const int active_single = occupancy_for<Acc32, false, false>(0);
    const float single_ms =
        run_case<Acc32, false, false>(sink, args, args.blocks, args.repeats, issuer_warps, 0);
    write_row(csv, "issue_parallelism", "single_thread", accum, issuer_warps, 0,
              active_single, args.blocks, args.repeats, 0, single_ms, "ok");

    const int active_multi = occupancy_for<Acc32, true, false>(0);
    const float multi_ms =
        run_case<Acc32, true, false>(sink, args, args.blocks, args.repeats, issuer_warps, 0);
    write_row(csv, "issue_parallelism", "multi_thread", accum, issuer_warps, 0,
              active_multi, args.blocks, args.repeats, 0, multi_ms, "ok");
  }
}

template <bool Acc32>
void run_a_tmem_issue_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int max_safe_issuer_warps = std::max(1, std::min(kMaxIssuerWarps, 128 / kTmemSlotStrideCols));
  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    const int active_multi = occupancy_for_a_tmem<Acc32, false>(0);
    const float multi_ms =
        run_a_tmem_case<Acc32, false>(sink, args, args.blocks, args.repeats, issuer_warps, 0);
    write_row(csv, "a_tmem_c_tmem", "multi_thread", accum, issuer_warps, 0,
              active_multi, args.blocks, args.repeats, 0, multi_ms, "a_tmem_b_smem_c_tmem");
  }
}

template <bool Acc32>
void run_rotate_accum_issue_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int max_safe_issuer_warps =
      std::max(1, std::min(2, std::min(kMaxIssuerWarps, 128 / kTmemSlotStrideCols)));
  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    const int active_multi = occupancy_for_rotate_accum<Acc32>(0);
    const float multi_ms =
        run_rotate_accum_case<Acc32>(sink, args, args.blocks, args.repeats, issuer_warps, 0);
    write_row(csv, "rotate_accum_groups", "multi_thread", accum, issuer_warps, 0,
              active_multi, args.blocks, args.repeats, 0, multi_ms, "group_size_8");
  }
}

template <bool Acc32>
void run_read_every8_issue_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int max_safe_issuer_warps =
      std::max(1, std::min(kBlockWarps, 128 / kTmemSlotStrideCols));
  char notes[64];
  std::snprintf(notes, sizeof(notes), "%d_mma_commit_wait_ldx32", args.group_mmas);
  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    const int active_multi = occupancy_for_read_every8<Acc32>(0);
    const float multi_ms =
        run_read_every8_case<Acc32>(sink, args, args.blocks, args.repeats, issuer_warps, 0);
    write_row(csv, "read_every8", "multi_thread", accum, issuer_warps, 0,
              active_multi, args.blocks, args.repeats, 0, multi_ms,
              notes);
  }
}

template <bool Acc32>
void run_full_consume_every8(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_every8<Acc32>(0);
  const float ms =
      run_full_consume_every8_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_every8", "multi_thread", accum, 1, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            kFullConsumeLdWidth == 128
                ? "one_m128n128_tile_mma8_then_4warp_full_read_ldx128"
                : "one_m128n128_tile_mma8_then_4warp_full_read_ldx32");
}

template <bool Acc32>
void run_full_consume_two_groups(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_two_groups<Acc32>(0);
  const float ms =
      run_full_consume_two_groups_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_two_groups", "multi_thread", accum, 2, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            "two_independent_4warp_groups_each_mma8_then_full_read_x128");
}

template <bool Acc32>
void run_full_consume_four_buffers(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_four_buffers<Acc32>(0);
  const float ms =
      run_full_consume_four_buffers_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_four_buffers", "multi_thread", accum, 4, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            kFourBufferReadsPerGroup == 2
                ? "four_buffers_four_producer_warps_requested_two_reads_per_group"
                : "four_buffers_four_producer_warps_one_full_read_per_group");
}

template <bool Acc32>
void run_full_consume_four_warpgroups(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  if constexpr (kBlockWarps < 16) {
    write_row(csv, "full_consume_four_warpgroups", "multi_thread", accum, 4, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_requires_BLOCK_WARPS_16");
  } else {
    const int active_multi = occupancy_for_full_consume_four_warpgroups<Acc32>(0);
    const float ms =
        run_full_consume_four_warpgroups_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
    write_row(csv, "full_consume_four_warpgroups", "multi_thread", accum, 4, 0,
              active_multi, args.blocks, args.repeats, 0, ms,
              "four_warpgroups_each_wg0_mma8_then_4warp_ldx128_once");
  }
}

template <bool Acc32>
void run_full_consume_quad512(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_quad512<Acc32>(0);
  const float ms =
      run_full_consume_quad512_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_quad512", "multi_thread", accum, kFullConsumeGroups, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            "one_512col_alloc_mma8_tiles_all_full_read");
}

template <bool Acc32>
void run_full_consume_double_buffer(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_double_buffer<Acc32>(0);
  const float ms =
      run_full_consume_double_buffer_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_double_buffer", "multi_thread", accum, 1, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            "producer_warp_overlaps_next_mma8_with_previous_full_read");
}

template <bool Acc32>
void run_full_consume_pair_barrier(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_pair_barrier<Acc32>(0);
  const float ms =
      run_full_consume_pair_barrier_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_pair_barrier", "multi_thread", accum, kPairBarrierGroups, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            "one_barrier_for_two_mma8_tiles_then_full_read_both");
}

template <bool Acc32>
void run_full_consume_two_db(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int active_multi = occupancy_for_full_consume_two_db<Acc32>(0);
  const float ms =
      run_full_consume_two_db_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
  write_row(csv, "full_consume_two_db", "multi_thread", accum, 2, 0,
            active_multi, args.blocks, args.repeats, 0, ms,
            "two_groups_double_buffer_mma8_full_read");
}

template <bool Acc32>
void run_split_read_every8_issue_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  const int max_safe_issuer_warps =
      std::max(1, std::min(kBlockWarps / 2, 128 / kTmemSlotStrideCols));
  char notes[80];
  std::snprintf(notes, sizeof(notes), "issuex%d_warps_separate_from_read_warps",
                args.group_mmas);
  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    const int active_multi = occupancy_for_split_read_every8<Acc32>(0);
    const float multi_ms =
        run_split_read_every8_case<Acc32>(sink, args, args.blocks, args.repeats, issuer_warps, 0);
    write_row(csv, "split_read_every8", "multi_thread", accum, issuer_warps, 0,
              active_multi, args.blocks, args.repeats, 0, multi_ms,
              notes);
  }
}

template <bool Acc32>
void run_all8_read_every8(FILE* csv, uint32_t* sink, const Args& args) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  if constexpr (kTmemSlotStrideCols > 32) {
    write_row(csv, "all8_read_every8", "multi_thread", accum, kAllReadWarps, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_requires_N_le_128");
  } else {
    const int active_multi = occupancy_for_all8_read_every8<Acc32>(0);
    const float multi_ms =
        run_all8_read_every8_case<Acc32>(sink, args, args.blocks, args.repeats, 0);
    write_row(csv, "all8_read_every8", "multi_thread", accum, kAllReadWarps, 0,
              active_multi, args.blocks, args.repeats, 0, multi_ms,
              kAllReadAllocCols == 128 ? "each_warp_mma8_then_ldx32_alloc128"
                                        : "each_warp_mma8_then_ldx32_alloc32");
  }
}

template <bool Acc32>
void run_occupancy_sweep(FILE* csv,
                         uint32_t* sink,
                         const Args& args,
                         const cudaDeviceProp& prop) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  for (int requested : {1, 2, 4, 8}) {
    const size_t dynamic_smem = dynamic_smem_for_target(prop, requested);
    int actual = 0;
    cudaError_t attr_err = cudaFuncSetAttribute(mma_issue_kernel<Acc32, true, false>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(dynamic_smem));
    if (attr_err != cudaSuccess) {
      std::fprintf(stderr, "skip occupancy target %d for %s: dynamic_smem=%zu attr=%s\n",
                   requested, accum, dynamic_smem, cudaGetErrorString(attr_err));
      write_row(csv, "occupancy", "multi_thread", accum, 1, requested, 0, args.blocks,
                args.repeats, dynamic_smem, 0.0f, "skipped_attr");
      continue;
    }
    cudaError_t occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &actual, mma_issue_kernel<Acc32, true, false>, kThreads, dynamic_smem);
    if (occ_err != cudaSuccess || actual <= 0) {
      std::fprintf(stderr, "skip occupancy target %d for %s: dynamic_smem=%zu occupancy=%s\n",
                   requested, accum, dynamic_smem, cudaGetErrorString(occ_err));
      write_row(csv, "occupancy", "multi_thread", accum, 1, requested, 0, args.blocks,
                args.repeats, dynamic_smem, 0.0f, "skipped_occupancy");
      continue;
    }
    const float ms = run_case<Acc32, true, false>(sink, args, args.blocks, args.repeats, 1,
                                                  dynamic_smem);
    write_row(csv, "occupancy", "multi_thread", accum, 1, requested, actual,
              args.blocks, args.repeats, dynamic_smem, ms, "ok");
  }
}

void write_acc16_issue_skips(FILE* csv, const Args& args) {
  const int max_safe_issuer_warps = std::max(1, std::min(kMaxIssuerWarps, 128 / kTmemSlotStrideCols));
  for (int issuer_warps = 1; issuer_warps <= max_safe_issuer_warps; issuer_warps *= 2) {
    write_row(csv, "issue_parallelism", "single_thread", "acc16", issuer_warps, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    write_row(csv, "issue_parallelism", "multi_thread", "acc16", issuer_warps, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
  }
}

void write_acc16_occupancy_skips(FILE* csv, const Args& args, const cudaDeviceProp& prop) {
  for (int requested : {1, 2, 4, 8}) {
    const size_t dynamic_smem = dynamic_smem_for_target(prop, requested);
    write_row(csv, "occupancy", "multi_thread", "acc16", 1, requested, 0, args.blocks,
              args.repeats, dynamic_smem, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
  }
}

template <bool Acc32, int MmaM, int MmaN, int IssueThreads, int BufferCount, int AccumulatesPerBuffer>
__global__ __launch_bounds__(kThreads, 1) void focused_mma_issue_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  static_assert(IssueThreads <= kBlockWarps, "IssueThreads maps to one lane0 per warp.");
  static_assert((AccumulatesPerBuffer & (AccumulatesPerBuffer - 1)) == 0,
                "AccumulatesPerBuffer must be a power of two.");
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  constexpr int kShapeSlotStrideCols = tmem_slot_stride_cols_for<MmaN>();
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc_shape<MmaM, MmaN>(Acc32);

  if (lane == 0 && warp_id < IssueThreads) {
    constexpr int kPatternMmas = BufferCount * AccumulatesPerBuffer;
    const int full_patterns = repeats / kPatternMmas;
    const int tail_mmas = repeats - full_patterns * kPatternMmas;

    if (full_patterns > 0) {
#pragma unroll
      for (int buffer = 0; buffer < BufferCount; ++buffer) {
        const int global_buffer = warp_id * BufferCount + buffer;
        const uint32_t d_taddr =
            tmem_base + static_cast<uint32_t>(global_buffer * kShapeSlotStrideCols);
#pragma unroll
        for (int k = 0; k < AccumulatesPerBuffer; ++k) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
        }
      }

#pragma unroll 1
      for (int cycle = 1; cycle < full_patterns; ++cycle) {
#pragma unroll
        for (int buffer = 0; buffer < BufferCount; ++buffer) {
          const int global_buffer = warp_id * BufferCount + buffer;
          const uint32_t d_taddr =
              tmem_base + static_cast<uint32_t>(global_buffer * kShapeSlotStrideCols);
#pragma unroll
          for (int k = 0; k < AccumulatesPerBuffer; ++k) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, true);
          }
        }
      }
    }

    int tail_issued = 0;
#pragma unroll
    for (int buffer = 0; buffer < BufferCount; ++buffer) {
      const int global_buffer = warp_id * BufferCount + buffer;
      const uint32_t d_taddr =
          tmem_base + static_cast<uint32_t>(global_buffer * kShapeSlotStrideCols);
#pragma unroll
      for (int k = 0; k < AccumulatesPerBuffer; ++k) {
        if (tail_issued < tail_mmas) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc,
                              full_patterns > 0 || k != 0);
          ++tail_issued;
        }
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] =
        tmem_base ^ static_cast<uint32_t>(IssueThreads * BufferCount *
                                          AccumulatesPerBuffer * repeats);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <bool Acc32, int MmaM, int MmaN, int IssueThreads, int BufferCount, int AccumulatesPerBuffer>
float run_focused_case(uint32_t* sink, const Args& args, int repeats) {
  dim3 grid(args.blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    focused_mma_issue_kernel<Acc32, MmaM, MmaN, IssueThreads, BufferCount, AccumulatesPerBuffer>
        <<<grid, block>>>(sink, repeats);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    focused_mma_issue_kernel<Acc32, MmaM, MmaN, IssueThreads, BufferCount, AccumulatesPerBuffer>
        <<<grid, block>>>(sink, repeats);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

#if defined(MMA_DUAL_N64_HYPOTHESIS) && MMA_DUAL_N64_HYPOTHESIS
template <int MmaN, int IssueThreads, bool DualN64>
__global__ __launch_bounds__(kThreads, 1) void dual_n64_hypothesis_kernel(
    uint32_t* __restrict__ sink,
    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t mma_barrier;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;

  if (threadIdx.x == 0) {
    mbarrier_init(&mma_barrier, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words, 4096);
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if ((threadIdx.x & 31) == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  constexpr int kStrideCols = tmem_slot_stride_cols_for<MmaN>();
  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words), 128, 32, 2);
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048), 128, 32, 2);
  const uint32_t idesc = make_bf16_idesc_shape<128, MmaN>(true);

  if (lane == 0 && warp_id < IssueThreads && repeats > 0) {
    const uint32_t d0 =
        tmem_base + static_cast<uint32_t>((DualN64 ? warp_id * 2 : warp_id) * kStrideCols);
    const uint32_t d1 = d0 + static_cast<uint32_t>(kStrideCols);
    tcgen05_mma_bf16_ss(d0, a_desc, b_desc, idesc, false);
    if constexpr (DualN64) {
      tcgen05_mma_bf16_ss(d1, a_desc, b_desc, idesc, false);
    }
#pragma unroll 1
    for (int i = 1; i < repeats; ++i) {
      tcgen05_mma_bf16_ss(d0, a_desc, b_desc, idesc, true);
      if constexpr (DualN64) {
        tcgen05_mma_bf16_ss(d1, a_desc, b_desc, idesc, true);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) tcgen05_commit(&mma_barrier);
  mbarrier_wait(&mma_barrier, 0);
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    sink[blockIdx.x] = tmem_base ^ static_cast<uint32_t>(repeats * IssueThreads);
  }
  __syncthreads();

  if ((threadIdx.x >> 5) == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if ((threadIdx.x >> 5) == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <int MmaN, int IssueThreads, bool DualN64>
float run_dual_n64_hypothesis_case(uint32_t* sink, const Args& args, int repeats) {
  dim3 grid(args.blocks);
  dim3 block(kThreads);
  for (int i = 0; i < args.warmup; ++i) {
    dual_n64_hypothesis_kernel<MmaN, IssueThreads, DualN64><<<grid, block>>>(sink, repeats);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    dual_n64_hypothesis_kernel<MmaN, IssueThreads, DualN64><<<grid, block>>>(sink, repeats);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <int MmaN, int IssueThreads, bool DualN64>
void write_dual_n64_hypothesis_case(FILE* csv,
                                    uint32_t* sink,
                                    const Args& args,
                                    const char* name) {
  const float ms = run_dual_n64_hypothesis_case<MmaN, IssueThreads, DualN64>(
      sink, args, args.repeats);
  constexpr int kMmasPerRepeat = DualN64 ? 2 : 1;
  constexpr double kCaseFlopsPerMma = flops_per_mma_for<128, MmaN>();
  const double total_mmas =
      static_cast<double>(args.blocks) * IssueThreads * args.repeats * kMmasPerRepeat;
  const double total_flops = total_mmas * kCaseFlopsPerMma;
  std::fprintf(csv, "%s,%d,%d,%d,%d,%d,%.6f,%.0f,%.3f\n", name, 128, MmaN,
               IssueThreads, args.repeats, kMmasPerRepeat, ms, total_mmas,
               tflops(total_flops, ms));
  std::printf("%s m128 n%d issue_threads=%d repeats=%d mmas_per_repeat=%d: "
              "%.3f TFLOP/s %.6f ms\n",
              name, MmaN, IssueThreads, args.repeats, kMmasPerRepeat,
              tflops(total_flops, ms), ms);
}

void run_dual_n64_hypothesis(FILE* csv, uint32_t* sink, const Args& args) {
  std::fprintf(csv,
               "case,mma_m,mma_n,issue_threads,repeats,mmas_per_repeat,"
               "elapsed_ms,total_mmas,TFLOP_per_s\n");
  write_dual_n64_hypothesis_case<64, 8, false>(csv, sink, args, "n64_single");
  write_dual_n64_hypothesis_case<64, 8, true>(csv, sink, args, "n64_dual_col64_64");
  write_dual_n64_hypothesis_case<128, 8, false>(csv, sink, args, "n128_single");
}
#endif

void write_focused_row(FILE* csv,
                       int mma_m,
                       int mma_n,
                       double flops_per_mma,
                       const char* accum,
                       int issue_threads,
                       int buffer_count,
                       int accumulates_per_buffer,
                       int repeats,
                       int actual_ctas_per_sm,
                       const Args& args,
                       float elapsed_ms,
                       const char* notes) {
  const int total_buffers = issue_threads * buffer_count;
  const double total_mmas =
      static_cast<double>(args.blocks) * static_cast<double>(issue_threads) *
      static_cast<double>(repeats);
  const double total_flops = total_mmas * flops_per_mma;
  const int pattern_mmas = buffer_count * accumulates_per_buffer;
  const double cycles_per_issue_thread =
      pattern_mmas > 0 ? static_cast<double>(repeats) / static_cast<double>(pattern_mmas) : 0.0;
  std::fprintf(csv,
               "%d,%d,%d,%s,%d,%d,%d,%d,%d,%d,%.3f,%d,%d,%d,%.6f,%.0f,%.3f,%.3f,%s\n",
               mma_m, mma_n, kMmaK, accum, issue_threads, buffer_count,
               accumulates_per_buffer, repeats, pattern_mmas, total_buffers,
               cycles_per_issue_thread, actual_ctas_per_sm, args.blocks, kThreads,
               elapsed_ms, total_mmas, mma_per_s(total_mmas, elapsed_ms),
               tflops(total_flops, elapsed_ms), notes);
  std::printf(
      "m%d n%d issue_threads=%d buffers=%d accum_per_buffer=%d repeats=%d: %.3f TFLOP/s %.6f ms %s\n",
      mma_m, mma_n, issue_threads, buffer_count, accumulates_per_buffer,
      repeats, tflops(total_flops, elapsed_ms), elapsed_ms, notes);
}

template <bool Acc32, int MmaM, int MmaN, int IssueThreads, int BufferCount, int AccumulatesPerBuffer>
void run_focused_case_if_valid(FILE* csv,
                               uint32_t* sink,
                               const Args& args,
                               int repeats) {
  const char* accum = Acc32 ? "acc32" : "acc16";
  constexpr int kShapeSlotStrideCols = tmem_slot_stride_cols_for<MmaN>();
  constexpr double kShapeFlopsPerMma = flops_per_mma_for<MmaM, MmaN>();
  const int total_buffers = IssueThreads * BufferCount;
  const int max_buffers = std::min(8, 256 / kShapeSlotStrideCols);
  if (total_buffers > max_buffers) {
    write_focused_row(csv, MmaM, MmaN, kShapeFlopsPerMma, accum, IssueThreads,
                      BufferCount, AccumulatesPerBuffer, repeats, 0, args, 0.0f,
                      "skipped_tmem_buffer_capacity");
    return;
  }
  int actual = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &actual,
      focused_mma_issue_kernel<Acc32, MmaM, MmaN, IssueThreads, BufferCount, AccumulatesPerBuffer>,
      kThreads, 0));
  const float ms =
      run_focused_case<Acc32, MmaM, MmaN, IssueThreads, BufferCount, AccumulatesPerBuffer>(
          sink, args, repeats);
  write_focused_row(csv, MmaM, MmaN, kShapeFlopsPerMma, accum, IssueThreads,
                    BufferCount, AccumulatesPerBuffer, repeats, actual, args, ms,
                    "one_lane0_per_issue_warp_no_midloop_wait");
}

template <bool Acc32, int MmaM, int MmaN, int IssueThreads, int BufferCount>
void run_focused_accum_sweep(FILE* csv,
                             uint32_t* sink,
                             const Args& args,
                             int repeats) {
  run_focused_case_if_valid<Acc32, MmaM, MmaN, IssueThreads, BufferCount, 1>(csv, sink, args, repeats);
  run_focused_case_if_valid<Acc32, MmaM, MmaN, IssueThreads, BufferCount, 2>(csv, sink, args, repeats);
  run_focused_case_if_valid<Acc32, MmaM, MmaN, IssueThreads, BufferCount, 4>(csv, sink, args, repeats);
  run_focused_case_if_valid<Acc32, MmaM, MmaN, IssueThreads, BufferCount, 8>(csv, sink, args, repeats);
  run_focused_case_if_valid<Acc32, MmaM, MmaN, IssueThreads, BufferCount, 16>(csv, sink, args, repeats);
  run_focused_case_if_valid<Acc32, MmaM, MmaN, IssueThreads, BufferCount, 32>(csv, sink, args, repeats);
}

template <bool Acc32, int MmaM, int MmaN, int IssueThreads>
void run_focused_buffer_sweep(FILE* csv,
                              uint32_t* sink,
                              const Args& args,
                              int repeats) {
  run_focused_accum_sweep<Acc32, MmaM, MmaN, IssueThreads, 1>(csv, sink, args, repeats);
  run_focused_accum_sweep<Acc32, MmaM, MmaN, IssueThreads, 2>(csv, sink, args, repeats);
  run_focused_accum_sweep<Acc32, MmaM, MmaN, IssueThreads, 4>(csv, sink, args, repeats);
  run_focused_accum_sweep<Acc32, MmaM, MmaN, IssueThreads, 8>(csv, sink, args, repeats);
}

template <bool Acc32, int MmaM, int MmaN>
void run_focused_repeat_sweep(FILE* csv,
                              uint32_t* sink,
                              const Args& args,
                              int repeats) {
  run_focused_buffer_sweep<Acc32, MmaM, MmaN, 1>(csv, sink, args, repeats);
  run_focused_buffer_sweep<Acc32, MmaM, MmaN, 2>(csv, sink, args, repeats);
  run_focused_buffer_sweep<Acc32, MmaM, MmaN, 4>(csv, sink, args, repeats);
  run_focused_buffer_sweep<Acc32, MmaM, MmaN, 8>(csv, sink, args, repeats);
}

template <bool Acc32, int MmaM, int MmaN>
void run_focused_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  std::vector<int> repeats_values;
  for (int r : {256, 1024, args.repeats}) {
    if (r > 0 && r <= args.repeats &&
        std::find(repeats_values.begin(), repeats_values.end(), r) ==
                     repeats_values.end()) {
      repeats_values.push_back(r);
    }
  }
  for (int repeats : repeats_values) {
    run_focused_repeat_sweep<Acc32, MmaM, MmaN>(csv, sink, args, repeats);
    std::fflush(csv);
  }
}

template <bool Acc32>
void run_focused_shape_sweep(FILE* csv, uint32_t* sink, const Args& args) {
#if defined(MMA_FOCUSED_ONLY_N64) && MMA_FOCUSED_ONLY_N64
  run_focused_sweep<Acc32, 64, 64>(csv, sink, args);
  run_focused_sweep<Acc32, 128, 64>(csv, sink, args);
#elif defined(MMA_FOCUSED_ONLY_M128N128) && MMA_FOCUSED_ONLY_M128N128
  run_focused_sweep<Acc32, 128, 128>(csv, sink, args);
#else
  run_focused_sweep<Acc32, 64, 32>(csv, sink, args);
  run_focused_sweep<Acc32, 64, 64>(csv, sink, args);
  run_focused_sweep<Acc32, 64, 128>(csv, sink, args);
  run_focused_sweep<Acc32, 64, 256>(csv, sink, args);
  run_focused_sweep<Acc32, 128, 32>(csv, sink, args);
  run_focused_sweep<Acc32, 128, 64>(csv, sink, args);
  run_focused_sweep<Acc32, 128, 128>(csv, sink, args);
  run_focused_sweep<Acc32, 128, 256>(csv, sink, args);
#endif
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
    std::fprintf(stderr, "This benchmark requires sm_100+ tcgen05 hardware, got sm_%d%d.\n",
                 prop.major, prop.minor);
    return 77;
  }

  uint32_t* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(sink, 0, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::fprintf(stderr, "Failed to open CSV path %s\n", args.csv);
    CUDA_CHECK(cudaFree(sink));
    return 1;
  }

#if defined(MMA_DUAL_N64_HYPOTHESIS) && MMA_DUAL_N64_HYPOTHESIS
  run_dual_n64_hypothesis(csv, sink, args);
  std::fclose(csv);
  CUDA_CHECK(cudaFree(sink));
  return 0;
#endif

  std::fprintf(csv,
               "mma_m,mma_n,mma_k,accum,issue_threads,buffer_count,"
               "accumulates_per_buffer,repeats,pattern_mmas_per_cycle,"
               "total_buffers,cycles_per_issue_thread,actual_ctas_per_sm,"
               "blocks,threads_per_cta,elapsed_ms,total_mmas,mma_per_s,"
               "TFLOP_per_s,notes\n");
  std::printf("mma_m,mma_n,mma_k,accum,issue_threads,buffer_count,"
              "accumulates_per_buffer,repeats,pattern_mmas_per_cycle,"
              "total_buffers,cycles_per_issue_thread,actual_ctas_per_sm,"
              "blocks,threads_per_cta,elapsed_ms,total_mmas,mma_per_s,"
              "TFLOP_per_s,notes\n");
  std::printf("device=%s sm_%d%d sms=%d blocks=%d repeats=%d warmup=%d iters=%d "
              "focused_shape_sweep_m64_m128_n32_n64_n128_n256 k%d\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, args.blocks,
	      args.repeats, args.warmup, args.iters, kMmaK);

  run_focused_shape_sweep<true>(csv, sink, args);
  if (args.try_acc16) {
    run_focused_shape_sweep<false>(csv, sink, args);
  }
  std::fclose(csv);
  CUDA_CHECK(cudaFree(sink));
  return 0;

  if (args.single_simple_only) {
    run_single_simple_sweep<true>(csv, sink, args);
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.pure_issue_only) {
    run_pure_issue_clock_sweep<true>(csv, sink, args, prop);
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.latency_only) {
    run_latency_clock_sweep<true>(csv, sink, args, prop);
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.a_tmem_only) {
    run_a_tmem_issue_sweep<true>(csv, sink, args);
    if (args.try_acc16) {
      run_a_tmem_issue_sweep<false>(csv, sink, args);
    } else {
      write_row(csv, "a_tmem_c_tmem", "multi_thread", "acc16", 1, 0, 0,
                args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    }
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.rotate_accum_only) {
    run_rotate_accum_issue_sweep<true>(csv, sink, args);
    if (args.try_acc16) {
      run_rotate_accum_issue_sweep<false>(csv, sink, args);
    } else {
      write_row(csv, "rotate_accum_groups", "multi_thread", "acc16", 1, 0, 0,
                args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    }
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.read_every8_only) {
    run_read_every8_issue_sweep<true>(csv, sink, args);
    write_row(csv, "read_every8", "multi_thread", "acc16", 1, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_every8_only) {
    run_full_consume_every8<true>(csv, sink, args);
    write_row(csv, "full_consume_every8", "multi_thread", "acc16", 1, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_two_groups_only) {
    run_full_consume_two_groups<true>(csv, sink, args);
    write_row(csv, "full_consume_two_groups", "multi_thread", "acc16", 2, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_four_buffers_only) {
    run_full_consume_four_buffers<true>(csv, sink, args);
    write_row(csv, "full_consume_four_buffers", "multi_thread", "acc16", 4, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_four_warpgroups_only) {
    run_full_consume_four_warpgroups<true>(csv, sink, args);
    write_row(csv, "full_consume_four_warpgroups", "multi_thread", "acc16", 4, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_quad512_only) {
    run_full_consume_quad512<true>(csv, sink, args);
    write_row(csv, "full_consume_quad512", "multi_thread", "acc16", kFullConsumeGroups, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_double_buffer_only) {
    run_full_consume_double_buffer<true>(csv, sink, args);
    write_row(csv, "full_consume_double_buffer", "multi_thread", "acc16", 1, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_two_db_only) {
    run_full_consume_two_db<true>(csv, sink, args);
    write_row(csv, "full_consume_two_db", "multi_thread", "acc16", 2, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.full_consume_pair_barrier_only) {
    run_full_consume_pair_barrier<true>(csv, sink, args);
    write_row(csv, "full_consume_pair_barrier", "multi_thread", "acc16", kPairBarrierGroups, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.split_read_every8_only) {
    run_split_read_every8_issue_sweep<true>(csv, sink, args);
    write_row(csv, "split_read_every8", "multi_thread", "acc16", 1, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (args.all8_read_every8_only) {
    run_all8_read_every8<true>(csv, sink, args);
    write_row(csv, "all8_read_every8", "multi_thread", "acc16", kMaxIssuerWarps, 0, 0,
              args.blocks, args.repeats, 0, 0.0f, "skipped_bf16_acc16_illegal_on_sm100");
    std::fclose(csv);
    CUDA_CHECK(cudaFree(sink));
    return 0;
  }

  if (!args.occupancy_only) {
    run_issue_sweep<true>(csv, sink, args);
    if (args.try_acc16) {
      run_issue_sweep<false>(csv, sink, args);
    } else {
      write_acc16_issue_skips(csv, args);
    }
  }
  if (!args.issue_only) {
    run_occupancy_sweep<true>(csv, sink, args, prop);
    if (args.try_acc16) {
      run_occupancy_sweep<false>(csv, sink, args, prop);
    } else {
      write_acc16_occupancy_skips(csv, args, prop);
    }
  }

  std::fclose(csv);
  CUDA_CHECK(cudaFree(sink));
  return 0;
}
