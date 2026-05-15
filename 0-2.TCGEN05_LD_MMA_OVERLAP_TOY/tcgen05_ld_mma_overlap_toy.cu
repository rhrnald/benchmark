#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
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
static constexpr int kWarps = 16;
static constexpr int kThreads = kWarps * kWarpSize;
static constexpr int kPeakWarps = 8;
static constexpr int kPeakThreads = kPeakWarps * kWarpSize;
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kMmaK = 16;
static constexpr int kMmasPerGroup = 8;
static constexpr int kTileBf16Elems = kTileM * kTileN;
static constexpr int kTileWords = kTileBf16Elems / 2;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kDynamicSmemBytes = 2 * kTileBytes + 1024;
static constexpr int kLdConsumerWarps = 4;
static constexpr double kFlopsPerMmaGroup =
    static_cast<double>(kMmasPerGroup) * 2.0 * kTileM * kTileN * kMmaK;
static constexpr double kFlopsPerMma =
    2.0 * static_cast<double>(kTileM) * static_cast<double>(kTileN) *
    static_cast<double>(kMmaK);
static constexpr double kAcc32TileBytes =
    static_cast<double>(kTileM) * static_cast<double>(kTileN) * sizeof(float);
static constexpr double kLdBytesPerWarp =
    2.0 * static_cast<double>(kWarpSize) * 64.0 * sizeof(uint32_t);
static constexpr double kLdBytesPerGroup = kLdBytesPerWarp * kLdConsumerWarps;

enum Mode : uint32_t {
  kModeLdOnly = 0,
  kModeMma1Only = 1,
  kModeOverlapMma1 = 2,
  kModeMma2Only = 3,
  kModeOverlapMma2 = 4,
  kModeMma3Only = 5,
  kModeOverlapMma3 = 6,
  kModeCount = 7,
};

struct Args {
  int blocks = 4096;
  int repeats = 256;
  int ld_repeats = -1;
  int skip = 16;
  int warmup = 2;
  int iters = 5;
  int sms = 148;
  double clock_mhz = 1155.0;
  double reference_mma_tflops = 2207.606;
  double reference_ld8_peak_tbps = 1807.896;
  bool peak_only = false;
  const char* csv = "0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_trace.csv";
  const char* summary_csv =
      "0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_summary.csv";
  const char* peak_csv =
      "0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_peak_overlap.csv";
  const char* claim_csv =
      "0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_contention_claim.csv";
};

struct TraceRow {
  uint64_t mma_start;
  uint64_t mma_end;
  uint64_t ld_start;
  uint64_t ld_end;
  uint64_t ld_warp_start[kLdConsumerWarps];
  uint64_t ld_warp_end[kLdConsumerWarps];
  uint32_t mode;
  uint32_t iter;
  uint32_t sink;
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
  constexpr uint32_t stride_dim_byte_offset = 32;
  const uint32_t lead_enc = (leading_dim_byte_offset & 0x3ffffu) >> 4;
  const uint32_t stride_enc = (stride_dim_byte_offset & 0x3ffffu) >> 4;
  uint64_t desc = 0;
  desc |= static_cast<uint64_t>(matrix_start_aligned >> 4);
  desc |= static_cast<uint64_t>(lead_enc) << 16;
  desc |= static_cast<uint64_t>(stride_enc) << 32;
  desc |= static_cast<uint64_t>(0x1u) << 46;
  desc |= static_cast<uint64_t>(0xB0u) << 53;
  desc |= static_cast<uint64_t>(2u) << 61;
  return desc;
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("setmaxnreg.dec.sync.aligned.u32 88;" ::: "memory");
#endif
}

__device__ __forceinline__ void setmaxnreg_inc_consumer() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 160;" ::: "memory");
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

__device__ __forceinline__ uint32_t tcgen05_ld_x64x2_acc(uint32_t src_taddr) {
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

__device__ __forceinline__ uint32_t tcgen05_ld_x128_mix(uint32_t src_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t acc;
  asm volatile(
      "{ .reg .b32 r<128>; .reg .b32 acc; "
      "tcgen05.ld.sync.aligned.32x32b.x128.b32 {" TMEM_REGS_0_127 "}, [%1]; "
      "xor.b32 acc, r0, r31; xor.b32 acc, acc, r63; xor.b32 acc, acc, r95; "
      "xor.b32 %0, acc, r127; }"
      : "=r"(acc)
      : "r"(src_taddr)
      : "memory");
  return acc;
#else
  (void)src_taddr;
  return 0;
#endif
}

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

#undef TMEM_REGS_0_63
#undef TMEM_REGS_64_127
#undef TMEM_REGS_0_127

__device__ __forceinline__ void run_mma8(uint32_t taddr,
                                         const uint64_t* q_desc,
                                         const uint64_t* k_desc,
                                         uint32_t idesc,
                                         bool input_d) {
#pragma unroll
  for (int mma = 0; mma < kMmasPerGroup; ++mma) {
    tcgen05_mma_bf16_ss(taddr, q_desc[mma], k_desc[mma], idesc, input_d || mma != 0);
  }
}

template <int ProducerWarps, bool DoLd>
__global__ __launch_bounds__(kThreads, 1)
void ld_mma_overlap_toy_kernel(TraceRow* __restrict__ rows, uint32_t* __restrict__ sink,
                               int repeats, uint32_t mode) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)rows;
  (void)sink;
  (void)repeats;
  (void)mode;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);

  __shared__ uint64_t init_done;
  __shared__ uint64_t mma_done;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint64_t base_clock_shared;
  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ uint64_t q_desc_shared[kMmasPerGroup];
  __shared__ uint64_t k_desc_shared[kMmasPerGroup];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tid = threadIdx.x;

  if (warp_id < 8) {
    setmaxnreg_dec_producer();
  } else {
    setmaxnreg_inc_consumer();
  }

  if (tid == 0) {
    mbarrier_init(&init_done, 1);
    mbarrier_init(&mma_done, ProducerWarps > 0 ? ProducerWarps : 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }

  for (int i = tid; i < 2 * kTileWords; i += kThreads) {
    const uint32_t low = 0x3f80u ^ static_cast<uint32_t>((i * 17 + 3) & 0x7fu);
    const uint32_t high = 0x4000u ^ static_cast<uint32_t>((i * 29 + 5) & 0x7fu);
    smem_words[i] = (high << 16) | low;
  }

  if (tid == 0) {
    for (int i = 0; i < repeats; ++i) {
      rows[i].mma_start = ~0ull;
      rows[i].mma_end = 0;
      rows[i].ld_start = ~0ull;
      rows[i].ld_end = 0;
#pragma unroll
      for (int w = 0; w < kLdConsumerWarps; ++w) {
        rows[i].ld_warp_start[w] = ~0ull;
        rows[i].ld_warp_end[w] = 0;
      }
      rows[i].mode = mode;
      rows[i].iter = static_cast<uint32_t>(i);
      rows[i].sink = 0;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t ld_taddr = tmem_base;
  const uint32_t mma_base_taddr = tmem_base + (DoLd ? 128u : 0u);
  const uint32_t idesc = make_qk_idesc();
  uint32_t* q_smem = smem_words;
  uint32_t* k_smem = smem_words + kTileWords;

  if (warp_id == 0 && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMmasPerGroup; ++mma) {
      q_desc_shared[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc_shared[mma] = make_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
    }
    run_mma8(ld_taddr, q_desc_shared, k_desc_shared, idesc, false);
    tcgen05_commit(&init_done);
    mbarrier_wait(&init_done, 0);
  }
  __syncthreads();

  if (tid == 0) base_clock_shared = clock64();
  __syncthreads();

  uint32_t local_sink = static_cast<uint32_t>(tid + 1);

  if constexpr (ProducerWarps > 0) {
    if (warp_id < ProducerWarps && lane == 0) {
      const uint32_t mma_taddr = mma_base_taddr + static_cast<uint32_t>(warp_id * 128);
      for (int iter = 0; iter < repeats; ++iter) {
        const uint64_t s = clock64() - base_clock_shared;
        atomicMin(reinterpret_cast<unsigned long long*>(&rows[iter].mma_start),
                  static_cast<unsigned long long>(s));
        run_mma8(mma_taddr, q_desc_shared, k_desc_shared, idesc, iter != 0);
        tcgen05_commit(&mma_done);
        mbarrier_wait(&mma_done, static_cast<uint32_t>(iter & 1));
        const uint64_t e = clock64() - base_clock_shared;
        atomicMax(reinterpret_cast<unsigned long long*>(&rows[iter].mma_end),
                  static_cast<unsigned long long>(e));
        local_sink ^= static_cast<uint32_t>(e - s);
      }
    }
  }

  if constexpr (DoLd) {
    if (warp_id >= 8 && warp_id < 12) {
      const int consumer = warp_id - 8;
      for (int iter = 0; iter < repeats; ++iter) {
        const uint64_t s = clock64() - base_clock_shared;
        rows[iter].ld_warp_start[consumer] = s;
        atomicMin(reinterpret_cast<unsigned long long*>(&rows[iter].ld_start),
                  static_cast<unsigned long long>(s));
        const uint32_t acc = tcgen05_ld_x64x2_acc(ld_taddr);
        const uint64_t e = clock64() - base_clock_shared;
        rows[iter].ld_warp_end[consumer] = e;
        atomicMax(reinterpret_cast<unsigned long long*>(&rows[iter].ld_end),
                  static_cast<unsigned long long>(e));
        local_sink ^= acc + static_cast<uint32_t>(e - s) + static_cast<uint32_t>(consumer);
      }
    }
  }

  warp_sinks[warp_id] = local_sink;
  __syncthreads();

  if (tid == 0) {
    uint32_t acc = 0;
    for (int w = 0; w < kWarps; ++w) acc ^= warp_sinks[w];
    sink[mode] = acc;
  }

  __syncthreads();
  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <int IssueWarps, bool DoLd>
__global__ __launch_bounds__(kPeakThreads, 1)
void peak_ld_mma_kernel(uint32_t* __restrict__ sink, int mma_repeats, int ld_repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink;
  (void)mma_repeats;
  (void)ld_repeats;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);

  __shared__ uint64_t init_done;
  __shared__ uint64_t mma_done;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kPeakWarps];
  __shared__ uint64_t q_desc_shared[kMmasPerGroup];
  __shared__ uint64_t k_desc_shared[kMmasPerGroup];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tid = threadIdx.x;

  if (tid == 0) {
    mbarrier_init(&init_done, 1);
    mbarrier_init(&mma_done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < 2 * kTileWords; i += kPeakThreads) {
    const uint32_t low = 0x3f80u ^ static_cast<uint32_t>((i * 17 + 3) & 0x7fu);
    const uint32_t high = 0x4000u ^ static_cast<uint32_t>((i * 29 + 5) & 0x7fu);
    smem_words[i] = (high << 16) | low;
  }
  __syncthreads();

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t idesc = make_qk_idesc();
  uint32_t* q_smem = smem_words;
  uint32_t* k_smem = smem_words + kTileWords;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(q_smem));
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(k_smem));
  if (warp_id == 0 && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMmasPerGroup; ++mma) {
      q_desc_shared[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc_shared[mma] = make_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
    }
    run_mma8(tmem_base + 0u, q_desc_shared, k_desc_shared, idesc, false);
    run_mma8(tmem_base + 128u, q_desc_shared, k_desc_shared, idesc, false);
    tcgen05_commit(&init_done);
    mbarrier_wait(&init_done, 0);
  }
  __syncthreads();

  uint32_t local_sink = static_cast<uint32_t>(threadIdx.x + 1);

  if constexpr (IssueWarps > 0) {
    if (lane == 0 && warp_id < IssueWarps) {
      const uint32_t d_taddr = tmem_base + (DoLd ? 128u : 0u) +
                               static_cast<uint32_t>(warp_id * 32);
#pragma unroll 1
      for (int i = 0; i < mma_repeats; ++i) {
        tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, i != 0);
      }
      local_sink ^= d_taddr ^ static_cast<uint32_t>(mma_repeats);
    }
  }

  if constexpr (DoLd) {
    if (warp_id >= 4 && warp_id < 8) {
      const uint32_t src_taddr = tmem_base;
      uint32_t acc0 = local_sink;
      uint32_t acc1 = local_sink ^ 0x9e3779b9u;
      uint32_t acc2 = local_sink ^ 0x7f4a7c15u;
      uint32_t acc3 = local_sink ^ 0x94d049bbu;
      uint32_t acc4 = local_sink ^ 0x2545f491u;
      uint32_t acc5 = local_sink ^ 0x369dea0fu;
      uint32_t acc6 = local_sink ^ 0xdb4f0b91u;
      uint32_t acc7 = local_sink ^ 0xbb67ae85u;
      int i = 0;
#pragma unroll 1
      for (; i + 7 < ld_repeats; i += 8) {
        acc0 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 0);
        acc1 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 1);
        acc2 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 2);
        acc3 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 3);
        acc4 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 4);
        acc5 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 5);
        acc6 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 6);
        acc7 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i + 7);
      }
#pragma unroll 1
      for (; i < ld_repeats; ++i) {
        acc0 ^= tcgen05_ld_x128_mix(src_taddr) + static_cast<uint32_t>(i);
      }
      tcgen05_wait_ld();
      local_sink ^= acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
    }
  }

  warp_sinks[warp_id] = local_sink;
  __syncthreads();

  if constexpr (IssueWarps > 0) {
    if (threadIdx.x == 0) tcgen05_commit(&mma_done);
    mbarrier_wait(&mma_done, 0);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t acc = tmem_base ^ static_cast<uint32_t>(IssueWarps * mma_repeats) ^
                   static_cast<uint32_t>(ld_repeats);
    for (int w = 0; w < kPeakWarps; ++w) acc ^= warp_sinks[w];
    sink[blockIdx.x] = acc;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

const char* mode_name(uint32_t mode) {
  switch (mode) {
    case kModeLdOnly:
      return "ld_only";
    case kModeMma1Only:
      return "mma1_only";
    case kModeOverlapMma1:
      return "overlap_mma1_ld";
    case kModeMma2Only:
      return "mma2_only";
    case kModeOverlapMma2:
      return "overlap_mma2_ld";
    case kModeMma3Only:
      return "mma3_only";
    case kModeOverlapMma3:
      return "overlap_mma3_ld";
    default:
      return "unknown";
  }
}

int mode_producer_warps(uint32_t mode) {
  switch (mode) {
    case kModeMma1Only:
    case kModeOverlapMma1:
      return 1;
    case kModeMma2Only:
    case kModeOverlapMma2:
      return 2;
    case kModeMma3Only:
    case kModeOverlapMma3:
      return 3;
    default:
      return 0;
  }
}

bool mode_has_ld(uint32_t mode) {
  return mode == kModeLdOnly || mode == kModeOverlapMma1 || mode == kModeOverlapMma2 ||
         mode == kModeOverlapMma3;
}

const char* arg_value(int argc, char** argv, const char* name, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return fallback;
}

bool has_flag(int argc, char** argv, const char* name) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return true;
  }
  return false;
}

void parse_args(int argc, char** argv, Args* args) {
  args->blocks = std::atoi(arg_value(argc, argv, "--blocks", "4096"));
  args->repeats = std::atoi(arg_value(argc, argv, "--repeats", "256"));
  args->ld_repeats = std::atoi(arg_value(argc, argv, "--ld-repeats", "-1"));
  if (args->ld_repeats < 0) args->ld_repeats = args->repeats;
  args->skip = std::atoi(arg_value(argc, argv, "--skip", "16"));
  args->warmup = std::atoi(arg_value(argc, argv, "--warmup", "2"));
  args->iters = std::atoi(arg_value(argc, argv, "--iters", "5"));
  args->sms = std::atoi(arg_value(argc, argv, "--sms", "148"));
  args->clock_mhz = std::atof(arg_value(argc, argv, "--clock-mhz", "1155.0"));
  args->reference_mma_tflops =
      std::atof(arg_value(argc, argv, "--reference-mma-tflops", "2207.606"));
  args->reference_ld8_peak_tbps =
      std::atof(arg_value(argc, argv, "--reference-ld8-peak-tbps", "1807.896"));
  args->csv = arg_value(argc, argv, "--csv", args->csv);
  args->summary_csv = arg_value(argc, argv, "--summary-csv", args->summary_csv);
  args->peak_csv = arg_value(argc, argv, "--peak-csv", args->peak_csv);
  args->claim_csv = arg_value(argc, argv, "--claim-csv", args->claim_csv);
  args->peak_only = has_flag(argc, argv, "--peak-only");
  if (has_flag(argc, argv, "--help")) {
    std::printf(
        "Usage: tcgen05_ld_mma_overlap_toy [--blocks N] [--repeats N] [--ld-repeats N] [--skip N] "
        "[--warmup N] [--iters N] [--sms N] [--clock-mhz MHz] [--csv path] "
        "[--summary-csv path] [--peak-csv path] [--claim-csv path] "
        "[--reference-mma-tflops X] [--reference-ld8-peak-tbps X] [--peak-only]\n");
    std::exit(0);
  }
}

template <int ProducerWarps, bool DoLd>
void configure_kernel() {
  CUDA_CHECK(cudaFuncSetAttribute(ld_mma_overlap_toy_kernel<ProducerWarps, DoLd>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
}

template <int ProducerWarps, bool DoLd>
void run_mode(uint32_t mode, const Args& args, std::vector<TraceRow>* all_rows) {
  TraceRow* d_rows = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rows, args.repeats * sizeof(TraceRow)));
  CUDA_CHECK(cudaMalloc(&d_sink, kModeCount * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_rows, 0, args.repeats * sizeof(TraceRow)));
  CUDA_CHECK(cudaMemset(d_sink, 0, kModeCount * sizeof(uint32_t)));

  configure_kernel<ProducerWarps, DoLd>();
  ld_mma_overlap_toy_kernel<ProducerWarps, DoLd>
      <<<1, kThreads, kDynamicSmemBytes>>>(d_rows, d_sink, args.repeats, mode);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const size_t old_size = all_rows->size();
  all_rows->resize(old_size + args.repeats);
  CUDA_CHECK(cudaMemcpy(all_rows->data() + old_size, d_rows, args.repeats * sizeof(TraceRow),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_rows));
  CUDA_CHECK(cudaFree(d_sink));
}

double avg_or_zero(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  double s = 0.0;
  for (double x : v) s += x;
  return s / static_cast<double>(v.size());
}

double min_or_zero(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return *std::min_element(v.begin(), v.end());
}

double max_or_zero(const std::vector<double>& v) {
  if (v.empty()) return 0.0;
  return *std::max_element(v.begin(), v.end());
}

void write_detail_csv(const char* path, const std::vector<TraceRow>& rows) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(f,
               "mode,iter,mma_start,mma_end,mma_cycles,ld_start,ld_end,ld_agg_cycles,"
               "ld_warp0_start,ld_warp0_end,ld_warp0_cycles,"
               "ld_warp1_start,ld_warp1_end,ld_warp1_cycles,"
               "ld_warp2_start,ld_warp2_end,ld_warp2_cycles,"
               "ld_warp3_start,ld_warp3_end,ld_warp3_cycles,sink\n");
  for (const TraceRow& r : rows) {
    const bool has_mma = r.mma_start != ~0ull && r.mma_end > r.mma_start;
    const bool has_ld = r.ld_start != ~0ull && r.ld_end > r.ld_start;
    std::fprintf(f, "%s,%u,%llu,%llu,%llu,%llu,%llu,%llu", mode_name(r.mode), r.iter,
                 static_cast<unsigned long long>(has_mma ? r.mma_start : 0),
                 static_cast<unsigned long long>(has_mma ? r.mma_end : 0),
                 static_cast<unsigned long long>(has_mma ? r.mma_end - r.mma_start : 0),
                 static_cast<unsigned long long>(has_ld ? r.ld_start : 0),
                 static_cast<unsigned long long>(has_ld ? r.ld_end : 0),
                 static_cast<unsigned long long>(has_ld ? r.ld_end - r.ld_start : 0));
    for (int w = 0; w < kLdConsumerWarps; ++w) {
      const bool hw = r.ld_warp_start[w] != ~0ull && r.ld_warp_end[w] > r.ld_warp_start[w];
      std::fprintf(f, ",%llu,%llu,%llu",
                   static_cast<unsigned long long>(hw ? r.ld_warp_start[w] : 0),
                   static_cast<unsigned long long>(hw ? r.ld_warp_end[w] : 0),
                   static_cast<unsigned long long>(hw ? r.ld_warp_end[w] -
                                                            r.ld_warp_start[w]
                                                      : 0));
    }
    std::fprintf(f, ",%u\n", r.sink);
  }
  std::fclose(f);
}

void write_summary_csv(const char* path, const std::vector<TraceRow>& rows, const Args& args) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::perror(path);
    std::exit(1);
  }
  const double clock_hz = args.clock_mhz * 1.0e6;
  std::fprintf(f,
               "mode,producer_warps,repeats,skip,sms,clock_mhz,avg_mma_cycles,"
               "min_mma_cycles,max_mma_cycles,"
               "avg_mma_TFLOP_per_s,avg_mma_cycles_while_ld_active,"
               "mma_samples_while_ld_active,avg_mma_TFLOP_per_s_while_ld_active,"
               "avg_ld_agg_cycles,min_ld_agg_cycles,max_ld_agg_cycles,"
               "avg_ld_TBps,avg_ld_single_warp_cycles,avg_ld_start_skew_cycles,"
               "avg_ld_end_skew_cycles,status,notes\n");
  for (uint32_t mode = 0; mode < kModeCount; ++mode) {
    uint64_t ld_active_end = 0;
    if (mode_has_ld(mode) && mode_producer_warps(mode) > 0) {
      for (const TraceRow& r : rows) {
        if (r.mode == mode && r.ld_end > ld_active_end) ld_active_end = r.ld_end;
      }
    }
    std::vector<double> mma_cycles;
    std::vector<double> mma_cycles_while_ld_active;
    std::vector<double> ld_cycles;
    std::vector<double> ld_warp_cycles;
    std::vector<double> ld_start_skew;
    std::vector<double> ld_end_skew;
    for (const TraceRow& r : rows) {
      if (r.mode != mode || static_cast<int>(r.iter) < args.skip) continue;
      if (r.mma_start != ~0ull && r.mma_end > r.mma_start) {
        mma_cycles.push_back(static_cast<double>(r.mma_end - r.mma_start));
        if (ld_active_end != 0 && r.mma_start < ld_active_end) {
          mma_cycles_while_ld_active.push_back(static_cast<double>(r.mma_end - r.mma_start));
        }
      }
      if (r.ld_start != ~0ull && r.ld_end > r.ld_start) {
        ld_cycles.push_back(static_cast<double>(r.ld_end - r.ld_start));
        uint64_t min_s = ~0ull;
        uint64_t max_s = 0;
        uint64_t min_e = ~0ull;
        uint64_t max_e = 0;
        for (int w = 0; w < kLdConsumerWarps; ++w) {
          if (r.ld_warp_start[w] != ~0ull && r.ld_warp_end[w] > r.ld_warp_start[w]) {
            ld_warp_cycles.push_back(static_cast<double>(r.ld_warp_end[w] -
                                                         r.ld_warp_start[w]));
            min_s = std::min(min_s, r.ld_warp_start[w]);
            max_s = std::max(max_s, r.ld_warp_start[w]);
            min_e = std::min(min_e, r.ld_warp_end[w]);
            max_e = std::max(max_e, r.ld_warp_end[w]);
          }
        }
        if (min_s != ~0ull) ld_start_skew.push_back(static_cast<double>(max_s - min_s));
        if (min_e != ~0ull) ld_end_skew.push_back(static_cast<double>(max_e - min_e));
      }
    }
    const double avg_mma = avg_or_zero(mma_cycles);
    const double avg_mma_while_ld = avg_or_zero(mma_cycles_while_ld_active);
    const double avg_ld = avg_or_zero(ld_cycles);
    const double mma_flops = kFlopsPerMmaGroup * static_cast<double>(mode_producer_warps(mode));
    const double mma_tflops =
        avg_mma > 0.0 ? mma_flops * clock_hz * args.sms / avg_mma / 1.0e12 : 0.0;
    const double mma_tflops_while_ld = avg_mma_while_ld > 0.0
                                           ? mma_flops * clock_hz * args.sms /
                                                 avg_mma_while_ld / 1.0e12
                                           : 0.0;
    const double ld_tbps =
        avg_ld > 0.0 ? kLdBytesPerGroup * clock_hz * args.sms / avg_ld / 1.0e12 : 0.0;
    std::fprintf(f,
                 "%s,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%zu,%.3f,%.3f,%.3f,%.3f,"
                 "%.3f,%.3f,%.3f,%.3f,ok,independent_tmem_slots_mma8_vs_4warp_ldx64x2\n",
                 mode_name(mode), mode_producer_warps(mode), args.repeats, args.skip, args.sms,
                 args.clock_mhz, avg_mma,
                 min_or_zero(mma_cycles), max_or_zero(mma_cycles), mma_tflops,
                 avg_mma_while_ld, mma_cycles_while_ld_active.size(), mma_tflops_while_ld,
                 avg_ld,
                 min_or_zero(ld_cycles), max_or_zero(ld_cycles), ld_tbps,
                 avg_or_zero(ld_warp_cycles), avg_or_zero(ld_start_skew),
                 avg_or_zero(ld_end_skew));
  }
  std::fclose(f);
}

void print_summary(const std::vector<TraceRow>& rows, const Args& args) {
  const double clock_hz = args.clock_mhz * 1.0e6;
  for (uint32_t mode = 0; mode < kModeCount; ++mode) {
    std::vector<double> mma_cycles;
    std::vector<double> ld_cycles;
    for (const TraceRow& r : rows) {
      if (r.mode != mode || static_cast<int>(r.iter) < args.skip) continue;
      if (r.mma_start != ~0ull && r.mma_end > r.mma_start) {
        mma_cycles.push_back(static_cast<double>(r.mma_end - r.mma_start));
      }
      if (r.ld_start != ~0ull && r.ld_end > r.ld_start) {
        ld_cycles.push_back(static_cast<double>(r.ld_end - r.ld_start));
      }
    }
    const double avg_mma = avg_or_zero(mma_cycles);
    const double avg_ld = avg_or_zero(ld_cycles);
    const double mma_flops = kFlopsPerMmaGroup * static_cast<double>(mode_producer_warps(mode));
    const double mma_tflops =
        avg_mma > 0.0 ? mma_flops * clock_hz * args.sms / avg_mma / 1.0e12 : 0.0;
    const double ld_tbps =
        avg_ld > 0.0 ? kLdBytesPerGroup * clock_hz * args.sms / avg_ld / 1.0e12 : 0.0;
    std::printf("%-10s  mma %.1f cyc %.1f TF/s   ld %.1f cyc %.1f TB/s\n",
                mode_name(mode), avg_mma, mma_tflops, avg_ld, ld_tbps);
  }
}

struct PeakResult {
  const char* mode;
  int issue_warps;
  int ld_warps;
  float elapsed_ms;
  double total_mmas;
  double mma_tflops;
  double total_ld_ops;
  double ld_tbps;
};

template <int IssueWarps, bool DoLd>
float run_peak_timed_case(uint32_t* sink, const Args& args) {
  CUDA_CHECK(cudaFuncSetAttribute(peak_ld_mma_kernel<IssueWarps, DoLd>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  dim3 grid(args.blocks);
  dim3 block(kPeakThreads);
  for (int i = 0; i < args.warmup; ++i) {
    peak_ld_mma_kernel<IssueWarps, DoLd>
        <<<grid, block, kDynamicSmemBytes>>>(sink, args.repeats, args.ld_repeats);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    peak_ld_mma_kernel<IssueWarps, DoLd>
        <<<grid, block, kDynamicSmemBytes>>>(sink, args.repeats, args.ld_repeats);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / static_cast<float>(args.iters);
}

template <int IssueWarps, bool DoLd>
PeakResult make_peak_result(const char* mode, uint32_t* sink, const Args& args) {
  const float ms = run_peak_timed_case<IssueWarps, DoLd>(sink, args);
  const double total_mmas =
      static_cast<double>(args.blocks) * static_cast<double>(IssueWarps) *
      static_cast<double>(args.repeats);
  const double total_flops =
      total_mmas * (2.0 * static_cast<double>(kTileM) * static_cast<double>(kTileN) *
                    static_cast<double>(kMmaK));
  constexpr int kPeakLdWarps = DoLd ? 4 : 0;
  const double total_ld_ops =
      static_cast<double>(args.blocks) * static_cast<double>(kPeakLdWarps) *
      static_cast<double>(args.ld_repeats);
  const double total_ld_bytes = total_ld_ops * 128.0 * 128.0;  // x128: 16 KiB per warp op.
  PeakResult r{};
  r.mode = mode;
  r.issue_warps = IssueWarps;
  r.ld_warps = kPeakLdWarps;
  r.elapsed_ms = ms;
  r.total_mmas = total_mmas;
  r.mma_tflops = ms > 0.0f ? total_flops / (static_cast<double>(ms) * 1.0e9) : 0.0;
  r.total_ld_ops = total_ld_ops;
  r.ld_tbps = ms > 0.0f ? total_ld_bytes / (static_cast<double>(ms) * 1.0e9) : 0.0;
  return r;
}

void run_peak_sweep(const Args& args) {
  uint32_t* sink = nullptr;
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));
  std::vector<PeakResult> results;
  results.push_back(make_peak_result<0, true>("ld_peak_only", sink, args));
  results.push_back(make_peak_result<4, false>("mma4_peak_only", sink, args));
  results.push_back(make_peak_result<4, true>("mma4_plus_ld_peak_overlap", sink, args));
  CUDA_CHECK(cudaFree(sink));

  double ld_peak = 0.0;
  double overlap_ld = 0.0;
  double overlap_mma = 0.0;
  for (const PeakResult& r : results) {
    if (std::strcmp(r.mode, "ld_peak_only") == 0) ld_peak = r.ld_tbps;
    if (std::strcmp(r.mode, "mma4_plus_ld_peak_overlap") == 0) {
      overlap_ld = r.ld_tbps;
      overlap_mma = r.mma_tflops;
    }
  }

  const double clock_hz = args.clock_mhz * 1.0e6;
  const double bytes_per_cycle_per_sm_scale = 1.0e12 / (static_cast<double>(args.sms) * clock_hz);
  const double observed_drop_tbps = std::max(0.0, ld_peak - overlap_ld);
  const double observed_drop_b_per_cycle_per_sm =
      observed_drop_tbps * bytes_per_cycle_per_sm_scale;
  const double reference_c_read_tbps =
      args.reference_mma_tflops * (kAcc32TileBytes / kFlopsPerMma);
  const double reference_c_read_b_per_cycle_per_sm =
      reference_c_read_tbps * bytes_per_cycle_per_sm_scale;
  const double reference_c_read_plus_write_tbps = 2.0 * reference_c_read_tbps;
  const double observed_to_cread_ratio =
      reference_c_read_tbps > 0.0 ? observed_drop_tbps / reference_c_read_tbps : 0.0;
  const double scaled_to_ld8_peak_tbps =
      ld_peak > 0.0 ? (observed_drop_tbps / ld_peak) * args.reference_ld8_peak_tbps : 0.0;

  FILE* f = std::fopen(args.peak_csv, "w");
  if (!f) {
    std::perror(args.peak_csv);
    std::exit(1);
  }
  std::fprintf(f,
               "mode,issue_warps,ld_warps,blocks,repeats,warmup,iters,elapsed_ms,"
               "ld_repeats,total_mmas,mma_TFLOP_per_s,total_ld_ops,logical_ld_TBps,"
               "ld_peak_baseline_TBps,mma_equiv_ld_TBps_from_ld_drop,"
               "mma_equiv_ld_fraction_of_peak,equiv_scaled_to_ld8_peak_1807_896_TBps,"
               "drop_B_per_cycle_per_SM,reference_mma_TFLOP_per_s,"
               "reference_mma_C_read_TBps,reference_mma_C_read_B_per_cycle_per_SM,"
               "drop_to_reference_C_read_ratio,"
               "status,notes\n");
  for (const PeakResult& r : results) {
    const double equiv = r.ld_warps > 0 && r.issue_warps > 0 ? std::max(0.0, ld_peak - r.ld_tbps)
                                                             : 0.0;
    const double frac = ld_peak > 0.0 ? equiv / ld_peak : 0.0;
    const double scaled_to_ld8_peak = frac * args.reference_ld8_peak_tbps;
    const double row_drop_b_per_cycle_per_sm = equiv * bytes_per_cycle_per_sm_scale;
    const double row_drop_to_cread =
        reference_c_read_tbps > 0.0 ? equiv / reference_c_read_tbps : 0.0;
    std::fprintf(f,
                 "%s,%d,%d,%d,%d,%d,%d,%.6f,%d,%.0f,%.3f,%.0f,%.3f,%.3f,%.3f,%.6f,"
                 "%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,"
                 "ok,peak_style_no_midloop_wait_ldx128_mma4_issue_warps\n",
                 r.mode, r.issue_warps, r.ld_warps, args.blocks, args.repeats,
                 args.warmup, args.iters, r.elapsed_ms, args.ld_repeats, r.total_mmas, r.mma_tflops,
                 r.total_ld_ops, r.ld_tbps, ld_peak, equiv, frac, scaled_to_ld8_peak,
                 row_drop_b_per_cycle_per_sm, args.reference_mma_tflops,
                 reference_c_read_tbps, reference_c_read_b_per_cycle_per_sm,
                 row_drop_to_cread);
    std::printf("%-26s %.6f ms  MMA %.1f TF/s  LD %.1f TB/s  equiv %.1f TB/s\n",
                r.mode, r.elapsed_ms, r.mma_tflops, r.ld_tbps, equiv);
  }
  std::fclose(f);
  std::printf("wrote %s\n", args.peak_csv);

  FILE* claim = std::fopen(args.claim_csv, "w");
  if (!claim) {
    std::perror(args.claim_csv);
    std::exit(1);
  }
  std::fprintf(
      claim,
      "blocks,repeats,ld_repeats,sms,clock_mhz,ld_peak_TBps,overlap_ld_TBps,"
      "observed_ld_drop_TBps,observed_ld_drop_B_per_cycle_per_SM,overlap_mma_TFLOP_per_s,"
      "reference_mma_TFLOP_per_s,flops_per_mma,acc32_tile_bytes,"
      "reference_mma_C_read_TBps,reference_mma_C_read_B_per_cycle_per_SM,"
      "reference_mma_C_read_plus_write_TBps,drop_to_C_read_ratio,"
      "reference_ld8_peak_TBps,drop_fraction_of_local_ld_peak,"
      "drop_scaled_to_reference_ld8_peak_TBps,claim,status\n");
  std::fprintf(
      claim,
      "%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,%.3f,%.3f,"
      "%.3f,%.6f,%.3f,%.6f,%.3f,%s,%s\n",
      args.blocks, args.repeats, args.ld_repeats, args.sms, args.clock_mhz, ld_peak,
      overlap_ld, observed_drop_tbps, observed_drop_b_per_cycle_per_sm, overlap_mma,
      args.reference_mma_tflops, kFlopsPerMma, kAcc32TileBytes, reference_c_read_tbps,
      reference_c_read_b_per_cycle_per_sm, reference_c_read_plus_write_tbps,
      observed_to_cread_ratio, args.reference_ld8_peak_tbps,
      ld_peak > 0.0 ? observed_drop_tbps / ld_peak : 0.0, scaled_to_ld8_peak_tbps,
      "observed_ld_drop_matches_reference_mma_C_read_demand",
      "ok");
  std::fclose(claim);
  std::printf("claim: LD drop %.3f TB/s (%.1f B/cyc/SM), reference C-read %.3f TB/s "
              "(%.1f B/cyc/SM), ratio %.3f\n",
              observed_drop_tbps, observed_drop_b_per_cycle_per_sm, reference_c_read_tbps,
              reference_c_read_b_per_cycle_per_sm, observed_to_cread_ratio);
  std::printf("wrote %s\n", args.claim_csv);
}

}  // namespace

int main(int argc, char** argv) {
  Args args{};
  parse_args(argc, argv, &args);

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This toy requires SM100+; got sm_%d%d\n", prop.major, prop.minor);
    return 1;
  }

  if (!args.peak_only) {
    std::vector<TraceRow> rows;
    rows.reserve(static_cast<size_t>(args.repeats) * kModeCount);
    run_mode<0, true>(kModeLdOnly, args, &rows);
    run_mode<1, false>(kModeMma1Only, args, &rows);
    run_mode<1, true>(kModeOverlapMma1, args, &rows);
    run_mode<2, false>(kModeMma2Only, args, &rows);
    run_mode<2, true>(kModeOverlapMma2, args, &rows);
    run_mode<3, false>(kModeMma3Only, args, &rows);
    run_mode<3, true>(kModeOverlapMma3, args, &rows);

    write_detail_csv(args.csv, rows);
    write_summary_csv(args.summary_csv, rows, args);
    print_summary(rows, args);
    std::printf("wrote %s\nwrote %s\n", args.csv, args.summary_csv);
  }
  run_peak_sweep(args);
  return 0;
}
