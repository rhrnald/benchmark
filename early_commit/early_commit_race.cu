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
static constexpr int kProducerWarp = 0;
static constexpr int kConsumerWarp = 1;
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kMaxTargetMmas = 8;
static constexpr int kTileBf16Elems = kTileM * kTileN;
static constexpr int kTileWords = kTileBf16Elems / 2;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kDynamicSmemBytes = 2 * kTileBytes + 1024;

#ifndef EARLY_COMMIT_TARGET_MMAS
#define EARLY_COMMIT_TARGET_MMAS 8
#endif

#ifndef EARLY_COMMIT_FULL_EXTRA_MMAS
#define EARLY_COMMIT_FULL_EXTRA_MMAS 0
#endif

#ifndef EARLY_COMMIT_WARPS
#define EARLY_COMMIT_WARPS 4
#endif

static constexpr int kCompileTimeTargetMmas = EARLY_COMMIT_TARGET_MMAS;
static constexpr int kCompileTimeFullExtraMmas = EARLY_COMMIT_FULL_EXTRA_MMAS;
static constexpr int kWarps = EARLY_COMMIT_WARPS;
static constexpr int kThreads = kWarps * kWarpSize;
static_assert(kCompileTimeTargetMmas >= 1, "EARLY_COMMIT_TARGET_MMAS must be positive");
static_assert(kCompileTimeTargetMmas <= kMaxTargetMmas,
              "EARLY_COMMIT_TARGET_MMAS exceeds descriptor capacity");
static_assert(kCompileTimeFullExtraMmas >= 0,
              "EARLY_COMMIT_FULL_EXTRA_MMAS must be non-negative");
static_assert(kCompileTimeFullExtraMmas <= kMaxTargetMmas,
              "EARLY_COMMIT_FULL_EXTRA_MMAS exceeds descriptor capacity");
static_assert(kWarps >= 2, "EARLY_COMMIT_WARPS must include producer and consumer warps");
static_assert(kWarps <= 32, "EARLY_COMMIT_WARPS is unexpectedly large");

struct Args {
  int blocks = 512;
  int target_mmas = kCompileTimeTargetMmas;
  int full_extra_mmas = kCompileTimeFullExtraMmas;
  int warmup = 1;
  const char* early_targets = "8";
  const char* early_extras = "0";
  const char* delays = "0,1,2,3,4,5,6,7,8,10,12,16,24,32,48,64,96,128";
  const char* csv = "early_commit/log/early_commit_race_detail.csv";
  const char* summary_csv = "early_commit/log/early_commit_race_summary.csv";
  bool model = false;
  int probe_gap_cycles = 2048;
  const char* model_csv = "early_commit/log/early_commit_model.csv";
};

struct Record {
  uint32_t combo_id;
  uint32_t block;
  uint32_t target_mmas;
  uint32_t early_target_mmas;
  uint32_t early_extra_mmas;
  uint32_t full_extra_mmas;
  uint32_t delay_cycles;
  uint64_t target_issue_end;
  uint64_t early_commit_end;
  uint64_t early_commit_issue_end;
  uint64_t remaining_target_issue_start;
  uint64_t early_wait_end;
  uint64_t ld_start;
  uint64_t ld_end;
  uint64_t full_issue_end;
  uint64_t full_done_end;
  uint64_t ref_ld_start;
  uint64_t ref_ld_end;
  uint32_t mismatch_words;
  uint32_t mismatch_lanes;
  uint32_t early_sig;
  uint32_t ref_sig;
};

struct Combo {
  int early_target_mmas;
  int early_extra_mmas;
  int delay_cycles;
};

struct ModelRecord {
  uint32_t block;
  uint32_t target_mmas;
  uint32_t early_target_mmas;
  uint32_t probe_gap_cycles;
  uint64_t early_commit_end;
  uint64_t early_commit_issue_end;
  uint64_t remaining_target_issue_start;
  uint64_t target_issue_end;
  uint64_t full_commit_start;
  uint64_t full_commit_issue_end;
  uint64_t producer_full_wait_end;
  uint64_t early_wait_end;
  uint64_t full_wait_start;
  uint64_t full_wait_end;
  uint64_t late_mma_issue_end;
  uint64_t late_commit_start;
  uint64_t late_commit_issue_end;
  uint64_t late_wait_start;
  uint64_t late_wait_end;
  uint64_t ready_mma_issue_end;
  uint64_t ready_commit_start;
  uint64_t ready_commit_issue_end;
  uint64_t ready_wait_start;
  uint64_t ready_wait_end;
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

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void clock_delay_cycles(uint32_t cycles) {
  if (cycles == 0) return;
  const uint64_t start = clock64();
  while (clock64() - start < static_cast<uint64_t>(cycles)) {
  }
}

#define TCGEN05_LD_X64_OUTPUTS(a)                                            \
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

#define TCGEN05_LD_X64_OPERANDS                                              \
  "%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "  \
  "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "  \
  "%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "  \
  "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "  \
  "%58, %59, %60, %61, %62, %63"

#define TCGEN05_LD_X64(src_taddr, out_regs)                                  \
  asm volatile(                                                              \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TCGEN05_LD_X64_OPERANDS     \
      "}, [%64];"                                                            \
      : TCGEN05_LD_X64_OUTPUTS(out_regs)                                     \
      : "r"(src_taddr)                                                       \
      : "memory")

#define TCGEN05_TIMED_LD_X64(src_taddr, out_regs, ld_clock)                   \
  asm volatile(                                                               \
      "{ .reg .u64 t; "                                                       \
      "mov.u64 t, %%clock64; "                                                \
      "mov.u64 %64, t; "                                                      \
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TCGEN05_LD_X64_OPERANDS      \
      "}, [%65]; }"                                                           \
      : TCGEN05_LD_X64_OUTPUTS(out_regs), "=&l"(ld_clock)                     \
      : "r"(src_taddr)                                                        \
      : "memory")

__device__ __forceinline__ void issue_target_mma(int mma,
                                                 uint32_t target_taddr,
                                                 const uint64_t* q_desc,
                                                 const uint64_t* k_desc,
                                                 uint32_t idesc) {
  tcgen05_mma_bf16_ss(target_taddr, q_desc[mma & 7], k_desc[mma & 7], idesc, mma != 0);
}

__device__ __forceinline__ void issue_extra_mma(int mma,
                                                uint32_t extra_taddr,
                                                const uint64_t* q_desc,
                                                const uint64_t* k_desc,
                                                uint32_t idesc) {
  tcgen05_mma_bf16_ss(extra_taddr, q_desc[mma & 7], k_desc[mma & 7], idesc, mma != 0);
}

template <int EarlyTargetMmas, int EarlyExtraMmas>
__global__ __launch_bounds__(kThreads, 1)
void early_commit_race_kernel(Record* __restrict__ records,
                              int delay_cycles_after_early_wait,
                              uint32_t combo_id) {
  static_assert(EarlyTargetMmas >= 0, "EarlyTargetMmas must be non-negative");
  static_assert(EarlyTargetMmas <= kCompileTimeTargetMmas,
                "EarlyTargetMmas exceeds target MMA count");
  static_assert(EarlyExtraMmas >= 0, "EarlyExtraMmas must be non-negative");
  static_assert(EarlyExtraMmas <= kCompileTimeFullExtraMmas,
                "EarlyExtraMmas exceeds full extra MMA count");
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)records;
  (void)delay_cycles_after_early_wait;
  (void)combo_id;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);

  __shared__ uint64_t early_done;
  __shared__ uint64_t full_done;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint64_t base_clock_shared;
  __shared__ uint64_t q_desc_shared[kMaxTargetMmas];
  __shared__ uint64_t k_desc_shared[kMaxTargetMmas];
  __shared__ uint32_t mismatch_words_shared[kWarpSize];
  __shared__ uint32_t mismatch_lanes_shared[kWarpSize];
  __shared__ uint32_t early_sig_shared[kWarpSize];
  __shared__ uint32_t ref_sig_shared[kWarpSize];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tid = threadIdx.x;
  Record& rec = records[blockIdx.x];

  if (warp_id == kProducerWarp) {
    setmaxnreg_dec_producer();
  } else if (warp_id == kConsumerWarp) {
    setmaxnreg_inc_consumer();
  }

  if (tid == 0) {
    rec.combo_id = combo_id;
    rec.block = blockIdx.x;
    rec.target_mmas = static_cast<uint32_t>(kCompileTimeTargetMmas);
    rec.early_target_mmas = static_cast<uint32_t>(EarlyTargetMmas);
    rec.early_extra_mmas = static_cast<uint32_t>(EarlyExtraMmas);
    rec.full_extra_mmas = static_cast<uint32_t>(kCompileTimeFullExtraMmas);
    rec.delay_cycles = static_cast<uint32_t>(delay_cycles_after_early_wait);
    rec.target_issue_end = 0;
    rec.early_commit_end = 0;
    rec.early_commit_issue_end = 0;
    rec.remaining_target_issue_start = 0;
    rec.early_wait_end = 0;
    rec.ld_start = 0;
    rec.ld_end = 0;
    rec.full_issue_end = 0;
    rec.full_done_end = 0;
    rec.ref_ld_start = 0;
    rec.ref_ld_end = 0;
    rec.mismatch_words = 0;
    rec.mismatch_lanes = 0;
    rec.early_sig = 0;
    rec.ref_sig = 0;
    mbarrier_init(&early_done, 1);
    mbarrier_init(&full_done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }

  for (int i = tid; i < 2 * kTileWords; i += kThreads) {
    const uint32_t lo = 0x3f80u ^ static_cast<uint32_t>((i * 17 + blockIdx.x * 13 + 3) & 0x7fu);
    const uint32_t hi = 0x4000u ^ static_cast<uint32_t>((i * 29 + blockIdx.x * 7 + 5) & 0x7fu);
    smem_words[i] = (hi << 16) | lo;
  }
  __syncthreads();

  if (warp_id == kProducerWarp) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t target_taddr = tmem_base;
  const uint32_t extra_taddr = tmem_base + 128u;
  const uint32_t idesc = make_qk_idesc();
  uint32_t* q_smem = smem_words;
  uint32_t* k_smem = smem_words + kTileWords;

  if (warp_id == kProducerWarp && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMaxTargetMmas; ++mma) {
      q_desc_shared[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc_shared[mma] = make_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
    }
  }
  __syncthreads();

  if (tid == 0) base_clock_shared = clock64();
  __syncthreads();

  const uint64_t base_clock = base_clock_shared;

  if (warp_id == kProducerWarp && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < EarlyTargetMmas; ++mma) {
      issue_target_mma(mma, target_taddr, q_desc_shared, k_desc_shared, idesc);
    }
#pragma unroll
    for (int mma = 0; mma < EarlyExtraMmas; ++mma) {
      issue_extra_mma(mma, extra_taddr, q_desc_shared, k_desc_shared, idesc);
    }
    rec.early_commit_end = clock64() - base_clock;
    tcgen05_commit(&early_done);
    rec.early_commit_issue_end = clock64() - base_clock;

    rec.remaining_target_issue_start = clock64() - base_clock;
#pragma unroll
    for (int mma = EarlyTargetMmas; mma < kCompileTimeTargetMmas; ++mma) {
      issue_target_mma(mma, target_taddr, q_desc_shared, k_desc_shared, idesc);
    }
    rec.target_issue_end = clock64() - base_clock;
#pragma unroll
    for (int mma = EarlyExtraMmas; mma < kCompileTimeFullExtraMmas; ++mma) {
      issue_extra_mma(mma, extra_taddr, q_desc_shared, k_desc_shared, idesc);
    }
    rec.full_issue_end = clock64() - base_clock;
    tcgen05_commit(&full_done);
    mbarrier_wait(&full_done, 0);
    rec.full_done_end = clock64() - base_clock;
  }

  if (warp_id == kConsumerWarp) {
    mbarrier_wait(&early_done, 0);
    const uint64_t early_wait_end = clock64() - base_clock;
    if (lane == 0) rec.early_wait_end = early_wait_end;
    clock_delay_cycles(static_cast<uint32_t>(delay_cycles_after_early_wait));

    uint32_t early_regs[64];
    uint32_t ref_regs[64];
    uint64_t ld_clock = 0;
    TCGEN05_TIMED_LD_X64(target_taddr, early_regs, ld_clock);
    const uint64_t ld_start = ld_clock - base_clock;
    tcgen05_wait_ld();
    const uint64_t ld_end = clock64() - base_clock;
    if (lane == 0) {
      rec.ld_start = ld_start;
      rec.ld_end = ld_end;
    }

    mbarrier_wait(&full_done, 0);
    const uint64_t ref_ld_start = clock64() - base_clock;
    TCGEN05_LD_X64(target_taddr, ref_regs);
    tcgen05_wait_ld();
    const uint64_t ref_ld_end = clock64() - base_clock;

    uint32_t mismatch = 0;
    const uint32_t sig_seed = 0x9e3779b9u ^ static_cast<uint32_t>(lane);
    uint32_t early_sig = sig_seed;
    uint32_t ref_sig = sig_seed;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      mismatch += early_regs[i] != ref_regs[i] ? 1u : 0u;
      early_sig ^= early_regs[i] + 0x9e3779b9u + (early_sig << 6) + (early_sig >> 2);
      ref_sig ^= ref_regs[i] + 0x9e3779b9u + (ref_sig << 6) + (ref_sig >> 2);
    }
    mismatch_words_shared[lane] = mismatch;
    mismatch_lanes_shared[lane] = mismatch != 0 ? 1u : 0u;
    early_sig_shared[lane] = early_sig;
    ref_sig_shared[lane] = ref_sig;
    __syncwarp();
    if (lane == 0) {
      uint32_t total_mismatch = 0;
      uint32_t total_lanes = 0;
      uint32_t all_early_sig = 0;
      uint32_t all_ref_sig = 0;
      for (int i = 0; i < kWarpSize; ++i) {
        total_mismatch += mismatch_words_shared[i];
        total_lanes += mismatch_lanes_shared[i];
        all_early_sig ^= early_sig_shared[i] + static_cast<uint32_t>(i * 0x45d9f3bu);
        all_ref_sig ^= ref_sig_shared[i] + static_cast<uint32_t>(i * 0x45d9f3bu);
      }
      rec.ref_ld_start = ref_ld_start;
      rec.ref_ld_end = ref_ld_end;
      rec.mismatch_words = total_mismatch;
      rec.mismatch_lanes = total_lanes;
      rec.early_sig = all_early_sig;
      rec.ref_sig = all_ref_sig;
    }
  }

  __syncthreads();
  if (warp_id == kProducerWarp) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == kProducerWarp) tcgen05_relinquish_alloc_permit();
#endif
}

template <int EarlyTargetMmas>
__global__ __launch_bounds__(kThreads, 1)
void early_commit_model_kernel(ModelRecord* __restrict__ records, int probe_gap_cycles) {
  static_assert(EarlyTargetMmas >= 0, "EarlyTargetMmas must be non-negative");
  static_assert(EarlyTargetMmas <= kCompileTimeTargetMmas,
                "EarlyTargetMmas exceeds target MMA count");
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)records;
  (void)probe_gap_cycles;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);

  __shared__ uint64_t early_done;
  __shared__ uint64_t full_done;
  __shared__ uint64_t late_done;
  __shared__ uint64_t ready_done;
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint64_t base_clock_shared;
  __shared__ uint64_t q_desc_shared[kMaxTargetMmas];
  __shared__ uint64_t k_desc_shared[kMaxTargetMmas];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tid = threadIdx.x;
  ModelRecord& rec = records[blockIdx.x];

  if (warp_id == kProducerWarp) {
    setmaxnreg_dec_producer();
  } else if (warp_id == kConsumerWarp) {
    setmaxnreg_inc_consumer();
  }

  if (tid == 0) {
    rec.block = blockIdx.x;
    rec.target_mmas = static_cast<uint32_t>(kCompileTimeTargetMmas);
    rec.early_target_mmas = static_cast<uint32_t>(EarlyTargetMmas);
    rec.probe_gap_cycles = static_cast<uint32_t>(probe_gap_cycles);
    rec.early_commit_end = 0;
    rec.early_commit_issue_end = 0;
    rec.remaining_target_issue_start = 0;
    rec.target_issue_end = 0;
    rec.full_commit_start = 0;
    rec.full_commit_issue_end = 0;
    rec.producer_full_wait_end = 0;
    rec.early_wait_end = 0;
    rec.full_wait_start = 0;
    rec.full_wait_end = 0;
    rec.late_mma_issue_end = 0;
    rec.late_commit_start = 0;
    rec.late_commit_issue_end = 0;
    rec.late_wait_start = 0;
    rec.late_wait_end = 0;
    rec.ready_mma_issue_end = 0;
    rec.ready_commit_start = 0;
    rec.ready_commit_issue_end = 0;
    rec.ready_wait_start = 0;
    rec.ready_wait_end = 0;
    mbarrier_init(&early_done, 1);
    mbarrier_init(&full_done, 1);
    mbarrier_init(&late_done, 1);
    mbarrier_init(&ready_done, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }

  for (int i = tid; i < 2 * kTileWords; i += kThreads) {
    const uint32_t lo = 0x3f80u ^ static_cast<uint32_t>((i * 17 + blockIdx.x * 13 + 3) & 0x7fu);
    const uint32_t hi = 0x4000u ^ static_cast<uint32_t>((i * 29 + blockIdx.x * 7 + 5) & 0x7fu);
    smem_words[i] = (hi << 16) | lo;
  }
  __syncthreads();

  if (warp_id == kProducerWarp) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t target_taddr = tmem_base;
  const uint32_t late_taddr = tmem_base + 128u;
  const uint32_t ready_taddr = tmem_base + 256u;
  const uint32_t idesc = make_qk_idesc();
  uint32_t* q_smem = smem_words;
  uint32_t* k_smem = smem_words + kTileWords;

  if (warp_id == kProducerWarp && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < kMaxTargetMmas; ++mma) {
      q_desc_shared[mma] = make_smem_desc(smem_ptr_u32(q_smem + mma * 1024));
      k_desc_shared[mma] = make_smem_desc(smem_ptr_u32(k_smem + mma * 1024));
    }
  }
  __syncthreads();

  if (tid == 0) base_clock_shared = clock64();
  __syncthreads();

  const uint64_t base_clock = base_clock_shared;

  if (warp_id == kProducerWarp && lane == 0) {
#pragma unroll
    for (int mma = 0; mma < EarlyTargetMmas; ++mma) {
      issue_target_mma(mma, target_taddr, q_desc_shared, k_desc_shared, idesc);
    }
    rec.early_commit_end = clock64() - base_clock;
    tcgen05_commit(&early_done);
    rec.early_commit_issue_end = clock64() - base_clock;

    rec.remaining_target_issue_start = clock64() - base_clock;
#pragma unroll
    for (int mma = EarlyTargetMmas; mma < kCompileTimeTargetMmas; ++mma) {
      issue_target_mma(mma, target_taddr, q_desc_shared, k_desc_shared, idesc);
    }
    rec.target_issue_end = clock64() - base_clock;
    rec.full_commit_start = clock64() - base_clock;
    tcgen05_commit(&full_done);
    rec.full_commit_issue_end = clock64() - base_clock;
    mbarrier_wait(&full_done, 0);
    rec.producer_full_wait_end = clock64() - base_clock;

    tcgen05_mma_bf16_ss(late_taddr, q_desc_shared[0], k_desc_shared[0], idesc, false);
    rec.late_mma_issue_end = clock64() - base_clock;
    rec.late_commit_start = clock64() - base_clock;
    tcgen05_commit(&late_done);
    rec.late_commit_issue_end = clock64() - base_clock;

    clock_delay_cycles(static_cast<uint32_t>(probe_gap_cycles));

    tcgen05_mma_bf16_ss(ready_taddr, q_desc_shared[1], k_desc_shared[1], idesc, false);
    rec.ready_mma_issue_end = clock64() - base_clock;
    clock_delay_cycles(static_cast<uint32_t>(probe_gap_cycles));
    rec.ready_commit_start = clock64() - base_clock;
    tcgen05_commit(&ready_done);
    rec.ready_commit_issue_end = clock64() - base_clock;
  }

  if (warp_id == kConsumerWarp) {
    mbarrier_wait(&early_done, 0);
    if (lane == 0) rec.early_wait_end = clock64() - base_clock;

    const uint64_t full_wait_start = clock64() - base_clock;
    if (lane == 0) rec.full_wait_start = full_wait_start;
    mbarrier_wait(&full_done, 0);
    if (lane == 0) rec.full_wait_end = clock64() - base_clock;

    clock_delay_cycles(static_cast<uint32_t>(probe_gap_cycles));
    const uint64_t late_wait_start = clock64() - base_clock;
    if (lane == 0) rec.late_wait_start = late_wait_start;
    mbarrier_wait(&late_done, 0);
    if (lane == 0) rec.late_wait_end = clock64() - base_clock;

    const uint64_t ready_wait_start = clock64() - base_clock;
    if (lane == 0) rec.ready_wait_start = ready_wait_start;
    mbarrier_wait(&ready_done, 0);
    if (lane == 0) rec.ready_wait_end = clock64() - base_clock;
  }

  __syncthreads();
  if (warp_id == kProducerWarp) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == kProducerWarp) tcgen05_relinquish_alloc_permit();
#endif
}

#undef TCGEN05_LD_X64
#undef TCGEN05_TIMED_LD_X64
#undef TCGEN05_LD_X64_OUTPUTS
#undef TCGEN05_LD_X64_OPERANDS

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

std::vector<int> parse_list(const char* text) {
  std::vector<int> values;
  const std::string s(text ? text : "");
  size_t start = 0;
  while (start < s.size()) {
    const size_t comma = s.find(',', start);
    const std::string part = s.substr(start, comma == std::string::npos ? comma : comma - start);
    if (!part.empty()) values.push_back(std::atoi(part.c_str()));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return values;
}

void parse_args(int argc, char** argv, Args* args) {
  if (has_flag(argc, argv, "--help")) {
    std::printf(
        "Usage: early_commit_race [--blocks N] [--target-mmas N] [--full-extra-mmas N]\n"
        "                         [--early-targets list] [--early-extras list]\n"
        "                         [--delays list] [--warmup N]\n"
        "                         [--csv path] [--summary-csv path]\n"
        "                         [--model] [--model-csv path]\n"
        "                         [--probe-gap-cycles N]\n\n"
        "target_mmas and full_extra_mmas are compile-time fixed by the build:\n"
        "target_mmas=%d, full_extra_mmas=%d.\n"
        "Default sweep: early_targets=8, early_extras=0, clock64 busy-wait delays=0..128.\n",
        kCompileTimeTargetMmas, kCompileTimeFullExtraMmas);
    std::exit(0);
  }
  args->blocks = std::atoi(arg_value(argc, argv, "--blocks", "512"));
  if (const char* target_mmas = arg_value(argc, argv, "--target-mmas", nullptr)) {
    args->target_mmas = std::atoi(target_mmas);
  }
  if (const char* full_extra_mmas = arg_value(argc, argv, "--full-extra-mmas", nullptr)) {
    args->full_extra_mmas = std::atoi(full_extra_mmas);
  }
  args->warmup = std::atoi(arg_value(argc, argv, "--warmup", "1"));
  args->early_targets = arg_value(argc, argv, "--early-targets", args->early_targets);
  args->early_extras = arg_value(argc, argv, "--early-extras", args->early_extras);
  args->delays = arg_value(argc, argv, "--delays", args->delays);
  args->csv = arg_value(argc, argv, "--csv", args->csv);
  args->summary_csv = arg_value(argc, argv, "--summary-csv", args->summary_csv);
  args->model = has_flag(argc, argv, "--model");
  args->probe_gap_cycles = std::atoi(arg_value(argc, argv, "--probe-gap-cycles", "2048"));
  args->model_csv = arg_value(argc, argv, "--model-csv", args->model_csv);
}

std::vector<Combo> make_combos(const Args& args) {
  const std::vector<int> early_targets = parse_list(args.early_targets);
  const std::vector<int> early_extras = parse_list(args.early_extras);
  const std::vector<int> delays = parse_list(args.delays);
  std::vector<Combo> combos;
  for (int early_target : early_targets) {
    for (int early_extra : early_extras) {
      for (int delay : delays) {
        if (early_target < 0 || early_target > args.target_mmas) continue;
        if (early_extra < 0 || early_extra > args.full_extra_mmas) continue;
        if (delay < 0) continue;
        combos.push_back({early_target, early_extra, delay});
      }
    }
  }
  return combos;
}

long long signed_delta(uint64_t end, uint64_t start) {
  if (end == 0 || start == 0) return 0;
  return static_cast<long long>(end) - static_cast<long long>(start);
}

void write_detail_csv(const char* path, const std::vector<Record>& rows) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(f,
               "combo_id,block,target_mmas,early_target_mmas,early_extra_mmas,"
               "full_extra_mmas,delay_cycles,target_issue_end,early_commit_end,"
               "early_commit_issue_end,remaining_target_issue_start,"
               "early_wait_end,ld_start,ld_end,full_issue_end,full_done_end,"
               "ref_ld_start,ref_ld_end,ld_start_ahead_of_full_done,"
               "ld_end_ahead_of_full_done,early_wait_to_ld_start,"
               "commit_issue_cycles,remaining_target_issue_delay,"
               "remaining_target_issue_cycles,ld_start_after_remaining_target_issue_start,"
               "ld_start_after_target_issue_end,"
               "mismatch_words,mismatch_lanes,safe,early_sig,ref_sig\n");
  for (const Record& r : rows) {
    const bool safe = r.mismatch_words == 0;
    std::fprintf(f,
                 "%u,%u,%u,%u,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%u,%u,%d,0x%08x,0x%08x\n",
                 r.combo_id, r.block, r.target_mmas, r.early_target_mmas,
                 r.early_extra_mmas, r.full_extra_mmas, r.delay_cycles,
                 static_cast<unsigned long long>(r.target_issue_end),
                 static_cast<unsigned long long>(r.early_commit_end),
                 static_cast<unsigned long long>(r.early_commit_issue_end),
                 static_cast<unsigned long long>(r.remaining_target_issue_start),
                 static_cast<unsigned long long>(r.early_wait_end),
                 static_cast<unsigned long long>(r.ld_start),
                 static_cast<unsigned long long>(r.ld_end),
                 static_cast<unsigned long long>(r.full_issue_end),
                 static_cast<unsigned long long>(r.full_done_end),
                 static_cast<unsigned long long>(r.ref_ld_start),
                 static_cast<unsigned long long>(r.ref_ld_end),
                 signed_delta(r.full_done_end, r.ld_start),
                 signed_delta(r.full_done_end, r.ld_end),
                 signed_delta(r.ld_start, r.early_wait_end),
                 signed_delta(r.early_commit_issue_end, r.early_commit_end),
                 signed_delta(r.remaining_target_issue_start, r.early_commit_issue_end),
                 signed_delta(r.target_issue_end, r.remaining_target_issue_start),
                 signed_delta(r.ld_start, r.remaining_target_issue_start),
                 signed_delta(r.ld_start, r.target_issue_end), r.mismatch_words,
                 r.mismatch_lanes, safe ? 1 : 0, r.early_sig, r.ref_sig);
  }
  std::fclose(f);
}

double mean_or_zero(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  double sum = 0.0;
  for (double v : values) sum += v;
  return sum / static_cast<double>(values.size());
}

double min_or_zero(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return *std::min_element(values.begin(), values.end());
}

double max_or_zero(const std::vector<double>& values) {
  if (values.empty()) return 0.0;
  return *std::max_element(values.begin(), values.end());
}

void write_summary_csv(const char* path,
                       const std::vector<Record>& rows,
                       const std::vector<Combo>& combos) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(f,
               "combo_id,early_target_mmas,early_extra_mmas,delay_cycles,n,safe_n,"
               "unsafe_n,safe_rate,avg_ld_start_ahead,avg_ld_end_ahead,"
               "max_safe_ld_start_ahead,min_unsafe_ld_start_ahead,"
               "avg_early_wait_to_ld_start,avg_commit_issue_cycles,"
               "avg_remaining_target_issue_delay,avg_remaining_target_issue_cycles,"
               "avg_ld_start_after_remaining_target_issue_start,"
               "avg_ld_start_after_target_issue_end,avg_mismatch_words,status\n");
  for (size_t combo_id = 0; combo_id < combos.size(); ++combo_id) {
    int n = 0;
    int safe_n = 0;
    int unsafe_n = 0;
    std::vector<double> ld_start_ahead;
    std::vector<double> ld_end_ahead;
    std::vector<double> safe_ld_start_ahead;
    std::vector<double> unsafe_ld_start_ahead;
    std::vector<double> early_wait_to_ld_start;
    std::vector<double> commit_issue_cycles;
    std::vector<double> remaining_target_issue_delay;
    std::vector<double> remaining_target_issue_cycles;
    std::vector<double> ld_start_after_remaining_target_issue_start;
    std::vector<double> ld_start_after_target_issue_end;
    std::vector<double> mismatch_words;
    for (const Record& r : rows) {
      if (r.combo_id != combo_id) continue;
      const bool safe = r.mismatch_words == 0;
      ++n;
      safe ? ++safe_n : ++unsafe_n;
      const double start_ahead = static_cast<double>(signed_delta(r.full_done_end, r.ld_start));
      const double end_ahead = static_cast<double>(signed_delta(r.full_done_end, r.ld_end));
      ld_start_ahead.push_back(start_ahead);
      ld_end_ahead.push_back(end_ahead);
      early_wait_to_ld_start.push_back(static_cast<double>(signed_delta(r.ld_start, r.early_wait_end)));
      commit_issue_cycles.push_back(
          static_cast<double>(signed_delta(r.early_commit_issue_end, r.early_commit_end)));
      remaining_target_issue_delay.push_back(static_cast<double>(
          signed_delta(r.remaining_target_issue_start, r.early_commit_issue_end)));
      remaining_target_issue_cycles.push_back(static_cast<double>(
          signed_delta(r.target_issue_end, r.remaining_target_issue_start)));
      ld_start_after_remaining_target_issue_start.push_back(static_cast<double>(
          signed_delta(r.ld_start, r.remaining_target_issue_start)));
      ld_start_after_target_issue_end.push_back(
          static_cast<double>(signed_delta(r.ld_start, r.target_issue_end)));
      mismatch_words.push_back(static_cast<double>(r.mismatch_words));
      if (safe) {
        safe_ld_start_ahead.push_back(start_ahead);
      } else {
        unsafe_ld_start_ahead.push_back(start_ahead);
      }
    }
    const Combo& c = combos[combo_id];
    std::fprintf(f,
                 "%zu,%d,%d,%d,%d,%d,%d,%.6f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
                 "%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
                 combo_id, c.early_target_mmas, c.early_extra_mmas, c.delay_cycles, n,
                 safe_n, unsafe_n, n > 0 ? static_cast<double>(safe_n) / n : 0.0,
                 mean_or_zero(ld_start_ahead), mean_or_zero(ld_end_ahead),
                 max_or_zero(safe_ld_start_ahead), min_or_zero(unsafe_ld_start_ahead),
                 mean_or_zero(early_wait_to_ld_start), mean_or_zero(commit_issue_cycles),
                 mean_or_zero(remaining_target_issue_delay),
                 mean_or_zero(remaining_target_issue_cycles),
                 mean_or_zero(ld_start_after_remaining_target_issue_start),
                 mean_or_zero(ld_start_after_target_issue_end), mean_or_zero(mismatch_words),
                 unsafe_n == 0 ? "safe" : (safe_n == 0 ? "unsafe" : "mixed"));
  }
  std::fclose(f);
}

void print_summary(const std::vector<Record>& rows, const std::vector<Combo>& combos) {
  std::printf("combo early_target early_extra delay safe/n avg_start_ahead status\n");
  for (size_t combo_id = 0; combo_id < combos.size(); ++combo_id) {
    int n = 0;
    int safe_n = 0;
    std::vector<double> start_ahead;
    for (const Record& r : rows) {
      if (r.combo_id != combo_id) continue;
      ++n;
      if (r.mismatch_words == 0) ++safe_n;
      start_ahead.push_back(static_cast<double>(signed_delta(r.full_done_end, r.ld_start)));
    }
    const Combo& c = combos[combo_id];
    std::printf("%5zu %12d %11d %5d %4d/%-4d %15.3f %s\n", combo_id,
                c.early_target_mmas, c.early_extra_mmas, c.delay_cycles, safe_n, n,
                mean_or_zero(start_ahead),
                safe_n == n ? "safe" : (safe_n == 0 ? "unsafe" : "mixed"));
  }
}

void write_model_csv(const char* path, const std::vector<ModelRecord>& rows) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(
      f,
      "block,target_mmas,early_target_mmas,probe_gap_cycles,"
      "early_commit_end,early_commit_issue_end,remaining_target_issue_start,"
      "target_issue_end,full_commit_start,full_commit_issue_end,"
      "producer_full_wait_end,early_wait_end,full_wait_start,full_wait_end,"
      "late_mma_issue_end,late_commit_start,late_commit_issue_end,"
      "late_wait_start,late_wait_end,ready_mma_issue_end,ready_commit_start,"
      "ready_commit_issue_end,ready_wait_start,ready_wait_end,"
      "early_wait_after_early_commit,full_wait_after_early_wait,"
      "full_wait_after_target_issue,t_late_wait,t_ready_from_commit_start,"
      "t_ready_from_commit_end,v_est_late_wait,v_est_ready_commit_start,"
      "v_est_ready_commit_end\n");
  for (const ModelRecord& r : rows) {
    const long long t_late_wait = signed_delta(r.late_wait_end, r.late_wait_start);
    const long long t_ready_from_start = signed_delta(r.ready_wait_end, r.ready_commit_start);
    const long long t_ready_from_end = signed_delta(r.ready_wait_end, r.ready_commit_issue_end);
    std::fprintf(
        f,
        "%u,%u,%u,%u,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
        "%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
        r.block, r.target_mmas, r.early_target_mmas, r.probe_gap_cycles,
        static_cast<unsigned long long>(r.early_commit_end),
        static_cast<unsigned long long>(r.early_commit_issue_end),
        static_cast<unsigned long long>(r.remaining_target_issue_start),
        static_cast<unsigned long long>(r.target_issue_end),
        static_cast<unsigned long long>(r.full_commit_start),
        static_cast<unsigned long long>(r.full_commit_issue_end),
        static_cast<unsigned long long>(r.producer_full_wait_end),
        static_cast<unsigned long long>(r.early_wait_end),
        static_cast<unsigned long long>(r.full_wait_start),
        static_cast<unsigned long long>(r.full_wait_end),
        static_cast<unsigned long long>(r.late_mma_issue_end),
        static_cast<unsigned long long>(r.late_commit_start),
        static_cast<unsigned long long>(r.late_commit_issue_end),
        static_cast<unsigned long long>(r.late_wait_start),
        static_cast<unsigned long long>(r.late_wait_end),
        static_cast<unsigned long long>(r.ready_mma_issue_end),
        static_cast<unsigned long long>(r.ready_commit_start),
        static_cast<unsigned long long>(r.ready_commit_issue_end),
        static_cast<unsigned long long>(r.ready_wait_start),
        static_cast<unsigned long long>(r.ready_wait_end),
        signed_delta(r.early_wait_end, r.early_commit_end),
        signed_delta(r.full_wait_end, r.early_wait_end),
        signed_delta(r.full_wait_end, r.target_issue_end),
        t_late_wait, t_ready_from_start, t_ready_from_end,
        static_cast<long long>(r.full_wait_end) - t_late_wait,
        static_cast<long long>(r.full_wait_end) - t_ready_from_start,
        static_cast<long long>(r.full_wait_end) - t_ready_from_end);
  }
  std::fclose(f);
}

void print_model_summary(const std::vector<ModelRecord>& rows) {
  std::vector<double> early_wait;
  std::vector<double> full_wait;
  std::vector<double> full_after_early;
  std::vector<double> t_late;
  std::vector<double> t_ready_start;
  std::vector<double> t_ready_end;
  std::vector<double> v_ready_start;
  for (const ModelRecord& r : rows) {
    early_wait.push_back(static_cast<double>(r.early_wait_end));
    full_wait.push_back(static_cast<double>(r.full_wait_end));
    full_after_early.push_back(static_cast<double>(signed_delta(r.full_wait_end, r.early_wait_end)));
    const double late = static_cast<double>(signed_delta(r.late_wait_end, r.late_wait_start));
    const double ready_start =
        static_cast<double>(signed_delta(r.ready_wait_end, r.ready_commit_start));
    const double ready_end =
        static_cast<double>(signed_delta(r.ready_wait_end, r.ready_commit_issue_end));
    t_late.push_back(late);
    t_ready_start.push_back(ready_start);
    t_ready_end.push_back(ready_end);
    v_ready_start.push_back(static_cast<double>(r.full_wait_end) - ready_start);
  }
  std::printf("model rows=%zu\n", rows.size());
  std::printf("E early_wait_end mean=%.3f min=%.3f max=%.3f\n", mean_or_zero(early_wait),
              min_or_zero(early_wait), max_or_zero(early_wait));
  std::printf("W full_wait_end mean=%.3f min=%.3f max=%.3f\n", mean_or_zero(full_wait),
              min_or_zero(full_wait), max_or_zero(full_wait));
  std::printf("W-E mean=%.3f min=%.3f max=%.3f\n", mean_or_zero(full_after_early),
              min_or_zero(full_after_early), max_or_zero(full_after_early));
  std::printf("T_late_wait mean=%.3f min=%.3f max=%.3f\n", mean_or_zero(t_late),
              min_or_zero(t_late), max_or_zero(t_late));
  std::printf("T_ready_commit_start mean=%.3f min=%.3f max=%.3f\n",
              mean_or_zero(t_ready_start), min_or_zero(t_ready_start),
              max_or_zero(t_ready_start));
  std::printf("T_ready_commit_end mean=%.3f min=%.3f max=%.3f\n",
              mean_or_zero(t_ready_end), min_or_zero(t_ready_end), max_or_zero(t_ready_end));
  std::printf("V_est=W-T_ready_start mean=%.3f min=%.3f max=%.3f\n",
              mean_or_zero(v_ready_start), min_or_zero(v_ready_start),
              max_or_zero(v_ready_start));
}

[[noreturn]] void die_unsupported_combo(const Combo& combo) {
  std::fprintf(stderr,
               "unsupported compile-time combo: early_target=%d early_extra=%d. "
               "Rebuild with larger EARLY_COMMIT_TARGET_MMAS/FULL_EXTRA_MMAS if needed.\n",
               combo.early_target_mmas, combo.early_extra_mmas);
  std::exit(1);
}

template <int EarlyTargetMmas, int EarlyExtraMmas>
void launch_specialized_combo(const Args& args,
                              const Combo& combo,
                              uint32_t combo_id,
                              Record* d_records) {
  CUDA_CHECK(cudaFuncSetAttribute(
      early_commit_race_kernel<EarlyTargetMmas, EarlyExtraMmas>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes));
  for (int i = 0; i < args.warmup; ++i) {
    early_commit_race_kernel<EarlyTargetMmas, EarlyExtraMmas>
        <<<args.blocks, kThreads, kDynamicSmemBytes>>>(d_records, combo.delay_cycles, combo_id);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  early_commit_race_kernel<EarlyTargetMmas, EarlyExtraMmas>
      <<<args.blocks, kThreads, kDynamicSmemBytes>>>(d_records, combo.delay_cycles, combo_id);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <int EarlyTargetMmas, int EarlyExtraMmas>
void dispatch_early_extra(const Args& args,
                          const Combo& combo,
                          uint32_t combo_id,
                          Record* d_records) {
  if (combo.early_extra_mmas == EarlyExtraMmas) {
    launch_specialized_combo<EarlyTargetMmas, EarlyExtraMmas>(args, combo, combo_id,
                                                             d_records);
    return;
  }
  if constexpr (EarlyExtraMmas < kCompileTimeFullExtraMmas) {
    dispatch_early_extra<EarlyTargetMmas, EarlyExtraMmas + 1>(args, combo, combo_id,
                                                             d_records);
  } else {
    die_unsupported_combo(combo);
  }
}

template <int EarlyTargetMmas>
void dispatch_early_target(const Args& args,
                           const Combo& combo,
                           uint32_t combo_id,
                           Record* d_records) {
  if (combo.early_target_mmas == EarlyTargetMmas) {
    dispatch_early_extra<EarlyTargetMmas, 0>(args, combo, combo_id, d_records);
    return;
  }
  if constexpr (EarlyTargetMmas < kCompileTimeTargetMmas) {
    dispatch_early_target<EarlyTargetMmas + 1>(args, combo, combo_id, d_records);
  } else {
    die_unsupported_combo(combo);
  }
}

[[noreturn]] void die_unsupported_model_target(int early_target_mmas) {
  std::fprintf(stderr,
               "unsupported model early_target=%d. "
               "Rebuild with larger EARLY_COMMIT_TARGET_MMAS if needed.\n",
               early_target_mmas);
  std::exit(1);
}

template <int EarlyTargetMmas>
void launch_model_specialized(const Args& args, ModelRecord* d_records) {
  CUDA_CHECK(cudaFuncSetAttribute(early_commit_model_kernel<EarlyTargetMmas>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  for (int i = 0; i < args.warmup; ++i) {
    early_commit_model_kernel<EarlyTargetMmas>
        <<<args.blocks, kThreads, kDynamicSmemBytes>>>(d_records, args.probe_gap_cycles);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  early_commit_model_kernel<EarlyTargetMmas>
      <<<args.blocks, kThreads, kDynamicSmemBytes>>>(d_records, args.probe_gap_cycles);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

template <int EarlyTargetMmas>
void dispatch_model_target(const Args& args, int early_target_mmas, ModelRecord* d_records) {
  if (early_target_mmas == EarlyTargetMmas) {
    launch_model_specialized<EarlyTargetMmas>(args, d_records);
    return;
  }
  if constexpr (EarlyTargetMmas < kCompileTimeTargetMmas) {
    dispatch_model_target<EarlyTargetMmas + 1>(args, early_target_mmas, d_records);
  } else {
    die_unsupported_model_target(early_target_mmas);
  }
}

void run_model(const Args& args, int early_target_mmas, std::vector<ModelRecord>* records) {
  ModelRecord* d_records = nullptr;
  CUDA_CHECK(cudaMalloc(&d_records, static_cast<size_t>(args.blocks) * sizeof(ModelRecord)));
  CUDA_CHECK(cudaMemset(d_records, 0, static_cast<size_t>(args.blocks) * sizeof(ModelRecord)));
  dispatch_model_target<0>(args, early_target_mmas, d_records);

  records->resize(args.blocks);
  CUDA_CHECK(cudaMemcpy(records->data(), d_records,
                        static_cast<size_t>(args.blocks) * sizeof(ModelRecord),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_records));
}

void run_combo(const Args& args,
               const Combo& combo,
               uint32_t combo_id,
               std::vector<Record>* all_records) {
  Record* d_records = nullptr;
  CUDA_CHECK(cudaMalloc(&d_records, static_cast<size_t>(args.blocks) * sizeof(Record)));
  CUDA_CHECK(cudaMemset(d_records, 0, static_cast<size_t>(args.blocks) * sizeof(Record)));
  dispatch_early_target<0>(args, combo, combo_id, d_records);

  const size_t old_size = all_records->size();
  all_records->resize(old_size + args.blocks);
  CUDA_CHECK(cudaMemcpy(all_records->data() + old_size, d_records,
                        static_cast<size_t>(args.blocks) * sizeof(Record),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_records));
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  parse_args(argc, argv, &args);
  if (args.target_mmas != kCompileTimeTargetMmas) {
    std::fprintf(stderr,
                 "--target-mmas=%d does not match compile-time target_mmas=%d; "
                 "rebuild with TARGET_MMAS=%d\n",
                 args.target_mmas, kCompileTimeTargetMmas, args.target_mmas);
    return 1;
  }
  if (args.full_extra_mmas != kCompileTimeFullExtraMmas) {
    std::fprintf(stderr,
                 "--full-extra-mmas=%d does not match compile-time full_extra_mmas=%d; "
                 "rebuild with FULL_EXTRA_MMAS=%d\n",
                 args.full_extra_mmas, kCompileTimeFullExtraMmas, args.full_extra_mmas);
    return 1;
  }
  if (args.blocks <= 0) {
    std::fprintf(stderr, "--blocks must be positive\n");
    return 1;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This benchmark requires SM100+; got sm_%d%d\n", prop.major,
                 prop.minor);
    return 1;
  }

  if (args.model) {
    const std::vector<int> early_targets = parse_list(args.early_targets);
    if (early_targets.empty()) {
      std::fprintf(stderr, "--model requires one --early-targets value\n");
      return 1;
    }
    if (early_targets.size() != 1) {
      std::fprintf(stderr, "--model accepts exactly one --early-targets value\n");
      return 1;
    }
    const int early_target_mmas = early_targets[0];
    if (early_target_mmas < 0 || early_target_mmas > args.target_mmas) {
      std::fprintf(stderr, "--early-targets value must be in [0,%d]\n", args.target_mmas);
      return 1;
    }
    if (args.probe_gap_cycles < 0) {
      std::fprintf(stderr, "--probe-gap-cycles must be non-negative\n");
      return 1;
    }
    std::vector<ModelRecord> model_records;
    std::printf("running model early_target=%d blocks=%d probe_gap_cycles=%d\n",
                early_target_mmas, args.blocks, args.probe_gap_cycles);
    run_model(args, early_target_mmas, &model_records);
    write_model_csv(args.model_csv, model_records);
    print_model_summary(model_records);
    std::printf("model_csv=%s\n", args.model_csv);
    return 0;
  }

  const std::vector<Combo> combos = make_combos(args);
  if (combos.empty()) {
    std::fprintf(stderr, "empty sweep; check --early-targets/--early-extras/--delays\n");
    return 1;
  }

  std::vector<Record> records;
  records.reserve(static_cast<size_t>(args.blocks) * combos.size());
  for (size_t i = 0; i < combos.size(); ++i) {
    const Combo& c = combos[i];
    std::printf("running combo=%zu early_target=%d early_extra=%d delay=%d blocks=%d\n",
                i, c.early_target_mmas, c.early_extra_mmas, c.delay_cycles, args.blocks);
    run_combo(args, c, static_cast<uint32_t>(i), &records);
  }

  write_detail_csv(args.csv, records);
  write_summary_csv(args.summary_csv, records, combos);
  print_summary(records, combos);
  std::printf("detail_csv=%s\nsummary_csv=%s\n", args.csv, args.summary_csv);
  return 0;
}
