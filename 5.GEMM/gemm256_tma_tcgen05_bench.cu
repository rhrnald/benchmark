#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#ifndef GEMM_CLOCK_TRACE
#define GEMM_CLOCK_TRACE 0
#endif

#ifndef GEMM_PIPE1_PHASE_SHIFT_CYCLES
#define GEMM_PIPE1_PHASE_SHIFT_CYCLES 24
#endif

#define CUDA_CHECK(stmt)                                                        \
  do {                                                                          \
    cudaError_t err__ = (stmt);                                                 \
    if (err__ != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt, __FILE__,    \
                   __LINE__, cudaGetErrorString(err__));                        \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

void driver_check(CUresult result, const char* what) {
  if (result != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* msg = nullptr;
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
static constexpr int kCtaM = 256;
static constexpr int kCtaN = 256;
static constexpr int kStageK = 64;
static constexpr int kMmaM = 128;
static constexpr int kMmaN = 128;
static constexpr int kMmaK = 16;
static constexpr int kStages = 3;
static constexpr int kPipes = 2;
static constexpr int kAStageWords = kCtaM * kStageK / 2;
static constexpr int kBStageWords = kStageK * kCtaN / 2;
static constexpr int kBPipeWords = kStageK * kMmaN / 2;
static constexpr int kStageWords = kAStageWords + kBStageWords;
static constexpr int kAStageBytes = kAStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kBPipeBytes = kBPipeWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kStageBytes = kStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kDynamicSmemBytes = kStages * kStageBytes + 1024;
static constexpr int kHalfTileWords = kMmaM * kStageK / 2;
static constexpr int kTmemTileStride = 128;
[[maybe_unused]] static constexpr int kTraceSlotsPerIter = 8;
[[maybe_unused]] static constexpr int kPipe1PhaseShiftCycles =
    GEMM_PIPE1_PHASE_SHIFT_CYCLES;

static_assert(kPipes * kBPipeWords == kBStageWords);

enum TraceStage {
  kTraceNone = 0,
  kTraceTmaIssue = 1,
  kTraceTmaWait = 2,
  kTraceMmaIssue = 3,
  kTraceMmaWait = 4,
  kTraceDrain = 5,
};

struct Args {
  int device = 0;
  int warmup = 2;
  int iters = 5;
  std::vector<int> sizes = {4096, 8192, 16384, 32768};
  const char* csv = "gemm256_tma_tcgen05_bench.csv";
  bool validate = false;
  int validate_size = 256;
  const char* validate_pattern = "pattern";
  bool clock_trace = false;
  int clock_trace_start = 56;
  int clock_trace_iters = 8;
  const char* trace_csv = "gemm256_tma_tcgen05_trace.csv";
};

struct ClockTraceRecord {
  int stage = 0;
  int iter = 0;
  int warp = 0;
  unsigned long long start = 0;
  unsigned long long end = 0;
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
}

__device__ __forceinline__ void wait_pipe1_phase_shift() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && \
    GEMM_PIPE1_PHASE_SHIFT_CYCLES > 0
  const unsigned long long start = clock64();
  while (clock64() - start <
         static_cast<unsigned long long>(kPipe1PhaseShiftCycles)) {
  }
#endif
}

__device__ __forceinline__ void write_trace_record(ClockTraceRecord* records,
                                                   int trace_start,
                                                   int trace_iters,
                                                   unsigned long long trace_base,
                                                   int stage,
                                                   int iter,
                                                   int slot,
                                                   int warp,
                                                   unsigned long long start,
                                                   unsigned long long end) {
#if GEMM_CLOCK_TRACE
  if (records == nullptr || end <= start) return;
  if (blockIdx.x != 0 || blockIdx.y != 0) return;
  const int idx = iter - trace_start;
  if (idx < 0 || idx >= trace_iters) return;
  ClockTraceRecord r;
  r.stage = stage;
  r.iter = iter;
  r.warp = warp;
  r.start = start - trace_base;
  r.end = end - trace_base;
  records[idx * kTraceSlotsPerIter + slot] = r;
#else
  (void)records;
  (void)trace_start;
  (void)trace_iters;
  (void)trace_base;
  (void)stage;
  (void)iter;
  (void)slot;
  (void)warp;
  (void)start;
  (void)end;
#endif
}

__device__ __forceinline__ void write_trace_extra_record(
    ClockTraceRecord* records,
    int trace_iters,
    unsigned long long trace_base,
    int stage,
    int iter,
    int extra_slot,
    int warp,
    unsigned long long start,
    unsigned long long end) {
#if GEMM_CLOCK_TRACE
  if (records == nullptr || end <= start) return;
  if (blockIdx.x != 0 || blockIdx.y != 0) return;
  ClockTraceRecord r;
  r.stage = stage;
  r.iter = iter;
  r.warp = warp;
  r.start = start - trace_base;
  r.end = end - trace_base;
  records[trace_iters * kTraceSlotsPerIter + extra_slot] = r;
#else
  (void)records;
  (void)trace_iters;
  (void)trace_base;
  (void)stage;
  (void)iter;
  (void)extra_slot;
  (void)warp;
  (void)start;
  (void)end;
#endif
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_k_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  constexpr uint64_t desc_base = (static_cast<uint64_t>(1u) << 16) |
                                 (static_cast<uint64_t>(64u) << 32) |
                                 (static_cast<uint64_t>(1u) << 46) |
                                 (static_cast<uint64_t>(2u) << 61);
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (32u >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_mn_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  constexpr uint64_t desc_base = (static_cast<uint64_t>(128u) << 16) |
                                 (static_cast<uint64_t>(64u) << 32) |
                                 (static_cast<uint64_t>(1u) << 46) |
                                 (static_cast<uint64_t>(2u) << 61);
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (4096u >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__host__ __device__ __forceinline__ uint32_t make_bf16_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(kMmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kMmaM >> 4) << 24;
  return desc;
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
                                            int c0,
                                            int c1,
                                            int c2,
                                            int c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
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

__device__ __forceinline__ void tcgen05_ld_32x32b_x64(uint32_t (&dst)[64],
                                                      uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x64.b32 {" TCGEN05_LD_X64_OPERANDS
      "}, [%64];"
      : TCGEN05_LD_X64_OUTPUTS(dst)
      : "r"(taddr)
      : "memory");
#else
  (void)taddr;
  for (int i = 0; i < 64; ++i) dst[i] = 0;
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

__device__ __forceinline__ uint32_t consume_128x128(uint32_t taddr) {
  uint32_t acc = tcgen05_ld_32x32b_x64_acc(taddr);
  acc ^= tcgen05_ld_32x32b_x64_acc(taddr + 64u);
  tcgen05_wait_ld();
  return acc;
}

__device__ __forceinline__ void store_128x128_float_tile(uint32_t src_taddr,
                                                         float* out,
                                                         int out_ld,
                                                         int row_offset,
                                                         int col_offset) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int row = row_offset + warp_id * 32 + lane;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint32_t r[64];
    const uint32_t row_taddr =
        src_taddr + (static_cast<uint32_t>(warp_id * 32) << 16) +
        static_cast<uint32_t>(half * 64);
    tcgen05_ld_32x32b_x64(r, row_taddr);
    tcgen05_wait_ld();
    float* dst = out + static_cast<size_t>(row) * out_ld + col_offset + half * 64;
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      dst[i] = __uint_as_float(r[i]);
    }
  }
#else
  (void)src_taddr;
  (void)out;
  (void)out_ld;
  (void)row_offset;
  (void)col_offset;
#endif
}

__device__ __forceinline__ void issue_a_stage_tma(const CUtensorMap* a_map,
                                                  uint32_t* a_smem,
                                                  uint64_t* ready,
                                                  int tile_m,
                                                  int ktile) {
  mbarrier_expect_tx(ready, kAStageBytes);
  const int a_col_words = ktile * (kStageK / 2);
  const int a_row = tile_m * kCtaM;
  tma_load_2d(a_map, smem_ptr_u32(a_smem), ready, a_col_words, a_row);
}

__device__ __forceinline__ void issue_b_pipe_stage_tma(
    const CUtensorMap* b_map,
    uint32_t* b_smem,
    uint64_t* ready,
    int tile_n,
    int ktile,
    int pipe,
    ClockTraceRecord* clock_trace,
    int clock_trace_start,
    int clock_trace_iters,
    unsigned long long trace_base,
    int trace_slot,
    int trace_warp) {
  const unsigned long long trace_start =
      clock_trace != nullptr ? clock64() : 0ull;
  mbarrier_expect_tx(ready, kBPipeBytes);
  const int b_col_words = tile_n * (kCtaN / 2) + pipe * (kMmaN / 2);
  const int b_k16 = ktile * (kStageK / kMmaK);
  tma_load_4d(b_map, smem_ptr_u32(b_smem), ready, b_col_words, 0, 0, b_k16);
  const unsigned long long trace_end =
      clock_trace != nullptr ? clock64() : 0ull;
  write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                     trace_base, kTraceTmaIssue, ktile, trace_slot, trace_warp,
                     trace_start, trace_end);
}

__global__ __launch_bounds__(kThreads, 1)
void gemm256_tma_tcgen05_kernel(const __grid_constant__ CUtensorMap a_map,
                                const __grid_constant__ CUtensorMap b_map,
                                uint32_t* __restrict__ sink,
                                float* __restrict__ out,
                                int out_ld,
                                int ktiles,
                                ClockTraceRecord* __restrict__ clock_trace,
                                int clock_trace_start,
                                int clock_trace_iters) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)sink;
  (void)out;
  (void)out_ld;
  (void)ktiles;
  (void)clock_trace;
  (void)clock_trace_start;
  (void)clock_trace_iters;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_addr);

  __shared__ uint64_t a_ready[kStages];
  __shared__ uint64_t b_ready[kPipes][kStages];
  __shared__ uint64_t mma_done[kPipes];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ unsigned long long trace_base_shared;

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
      mbarrier_init(&mma_done[p], 1);
    }
    trace_base_shared = clock64();
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;
  const int tile_n = static_cast<int>(blockIdx.x);
  const int tile_m = static_cast<int>(blockIdx.y);
  const int ntile = static_cast<int>(gridDim.x);

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t c_taddr[4] = {
      tmem_base + 0u * kTmemTileStride,
      tmem_base + 1u * kTmemTileStride,
      tmem_base + 2u * kTmemTileStride,
      tmem_base + 3u * kTmemTileStride,
  };
  const uint32_t idesc = make_bf16_idesc() | (1u << 16);

  if (warp_id == 0 && lane0) {
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage = kt % kStages;
      uint32_t* stage_smem = smem + stage * kStageWords;
      uint32_t* a_smem = stage_smem;
      uint32_t* b_smem = stage_smem + kAStageWords;
      if (kt >= kStages) {
        const uint32_t reuse_phase = static_cast<uint32_t>((kt - kStages) & 1);
#pragma unroll
        for (int p = 0; p < kPipes; ++p) {
          mbarrier_wait(&mma_done[p], reuse_phase);
        }
      }
      const unsigned long long trace_start =
          clock_trace != nullptr ? clock64() : 0ull;
      issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
      issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n,
                             kt, 0, nullptr, 0, 0, trace_base_shared, 0, 0);
      const unsigned long long trace_end =
          clock_trace != nullptr ? clock64() : 0ull;
      write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                         trace_base_shared, kTraceTmaIssue, kt, 0, 0,
                         trace_start, trace_end);
    }
  }

  if (warp_id == 1 && lane0) {
    wait_pipe1_phase_shift();
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage = kt % kStages;
      uint32_t* stage_smem = smem + stage * kStageWords;
      uint32_t* b_smem = stage_smem + kAStageWords + kBPipeWords;
      if (kt >= kStages) {
        mbarrier_wait(&mma_done[1], static_cast<uint32_t>((kt - kStages) & 1));
      }
      issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n,
                             kt, 1, clock_trace, clock_trace_start,
                             clock_trace_iters, trace_base_shared, 1, 1);
    }
  }

  if ((warp_id == 2 || warp_id == 3) && lane0) {
    const int pipe = warp_id - 2;
    if (pipe == 1) wait_pipe1_phase_shift();
    const int top_c = pipe;
    const int bottom_c = pipe + 2;
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage = kt % kStages;
      const uint32_t tma_phase = static_cast<uint32_t>((kt / kStages) & 1);
      uint32_t* stage_smem = smem + stage * kStageWords;
      uint32_t* a_smem = stage_smem;
      uint32_t* b_smem = stage_smem + kAStageWords + pipe * kBPipeWords;

      const unsigned long long tma_wait_start =
          clock_trace != nullptr ? clock64() : 0ull;
      mbarrier_wait(&a_ready[stage], tma_phase);
      mbarrier_wait(&b_ready[pipe][stage], tma_phase);
      const unsigned long long tma_wait_end =
          clock_trace != nullptr ? clock64() : 0ull;
      write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                         trace_base_shared, kTraceTmaWait, kt, 2 + pipe,
                         warp_id,
                         tma_wait_start, tma_wait_end);

      const unsigned long long mma_issue_start =
          clock_trace != nullptr ? clock64() : 0ull;
#pragma unroll
      for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
        const uint32_t a0 = smem_ptr_u32(a_smem);
        const uint32_t a1 = smem_ptr_u32(a_smem + kHalfTileWords);
        const uint32_t b0 = smem_ptr_u32(b_smem);
        const uint64_t a0_desc = make_sw128_major_k_smem_desc(a0, kk);
        const uint64_t a1_desc = make_sw128_major_k_smem_desc(a1, kk);
        const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
        const bool input_d = (kt != 0) || (kk != 0);

        tcgen05_mma_bf16_ss(c_taddr[top_c], a0_desc, b0_desc, idesc, input_d);
        tcgen05_mma_bf16_ss(c_taddr[bottom_c], a1_desc, b0_desc, idesc, input_d);
      }
      tcgen05_commit(&mma_done[pipe]);
      const unsigned long long mma_issue_end =
          clock_trace != nullptr ? clock64() : 0ull;
      write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                         trace_base_shared, kTraceMmaIssue, kt, 4 + pipe,
                         warp_id,
                         mma_issue_start, mma_issue_end);

      const unsigned long long mma_wait_start =
          clock_trace != nullptr ? clock64() : 0ull;
      mbarrier_wait(&mma_done[pipe], static_cast<uint32_t>(kt & 1));
      const unsigned long long mma_wait_end =
          clock_trace != nullptr ? clock64() : 0ull;
      write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                         trace_base_shared, kTraceMmaWait, kt, 6 + pipe,
                         warp_id,
                         mma_wait_start, mma_wait_end);
    }
  }
  __syncthreads();

  uint32_t acc = static_cast<uint32_t>(threadIdx.x + 0x9e3779b9u);
  if (warp_id < kWarps) {
    const unsigned long long drain_start =
        lane0 && clock_trace != nullptr ? clock64() : 0ull;
    acc ^= consume_128x128(c_taddr[warp_id]);
    const unsigned long long drain_end =
        lane0 && clock_trace != nullptr ? clock64() : 0ull;
    if (lane0) {
      write_trace_extra_record(clock_trace, clock_trace_iters, trace_base_shared,
                               kTraceDrain, ktiles, warp_id, warp_id,
                               drain_start, drain_end);
    }
    if (lane0) warp_sinks[warp_id] = acc;
  }
  __syncthreads();

  if (out != nullptr && warp_id < kWarps) {
    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    store_128x128_float_tile(c_taddr[0], out, out_ld, global_row_base,
                             global_col_base);
    store_128x128_float_tile(c_taddr[1], out, out_ld, global_row_base,
                             global_col_base + 128);
    store_128x128_float_tile(c_taddr[2], out, out_ld, global_row_base + 128,
                             global_col_base);
    store_128x128_float_tile(c_taddr[3], out, out_ld, global_row_base + 128,
                             global_col_base + 128);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(ktiles);
#pragma unroll
    for (int w = 0; w < kWarps; ++w) {
      out ^= warp_sinks[w];
    }
    sink[tile_m * ntile + tile_n] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

void encode_a_row_major_sw128_tma_map(CUtensorMap* map,
                                      void* base,
                                      uint64_t rows,
                                      uint64_t cols_bf16) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  const cuuint64_t global_dim[2] = {cols_words, rows};
  const cuuint64_t global_stride[1] = {cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[2] = {kStageK / 2, kCtaM};
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
               "cuTensorMapEncodeTiled(a_row_major_sw128)");
}

void encode_b_row_major_sw128_k16_tma_map(CUtensorMap* map,
                                          void* base,
                                          uint64_t rows,
                                          uint64_t cols_bf16) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  const cuuint64_t global_dim[4] = {cols_words, kMmaK, 2, rows / kMmaK};
  const cuuint64_t global_stride[3] = {
      cols_words * sizeof(uint32_t),
      static_cast<cuuint64_t>(kMmaN / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kMmaK) * cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kMmaN / 4, kMmaK, 2, kStageK / kMmaK};
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
               "cuTensorMapEncodeTiled(b_row_major_sw128_k16)");
}

std::vector<int> parse_sizes(const char* s) {
  std::vector<int> out;
  const char* p = s;
  while (*p) {
    char* end = nullptr;
    long v = std::strtol(p, &end, 10);
    if (end == p || v <= 0 || v > (1 << 20)) {
      std::fprintf(stderr, "Invalid size list: %s\n", s);
      std::exit(EXIT_FAILURE);
    }
    out.push_back(static_cast<int>(v));
    p = *end == ',' ? end + 1 : end;
    if (*end == '\0') break;
  }
  return out;
}

void usage(const char* argv0) {
  std::printf("Usage: %s [--device N] [--sizes 4096,8192,16384,32768] "
              "[--warmup W] [--iters I] [--csv PATH] "
              "[--validate] [--validate-size N] [--validate-pattern pattern|ones] "
              "[--clock-trace] [--clock-trace-start N] "
              "[--clock-trace-iters N] [--trace-csv PATH]\n",
              argv0);
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    auto need_arg = [&](const char* name) {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", name);
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--device") == 0) {
      args.device = std::atoi(need_arg("--device"));
    } else if (std::strcmp(argv[i], "--sizes") == 0) {
      args.sizes = parse_sizes(need_arg("--sizes"));
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      args.warmup = std::atoi(need_arg("--warmup"));
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      args.iters = std::atoi(need_arg("--iters"));
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      args.csv = need_arg("--csv");
    } else if (std::strcmp(argv[i], "--validate") == 0) {
      args.validate = true;
    } else if (std::strcmp(argv[i], "--validate-size") == 0) {
      args.validate_size = std::atoi(need_arg("--validate-size"));
    } else if (std::strcmp(argv[i], "--validate-pattern") == 0) {
      args.validate_pattern = need_arg("--validate-pattern");
    } else if (std::strcmp(argv[i], "--clock-trace") == 0) {
      args.clock_trace = true;
    } else if (std::strcmp(argv[i], "--clock-trace-start") == 0) {
      args.clock_trace_start = std::atoi(need_arg("--clock-trace-start"));
    } else if (std::strcmp(argv[i], "--clock-trace-iters") == 0) {
      args.clock_trace_iters = std::atoi(need_arg("--clock-trace-iters"));
    } else if (std::strcmp(argv[i], "--trace-csv") == 0) {
      args.trace_csv = need_arg("--trace-csv");
    } else if (std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (args.warmup < 0 || args.iters <= 0 || args.sizes.empty()) {
    std::fprintf(stderr, "warmup must be >= 0, iters > 0, and sizes non-empty\n");
    std::exit(EXIT_FAILURE);
  }
  if (args.validate_size <= 0 || args.validate_size % kCtaM != 0 ||
      args.validate_size % kStageK != 0) {
    std::fprintf(stderr, "validate size must be a positive multiple of 256\n");
    std::exit(EXIT_FAILURE);
  }
  if (std::strcmp(args.validate_pattern, "pattern") != 0 &&
      std::strcmp(args.validate_pattern, "ones") != 0) {
    std::fprintf(stderr, "validate pattern must be 'pattern' or 'ones'\n");
    std::exit(EXIT_FAILURE);
  }
  if (args.clock_trace_start < 0 || args.clock_trace_iters <= 0) {
    std::fprintf(stderr, "clock trace start must be >= 0 and iters > 0\n");
    std::exit(EXIT_FAILURE);
  }
  return args;
}

double elapsed_ms(std::chrono::steady_clock::time_point start,
                  std::chrono::steady_clock::time_point stop) {
  return std::chrono::duration<double, std::milli>(stop - start).count();
}

struct CaseResult {
  int size = 0;
  int mtile = 0;
  int ntile = 0;
  int ktiles = 0;
  int ctas = 0;
  float event_ms = 0.0f;
  double wall_ms = 0.0;
  double event_tflops = 0.0;
  double wall_tflops = 0.0;
  uint32_t checksum = 0;
};

CaseResult run_case(int size, int warmup, int iters) {
  if (size % kCtaM != 0 || size % kCtaN != 0 || size % kStageK != 0) {
    std::fprintf(stderr, "size must be a multiple of 256 and 64; got %d\n", size);
    std::exit(EXIT_FAILURE);
  }

  const int m = size;
  const int n = size;
  const int k = size;
  const int mtile = m / kCtaM;
  const int ntile = n / kCtaN;
  const int ktiles = k / kStageK;
  const int ctas = mtile * ntile;
  const size_t a_words = static_cast<size_t>(m) * k / 2;
  const size_t b_words = static_cast<size_t>(k) * n / 2;

  uint32_t* d_a = nullptr;
  uint32_t* d_b = nullptr;
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_b, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(ctas) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_a, 0x3f, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_b, 0x11, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, static_cast<size_t>(ctas) * sizeof(uint32_t)));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUtensorMap a_map{}, b_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);

  CUDA_CHECK(cudaFuncSetAttribute(gemm256_tma_tcgen05_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));

  dim3 grid(ntile, mtile, 1);
  dim3 block(kThreads, 1, 1);

  for (int i = 0; i < warmup; ++i) {
    gemm256_tma_tcgen05_kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map,
                                                                   d_sink, nullptr,
                                                                   0, ktiles,
                                                                   nullptr, 0, 0);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const auto wall_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    gemm256_tma_tcgen05_kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map,
                                                                   d_sink, nullptr,
                                                                   0, ktiles,
                                                                   nullptr, 0, 0);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  const auto wall_stop = std::chrono::steady_clock::now();

  float total_event_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&total_event_ms, start, stop));

  std::vector<uint32_t> h_sink(std::min(ctas, 1024));
  CUDA_CHECK(cudaMemcpy(h_sink.data(), d_sink, h_sink.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  uint32_t checksum = 0;
  for (uint32_t v : h_sink) checksum ^= v;

  const double avg_event_ms = static_cast<double>(total_event_ms) / iters;
  const double avg_wall_ms = elapsed_ms(wall_start, wall_stop) / iters;
  const double flops = 2.0 * static_cast<double>(m) * n * k;

  CaseResult result;
  result.size = size;
  result.mtile = mtile;
  result.ntile = ntile;
  result.ktiles = ktiles;
  result.ctas = ctas;
  result.event_ms = static_cast<float>(avg_event_ms);
  result.wall_ms = avg_wall_ms;
  result.event_tflops = flops / (avg_event_ms * 1.0e-3) / 1.0e12;
  result.wall_tflops = flops / (avg_wall_ms * 1.0e-3) / 1.0e12;
  result.checksum = checksum;

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_sink));
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
  if (use_ones) return 1.0f;
  return (static_cast<float>((row % 17) - 8) * 0.015625f) +
         (static_cast<float>((col % 11) - 5) * 0.0078125f);
}

float validation_b_value(bool use_ones, int row, int col) {
  if (use_ones) return 1.0f;
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

ValidateResult run_validation(int size, const char* pattern) {
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
      a_ref[static_cast<size_t>(row) * k + col] = bf16_bits_to_float_host(lo_bits);
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
      b_ref[static_cast<size_t>(row) * n + col] = bf16_bits_to_float_host(lo_bits);
      b_ref[static_cast<size_t>(row) * n + col + 1] =
          bf16_bits_to_float_host(hi_bits);
      h_b[static_cast<size_t>(row) * (n / 2) + col / 2] =
          pack_bf16_pair_host(lo_bits, hi_bits);
    }
  }

  uint32_t* d_a = nullptr;
  uint32_t* d_b = nullptr;
  uint32_t* d_sink = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, h_a.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_b, h_b.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(ctas) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sink, 0, static_cast<size_t>(ctas) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_c, 0, static_cast<size_t>(m) * n * sizeof(float)));

  CUtensorMap a_map{}, b_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);
  CUDA_CHECK(cudaFuncSetAttribute(gemm256_tma_tcgen05_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));

  dim3 grid(ntile, mtile, 1);
  dim3 block(kThreads, 1, 1);
  gemm256_tma_tcgen05_kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map,
                                                                 d_sink, d_c, n,
                                                                 ktiles, nullptr,
                                                                 0, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> got(static_cast<size_t>(m) * n);
  CUDA_CHECK(cudaMemcpy(got.data(), d_c, got.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  ValidateResult result;
  constexpr double kAbsTol = 2.0e-2;
  constexpr double kRelTol = 2.0e-2;
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

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_c));
  return result;
}

#if GEMM_CLOCK_TRACE
const char* trace_stage_name(int stage) {
  switch (stage) {
    case kTraceTmaIssue:
      return "tma_issue";
    case kTraceTmaWait:
      return "tma_wait";
    case kTraceMmaIssue:
      return "mma_issue";
    case kTraceMmaWait:
      return "mma_wait";
    case kTraceDrain:
      return "tmem_drain";
    default:
      return "unknown";
  }
}

int trace_record_count(int trace_iters) {
  return trace_iters * kTraceSlotsPerIter + kWarps;
}

void write_trace_csv(const char* path,
                     int size,
                     int ktiles,
                     int trace_start,
                     int trace_iters,
                     const std::vector<ClockTraceRecord>& records) {
  FILE* csv = std::fopen(path, "w");
  if (!csv) {
    std::perror(path);
    std::exit(EXIT_FAILURE);
  }
  std::fprintf(csv, "size,ktiles,trace_start,trace_iters,stage,iter,warp,start,end,cycles\n");
  for (const ClockTraceRecord& r : records) {
    if (r.stage == kTraceNone || r.end <= r.start) continue;
    std::fprintf(csv, "%d,%d,%d,%d,%s,%d,%d,%llu,%llu,%llu\n", size, ktiles,
                 trace_start, trace_iters, trace_stage_name(r.stage), r.iter,
                 r.warp, r.start, r.end, r.end - r.start);
  }
  std::fclose(csv);
}

void run_trace_case(const Args& args) {
  const int size = args.sizes.front();
  if (size % kCtaM != 0 || size % kCtaN != 0 || size % kStageK != 0) {
    std::fprintf(stderr, "trace size must be a multiple of 256 and 64; got %d\n",
                 size);
    std::exit(EXIT_FAILURE);
  }

  const int m = size;
  const int n = size;
  const int k = size;
  const int mtile = m / kCtaM;
  const int ntile = n / kCtaN;
  const int ktiles = k / kStageK;
  if (args.clock_trace_start >= ktiles) {
    std::fprintf(stderr,
                 "clock trace start must be < ktiles; start=%d ktiles=%d\n",
                 args.clock_trace_start, ktiles);
    std::exit(EXIT_FAILURE);
  }
  const int trace_iters =
      std::min(args.clock_trace_iters, ktiles - args.clock_trace_start);
  const int ctas = mtile * ntile;
  const size_t a_words = static_cast<size_t>(m) * k / 2;
  const size_t b_words = static_cast<size_t>(k) * n / 2;
  const int records_count = trace_record_count(trace_iters);

  uint32_t* d_a = nullptr;
  uint32_t* d_b = nullptr;
  uint32_t* d_sink = nullptr;
  ClockTraceRecord* d_trace = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_b, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(ctas) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_trace,
                        static_cast<size_t>(records_count) *
                            sizeof(ClockTraceRecord)));
  CUDA_CHECK(cudaMemset(d_a, 0x3f, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_b, 0x11, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, static_cast<size_t>(ctas) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_trace, 0,
                        static_cast<size_t>(records_count) *
                            sizeof(ClockTraceRecord)));

  CUtensorMap a_map{}, b_map{};
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n);
  CUDA_CHECK(cudaFuncSetAttribute(gemm256_tma_tcgen05_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));

  dim3 grid(ntile, mtile, 1);
  dim3 block(kThreads, 1, 1);
  for (int i = 0; i < args.warmup; ++i) {
    gemm256_tma_tcgen05_kernel<<<grid, block, kDynamicSmemBytes>>>(
        a_map, b_map, d_sink, nullptr, 0, ktiles, nullptr, 0, 0);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  gemm256_tma_tcgen05_kernel<<<grid, block, kDynamicSmemBytes>>>(
      a_map, b_map, d_sink, nullptr, 0, ktiles, d_trace,
      args.clock_trace_start, trace_iters);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ClockTraceRecord> h_trace(records_count);
  CUDA_CHECK(cudaMemcpy(h_trace.data(), d_trace,
                        static_cast<size_t>(records_count) *
                            sizeof(ClockTraceRecord),
                        cudaMemcpyDeviceToHost));
  write_trace_csv(args.trace_csv, size, ktiles, args.clock_trace_start,
                  trace_iters, h_trace);
  std::printf("trace_csv=%s size=%d ktiles=%d start=%d iters=%d records=%d\n",
              args.trace_csv, size, ktiles, args.clock_trace_start,
              trace_iters, records_count);

  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_sink));
  CUDA_CHECK(cudaFree(d_trace));
}
#endif

}  // namespace

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.device));
  CUDA_CHECK(cudaFree(nullptr));
  driver_check(cuInit(0), "cuInit");

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, args.device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This benchmark requires SM100+; got sm_%d%d\n",
                 prop.major, prop.minor);
    return 77;
  }

  if (args.clock_trace) {
#if !GEMM_CLOCK_TRACE
    std::fprintf(stderr,
                 "--clock-trace requires compiling with -DGEMM_CLOCK_TRACE=1\n");
    return 77;
#else
    run_trace_case(args);
    return 0;
#endif
  }

  if (args.validate) {
    ValidateResult r = run_validation(args.validate_size, args.validate_pattern);
    std::printf("validation size=%d pattern=%s status=%s max_abs=%g max_rel=%g bad=%zu\n",
                args.validate_size, args.validate_pattern, r.ok ? "ok" : "fail",
                r.max_abs, r.max_rel, r.bad_count);
    if (!r.ok) {
      std::printf("first_bad row=%d col=%d got=%g ref=%g\n", r.first_bad_row,
                  r.first_bad_col, r.first_bad_got, r.first_bad_ref);
    }
    return r.ok ? 0 : 1;
  }

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    return 1;
  }
  std::fprintf(csv,
               "size,m,n,k,cta_m,cta_n,stage_k,mtile,ntile,ktiles,ctas,"
               "warmup,iters,dynamic_smem_bytes,event_ms,wall_ms,"
               "event_TFLOPS,wall_TFLOPS,checksum,device\n");

  std::printf("device=%d name=\"%s\" cc=%d.%d dynamic_smem=%d bytes\n",
              args.device, prop.name, prop.major, prop.minor, kDynamicSmemBytes);
  std::printf("mode=bf16_tcgen05_compute_sink layout=row_major_sw128 "
              "cta_tile=%dx%d stage_k=%d stages=%d pipes=%d "
              "pipe1_phase_shift_cycles=%d\n",
              kCtaM, kCtaN, kStageK, kStages, kPipes,
              kPipe1PhaseShiftCycles);

  for (int size : args.sizes) {
    CaseResult r = run_case(size, args.warmup, args.iters);
    std::printf("size=%d mtile=%d ntile=%d ktiles=%d ctas=%d "
                "event_ms=%.6f wall_ms=%.6f event_TFLOPS=%.3f "
                "wall_TFLOPS=%.3f checksum=%08x\n",
                r.size, r.mtile, r.ntile, r.ktiles, r.ctas, r.event_ms,
                r.wall_ms, r.event_tflops, r.wall_tflops, r.checksum);
    std::fprintf(csv,
                 "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,"
                 "%.3f,%.3f,%08x,%s\n",
                 r.size, r.size, r.size, r.size, kCtaM, kCtaN, kStageK,
                 r.mtile, r.ntile, r.ktiles, r.ctas, args.warmup, args.iters,
                 kDynamicSmemBytes, r.event_ms, r.wall_ms, r.event_tflops,
                 r.wall_tflops, r.checksum, prop.name);
    std::fflush(csv);
  }

  std::fclose(csv);
  return 0;
}
