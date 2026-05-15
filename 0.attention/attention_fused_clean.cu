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

#define ATTENTION_PRODUCER_REGS 128
#define ATTENTION_CONSUMER_REGS 184
#define ATTENTION_USE_SETMAXNREG 1

#ifndef ATTENTION_STORE_OUTPUT
#define ATTENTION_STORE_OUTPUT 0
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
static constexpr int kPipeCount = 2;
static constexpr int kConsumerWarpsPerPipe = 4;
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
static constexpr double kFlopsPerMma =
    2.0 * static_cast<double>(kTileM) * static_cast<double>(kTileN) *
    static_cast<double>(kMmaK);

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int k_tiles = 8192;
  int warmup = 2;
  int iters = 5;
  bool store_output = false;
  std::string stage = "benchmark";
  std::string pattern = "constant";
  const char* csv = "0.attention/attention_fused_clean.csv";
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
  return byte_offset;
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

#if ATTENTION_STORE_OUTPUT
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
#endif

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

#if ATTENTION_STORE_OUTPUT
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


#define TCGEN05_LD_X64_WAIT_PACK_STORE(src_taddr, s_smem, consumer_warp, consumer_half, p_done_barrier, arrive_p_done, row_sum_pipe, acc_out) \
do {                                                                        \
  (acc_out) = tcgen05_ld_x64_wait_pack_store_nvcc(                          \
      (src_taddr), (s_smem), (consumer_warp), (consumer_half),              \
      (p_done_barrier), (arrive_p_done), (row_sum_pipe));                   \
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
      if (lane0) {
        mbarrier_expect_tx(&v_ready[pipe], kTileBytes);
        tma_load_4d(&v_map, smem_ptr_u32(v_smem[pipe]), &v_ready[pipe], 0, 0,
                    (iter % k_tiles) * 64, 0);
        if (pipe == 0) {
          pipe0_vtma_local_shared = local;
        }
      }
      mbarrier_wait(&v_ready[pipe], phase);
      mbarrier_wait(&s_ready[pipe], phase);
      if (pipe == 0) {
        if (local > 0) {
          mbarrier_wait(&pv_done[1], static_cast<uint32_t>((local - 1) & 1));
        }
      } else {
        mbarrier_wait(&pv_done[0], phase);
      }
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

uint64_t checksum_sink(uint32_t* d_sink, int blocks) {
  std::vector<uint32_t> h_sink(blocks);
  CUDA_CHECK(cudaMemcpy(h_sink.data(), d_sink, static_cast<size_t>(blocks) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  uint64_t checksum = 1469598103934665603ull;
  for (int i = 0; i < blocks; ++i) {
    checksum ^= static_cast<uint64_t>(h_sink[i]) + (static_cast<uint64_t>(i) << 32);
    checksum *= 1099511628211ull;
  }
  return checksum;
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

  if (output != nullptr) {
    CUDA_CHECK(cudaMemset(output, 0, static_cast<size_t>(args.blocks) * kTileWords * sizeof(uint32_t)));
  }
  CUDA_CHECK(cudaMemset(sink, 0, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));

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

void write_benchmark_csv(const Args& args,
                         int active,
                         const RunResult& result,
                         uint64_t sink_checksum) {
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
               "sink_checksum,status,cuda_error,notes\n");

  const double groups = static_cast<double>(args.blocks) * args.repeats;
  const double total_mmas = groups * kMmasPerTile * 2.0;
  const double q_tma_bytes = static_cast<double>(args.blocks) * kTileBytes;
  const double k_tma_bytes = groups * kTileBytes;
  const double v_tma_bytes = groups * kTileBytes;
  const double p_read_bytes = groups * 2.0 * kTileBytes;
  const double s_store_bytes = groups * kTileBytes;
  const double qk_flops = groups * kMmasPerTile * kFlopsPerMma;
  const double pv_flops = qk_flops;
  const char* mode = args.store_output ? "qk_pack_smem_pv_2pipe_bf16_output"
                                       : "qk_pack_smem_pv_2pipe";
  const char* notes =
      "active_path_p128_c184_nvcc_ld_regs_pv_pingpong_dep_bf16_output_when_enabled";

  std::fprintf(csv,
               "%s,Q[%d,128,128]_bf16,K[%d,128,128]_bf16,V[%d,128,128]_bf16,sink[%d]_u32_checksum,%d,%d,%d,%d,%d,%d,%d,"
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%llu,%s,%s,%s\n",
               mode, args.blocks, args.k_tiles, args.k_tiles, args.blocks, kTileM, kTileN, kMmaK,
               kMmasPerTile, args.blocks, args.repeats, args.k_tiles, args.warmup, args.iters,
               kMainThreads, active, kDynamicSmemBytes, kTmemAllocCols, kTmemUsedCols, 256, 256,
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

int logical_word_offset(int row, int col_pair) {
  return atom_major_k_word_offset(row, col_pair);
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
  const char* stage;
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
  CompareResult r{stage, true, 0.0f, 0.0f, 0};
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
  CompareResult r{stage, true, 0.0f, 0.0f, 0};
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
    std::fprintf(csv, "%s,%s,%g,%g,%zu\n", r.stage, r.ok ? "ok" : "fail",
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
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--store-output")) {
      args->store_output = true;
    } else if (!std::strcmp(argv[i], "--validate")) {
      args->stage = "validate";
      args->store_output = true;
    } else if (!std::strcmp(argv[i], "--pattern") && i + 1 < argc) {
      args->pattern = argv[++i];
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf("Usage: %s [--validate] [--pattern constant|rank1|onehot] [--blocks N] [--repeats N] [--k-tiles N] [--warmup N] [--iters N] [--store-output] [--csv PATH]\n", argv[0]);
      std::exit(0);
    }
  }
  if (args->blocks < 1) args->blocks = 1;
  if (args->repeats < 1) args->repeats = 1;
  if (args->k_tiles < 1) args->k_tiles = 1;
}

int run_validation(const Args& args) {
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
  uint32_t* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_o, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), kTileWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), static_cast<size_t>(args.k_tiles) * kTileWords * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_o, 0, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0, sizeof(uint32_t)));

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  encode_atom_tma_map(&q_map, d_q, 64);
  encode_atom_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_bf16_output_tma_map(&o_map, d_o, 1);

  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  qk_tma_mma_ld_kernel<<<1, kMainThreads, kDynamicSmemBytes>>>(
      q_map, k_map, v_map, o_map, d_sink, args.k_tiles, args.k_tiles, d_o);
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
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu\n", r.stage,
                r.ok ? "ok" : "fail", r.max_abs, r.max_rel, r.bad_count);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));
  CUDA_CHECK(cudaFree(d_sink));
  return std::all_of(results.begin(), results.end(), [](const CompareResult& r) { return r.ok; }) ? 0 : 1;
}

int run_benchmark(const Args& args_in) {
  Args args = args_in;
#if ATTENTION_STORE_OUTPUT
  if (args.store_output) args.repeats = args.k_tiles;
#endif
  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_o = nullptr;
  uint32_t* d_sink = nullptr;
  const size_t q_words = static_cast<size_t>(args.blocks) * kTileWords;
  const size_t k_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  const size_t v_words = static_cast<size_t>(args.k_tiles) * kTileWords;
  CUDA_CHECK(cudaMalloc(&d_q, q_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, k_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, v_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, static_cast<size_t>(args.blocks) * sizeof(uint32_t)));
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
  encode_atom_tma_map(&q_map, d_q, static_cast<uint64_t>(args.blocks) * 64);
  encode_atom_tma_map(&k_map, d_k, static_cast<uint64_t>(args.k_tiles) * 64);
  encode_atom_tma_map(&v_map, d_v, static_cast<uint64_t>(args.k_tiles) * 64);
  if (d_o) encode_bf16_output_tma_map(&o_map, d_o, static_cast<uint64_t>(args.blocks));

  int active = 0;
  RunResult result = run_kernel(args, q_map, k_map, v_map, o_map, d_sink, d_o, &active);
  const uint64_t sink_checksum = result.error == cudaSuccess ? checksum_sink(d_sink, args.blocks) : 0ull;
  write_benchmark_csv(args, active, result, sink_checksum);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_sink));
  if (d_o) CUDA_CHECK(cudaFree(d_o));
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
  if (prop.major < 10) {
    std::fprintf(stderr, "This benchmark requires SM100+; got sm_%d%d\n", prop.major, prop.minor);
    return 77;
  }
  driver_check(cuInit(0), "cuInit");
  return args.stage == "validate" ? run_validation(args) : run_benchmark(args);
}
