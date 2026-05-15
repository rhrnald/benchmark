#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cccl/cuda/__ptx/instructions/tcgen05_ld.h>
#include <cccl/cuda/__ptx/instructions/tcgen05_st.h>

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
static constexpr int kMaxWarps = 8;
static constexpr int kTmemRowShift = 16;

enum class Op : int {
  kLd = 0,
  kSt = 1,
  kLdSt = 2,
  kSt128 = 3,
  kLdX2 = 4,
  kLdX4 = 5,
  kLdX8 = 6,
  kLdX16 = 7,
  kLdX32 = 8,
  kLdX64 = 9,
  kLdX128 = 10,
  kStX2 = 11,
  kStX4 = 12,
  kStX8 = 13,
  kStX16 = 14,
  kStX32 = 15,
  kStX64 = 16,
  kStX128 = 17
};

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
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

__device__ __forceinline__ void tcgen05_dealloc_32cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
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

__device__ __forceinline__ void tcgen05_dealloc_128cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;"
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

__device__ __forceinline__ void tcgen05_st_32x32b_x1(uint32_t taddr, uint32_t v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
               :: "r"(taddr), "r"(v)
               : "memory");
#else
  (void)taddr;
  (void)v;
#endif
}

__device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x1(uint32_t taddr) {
  uint32_t r = 0;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
               : "=r"(r)
               : "r"(taddr)
               : "memory");
#else
  (void)taddr;
#endif
  return r;
}

#define TMEM_VREGS_0_1 "v0, v1"
#define TMEM_VREGS_0_3 "v0, v1, v2, v3"
#define TMEM_VREGS_0_7 "v0, v1, v2, v3, v4, v5, v6, v7"
#define TMEM_VREGS_0_15 \
  "v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15"
#define TMEM_VREGS_0_31 \
  "v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, " \
  "v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31"
#define TMEM_VREGS_32_63 \
  "v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, " \
  "v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63"
#define TMEM_VREGS_64_95 \
  "v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, " \
  "v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95"
#define TMEM_VREGS_96_127 \
  "v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109, v110, v111, " \
  "v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127"
#define TMEM_VREGS_0_63 TMEM_VREGS_0_31 ", " TMEM_VREGS_32_63
#define TMEM_VREGS_0_127 TMEM_VREGS_0_63 ", " TMEM_VREGS_64_95 ", " TMEM_VREGS_96_127

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#define DEFINE_TCGEN05_LD_MIX(WIDTH, REGS, MIX)                                 \
  __device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x##WIDTH##_mix(          \
      uint32_t taddr) {                                                          \
    uint32_t r = 0;                                                              \
    asm volatile("{ .reg .b32 v<128>; .reg .b32 acc; "                          \
                 "tcgen05.ld.sync.aligned.32x32b.x" #WIDTH ".b32 {" REGS       \
                 "}, [%1]; " MIX " }"                                           \
                 : "=r"(r)                                                       \
                 : "r"(taddr)                                                   \
                 : "memory");                                                   \
    return r;                                                                    \
  }
#else
#define DEFINE_TCGEN05_LD_MIX(WIDTH, REGS, MIX)                                 \
  __device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x##WIDTH##_mix(          \
      uint32_t taddr) {                                                          \
    (void)taddr;                                                                 \
    return 0;                                                                    \
  }
#endif

DEFINE_TCGEN05_LD_MIX(2, TMEM_VREGS_0_1,
                      "xor.b32 %0, v0, v1;")
DEFINE_TCGEN05_LD_MIX(4, TMEM_VREGS_0_3,
                      "xor.b32 acc, v0, v1; xor.b32 acc, acc, v2; xor.b32 %0, acc, v3;")
DEFINE_TCGEN05_LD_MIX(8, TMEM_VREGS_0_7,
                      "xor.b32 acc, v0, v3; xor.b32 %0, acc, v7;")
DEFINE_TCGEN05_LD_MIX(16, TMEM_VREGS_0_15,
                      "xor.b32 acc, v0, v7; xor.b32 %0, acc, v15;")
DEFINE_TCGEN05_LD_MIX(32, TMEM_VREGS_0_31,
                      "xor.b32 acc, v0, v15; xor.b32 %0, acc, v31;")
DEFINE_TCGEN05_LD_MIX(64, TMEM_VREGS_0_63,
                      "xor.b32 acc, v0, v15; xor.b32 acc, acc, v31; xor.b32 acc, acc, v47; xor.b32 %0, acc, v63;")
DEFINE_TCGEN05_LD_MIX(128, TMEM_VREGS_0_127,
                      "xor.b32 acc, v0, v31; xor.b32 acc, acc, v63; xor.b32 acc, acc, v95; xor.b32 %0, acc, v127;")

#undef DEFINE_TCGEN05_LD_MIX
#undef TMEM_VREGS_0_127
#undef TMEM_VREGS_0_63
#undef TMEM_VREGS_96_127
#undef TMEM_VREGS_64_95
#undef TMEM_VREGS_32_63
#undef TMEM_VREGS_0_31
#undef TMEM_VREGS_0_15
#undef TMEM_VREGS_0_7
#undef TMEM_VREGS_0_3
#undef TMEM_VREGS_0_1

__device__ __forceinline__ void tcgen05_wait_st() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ void tcgen05_wait_ld() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ uint32_t layout_taddr(int layout,
                                                 uint32_t base,
                                                 int warp_id,
                                                 int lane,
                                                 int iter) {
  const uint32_t row = static_cast<uint32_t>(lane);
  switch (layout) {
    case 0:  // A: all active warps hit the same 32-row strip.
      return base;
    case 1:  // B: one physical column per warp.
      return base + static_cast<uint32_t>(warp_id);
    case 2:  // C: two columns per warp, alternated by lane parity.
      return base + static_cast<uint32_t>(warp_id * 2 + (lane & 1));
    case 3:  // D: row-addressed strip, lane selects row.
      return base + (row << kTmemRowShift);
    case 4:  // E: row-addressed strip plus per-warp column.
      return base + (row << kTmemRowShift) + static_cast<uint32_t>(warp_id);
    case 5:  // F: row block split similar to 64x8 result extraction.
      return base + (static_cast<uint32_t>((warp_id & 1) * 32 + lane) << kTmemRowShift) +
             static_cast<uint32_t>(warp_id >> 1);
    case 6:  // G: rotating column pressure across the 32-column allocation.
      return base + static_cast<uint32_t>((warp_id * 8 + ((iter + lane) & 7)) & 31);
    case 8:  // I: canonical tcgen05.ld/st consume, natural 4-warp row ownership.
      return base + static_cast<uint32_t>((warp_id >> 2) * 128);
    default:  // H: each warp starts from a different 32-column region.
      return base + static_cast<uint32_t>(warp_id * 32);
  }
}

__device__ __forceinline__ uint32_t validation_value(int lane, int col) {
  return 0x5a000000u ^ (static_cast<uint32_t>(lane) << 8) ^
         static_cast<uint32_t>(col);
}

template <int Width>
__device__ __forceinline__ void tcgen05_st_32x32b_xN(uint32_t taddr,
                                                     int lane,
                                                     int iter) {
  uint32_t values[Width];
#pragma unroll
  for (int i = 0; i < Width; ++i) {
    values[i] = 0xc3000000u ^ (static_cast<uint32_t>(lane) << 10) ^
                (static_cast<uint32_t>(iter) << 1) ^ static_cast<uint32_t>(i);
  }
  cuda::ptx::tcgen05_st_32x32b(taddr, values);
}

template <int Width>
__device__ __forceinline__ void validate_ld_width(uint32_t base,
                                                  int lane,
                                                  uint32_t* errors,
                                                  uint32_t* samples,
                                                  int sample_slot) {
  uint32_t out[Width];
  cuda::ptx::tcgen05_ld_32x32b(out, base);
  tcgen05_wait_ld();

  uint32_t local_errors = 0;
#pragma unroll
  for (int i = 0; i < Width; ++i) {
    const uint32_t expected = validation_value(lane, i);
    local_errors += out[i] == expected ? 0u : 1u;
  }
  if (local_errors != 0) atomicAdd(errors, local_errors);
  if (lane == 0) {
    samples[sample_slot * 3 + 0] = out[0];
    samples[sample_slot * 3 + 1] = out[Width / 2];
    samples[sample_slot * 3 + 2] = out[Width - 1];
  }
}

__global__ __launch_bounds__(kThreadsPerWarp, 1) void validate_ld_values_kernel(
    uint32_t* errors,
    uint32_t* samples) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)errors;
  (void)samples;
#else
  const int lane = threadIdx.x & 31;
  __shared__ uint32_t tmem_base_smem;

  uint32_t taddr = tcgen05_alloc_128cols(&tmem_base_smem);
  if (lane == 0) tmem_base_smem = taddr;
  __syncwarp();
  const uint32_t base = tmem_base_smem;

#pragma unroll
  for (int col = 0; col < 128; ++col) {
    tcgen05_st_32x32b_x1(base + static_cast<uint32_t>(col),
                         validation_value(lane, col));
  }
  tcgen05_wait_st();
  __syncwarp();

  validate_ld_width<1>(base, lane, errors, samples, 0);
  validate_ld_width<2>(base, lane, errors, samples, 1);
  validate_ld_width<4>(base, lane, errors, samples, 2);
  validate_ld_width<8>(base, lane, errors, samples, 3);
  validate_ld_width<16>(base, lane, errors, samples, 4);
  validate_ld_width<32>(base, lane, errors, samples, 5);
  validate_ld_width<64>(base, lane, errors, samples, 6);
  validate_ld_width<128>(base, lane, errors, samples, 7);

  __syncwarp();
  tcgen05_dealloc_128cols(base);
  __syncwarp();
  tcgen05_relinquish_alloc_permit();
#endif
}

template <int Warps, int Layout, Op WhichOp, bool WaitEach>
__global__ __launch_bounds__(Warps * kThreadsPerWarp, 1) void tmem_ldst_kernel(
    uint32_t* __restrict__ sink,
    int repeats,
    uint32_t anti_cse_mask) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink;
  (void)repeats;
  (void)anti_cse_mask;
#else
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  __shared__ uint32_t tmem_base_smem;

  if (warp_id == 0) {
    uint32_t taddr;
    if constexpr (Layout == 7 || Layout == 8) {
      taddr = tcgen05_alloc_512cols(&tmem_base_smem);
    } else if constexpr (WhichOp == Op::kLdSt || WhichOp == Op::kSt128 ||
                         WhichOp == Op::kLdX64 || WhichOp == Op::kLdX128 ||
                         WhichOp == Op::kStX64 || WhichOp == Op::kStX128) {
      taddr = tcgen05_alloc_128cols(&tmem_base_smem);
    } else {
      taddr = tcgen05_alloc_32cols(&tmem_base_smem);
    }
    if (lane == 0) tmem_base_smem = taddr;
  }
  __syncthreads();
  const uint32_t base = tmem_base_smem;

  uint32_t acc0 = static_cast<uint32_t>(threadIdx.x + 1);
  uint32_t acc1 = acc0 ^ 0x9e3779b9u;
  uint32_t acc2 = acc0 ^ 0x7f4a7c15u;
  uint32_t acc3 = acc0 ^ 0x94d049bbu;
  uint32_t acc4 = acc0 ^ 0x2545f491u;
  uint32_t acc5 = acc0 ^ 0x369dea0fu;
  uint32_t acc6 = acc0 ^ 0xdb4f0b91u;
  uint32_t acc7 = acc0 ^ 0xbb67ae85u;

  for (int i = 0; i < 32; ++i) {
    const uint32_t taddr = layout_taddr(Layout, base, warp_id, lane, i);
    tcgen05_st_32x32b_x1(taddr, acc0 + static_cast<uint32_t>(i));
  }
  tcgen05_wait_st();
  __syncthreads();

  auto issue_once = [&](int i, uint32_t& acc_slot) {
    const uint32_t taddr = layout_taddr(Layout, base, warp_id, lane, i);
    const uint32_t ld_anti_cse = static_cast<uint32_t>(i & 7) & anti_cse_mask;
    if constexpr (WhichOp == Op::kSt || WhichOp == Op::kSt128) {
      const uint32_t dst_taddr =
          WhichOp == Op::kSt128 ? layout_taddr(Layout, base + 64, warp_id, lane, i) : taddr;
      tcgen05_st_32x32b_x1(dst_taddr, acc_slot + static_cast<uint32_t>(i));
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX2) {
      tcgen05_st_32x32b_xN<2>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX4) {
      tcgen05_st_32x32b_xN<4>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX8) {
      tcgen05_st_32x32b_xN<8>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX16) {
      tcgen05_st_32x32b_xN<16>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX32) {
      tcgen05_st_32x32b_xN<32>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX64) {
      tcgen05_st_32x32b_xN<64>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kStX128) {
      tcgen05_st_32x32b_xN<128>(taddr, lane, i);
      if constexpr (WaitEach) tcgen05_wait_st();
    } else if constexpr (WhichOp == Op::kLdSt) {
      const uint32_t src_taddr = taddr + ld_anti_cse;
      const uint32_t dst_taddr = layout_taddr(Layout, base + 64, warp_id, lane, i);
      acc_slot ^= tcgen05_ld_32x32b_x1(src_taddr) + static_cast<uint32_t>(i);
      tcgen05_st_32x32b_x1(dst_taddr, static_cast<uint32_t>(i) ^ (threadIdx.x * 0x9e3779b9u));
      if constexpr (WaitEach) {
        tcgen05_wait_ld();
        tcgen05_wait_st();
      }
    } else if constexpr (WhichOp == Op::kLdX2) {
      acc_slot ^= tcgen05_ld_32x32b_x2_mix(taddr + ld_anti_cse) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else if constexpr (WhichOp == Op::kLdX4) {
      acc_slot ^= tcgen05_ld_32x32b_x4_mix(taddr + ld_anti_cse) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else if constexpr (WhichOp == Op::kLdX8) {
      acc_slot ^= tcgen05_ld_32x32b_x8_mix(taddr + ld_anti_cse) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else if constexpr (WhichOp == Op::kLdX16) {
      acc_slot ^= tcgen05_ld_32x32b_x16_mix(taddr + ld_anti_cse) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else if constexpr (WhichOp == Op::kLdX32) {
      acc_slot ^= tcgen05_ld_32x32b_x32_mix(taddr + ld_anti_cse) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else if constexpr (WhichOp == Op::kLdX64) {
      acc_slot ^= tcgen05_ld_32x32b_x64_mix(taddr) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else if constexpr (WhichOp == Op::kLdX128) {
      acc_slot ^= tcgen05_ld_32x32b_x128_mix(taddr) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    } else {
      acc_slot ^= tcgen05_ld_32x32b_x1(taddr + ld_anti_cse) + static_cast<uint32_t>(i);
      if constexpr (WaitEach) tcgen05_wait_ld();
    }
  };

  int i = 0;
#pragma unroll 1
  for (; i + 7 < repeats; i += 8) {
    issue_once(i + 0, acc0);
    issue_once(i + 1, acc1);
    issue_once(i + 2, acc2);
    issue_once(i + 3, acc3);
    issue_once(i + 4, acc4);
    issue_once(i + 5, acc5);
    issue_once(i + 6, acc6);
    issue_once(i + 7, acc7);
  }
#pragma unroll 1
  for (; i < repeats; ++i) {
    switch (i & 7) {
      case 0: issue_once(i, acc0); break;
      case 1: issue_once(i, acc1); break;
      case 2: issue_once(i, acc2); break;
      case 3: issue_once(i, acc3); break;
      case 4: issue_once(i, acc4); break;
      case 5: issue_once(i, acc5); break;
      case 6: issue_once(i, acc6); break;
      default: issue_once(i, acc7); break;
    }
  }
  if constexpr (WhichOp == Op::kSt || WhichOp == Op::kSt128) {
    if constexpr (!WaitEach) tcgen05_wait_st();
    const uint32_t read_base = WhichOp == Op::kSt128 ? base + 64 : base;
    const uint32_t taddr = layout_taddr(Layout, read_base, warp_id, lane, repeats - 1);
    acc0 ^= tcgen05_ld_32x32b_x1(taddr);
    tcgen05_wait_ld();
  } else if constexpr (WhichOp == Op::kStX2 || WhichOp == Op::kStX4 ||
                       WhichOp == Op::kStX8 || WhichOp == Op::kStX16 ||
                       WhichOp == Op::kStX32 || WhichOp == Op::kStX64 ||
                       WhichOp == Op::kStX128) {
    if constexpr (!WaitEach) tcgen05_wait_st();
    const uint32_t taddr = layout_taddr(Layout, base, warp_id, lane, repeats - 1);
    acc0 ^= tcgen05_ld_32x32b_x1(taddr);
    tcgen05_wait_ld();
  } else if constexpr (WhichOp == Op::kLdSt) {
    if constexpr (!WaitEach) {
      tcgen05_wait_ld();
      tcgen05_wait_st();
    }
    const uint32_t taddr = layout_taddr(Layout, base + 64, warp_id, lane, repeats - 1);
    acc0 ^= tcgen05_ld_32x32b_x1(taddr);
    tcgen05_wait_ld();
  } else {
    if constexpr (!WaitEach) tcgen05_wait_ld();
  }

  if (lane == 0) {
    sink[blockIdx.x * Warps + warp_id] =
        acc0 ^ acc1 ^ acc2 ^ acc3 ^ acc4 ^ acc5 ^ acc6 ^ acc7;
  }
  __syncthreads();
  if (warp_id == 0) {
    if constexpr (Layout == 7 || Layout == 8) {
      tcgen05_dealloc_512cols(base);
    } else if constexpr (WhichOp == Op::kLdSt || WhichOp == Op::kSt128 ||
                         WhichOp == Op::kLdX64 || WhichOp == Op::kLdX128 ||
                         WhichOp == Op::kStX64 || WhichOp == Op::kStX128) {
      tcgen05_dealloc_128cols(base);
    } else {
      tcgen05_dealloc_32cols(base);
    }
  }
  __syncthreads();
  if (warp_id == 0) {
    tcgen05_relinquish_alloc_permit();
  }
#endif
}

struct Args {
  int blocks = 512;
  int repeats = 4096;
  int warmup = 5;
  int iters = 20;
  bool wait_each = false;
  const char* csv_path = nullptr;
  uint32_t value_check_errors = 0;
  const char* value_check_status = "not_run";
};

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
    } else if (!std::strcmp(argv[i], "--wait-each")) {
      args->wait_each = true;
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv_path = argv[++i];
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf("Usage: %s [--blocks N] [--repeats N] [--warmup N] [--iters N] [--wait-each] [--csv PATH]\n", argv[0]);
      std::exit(0);
    }
  }
}

template <int Warps, int Layout, Op WhichOp, bool WaitEach>
float run_one(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block(Warps * kThreadsPerWarp);
  for (int i = 0; i < args.warmup; ++i) {
    tmem_ldst_kernel<Warps, Layout, WhichOp, WaitEach><<<grid, block>>>(
        sink, args.repeats, 0u);
  }
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    tmem_ldst_kernel<Warps, Layout, WhichOp, WaitEach><<<grid, block>>>(
        sink, args.repeats, 0u);
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

template <int Warps, int Layout, bool WaitEach>
void run_layout(FILE* out, uint32_t* sink, const Args& args) {
  const float ld_ms = run_one<Warps, Layout, Op::kLd, WaitEach>(sink, args);
  const float ldx2_ms = run_one<Warps, Layout, Op::kLdX2, WaitEach>(sink, args);
  const float ldx4_ms = run_one<Warps, Layout, Op::kLdX4, WaitEach>(sink, args);
  const float ldx8_ms = run_one<Warps, Layout, Op::kLdX8, WaitEach>(sink, args);
  const float ldx16_ms = run_one<Warps, Layout, Op::kLdX16, WaitEach>(sink, args);
  const float ldx32_ms = run_one<Warps, Layout, Op::kLdX32, WaitEach>(sink, args);
  const float ldx64_ms = run_one<Warps, Layout, Op::kLdX64, WaitEach>(sink, args);
  const float ldx128_ms = run_one<Warps, Layout, Op::kLdX128, WaitEach>(sink, args);
  const float st_ms = run_one<Warps, Layout, Op::kSt, WaitEach>(sink, args);
  const float stx2_ms = run_one<Warps, Layout, Op::kStX2, WaitEach>(sink, args);
  const float stx4_ms = run_one<Warps, Layout, Op::kStX4, WaitEach>(sink, args);
  const float stx8_ms = run_one<Warps, Layout, Op::kStX8, WaitEach>(sink, args);
  const float stx16_ms = run_one<Warps, Layout, Op::kStX16, WaitEach>(sink, args);
  const float stx32_ms = run_one<Warps, Layout, Op::kStX32, WaitEach>(sink, args);
  const float stx64_ms = run_one<Warps, Layout, Op::kStX64, WaitEach>(sink, args);
  const float stx128_ms = run_one<Warps, Layout, Op::kStX128, WaitEach>(sink, args);
  const float st128_ms = run_one<Warps, Layout, Op::kSt128, WaitEach>(sink, args);
  const float ldst_ms = run_one<Warps, Layout, Op::kLdSt, WaitEach>(sink, args);
  const double ops = static_cast<double>(args.blocks) * Warps * args.repeats;

  auto layout_name = []() {
    if constexpr (Layout == 8) {
      return "canonical_4warp_rows";
    } else {
      static const char names[][2] = {"A", "B", "C", "D", "E", "F", "G", "H"};
      return names[Layout];
    }
  };

  auto emit_row = [&](const char* op,
                      int width,
	                      const char* variant,
	                      float elapsed_ms,
	                      int bytes_per_op) {
    const double gops = ops / (static_cast<double>(elapsed_ms) * 1.0e6);
    const double tbps = gops * static_cast<double>(bytes_per_op) / 1000.0;
    const int canonical_tiles = (Warps + 3) / 4;
    const int canonical_rows = Warps * 32;
    std::fprintf(out,
                 "%s,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%.6f,%.3f,%d,%.3f,%u,%s\n",
                 op, width, variant, layout_name(), Warps, canonical_tiles,
                 canonical_rows,
                 args.blocks, args.repeats, args.warmup, args.iters,
                 WaitEach ? 1 : 0, elapsed_ms, gops, bytes_per_op, tbps,
                 args.value_check_errors, args.value_check_status);
  };

  emit_row("ld", 1, "x1", ld_ms, 128);
  emit_row("ld", 2, "x2", ldx2_ms, 256);
  emit_row("ld", 4, "x4", ldx4_ms, 512);
  emit_row("ld", 8, "x8", ldx8_ms, 1024);
  emit_row("ld", 16, "x16", ldx16_ms, 2048);
  emit_row("ld", 32, "x32", ldx32_ms, 4096);
  emit_row("ld", 64, "x64", ldx64_ms, 8192);
  emit_row("ld", 128, "x128", ldx128_ms, 16384);
  emit_row("st", 1, "x1", st_ms, 128);
  emit_row("st", 2, "x2", stx2_ms, 256);
  emit_row("st", 4, "x4", stx4_ms, 512);
  emit_row("st", 8, "x8", stx8_ms, 1024);
  emit_row("st", 16, "x16", stx16_ms, 2048);
  emit_row("st", 32, "x32", stx32_ms, 4096);
  emit_row("st", 64, "x64", stx64_ms, 8192);
  emit_row("st", 128, "x128", stx128_ms, 16384);
  emit_row("st", 1, "x1_alloc128_dst_plus64", st128_ms, 128);
  emit_row("ldst", 1, "ld.x1_plus_st.x1", ldst_ms, 256);
  std::fflush(out);
}

template <int Warps, bool WaitEach>
void run_warps(FILE* out, uint32_t* sink, const Args& args) {
  run_layout<Warps, 8, WaitEach>(out, sink, args);
}

template <bool WaitEach>
void run_all(FILE* out, uint32_t* sink, const Args& args) {
  run_warps<1, WaitEach>(out, sink, args);
  run_warps<2, WaitEach>(out, sink, args);
  run_warps<4, WaitEach>(out, sink, args);
  run_warps<8, WaitEach>(out, sink, args);
}

struct ValidationResult {
  uint32_t errors = 0;
  uint32_t samples[24] = {};
};

ValidationResult run_value_validation() {
  ValidationResult result;
  uint32_t* device_errors = nullptr;
  uint32_t* device_samples = nullptr;
  CUDA_CHECK(cudaMalloc(&device_errors, sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&device_samples, sizeof(result.samples)));
  CUDA_CHECK(cudaMemset(device_errors, 0, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(device_samples, 0, sizeof(result.samples)));

  validate_ld_values_kernel<<<1, kThreadsPerWarp>>>(device_errors, device_samples);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(&result.errors, device_errors, sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(result.samples, device_samples, sizeof(result.samples),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_errors));
  CUDA_CHECK(cudaFree(device_samples));
  return result;
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
  CUDA_CHECK(cudaMalloc(&sink, static_cast<size_t>(args.blocks) * kMaxWarps * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(sink, 0, static_cast<size_t>(args.blocks) * kMaxWarps * sizeof(uint32_t)));

  const ValidationResult validation = run_value_validation();
  args.value_check_errors = validation.errors;
  args.value_check_status = validation.errors == 0 ? "ok" : "fail";

  FILE* out = stdout;
  if (args.csv_path != nullptr) {
    out = std::fopen(args.csv_path, "w");
    if (out == nullptr) {
      std::fprintf(stderr, "Failed to open CSV path: %s\n", args.csv_path);
      return 1;
    }
  }

  std::fprintf(stderr,
               "device=%s sm_%d%d blocks=%d repeats=%d warmup=%d iters=%d wait_each=%d csv=%s value_check=%s errors=%u samples_x128=(0x%08x,0x%08x,0x%08x)\n",
               prop.name, prop.major, prop.minor, args.blocks, args.repeats,
               args.warmup, args.iters, args.wait_each ? 1 : 0,
               args.csv_path == nullptr ? "<stdout>" : args.csv_path,
               args.value_check_status, args.value_check_errors,
               validation.samples[21], validation.samples[22], validation.samples[23]);
  std::fprintf(out,
               "op,width,variant,layout,warps,canonical_tiles,canonical_rows,blocks,repeats,warmup,iters,wait_each,elapsed_ms,Gops_per_s,canonical_payload_bytes_per_op,canonical_payload_TBps,value_check_errors,value_check_status\n");
  if (args.wait_each) {
    run_all<true>(out, sink, args);
  } else {
    run_all<false>(out, sink, args);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaFree(sink));
  if (out != stdout) std::fclose(out);
  return 0;
}
