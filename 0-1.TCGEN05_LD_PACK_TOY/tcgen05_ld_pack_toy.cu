#include <cccl/cuda/__ptx/instructions/tcgen05_ld.h>
#include <cccl/cuda/__ptx/instructions/tcgen05_st.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static constexpr int kWarpSize = 32;
static constexpr int kRegs = 128;
static constexpr int kOutputWords = kWarpSize * kRegs;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(_err));                                  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t out;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(out)
               : "l"(ptr));
  return out;
}

__device__ __forceinline__ void tcgen05_fence_after_thread_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
#endif
}

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

__device__ __forceinline__ uint32_t tcgen05_alloc_256cols(uint32_t* smem_out_taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t smem_addr = smem_ptr_u32(smem_out_taddr);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 256;"
               :: "r"(smem_addr)
               : "memory");
  tcgen05_fence_after_thread_sync();
  __syncthreads();
  uint32_t taddr = *smem_out_taddr;
  __syncthreads();
  return taddr;
#else
  (void)smem_out_taddr;
  return 0;
#endif
}

__device__ __forceinline__ void tcgen05_dealloc_256cols(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 256;"
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

__host__ __device__ __forceinline__ uint32_t source_pattern(int lane, int col) {
  const uint32_t tag = static_cast<uint32_t>((lane & 31) << 8) | static_cast<uint32_t>(col & 255);
  const uint32_t lo = 0x4000u | tag;
  const uint32_t hi = 0x8000u | tag;
  return (hi << 16) | lo;
}

__global__ __launch_bounds__(kWarpSize, 1)
void pack_x128_probe_kernel(uint32_t* __restrict__ packed_out, uint32_t* __restrict__ meta) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)packed_out;
  (void)meta;
#else
  __shared__ uint32_t tmem_smem;
  const int lane = threadIdx.x & 31;
  uint32_t taddr = tcgen05_alloc_256cols(&tmem_smem);
  if (lane == 0) meta[0] = taddr;

  {
    uint32_t values[kRegs];
#pragma unroll
    for (int col = 0; col < kRegs; ++col) {
      values[col] = source_pattern(lane, col);
    }
    cuda::ptx::tcgen05_st_32x32b(taddr, values);
    tcgen05_wait_st();
  }
  {
    uint32_t values[kRegs];
#pragma unroll
    for (int col = 0; col < kRegs; ++col) {
      values[col] = source_pattern(lane, kRegs + col);
    }
    cuda::ptx::tcgen05_st_32x32b(taddr + kRegs, values);
    tcgen05_wait_st();
  }

  {
    uint32_t packed[kRegs];
    cuda::ptx::tcgen05_ld_32x32b_pack_16b(packed, taddr);
    tcgen05_wait_ld();
#pragma unroll
    for (int reg = 0; reg < kRegs; ++reg) {
      packed_out[lane * kRegs + reg] = packed[reg];
    }
  }

  __syncthreads();
  tcgen05_dealloc_256cols(taddr);
  tcgen05_relinquish_alloc_permit();
#endif
}

struct HalfDecode {
  const char* half;
  int lane;
  int col;
};

HalfDecode decode_half(uint16_t h) {
  HalfDecode d{};
  d.half = (h & 0x8000u) ? "hi16" : ((h & 0x4000u) ? "lo16" : "unknown");
  d.lane = (h >> 8) & 31;
  d.col = h & 255;
  return d;
}

const char* arg_value(int argc, char** argv, const char* name, const char* fallback) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  }
  return fallback;
}

void write_csv(const char* path, const std::vector<uint32_t>& packed, uint32_t taddr) {
  FILE* f = std::fopen(path, "w");
  if (!f) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(f,
               "lane,reg,packed_hex,packed_lo16,packed_hi16,lo_source_half,lo_source_lane,"
               "lo_source_col,hi_source_half,hi_source_lane,hi_source_col,raw_same_col_hex,"
               "raw_next_col_hex\n");
  for (int lane = 0; lane < kWarpSize; ++lane) {
    for (int reg = 0; reg < kRegs; ++reg) {
      const uint32_t v = packed[lane * kRegs + reg];
      const uint16_t lo = static_cast<uint16_t>(v & 0xffffu);
      const uint16_t hi = static_cast<uint16_t>(v >> 16);
      const HalfDecode dlo = decode_half(lo);
      const HalfDecode dhi = decode_half(hi);
      std::fprintf(f,
                   "%d,%d,0x%08x,0x%04x,0x%04x,%s,%d,%d,%s,%d,%d,0x%08x,0x%08x\n",
                   lane, reg, v, lo, hi, dlo.half, dlo.lane, dlo.col, dhi.half, dhi.lane,
                   dhi.col, source_pattern(lane, (reg * 2) & 255),
                   source_pattern(lane, (reg * 2 + 1) & 255));
    }
  }
  std::fclose(f);
  std::printf("wrote %s (taddr=0x%08x)\n", path, taddr);
}

}  // namespace

int main(int argc, char** argv) {
  const char* csv = arg_value(argc, argv, "--csv",
                              "0-1.TCGEN05_LD_PACK_TOY/tcgen05_ld_pack_x128_probe.csv");
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 10) {
    std::fprintf(stderr, "This toy requires SM100+; got sm_%d%d\n", prop.major, prop.minor);
    return 1;
  }

  uint32_t* d_packed = nullptr;
  uint32_t* d_meta = nullptr;
  CUDA_CHECK(cudaMalloc(&d_packed, kOutputWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_meta, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_packed, 0, kOutputWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_meta, 0, sizeof(uint32_t)));

  pack_x128_probe_kernel<<<1, kWarpSize>>>(d_packed, d_meta);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> packed(kOutputWords);
  uint32_t meta = 0;
  CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, kOutputWords * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&meta, d_meta, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  write_csv(csv, packed, meta);

  std::printf("lane0 first 16 packed registers:\n");
  for (int reg = 0; reg < 16; ++reg) {
    const uint32_t v = packed[reg];
    const HalfDecode lo = decode_half(static_cast<uint16_t>(v & 0xffffu));
    const HalfDecode hi = decode_half(static_cast<uint16_t>(v >> 16));
    std::printf("  r%-3d = 0x%08x  lo=%s lane%d col%d  hi=%s lane%d col%d\n", reg, v,
                lo.half, lo.lane, lo.col, hi.half, hi.lane, hi.col);
  }

  CUDA_CHECK(cudaFree(d_packed));
  CUDA_CHECK(cudaFree(d_meta));
  return 0;
}
