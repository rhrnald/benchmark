#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kTileElems = kTileM * kTileN;
static constexpr int kThreads = 256;
static constexpr int kDynamicSmemBytes = kTileElems * static_cast<int>(sizeof(uint16_t)) + 1024;

__device__ __forceinline__ uint32_t smem_ptr_u32(const void* ptr) {
  uint32_t addr;
  asm volatile("{ .reg .u64 u64addr; cvta.to.shared.u64 u64addr, %1; cvt.u32.u64 %0, u64addr; }"
               : "=r"(addr)
               : "l"(ptr));
  return addr;
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

__global__ __launch_bounds__(kThreads, 1)
void tma_swizzle_map_kernel(const __grid_constant__ CUtensorMap map,
                            uint16_t* __restrict__ sink) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint16_t* smem = reinterpret_cast<uint16_t*>(smem_addr);
  for (int i = threadIdx.x; i < kTileElems; i += blockDim.x) {
    smem[i] = static_cast<uint16_t>(i);
  }
  __syncthreads();
  tma_store_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    tma_store_4d(&map, smem_ptr_u32(smem), 0, 0, 0, 0);
    tma_store_4d(&map, smem_ptr_u32(smem + kTileElems / 2), kTileN / 2, 0, 0, 0);
    tma_store_commit_group();
    tma_store_wait_group_read();
  }
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

void encode_sw128_half_tma_map(CUtensorMap* map, void* base) {
  const cuuint64_t global_dim[4] = {kTileN, kTileM, 1, 1};
  const cuuint64_t global_stride[3] = {
      kTileN * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(uint16_t)};
  const cuuint32_t box_dim[4] = {kTileN / 2, kTileM, 1, 1};
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
                                      CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(sw128_half)");
}

}  // namespace

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  driver_check(cuInit(0), "cuInit");

  uint16_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, kTileElems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemset(d_out, 0xff, kTileElems * sizeof(uint16_t)));

  CUtensorMap map{};
  encode_sw128_half_tma_map(&map, d_out);
  CUDA_CHECK(cudaFuncSetAttribute(tma_swizzle_map_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  tma_swizzle_map_kernel<<<1, kThreads, kDynamicSmemBytes>>>(map, d_out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint16_t> h(kTileElems);
  CUDA_CHECK(cudaMemcpy(h.data(), d_out, kTileElems * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));
  for (int row = 0; row < 16; ++row) {
    std::printf("row %02d:", row);
    for (int col = 0; col < 16; ++col) {
      std::printf(" %04x", h[row * kTileN + col]);
    }
    std::printf("\n");
  }
  std::printf("pairs row,col_pair -> smem_word_offset for first 8 rows x 8 pairs\n");
  for (int row = 0; row < 8; ++row) {
    std::printf("row %d:", row);
    for (int col_pair = 0; col_pair < 8; ++col_pair) {
      const uint16_t lo = h[row * kTileN + col_pair * 2];
      const uint16_t hi = h[row * kTileN + col_pair * 2 + 1];
      std::printf(" %04x/%04x", lo / 2, hi / 2);
    }
    std::printf("\n");
  }
  CUDA_CHECK(cudaFree(d_out));
  return 0;
}
