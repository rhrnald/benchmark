#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
static constexpr int kMmaM = 128;
static constexpr int kMmaN = 128;
static constexpr int kMmaK = 16;
static constexpr int kConsumerWarps = 8;
static constexpr int kProducerWarps = 2;
static constexpr int kSharedConsumerWarps = 4;
static constexpr int kStreams = 2;
static constexpr int kSplitHalves = 2;
static constexpr int kSplitMmaN = 64;
static constexpr int kSplitProducerWarps = kStreams * kSplitHalves;
static constexpr int kAccumulatesPerConsume = 8;
static constexpr int kThreadsPerCta = (kConsumerWarps + kProducerWarps) * kWarpSize;
static constexpr int kSharedConsumerThreadsPerCta =
    (kSharedConsumerWarps + kProducerWarps) * kWarpSize;
static constexpr int kSplit4ProducerThreadsPerCta =
    (kConsumerWarps + kSplitProducerWarps) * kWarpSize;
static constexpr int kTmemColumnsUsed = kStreams * kMmaN;
static constexpr int kAllocated128ColChunks = (kTmemColumnsUsed + 127) / 128;
static constexpr double kPeakTflops = 2200.0;

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int warmup = 2;
  int iters = 5;
  const char* csv = "3-1.MMA_LD_PIPELINE_128KB/mma_ld_pipeline_128kb_case.csv";
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

__host__ __device__ __forceinline__ uint32_t make_bf16_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(kMmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kMmaM >> 4) << 24;
  return desc;
}

template <int MmaN>
__host__ __device__ __forceinline__ uint32_t make_bf16_idesc_n() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(MmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kMmaM >> 4) << 24;
  return desc;
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

__device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x128_acc(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t acc;
  asm volatile(
      "{ .reg .b32 v<128>; .reg .b32 acc; "
      "tcgen05.ld.sync.aligned.32x32b.x128.b32 "
      "{v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, "
      "v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, "
      "v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, "
      "v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, "
      "v64, v65, v66, v67, v68, v69, v70, v71, v72, v73, v74, v75, v76, v77, v78, v79, "
      "v80, v81, v82, v83, v84, v85, v86, v87, v88, v89, v90, v91, v92, v93, v94, v95, "
      "v96, v97, v98, v99, v100, v101, v102, v103, v104, v105, v106, v107, v108, v109, v110, v111, "
      "v112, v113, v114, v115, v116, v117, v118, v119, v120, v121, v122, v123, v124, v125, v126, v127}, [%1]; "
      "xor.b32 acc, v0, v31; "
      "xor.b32 acc, acc, v63; "
      "xor.b32 acc, acc, v95; "
      "xor.b32 %0, acc, v127; }"
      : "=r"(acc)
      : "r"(taddr)
      : "memory");
  return acc;
#else
  (void)taddr;
  return 0;
#endif
}

__device__ __forceinline__ uint32_t consume_128x128_accumulator_tile(uint32_t taddr) {
  uint32_t acc = tcgen05_ld_32x32b_x128_acc(taddr);
  tcgen05_wait_ld();
  return acc;
}

__device__ __forceinline__ uint32_t consume_128x64_accumulator_tile(uint32_t taddr) {
  uint32_t acc = tcgen05_ld_32x32b_x64_acc(taddr);
  tcgen05_wait_ld();
  return acc;
}

__device__ __forceinline__ void init_bf16_smem(uint32_t* smem_words) {
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    smem_words[i] = 0x3f803f80u ^ static_cast<uint32_t>((i + 17 * blockIdx.x) & 0x000f000fu);
  }
}

__device__ __forceinline__ uint32_t stream_taddr(uint32_t base, int stream) {
  return base + static_cast<uint32_t>(stream * kMmaN);
}

__device__ __forceinline__ uint32_t split_half_taddr(uint32_t base, int stream, int half) {
  return base + static_cast<uint32_t>((stream * kSplitHalves + half) * kSplitMmaN);
}

__global__ __launch_bounds__(kThreadsPerCta, 1)
void mma_ld_pipeline_128kb_case_kernel(uint32_t* __restrict__ sink, int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink;
  (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kStreams];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kConsumerWarps];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      mbarrier_init(&ready[s], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int consumer_id = warp_id;
  const int producer_id = warp_id - kConsumerWarps;

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint32_t idesc = make_bf16_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = repeats / kAccumulatesPerConsume;

  for (int group = 0; group < groups; ++group) {
    if (lane == 0 && producer_id >= 0 && producer_id < kProducerWarps) {
      const int stream = producer_id;
      const uint32_t d_taddr = stream_taddr(tmem_base, stream);
#pragma unroll
      for (int k = 0; k < kAccumulatesPerConsume; ++k) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
      }
      tcgen05_commit(&ready[stream]);
    }

    if (consumer_id >= 0 && consumer_id < kConsumerWarps) {
      const int stream = consumer_id >> 2;
      mbarrier_wait(&ready[stream], static_cast<uint32_t>(group & 1));
      read_acc ^= consume_128x128_accumulator_tile(stream_taddr(tmem_base, stream));
    }
    __syncthreads();
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams);
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) out ^= warp_sinks[w];
    sink[blockIdx.x] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kThreadsPerCta, 1)
void split_n64_half_pipeline_128kb_case_kernel(uint32_t* __restrict__ sink, int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink;
  (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kStreams][kSplitHalves];
  __shared__ uint64_t done[kStreams][kSplitHalves];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kConsumerWarps];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
#pragma unroll
      for (int h = 0; h < kSplitHalves; ++h) {
        mbarrier_init(&ready[s][h], 1);
        mbarrier_init(&done[s][h], 4);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int consumer_id = warp_id;
  const int producer_id = warp_id - kConsumerWarps;

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc0 = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint64_t b_desc1 = make_smem_desc(smem_ptr_u32(smem_words + 3072));
  const uint32_t idesc = make_bf16_idesc_n<kSplitMmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = repeats / kAccumulatesPerConsume;

  if (lane == 0 && producer_id >= 0 && producer_id < kProducerWarps) {
    const int stream = producer_id;
    for (int group = 0; group < groups; ++group) {
#pragma unroll
      for (int half = 0; half < kSplitHalves; ++half) {
        if (group > 0) {
          mbarrier_wait(&done[stream][half], static_cast<uint32_t>((group - 1) & 1));
        }
        const uint32_t d_taddr = split_half_taddr(tmem_base, stream, half);
        const uint64_t b_desc = half == 0 ? b_desc0 : b_desc1;
#pragma unroll
        for (int k = 0; k < kAccumulatesPerConsume; ++k) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
        }
        tcgen05_commit(&ready[stream][half]);
      }
    }
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps) {
    const int stream = consumer_id >> 2;
    for (int group = 0; group < groups; ++group) {
#pragma unroll
      for (int half = 0; half < kSplitHalves; ++half) {
        mbarrier_wait(&ready[stream][half], static_cast<uint32_t>(group & 1));
        read_acc ^= consume_128x64_accumulator_tile(split_half_taddr(tmem_base, stream, half));
        if (lane == 0) mbarrier_arrive(&done[stream][half]);
      }
    }
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams * kSplitHalves);
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) out ^= warp_sinks[w];
    sink[blockIdx.x] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kSharedConsumerThreadsPerCta, 1)
void shared_consumer4_two_streams_128kb_case_kernel(uint32_t* __restrict__ sink,
                                                    int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink;
  (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kStreams];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kSharedConsumerWarps];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      mbarrier_init(&ready[s], 1);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int consumer_id = warp_id;
  const int producer_id = warp_id - kSharedConsumerWarps;

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint32_t idesc = make_bf16_idesc();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = repeats / kAccumulatesPerConsume;

  for (int group = 0; group < groups; ++group) {
    if (lane == 0 && producer_id >= 0 && producer_id < kProducerWarps) {
      const int stream = producer_id;
      const uint32_t d_taddr = stream_taddr(tmem_base, stream);
#pragma unroll
      for (int k = 0; k < kAccumulatesPerConsume; ++k) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
      }
      tcgen05_commit(&ready[stream]);
    }

    if (consumer_id >= 0 && consumer_id < kSharedConsumerWarps) {
#pragma unroll
      for (int stream = 0; stream < kStreams; ++stream) {
        mbarrier_wait(&ready[stream], static_cast<uint32_t>(group & 1));
        read_acc ^= consume_128x128_accumulator_tile(stream_taddr(tmem_base, stream));
      }
    }
    __syncthreads();
  }

  if (consumer_id >= 0 && consumer_id < kSharedConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams);
#pragma unroll
    for (int w = 0; w < kSharedConsumerWarps; ++w) out ^= warp_sinks[w];
    sink[blockIdx.x] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

__global__ __launch_bounds__(kSplit4ProducerThreadsPerCta, 1)
void split_n64_4producer_128kb_case_kernel(uint32_t* __restrict__ sink, int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink;
  (void)repeats;
#else
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kStreams][kSplitHalves];
  __shared__ uint64_t done[kStreams][kSplitHalves];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kConsumerWarps];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
#pragma unroll
      for (int h = 0; h < kSplitHalves; ++h) {
        mbarrier_init(&ready[s][h], 1);
        mbarrier_init(&done[s][h], 4);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int consumer_id = warp_id;
  const int producer_id = warp_id - kConsumerWarps;

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc0 = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint64_t b_desc1 = make_smem_desc(smem_ptr_u32(smem_words + 3072));
  const uint32_t idesc = make_bf16_idesc_n<kSplitMmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = repeats / kAccumulatesPerConsume;

  if (lane == 0 && producer_id >= 0 && producer_id < kSplitProducerWarps) {
    const int stream = producer_id >> 1;
    const int half = producer_id & 1;
    const uint64_t b_desc = half == 0 ? b_desc0 : b_desc1;
    const uint32_t d_taddr = split_half_taddr(tmem_base, stream, half);
    for (int group = 0; group < groups; ++group) {
      if (group > 0) {
        mbarrier_wait(&done[stream][half], static_cast<uint32_t>((group - 1) & 1));
      }
#pragma unroll
      for (int k = 0; k < kAccumulatesPerConsume; ++k) {
        tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
      }
      tcgen05_commit(&ready[stream][half]);
    }
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps) {
    const int stream = consumer_id >> 2;
    for (int group = 0; group < groups; ++group) {
#pragma unroll
      for (int half = 0; half < kSplitHalves; ++half) {
        mbarrier_wait(&ready[stream][half], static_cast<uint32_t>(group & 1));
        read_acc ^= consume_128x64_accumulator_tile(split_half_taddr(tmem_base, stream, half));
        if (lane == 0) mbarrier_arrive(&done[stream][half]);
      }
    }
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams * kSplitHalves);
#pragma unroll
    for (int w = 0; w < kConsumerWarps; ++w) out ^= warp_sinks[w];
    sink[blockIdx.x] = out;
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
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf("Usage: %s [--blocks N] [--repeats N] [--warmup N] [--iters N] [--csv PATH]\n",
                  argv[0]);
      std::exit(0);
    }
  }
}

double elapsed_peak_ms(double total_flops, double peak_tflops) {
  return total_flops / (peak_tflops * 1.0e12) * 1.0e3;
}

float run_timed_case(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block(kThreadsPerCta);
  for (int i = 0; i < args.warmup; ++i) {
    mma_ld_pipeline_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    mma_ld_pipeline_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
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

float run_timed_split_n64_case(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block(kThreadsPerCta);
  for (int i = 0; i < args.warmup; ++i) {
    split_n64_half_pipeline_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    split_n64_half_pipeline_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
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

float run_timed_shared_consumer4_case(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block(kSharedConsumerThreadsPerCta);
  for (int i = 0; i < args.warmup; ++i) {
    shared_consumer4_two_streams_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    shared_consumer4_two_streams_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
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

float run_timed_split_n64_4producer_case(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block(kSplit4ProducerThreadsPerCta);
  for (int i = 0; i < args.warmup; ++i) {
    split_n64_4producer_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    split_n64_4producer_128kb_case_kernel<<<grid, block>>>(sink, args.repeats);
  }
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

void write_result_row(FILE* csv,
                      const char* mode,
                      const char* consume_pattern,
                      const char* notes,
                      int producer_warps,
                      int mma_n,
                      int mmas_per_repeat,
                      int consumer_warps,
                      int threads_per_cta,
                      int active,
                      float ms,
                      const Args& args) {
  const double groups = static_cast<double>(args.repeats / kAccumulatesPerConsume);
  const double total_mmas =
      static_cast<double>(args.blocks) * static_cast<double>(args.repeats) *
      static_cast<double>(kStreams) * static_cast<double>(mmas_per_repeat);
  const double total_flops =
      total_mmas * 2.0 * static_cast<double>(kMmaM) * static_cast<double>(mma_n) *
      static_cast<double>(kMmaK);
  const double logical_read_bytes =
      static_cast<double>(args.blocks) * groups * static_cast<double>(kStreams) *
      static_cast<double>(mmas_per_repeat) * 128.0 * static_cast<double>(mma_n) * 4.0;
  const double mma_per_s = total_mmas / (static_cast<double>(ms) * 1.0e-3);
  const double tflops = total_flops / (static_cast<double>(ms) * 1.0e9);
  const double logical_read_gb = logical_read_bytes / 1.0e9;
  const double logical_read_tbps = logical_read_bytes / (static_cast<double>(ms) * 1.0e9);
  const double ideal_ms = elapsed_peak_ms(total_flops, kPeakTflops);
  const double overhead_ms = static_cast<double>(ms) - ideal_ms;
  const double util = ms > 0.0f ? ideal_ms / static_cast<double>(ms) : 0.0;

  std::fprintf(csv,
               "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.3f,"
               "%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%s,%s\n",
               mode, kMmaM, mma_n, kMmaK, producer_warps, consumer_warps, kStreams,
               kAccumulatesPerConsume, args.repeats, args.blocks, threads_per_cta, active,
               kTmemColumnsUsed, kAllocated128ColChunks, consume_pattern, ms, total_mmas,
               mma_per_s, tflops, logical_read_gb, logical_read_tbps, ideal_ms, overhead_ms,
               util, "ok", notes);

  std::printf("%s: %.3f TFLOP/s read %.3f TB/s %.6f ms\n",
              mode, tflops, logical_read_tbps, ms);
}

void write_csv(FILE* csv, uint32_t* sink, const Args& args) {
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, mma_ld_pipeline_128kb_case_kernel, kThreadsPerCta, 0));

  const float ms = run_timed_case(sink, args);

  std::fprintf(csv,
               "mode,mma_m,mma_n,mma_k,producer_warps,consumer_warps,streams,"
               "accumulates_per_consume,repeats,blocks,threads_per_cta,"
               "actual_ctas_per_sm,tmem_columns_used,allocated_128col_chunks,"
               "consume_pattern,elapsed_ms,total_mmas,mma_per_s,mma_TFLOP_per_s,"
               "logical_read_GB,logical_read_TBps,ideal_mma_ms_2200,"
               "overhead_vs_2200_ms,utilization_vs_2200,status,notes\n");
  write_result_row(csv, "serial_128kb_fixed", "ldx128",
                   "fixed_case_producers_warp8_9_consumers_warp0_7_mmax8_then_full_consume",
                   kProducerWarps, kMmaN, 1, kConsumerWarps, kThreadsPerCta, active, ms, args);

  int split_active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &split_active, split_n64_half_pipeline_128kb_case_kernel, kThreadsPerCta, 0));
  const float split_ms = run_timed_split_n64_case(sink, args);
  write_result_row(csv, "split_n64_half_pipeline_128kb", "ldx64",
                   "same_128kb_footprint_two_64col_halves_overlap_half_consume_with_next_half_mma",
                   kProducerWarps, kSplitMmaN, kSplitHalves, kConsumerWarps, kThreadsPerCta,
                   split_active, split_ms, args);

  int shared_active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &shared_active, shared_consumer4_two_streams_128kb_case_kernel,
      kSharedConsumerThreadsPerCta, 0));
  const float shared_ms = run_timed_shared_consumer4_case(sink, args);
  write_result_row(csv, "shared_consumer4_two_streams_128kb", "ldx128",
                   "same_mma_work_four_consumer_warps_consume_both_streams_sequentially",
                   kProducerWarps, kMmaN, 1, kSharedConsumerWarps, kSharedConsumerThreadsPerCta,
                   shared_active, shared_ms, args);

  int split4_active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &split4_active, split_n64_4producer_128kb_case_kernel,
      kSplit4ProducerThreadsPerCta, 0));
  const float split4_ms = run_timed_split_n64_4producer_case(sink, args);
  write_result_row(csv, "split_n64_4producer_128kb", "ldx64",
                   "four_producer_warps_each_issue_one_64col_half_same_128kb_footprint",
                   kSplitProducerWarps, kSplitMmaN, kSplitHalves, kConsumerWarps,
                   kSplit4ProducerThreadsPerCta, split4_active, split4_ms, args);
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
  if (args.repeats % kAccumulatesPerConsume != 0) {
    std::fprintf(stderr, "This fixed benchmark requires repeats to be a multiple of %d.\n",
                 kAccumulatesPerConsume);
    return 1;
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

  std::printf("device=%s sm_%d%d blocks=%d repeats=%d warmup=%d iters=%d csv=%s\n",
              prop.name, prop.major, prop.minor, args.blocks, args.repeats,
              args.warmup, args.iters, args.csv);
  write_csv(csv, sink, args);

  std::fclose(csv);
  CUDA_CHECK(cudaFree(sink));
  return 0;
}
