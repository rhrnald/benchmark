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
static constexpr int kMmaK = 16;
static constexpr int kMaxConsumerWarps = 12;
static constexpr int kMaxStreams = kMaxConsumerWarps / 4;
static constexpr int kMaxProducerWarps = kMaxStreams;
static constexpr int kMaxTotalWarps = kMaxConsumerWarps + kMaxProducerWarps;
static constexpr int kMaxSlots = kMaxStreams * 2;
static constexpr int kMaxTmemColumns = 512;
static constexpr int kPeakTflops = 2200;

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int warmup = 2;
  int iters = 5;
  const char* csv = "3.MMA_LD_PIPELINE/mma_ld_pipeline_bench.csv";
};

enum class Mode {
  kSerial,
  kDoubleBuffer,
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

template <int MmaN>
__host__ __device__ __forceinline__ uint32_t make_bf16_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(MmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kMmaM >> 4) << 24;
  return desc;
}

template <int MmaN>
__host__ __device__ __forceinline__ constexpr double flops_per_mma() {
  return 2.0 * static_cast<double>(kMmaM) * static_cast<double>(MmaN) *
         static_cast<double>(kMmaK);
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
  (void)d_taddr; (void)a_desc; (void)b_desc; (void)idesc; (void)input_d;
#endif
}

__device__ __forceinline__ uint32_t tcgen05_ld_32x32b_x32_acc(uint32_t taddr) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t acc;
  asm volatile(
      "{ .reg .b32 r<32>; .reg .b32 acc; "
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, "
      "r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31}, [%1]; "
      "xor.b32 acc, r0, r7; "
      "xor.b32 acc, acc, r15; "
      "xor.b32 acc, acc, r23; "
      "xor.b32 %0, acc, r31; }"
      : "=r"(acc)
      : "r"(taddr)
      : "memory");
  return acc;
#else
  (void)taddr;
  return 0;
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

template <int MmaN>
__device__ __forceinline__ uint32_t consume_accumulator_tile(uint32_t taddr) {
  if constexpr (MmaN == 32) {
    const uint32_t acc = tcgen05_ld_32x32b_x32_acc(taddr);
    tcgen05_wait_ld();
    return acc;
  } else if constexpr (MmaN == 64) {
    const uint32_t acc = tcgen05_ld_32x32b_x64_acc(taddr);
    tcgen05_wait_ld();
    return acc;
  } else {
    uint32_t acc = tcgen05_ld_32x32b_x64_acc(taddr);
    acc ^= tcgen05_ld_32x32b_x64_acc(taddr + 64);
    tcgen05_wait_ld();
    return acc;
  }
}

template <int MmaN>
const char* consume_pattern_name() {
  if constexpr (MmaN == 32) return "ldx32";
  if constexpr (MmaN == 64) return "ldx64";
  return "ldx64x2";
}

__device__ __forceinline__ void init_bf16_smem(uint32_t* smem_words) {
  for (int i = threadIdx.x; i < 4096; i += blockDim.x) {
    smem_words[i] = 0x3f803f80u ^ static_cast<uint32_t>((i + 17 * blockIdx.x) & 0x000f000fu);
  }
}

template <int MmaN>
__host__ __device__ __forceinline__ constexpr int slot_cols() {
  return MmaN;
}

template <int MmaN>
__device__ __forceinline__ uint32_t slot_taddr(uint32_t base, int stream, int buffer, int depth) {
  return base + static_cast<uint32_t>((stream * depth + buffer) * slot_cols<MmaN>());
}

template <int MmaN, int ConsumerWarps, int ProducerWarps, int AccumulatesPerConsume>
__global__ __launch_bounds__(kMaxTotalWarps * kWarpSize, 1)
void serial_kernel(uint32_t* __restrict__ sink, int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  static_assert(ConsumerWarps == 4 || ConsumerWarps == 8 || ConsumerWarps == 12,
                "ConsumerWarps must be 4, 8, or 12.");
  static_assert(ProducerWarps >= 1 && ProducerWarps <= ConsumerWarps / 4,
                "ProducerWarps must be in 1..streams.");
  static_assert(AccumulatesPerConsume == 1 || AccumulatesPerConsume == 2 ||
                    AccumulatesPerConsume == 4 || AccumulatesPerConsume == 8 ||
                    AccumulatesPerConsume == 16 || AccumulatesPerConsume == 32,
                "Unsupported accumulate group.");
  constexpr int kStreams = ConsumerWarps / 4;
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kMaxStreams];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kMaxConsumerWarps];

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
  const int producer_id = warp_id - ConsumerWarps;
  const int consumer_id = warp_id;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint32_t idesc = make_bf16_idesc<MmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = (repeats + AccumulatesPerConsume - 1) / AccumulatesPerConsume;

  for (int group = 0; group < groups; ++group) {
    if (lane == 0 && producer_id >= 0 && producer_id < ProducerWarps) {
      for (int stream = producer_id; stream < kStreams; stream += ProducerWarps) {
        const uint32_t d_taddr = slot_taddr<MmaN>(tmem_base, stream, 0, 1);
#pragma unroll
        for (int k = 0; k < AccumulatesPerConsume; ++k) {
          const int i = group * AccumulatesPerConsume + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
        tcgen05_commit(&ready[stream]);
      }
    }

    __syncthreads();
    if (consumer_id >= 0 && consumer_id < ConsumerWarps) {
      const int stream = consumer_id >> 2;
      mbarrier_wait(&ready[stream], static_cast<uint32_t>(group & 1));
      read_acc ^= consume_accumulator_tile<MmaN>(slot_taddr<MmaN>(tmem_base, stream, 0, 1));
    }
    __syncthreads();
  }

  if (consumer_id >= 0 && consumer_id < ConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams);
#pragma unroll
    for (int w = 0; w < ConsumerWarps; ++w) out ^= warp_sinks[w];
    sink[blockIdx.x] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <int MmaN, int ConsumerWarps, int ProducerWarps, int AccumulatesPerConsume>
__global__ __launch_bounds__(kMaxTotalWarps * kWarpSize, 1)
void double_buffer_kernel(uint32_t* __restrict__ sink, int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  static_assert(ConsumerWarps == 4 || ConsumerWarps == 8 || ConsumerWarps == 12,
                "ConsumerWarps must be 4, 8, or 12.");
  static_assert(ProducerWarps >= 1 && ProducerWarps <= ConsumerWarps / 4,
                "ProducerWarps must be in 1..streams.");
  static_assert(AccumulatesPerConsume == 1 || AccumulatesPerConsume == 2 ||
                    AccumulatesPerConsume == 4 || AccumulatesPerConsume == 8 ||
                    AccumulatesPerConsume == 16 || AccumulatesPerConsume == 32,
                "Unsupported accumulate group.");
  constexpr int kStreams = ConsumerWarps / 4;
  constexpr int kDepth = 2;
  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kMaxSlots];
  __shared__ uint64_t done[kMaxSlots];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kMaxConsumerWarps];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
#pragma unroll
      for (int b = 0; b < kDepth; ++b) {
        const int idx = s * kDepth + b;
        mbarrier_init(&ready[idx], 1);
        mbarrier_init(&done[idx], 4);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int producer_id = warp_id - ConsumerWarps;
  const int consumer_id = warp_id;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint32_t idesc = make_bf16_idesc<MmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = (repeats + AccumulatesPerConsume - 1) / AccumulatesPerConsume;

  for (int group = 0; group < groups; ++group) {
    const int buffer = group & 1;
    if (lane == 0 && producer_id >= 0 && producer_id < ProducerWarps) {
      for (int stream = producer_id; stream < kStreams; stream += ProducerWarps) {
        const int idx = stream * kDepth + buffer;
        if (group >= kDepth) {
          mbarrier_wait(&done[idx], static_cast<uint32_t>(((group >> 1) - 1) & 1));
        }
        const uint32_t d_taddr = slot_taddr<MmaN>(tmem_base, stream, buffer, kDepth);
#pragma unroll
        for (int k = 0; k < AccumulatesPerConsume; ++k) {
          const int i = group * AccumulatesPerConsume + k;
          if (i < repeats) {
            tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
          }
        }
        tcgen05_commit(&ready[idx]);
      }
    }

    if (consumer_id >= 0 && consumer_id < ConsumerWarps) {
      const int stream = consumer_id >> 2;
      const int idx = stream * kDepth + buffer;
      mbarrier_wait(&ready[idx], static_cast<uint32_t>((group >> 1) & 1));
      read_acc ^= consume_accumulator_tile<MmaN>(slot_taddr<MmaN>(tmem_base, stream, buffer, kDepth));
      if (lane == 0) mbarrier_arrive(&done[idx]);
    }
  }

  if (consumer_id >= 0 && consumer_id < ConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams);
#pragma unroll
    for (int w = 0; w < ConsumerWarps; ++w) out ^= warp_sinks[w];
    sink[blockIdx.x] = out;
  }
  __syncthreads();

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

template <int AccumulatesPerConsume>
__global__ __launch_bounds__((8 + 2) * kWarpSize, 1)
void stream_pipeline_128kb_kernel(uint32_t* __restrict__ sink, int repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)sink; (void)repeats;
#else
  static_assert(AccumulatesPerConsume == 8 || AccumulatesPerConsume == 16 ||
                    AccumulatesPerConsume == 32,
                "Optimized pipeline is intended for large enough groups.");
  constexpr int kMmaN = 128;
  constexpr int kStreams = 2;
  constexpr int kConsumerWarps = 8;
  constexpr int kProducerWarps = 2;
  constexpr int kTmemColumns = kStreams * slot_cols<kMmaN>();

  __shared__ __align__(1024) uint32_t smem_words[4096];
  __shared__ uint64_t ready[kStreams];
  __shared__ uint64_t done[kStreams];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kConsumerWarps];

  if (threadIdx.x == 0) {
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      mbarrier_init(&ready[s], 1);
      mbarrier_init(&done[s], 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  init_bf16_smem(smem_words);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int producer_id = warp_id - kConsumerWarps;
  const int consumer_id = warp_id;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint64_t a_desc = make_smem_desc(smem_ptr_u32(smem_words));
  const uint64_t b_desc = make_smem_desc(smem_ptr_u32(smem_words + 2048));
  const uint32_t idesc = make_bf16_idesc<kMmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);
  const int groups = (repeats + AccumulatesPerConsume - 1) / AccumulatesPerConsume;

  if (lane == 0 && producer_id >= 0 && producer_id < kProducerWarps) {
    const int stream = producer_id;
    const uint32_t d_taddr = slot_taddr<kMmaN>(tmem_base, stream, 0, 1);
    for (int group = 0; group < groups; ++group) {
      if (group > 0) {
        mbarrier_wait(&done[stream], static_cast<uint32_t>((group - 1) & 1));
      }
#pragma unroll
      for (int k = 0; k < AccumulatesPerConsume; ++k) {
        const int i = group * AccumulatesPerConsume + k;
        if (i < repeats) {
          tcgen05_mma_bf16_ss(d_taddr, a_desc, b_desc, idesc, k != 0);
        }
      }
      tcgen05_commit(&ready[stream]);
    }
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps) {
    const int stream = consumer_id >> 2;
    const uint32_t read_taddr = slot_taddr<kMmaN>(tmem_base, stream, 0, 1);
    for (int group = 0; group < groups; ++group) {
      mbarrier_wait(&ready[stream], static_cast<uint32_t>(group & 1));
      read_acc ^= consume_accumulator_tile<kMmaN>(read_taddr);
      if (lane == 0) mbarrier_arrive(&done[stream]);
    }
  }

  if (consumer_id >= 0 && consumer_id < kConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams) ^
                   static_cast<uint32_t>(kTmemColumns);
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

template <int MmaN, int ConsumerWarps, int ProducerWarps, int AccumulatesPerConsume, Mode KernelMode>
float run_timed_case(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block((ProducerWarps + ConsumerWarps) * kWarpSize);
  for (int i = 0; i < args.warmup; ++i) {
    if constexpr (KernelMode == Mode::kSerial) {
      serial_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume>
          <<<grid, block>>>(sink, args.repeats);
    } else {
      double_buffer_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume>
          <<<grid, block>>>(sink, args.repeats);
    }
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    if constexpr (KernelMode == Mode::kSerial) {
      serial_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume>
          <<<grid, block>>>(sink, args.repeats);
    } else {
      double_buffer_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume>
          <<<grid, block>>>(sink, args.repeats);
    }
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

template <int AccumulatesPerConsume>
float run_stream_pipeline_128kb_case(uint32_t* sink, const Args& args) {
  dim3 grid(args.blocks);
  dim3 block((8 + 2) * kWarpSize);
  for (int i = 0; i < args.warmup; ++i) {
    stream_pipeline_128kb_kernel<AccumulatesPerConsume><<<grid, block>>>(sink, args.repeats);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    stream_pipeline_128kb_kernel<AccumulatesPerConsume><<<grid, block>>>(sink, args.repeats);
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

template <int MmaN, int ConsumerWarps, int ProducerWarps, int AccumulatesPerConsume, Mode KernelMode>
void write_case(FILE* csv, uint32_t* sink, const Args& args) {
  constexpr int kStreams = ConsumerWarps / 4;
  constexpr int kBufferDepth = KernelMode == Mode::kSerial ? 1 : 2;
  constexpr int kTmemColumns = kStreams * kBufferDepth * slot_cols<MmaN>();
  constexpr int kAllocatedChunks = (kTmemColumns + 127) / 128;
  const char* mode = KernelMode == Mode::kSerial ? "serial" : "double_buffer";

  const double mma_groups = static_cast<double>((args.repeats + AccumulatesPerConsume - 1) /
                                                AccumulatesPerConsume);
  const double total_mmas =
      static_cast<double>(args.blocks) * static_cast<double>(args.repeats) *
      static_cast<double>(kStreams);
  const double total_flops = total_mmas * flops_per_mma<MmaN>();
  const double logical_read_bytes =
      static_cast<double>(args.blocks) * mma_groups * static_cast<double>(kStreams) *
      128.0 * static_cast<double>(MmaN) * 4.0;
  const double logical_read_gb = logical_read_bytes / 1.0e9;
  const double ideal_ms = elapsed_peak_ms(total_flops, static_cast<double>(kPeakTflops));

  if constexpr (kTmemColumns > kMaxTmemColumns) {
    std::fprintf(csv,
                 "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.3f,"
                 "%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%s,%s\n",
                 mode, kMmaM, MmaN, kMmaK, ConsumerWarps, kStreams, ProducerWarps,
                 AccumulatesPerConsume, args.repeats, args.blocks,
                 (ProducerWarps + ConsumerWarps) * kWarpSize,
                 0, kBufferDepth, kTmemColumns, kAllocatedChunks, consume_pattern_name<MmaN>(), 0.0,
                 total_mmas, 0.0, 0.0, logical_read_gb, 0.0, ideal_ms, 0.0, 0.0,
                 "skipped_tmem_capacity", "requires_more_than_512_tmem_columns");
    return;
  }

  int active = 0;
  if constexpr (KernelMode == Mode::kSerial) {
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, serial_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume>,
        (ProducerWarps + ConsumerWarps) * kWarpSize, 0));
  } else {
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, double_buffer_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume>,
        (ProducerWarps + ConsumerWarps) * kWarpSize, 0));
  }

  const float ms = run_timed_case<MmaN, ConsumerWarps, ProducerWarps,
                                  AccumulatesPerConsume, KernelMode>(sink, args);
  const double mma_per_s = total_mmas / (static_cast<double>(ms) * 1.0e-3);
  const double tflops = total_flops / (static_cast<double>(ms) * 1.0e9);
  const double logical_read_tbps = logical_read_bytes / (static_cast<double>(ms) * 1.0e9);
  const double overhead_ms = static_cast<double>(ms) - ideal_ms;
  const double util = ms > 0.0f ? ideal_ms / static_cast<double>(ms) : 0.0;

  std::fprintf(csv,
               "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.3f,"
               "%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%s,%s\n",
               mode, kMmaM, MmaN, kMmaK, ConsumerWarps, kStreams, ProducerWarps,
               AccumulatesPerConsume, args.repeats, args.blocks,
               (ProducerWarps + ConsumerWarps) * kWarpSize,
               active, kBufferDepth, kTmemColumns, kAllocatedChunks, consume_pattern_name<MmaN>(), ms,
               total_mmas, mma_per_s, tflops, logical_read_gb, logical_read_tbps,
               ideal_ms, overhead_ms, util, "ok",
               "consumer_warps_prefix_producers_after_consumers");
  std::printf("N=%d cw=%d pw=%d %s acc=%d: %.3f TFLOP/s read %.3f TB/s %.6f ms\n",
              MmaN, ConsumerWarps, ProducerWarps, mode, AccumulatesPerConsume,
              tflops, logical_read_tbps, ms);
}

template <int AccumulatesPerConsume>
void write_stream_pipeline_128kb_case(FILE* csv, uint32_t* sink, const Args& args) {
  constexpr int kMmaN = 128;
  constexpr int kConsumerWarps = 8;
  constexpr int kStreams = 2;
  constexpr int kProducerWarps = 2;
  constexpr int kBufferDepth = 1;
  constexpr int kTmemColumns = 256;
  constexpr int kAllocatedChunks = 2;
  int active = 0;
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, stream_pipeline_128kb_kernel<AccumulatesPerConsume>,
      (kConsumerWarps + kProducerWarps) * kWarpSize, 0));

  const float ms = run_stream_pipeline_128kb_case<AccumulatesPerConsume>(sink, args);
  const double mma_groups = static_cast<double>((args.repeats + AccumulatesPerConsume - 1) /
                                                AccumulatesPerConsume);
  const double total_mmas =
      static_cast<double>(args.blocks) * static_cast<double>(args.repeats) *
      static_cast<double>(kStreams);
  const double total_flops = total_mmas * flops_per_mma<kMmaN>();
  const double logical_read_bytes =
      static_cast<double>(args.blocks) * mma_groups * static_cast<double>(kStreams) *
      128.0 * static_cast<double>(kMmaN) * 4.0;
  const double mma_per_s = total_mmas / (static_cast<double>(ms) * 1.0e-3);
  const double tflops = total_flops / (static_cast<double>(ms) * 1.0e9);
  const double logical_read_gb = logical_read_bytes / 1.0e9;
  const double logical_read_tbps = logical_read_bytes / (static_cast<double>(ms) * 1.0e9);
  const double ideal_ms = elapsed_peak_ms(total_flops, static_cast<double>(kPeakTflops));
  const double overhead_ms = static_cast<double>(ms) - ideal_ms;
  const double util = ms > 0.0f ? ideal_ms / static_cast<double>(ms) : 0.0;

  std::fprintf(csv,
               "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.3f,"
               "%.3f,%.3f,%.3f,%.6f,%.6f,%.3f,%s,%s\n",
               "stream_pipeline_128kb", kMmaM, kMmaN, kMmaK, kConsumerWarps, kStreams,
               kProducerWarps, AccumulatesPerConsume, args.repeats, args.blocks,
               (kConsumerWarps + kProducerWarps) * kWarpSize, active, kBufferDepth,
               kTmemColumns, kAllocatedChunks, consume_pattern_name<kMmaN>(), ms,
               total_mmas, mma_per_s, tflops, logical_read_gb, logical_read_tbps,
               ideal_ms, overhead_ms, util, "ok",
               "128kb_two_stream_per_stream_ready_done_no_cta_sync");
  std::printf("optimized_128kb acc=%d: %.3f TFLOP/s read %.3f TB/s %.6f ms\n",
              AccumulatesPerConsume, tflops, logical_read_tbps, ms);
}

template <int MmaN, int ConsumerWarps, int ProducerWarps, Mode KernelMode>
void write_accumulate_sweep_for_mode(FILE* csv, uint32_t* sink, const Args& args) {
  write_case<MmaN, ConsumerWarps, ProducerWarps, 1, KernelMode>(csv, sink, args);
  std::fflush(csv);
  write_case<MmaN, ConsumerWarps, ProducerWarps, 2, KernelMode>(csv, sink, args);
  std::fflush(csv);
  write_case<MmaN, ConsumerWarps, ProducerWarps, 4, KernelMode>(csv, sink, args);
  std::fflush(csv);
  write_case<MmaN, ConsumerWarps, ProducerWarps, 8, KernelMode>(csv, sink, args);
  std::fflush(csv);
  write_case<MmaN, ConsumerWarps, ProducerWarps, 16, KernelMode>(csv, sink, args);
  std::fflush(csv);
  write_case<MmaN, ConsumerWarps, ProducerWarps, 32, KernelMode>(csv, sink, args);
  std::fflush(csv);
}

template <int MmaN, int ConsumerWarps, int ProducerWarps>
void write_mode_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  write_accumulate_sweep_for_mode<MmaN, ConsumerWarps, ProducerWarps, Mode::kSerial>(
      csv, sink, args);
  write_accumulate_sweep_for_mode<MmaN, ConsumerWarps, ProducerWarps, Mode::kDoubleBuffer>(
      csv, sink, args);
}

template <int MmaN, int ConsumerWarps>
void write_producer_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  write_mode_sweep<MmaN, ConsumerWarps, 1>(csv, sink, args);
  if constexpr (ConsumerWarps >= 8) {
    write_mode_sweep<MmaN, ConsumerWarps, 2>(csv, sink, args);
  }
  if constexpr (ConsumerWarps >= 12) {
    write_mode_sweep<MmaN, ConsumerWarps, 3>(csv, sink, args);
  }
}

template <int MmaN>
void write_n_sweep(FILE* csv, uint32_t* sink, const Args& args) {
  write_producer_sweep<MmaN, 4>(csv, sink, args);
  write_producer_sweep<MmaN, 8>(csv, sink, args);
  write_producer_sweep<MmaN, 12>(csv, sink, args);
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

  std::fprintf(csv,
               "mode,mma_m,mma_n,mma_k,consumer_warps,streams,producer_warps,"
               "accumulates_per_consume,repeats,blocks,threads_per_cta,"
               "actual_ctas_per_sm,buffer_depth,tmem_columns_used,"
               "allocated_128col_chunks,consume_pattern,elapsed_ms,total_mmas,"
               "mma_per_s,mma_TFLOP_per_s,"
               "logical_read_GB,logical_read_TBps,ideal_mma_ms_2200,"
               "overhead_vs_2200_ms,utilization_vs_2200,status,notes\n");

  std::printf("device=%s sm_%d%d sms=%d blocks=%d repeats=%d warmup=%d iters=%d csv=%s\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount,
              args.blocks, args.repeats, args.warmup, args.iters, args.csv);

  write_n_sweep<32>(csv, sink, args);
  write_n_sweep<64>(csv, sink, args);
  write_n_sweep<128>(csv, sink, args);
  write_stream_pipeline_128kb_case<8>(csv, sink, args);
  write_stream_pipeline_128kb_case<16>(csv, sink, args);
  write_stream_pipeline_128kb_case<32>(csv, sink, args);

  std::fclose(csv);
  CUDA_CHECK(cudaFree(sink));
  return 0;
}
