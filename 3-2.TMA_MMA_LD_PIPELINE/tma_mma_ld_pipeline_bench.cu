#include <cuda.h>
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
static constexpr int kMaxConsumerWarps = 8;
static constexpr int kMaxStreams = kMaxConsumerWarps / 4;
static constexpr int kMaxProducerWarps = kMaxStreams;
static constexpr int kMaxTmaProducerWarps = kMaxStreams;
static constexpr int kMaxTotalWarps =
    kMaxConsumerWarps + kMaxProducerWarps + kMaxTmaProducerWarps;
static constexpr int kMaxSlots = kMaxStreams * 2;
static constexpr int kMaxTmemColumns = 512;
static constexpr int kTileWords = (128 * 128) / 2;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kMmaSliceWords = (128 * 16) / 2;
static constexpr int kMmaSliceBytes = kMmaSliceWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kDynamicSmemBytes = 192 * 1024;
static constexpr int kPeakTflops = 2200;

struct Args {
  int blocks = 4096;
  int repeats = 8192;
  int warmup = 2;
  int iters = 5;
  const char* csv = "3-2.TMA_MMA_LD_PIPELINE/tma_mma_ld_pipeline_bench.csv";
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

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier, uint32_t count) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(addr), "r"(count) : "memory");
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
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
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

template <int BChunkSlices>
const char* b_tma_mode_name() {
  if constexpr (BChunkSlices == 1) return "chunk1_k16";
  if constexpr (BChunkSlices == 2) return "chunk2_k32";
  if constexpr (BChunkSlices == 4) return "chunk4_k64";
  return "chunk8_k128";
}

template <int MmaN>
__host__ __device__ __forceinline__ constexpr int slot_cols() {
  return MmaN;
}

template <int MmaN>
__device__ __forceinline__ uint32_t slot_taddr(uint32_t base, int stream, int buffer, int depth) {
  return base + static_cast<uint32_t>((stream * depth + buffer) * slot_cols<MmaN>());
}

__host__ __device__ __forceinline__ uint32_t ready_phase(int group, int depth) {
  return static_cast<uint32_t>((group / depth) & 1);
}

__host__ __device__ __forceinline__ uint32_t reuse_done_phase(int group, int depth) {
  return static_cast<uint32_t>(((group - depth) / depth) & 1);
}

template <int MmaN,
          int ConsumerWarps,
          int ProducerWarps,
          int AccumulatesPerConsume,
          Mode KernelMode,
          int BChunkSlices>
__global__ __launch_bounds__(kMaxTotalWarps * kWarpSize, 1)
void tma_mma_ld_kernel(const __grid_constant__ CUtensorMap a_map,
                       const __grid_constant__ CUtensorMap b_map,
                       uint32_t* __restrict__ sink,
                       int repeats,
                       int b_stride_repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)sink;
  (void)repeats;
#else
  static_assert(ConsumerWarps == 4 || ConsumerWarps == 8, "ConsumerWarps must be 4 or 8.");
  static_assert(ProducerWarps >= 1 && ProducerWarps <= ConsumerWarps / 4,
                "ProducerWarps must be in 1..streams.");
  static_assert(AccumulatesPerConsume == 1 || AccumulatesPerConsume == 2 ||
                    AccumulatesPerConsume == 4 || AccumulatesPerConsume == 8 ||
                    AccumulatesPerConsume == 16 || AccumulatesPerConsume == 32,
                "Unsupported accumulate group.");
  static_assert(BChunkSlices == 1 || BChunkSlices == 2 || BChunkSlices == 4 ||
                    BChunkSlices == 8,
                "B chunk must be 1, 2, 4, or 8 k16 slices.");
  constexpr int kStreams = ConsumerWarps / 4;
  constexpr int kDepth = KernelMode == Mode::kSerial ? 1 : 2;

  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* a_smem = smem_words;
  uint32_t* b_smem = a_smem + kTileWords;

  __shared__ uint64_t a_ready;
  __shared__ uint64_t b_ready[kMaxStreams];
  __shared__ uint64_t ready[kMaxSlots];
  __shared__ uint64_t done[kMaxSlots];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kMaxConsumerWarps];

  if (threadIdx.x == 0) {
    mbarrier_init(&a_ready, 1);
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
      mbarrier_init(&b_ready[s], 1);
#pragma unroll
      for (int b = 0; b < kDepth; ++b) {
        const int idx = s * kDepth + b;
        mbarrier_init(&ready[idx], 1);
        mbarrier_init(&done[idx], 4);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
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
  const int groups = (repeats + AccumulatesPerConsume - 1) / AccumulatesPerConsume;

  if (warp_id == 0 && lane == 0) {
    mbarrier_expect_tx(&a_ready, kTileBytes);
    tma_load_4d(&a_map, smem_ptr_u32(a_smem), &a_ready, 0, blockIdx.x * 64, 0, 0);
  }

  const uint32_t idesc = make_bf16_idesc<MmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  for (int group = 0; group < groups; ++group) {
    const int buffer = group & (kDepth - 1);

    if (lane == 0 && producer_id >= 0 && producer_id < ProducerWarps) {
      for (int stream = producer_id; stream < kStreams; stream += ProducerWarps) {
        const int slot_idx = stream * kDepth + buffer;
        if (group >= kDepth) {
          mbarrier_wait(&done[slot_idx], reuse_done_phase(group, kDepth));
        }
        const uint32_t d_taddr = slot_taddr<MmaN>(tmem_base, stream, buffer, kDepth);
        {
          uint32_t* stream_b = b_smem + stream * kTileWords;
          constexpr int kChunksPerGroup =
              (AccumulatesPerConsume + BChunkSlices - 1) / BChunkSlices;
#pragma unroll
          for (int chunk = 0; chunk < AccumulatesPerConsume; chunk += BChunkSlices) {
            const int iter0 = group * AccumulatesPerConsume + chunk;
            if (iter0 < repeats) {
              mbarrier_expect_tx(&b_ready[stream], BChunkSlices * kMmaSliceBytes);
              tma_load_4d(&b_map, smem_ptr_u32(stream_b), &b_ready[stream], 0,
                          (stream * b_stride_repeats + iter0) * 8, 0, 0);
              mbarrier_wait(&a_ready, 0);
              const int chunk_id = chunk / BChunkSlices;
              const uint32_t b_phase =
                  static_cast<uint32_t>((group * kChunksPerGroup + chunk_id) & 1);
              mbarrier_wait(&b_ready[stream], b_phase);
#pragma unroll
              for (int kk = 0; kk < BChunkSlices; ++kk) {
                const int iter = iter0 + kk;
                if (kk < AccumulatesPerConsume - chunk && iter < repeats) {
                  const uint64_t a_slice_desc =
                      make_smem_desc(smem_ptr_u32(a_smem + (iter & 7) * kMmaSliceWords));
                  const uint64_t b_slice_desc =
                      make_smem_desc(smem_ptr_u32(stream_b + kk * kMmaSliceWords));
                  tcgen05_mma_bf16_ss(d_taddr, a_slice_desc, b_slice_desc, idesc,
                                      (chunk + kk) != 0);
                }
              }
            }
          }
        }
        tcgen05_commit(&ready[slot_idx]);
      }
    }

    if (consumer_id >= 0 && consumer_id < ConsumerWarps) {
      const int stream = consumer_id >> 2;
      const int slot_idx = stream * kDepth + buffer;
      mbarrier_wait(&ready[slot_idx], ready_phase(group, kDepth));
      read_acc ^= consume_accumulator_tile<MmaN>(
          slot_taddr<MmaN>(tmem_base, stream, buffer, kDepth));
      if (lane == 0) mbarrier_arrive(&done[slot_idx]);
    }
  }

  if (consumer_id >= 0 && consumer_id < ConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams) ^
                   static_cast<uint32_t>(kDepth * MmaN);
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

template <int MmaN,
          int ConsumerWarps,
          int MmaProducerWarps,
          int TmaProducerWarps,
          Mode KernelMode,
          int BChunkSlices>
__global__ __launch_bounds__(kMaxTotalWarps * kWarpSize, 1)
void tma_prefetch_mma_ld_kernel(const __grid_constant__ CUtensorMap a_map,
                                const __grid_constant__ CUtensorMap b_map,
                                uint32_t* __restrict__ sink,
                                int repeats,
                                int b_stride_repeats) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)sink;
  (void)repeats;
  (void)b_stride_repeats;
#else
  static_assert(ConsumerWarps == 4 || ConsumerWarps == 8, "ConsumerWarps must be 4 or 8.");
  static_assert(MmaProducerWarps == 1 || MmaProducerWarps == 2,
                "MmaProducerWarps must be 1 or 2.");
  static_assert(TmaProducerWarps == 1 || TmaProducerWarps == 2,
                "TmaProducerWarps must be 1 or 2.");
  static_assert(BChunkSlices == 1 || BChunkSlices == 2 || BChunkSlices == 4 ||
                    BChunkSlices == 8,
                "B chunk must be 1, 2, 4, or 8 k16 slices.");
  constexpr int kStreams = ConsumerWarps / 4;
  constexpr int kAccumulatesPerConsume = 8;
  constexpr int kBDepth = 2;
  constexpr int kOutDepth = KernelMode == Mode::kSerial ? 1 : 2;

  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem_words = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* a_smem = smem_words;
  uint32_t* b_smem = a_smem + kTileWords;

  __shared__ uint64_t a_ready;
  __shared__ uint64_t b_ready[kMaxSlots];
  __shared__ uint64_t b_done[kMaxSlots];
  __shared__ uint64_t out_ready[kMaxSlots];
  __shared__ uint64_t out_done[kMaxSlots];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kMaxConsumerWarps];

  if (threadIdx.x == 0) {
    mbarrier_init(&a_ready, 1);
#pragma unroll
    for (int s = 0; s < kStreams; ++s) {
#pragma unroll
      for (int b = 0; b < kBDepth; ++b) {
        const int idx = s * kBDepth + b;
        mbarrier_init(&b_ready[idx], 1);
        mbarrier_init(&b_done[idx], 4);
      }
#pragma unroll
      for (int b = 0; b < kOutDepth; ++b) {
        const int idx = s * kOutDepth + b;
        mbarrier_init(&out_ready[idx], 1);
        mbarrier_init(&out_done[idx], 4);
      }
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int consumer_id = warp_id;
  const int mma_id = warp_id - ConsumerWarps;
  const int tma_id = warp_id - ConsumerWarps - MmaProducerWarps;

  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane == 0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const int groups = (repeats + kAccumulatesPerConsume - 1) / kAccumulatesPerConsume;

  if (tma_id == 0 && lane == 0) {
    mbarrier_expect_tx(&a_ready, kTileBytes);
    tma_load_4d(&a_map, smem_ptr_u32(a_smem), &a_ready, 0, blockIdx.x * 64, 0, 0);
  }

  const uint32_t idesc = make_bf16_idesc<MmaN>();
  uint32_t read_acc = static_cast<uint32_t>(threadIdx.x + 1);

  if (lane == 0 && tma_id >= 0 && tma_id < TmaProducerWarps) {
    for (int group = 0; group < groups; ++group) {
      const int b_buffer = group & 1;
      for (int stream = tma_id; stream < kStreams; stream += TmaProducerWarps) {
        const int b_idx = stream * kBDepth + b_buffer;
        if (group >= kBDepth) {
          mbarrier_wait(&b_done[b_idx], reuse_done_phase(group, kBDepth));
        }
        const int iter0 = group * kAccumulatesPerConsume;
        const int remaining = repeats - iter0;
        const int valid = remaining < kAccumulatesPerConsume ? remaining : kAccumulatesPerConsume;
        const int chunk_count = (valid + BChunkSlices - 1) / BChunkSlices;
        uint32_t* stream_b = b_smem + (stream * kBDepth + b_buffer) * kTileWords;
        mbarrier_expect_tx(&b_ready[b_idx], chunk_count * BChunkSlices * kMmaSliceBytes);
#pragma unroll
        for (int chunk = 0; chunk < kAccumulatesPerConsume; chunk += BChunkSlices) {
          if (chunk < valid) {
            tma_load_4d(&b_map, smem_ptr_u32(stream_b + chunk * kMmaSliceWords),
                        &b_ready[b_idx], 0,
                        (stream * b_stride_repeats + iter0 + chunk) * 8, 0, 0);
          }
        }
      }
    }
  }

  if (lane == 0 && mma_id >= 0 && mma_id < MmaProducerWarps) {
    for (int group = 0; group < groups; ++group) {
      const int b_buffer = group & 1;
      const int out_buffer = group & (kOutDepth - 1);
      for (int stream = mma_id; stream < kStreams; stream += MmaProducerWarps) {
        const int b_idx = stream * kBDepth + b_buffer;
        const int out_idx = stream * kOutDepth + out_buffer;
        if (group >= kOutDepth) {
          mbarrier_wait(&out_done[out_idx], reuse_done_phase(group, kOutDepth));
        }
        mbarrier_wait(&a_ready, 0);
        mbarrier_wait(&b_ready[b_idx], ready_phase(group, kBDepth));

        uint32_t* stream_b = b_smem + (stream * kBDepth + b_buffer) * kTileWords;
        const uint32_t d_taddr = slot_taddr<MmaN>(tmem_base, stream, out_buffer, kOutDepth);
        int local_mma = 0;
#pragma unroll
        for (int kk = 0; kk < kAccumulatesPerConsume; ++kk) {
          const int iter = group * kAccumulatesPerConsume + kk;
          if (iter < repeats) {
            const uint64_t a_slice_desc =
                make_smem_desc(smem_ptr_u32(a_smem + (iter & 7) * kMmaSliceWords));
            const uint64_t b_slice_desc =
                make_smem_desc(smem_ptr_u32(stream_b + kk * kMmaSliceWords));
            tcgen05_mma_bf16_ss(d_taddr, a_slice_desc, b_slice_desc, idesc, local_mma != 0);
            ++local_mma;
          }
        }
        tcgen05_commit(&out_ready[out_idx]);
      }
    }
  }

  if (consumer_id >= 0 && consumer_id < ConsumerWarps) {
    const int stream = consumer_id >> 2;
    for (int group = 0; group < groups; ++group) {
      const int b_buffer = group & 1;
      const int out_buffer = group & (kOutDepth - 1);
      const int b_idx = stream * kBDepth + b_buffer;
      const int out_idx = stream * kOutDepth + out_buffer;
      mbarrier_wait(&out_ready[out_idx], ready_phase(group, kOutDepth));
      read_acc ^=
          consume_accumulator_tile<MmaN>(slot_taddr<MmaN>(tmem_base, stream, out_buffer,
                                                          kOutDepth));
      if (lane == 0) {
        mbarrier_arrive(&out_done[out_idx]);
        mbarrier_arrive(&b_done[b_idx]);
      }
    }
  }

  if (consumer_id >= 0 && consumer_id < ConsumerWarps && lane == 0) {
    warp_sinks[consumer_id] = read_acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    tcgen05_fence_after_thread_sync();
    uint32_t out = tmem_base ^ static_cast<uint32_t>(repeats * kStreams) ^
                   static_cast<uint32_t>(kOutDepth * MmaN);
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

void encode_tma_map(CUtensorMap* map,
                    void* base,
                    uint64_t dim0_words,
                    uint64_t dim1_rows,
                    uint32_t box0_words,
                    uint32_t box1_rows) {
  const cuuint64_t global_dim[4] = {dim0_words, dim1_rows, 1, 1};
  const cuuint64_t global_stride[3] = {
      dim0_words * sizeof(uint32_t),
      dim0_words * dim1_rows * sizeof(uint32_t),
      dim0_words * dim1_rows * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {box0_words, box1_rows, 1, 1};
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
               "cuTensorMapEncodeTiled");
}

double elapsed_peak_ms(double total_flops, double peak_tflops) {
  return total_flops / (peak_tflops * 1.0e12) * 1.0e3;
}

double tbps_from_bytes(double bytes, double ms) {
  return ms > 0.0 ? bytes / (ms * 1.0e-3) / 1.0e12 : 0.0;
}

void write_result_row(FILE* csv,
                      const char* mode,
                      const char* b_tma_mode,
                      int b_chunk_slices,
                      int b_chunk_k,
                      const char* pipeline,
                      int mma_m,
                      int mma_n,
                      int mma_k,
                      int consumer_warps,
                      int streams,
                      int mma_producer_warps,
                      int tma_producer_warps,
                      int b_smem_buffers,
                      int accumulates_per_consume,
                      int repeats,
                      int blocks,
                      int threads,
                      int actual_ctas_per_sm,
                      int buffer_depth,
                      int tmem_columns,
                      int dynamic_smem_bytes,
                      const char* consume_pattern,
                      double elapsed_ms,
                      double total_mmas,
                      double mma_tflops,
                      double logical_read_gb,
                      double logical_read_tbps,
                      double a_tma_gb,
                      double b_tma_gb,
                      double total_tma_tbps,
                      double ideal_mma_ms,
                      double utilization,
                      const char* status,
                      const char* notes) {
  std::fprintf(csv,
               "%s,%s,%d,%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,"
               "%.6f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.6f,%.3f,%s,%s\n",
               mode, b_tma_mode, b_chunk_slices, b_chunk_k, pipeline, mma_m, mma_n, mma_k,
               consumer_warps, streams, mma_producer_warps, tma_producer_warps,
               b_smem_buffers, accumulates_per_consume, repeats, blocks, threads,
               actual_ctas_per_sm, buffer_depth, tmem_columns, dynamic_smem_bytes,
               consume_pattern, elapsed_ms, total_mmas, mma_tflops, logical_read_gb,
               logical_read_tbps, a_tma_gb, b_tma_gb, total_tma_tbps, ideal_mma_ms,
               utilization, status, notes);
}

template <int MmaN,
          int ConsumerWarps,
          int ProducerWarps,
          int AccumulatesPerConsume,
          Mode KernelMode,
          int BChunkSlices>
float run_timed_case(const Args& args,
                     const CUtensorMap& a_map,
                     const CUtensorMap& b_map,
                     uint32_t* sink) {
  dim3 grid(args.blocks);
  dim3 block((ProducerWarps + ConsumerWarps) * kWarpSize);
  auto kernel = tma_mma_ld_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume,
                                  KernelMode, BChunkSlices>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map, sink, args.repeats,
                                               args.repeats + 7);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map, sink, args.repeats,
                                               args.repeats + 7);
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

template <int MmaN,
          int ConsumerWarps,
          int ProducerWarps,
          int AccumulatesPerConsume,
          Mode KernelMode,
          int BChunkSlices>
void write_case(FILE* csv,
                const Args& args,
                const CUtensorMap& a_map,
                const CUtensorMap& b_map,
                uint32_t* sink) {
  constexpr int kStreams = ConsumerWarps / 4;
  constexpr int kDepth = KernelMode == Mode::kSerial ? 1 : 2;
  constexpr int kTmemColumns = kStreams * kDepth * slot_cols<MmaN>();
  const char* mode = KernelMode == Mode::kSerial ? "serial" : "double_buffer";
  const char* b_tma_mode = b_tma_mode_name<BChunkSlices>();
  const int threads = (ProducerWarps + ConsumerWarps) * kWarpSize;
  const int groups = (args.repeats + AccumulatesPerConsume - 1) / AccumulatesPerConsume;
  const double total_mmas = static_cast<double>(args.blocks) * args.repeats * kStreams;
  const double total_flops = total_mmas * flops_per_mma<MmaN>();
  const double logical_read_bytes =
      static_cast<double>(args.blocks) * groups * kStreams * 128.0 * MmaN * 4.0;
  const double a_tma_bytes = static_cast<double>(args.blocks) * kTileBytes;
  int b_tma_loads_per_stream = 0;
  for (int g = 0; g < groups; ++g) {
    const int remaining = args.repeats - g * AccumulatesPerConsume;
    const int valid = remaining < AccumulatesPerConsume ? remaining : AccumulatesPerConsume;
    if (valid > 0) {
      b_tma_loads_per_stream += (valid + BChunkSlices - 1) / BChunkSlices;
    }
  }
  const double b_tma_bytes =
      static_cast<double>(args.blocks) * kStreams * b_tma_loads_per_stream *
      static_cast<double>(BChunkSlices * kMmaSliceBytes);
  const double ideal_ms = elapsed_peak_ms(total_flops, static_cast<double>(kPeakTflops));

  if constexpr (kTmemColumns > kMaxTmemColumns) {
    write_result_row(csv, mode, b_tma_mode, BChunkSlices, BChunkSlices * 16,
                     "coupled_tma_mma", kMmaM, MmaN, kMmaK, ConsumerWarps, kStreams,
                     ProducerWarps, 0, 0, AccumulatesPerConsume, args.repeats, args.blocks,
                     threads, 0, kDepth, kTmemColumns, kDynamicSmemBytes,
                     consume_pattern_name<MmaN>(), 0.0, total_mmas, 0.0,
                     logical_read_bytes / 1.0e9, 0.0, a_tma_bytes / 1.0e9,
                     b_tma_bytes / 1.0e9, 0.0, ideal_ms, 0.0, "skipped_tmem_capacity",
                     "requires_more_than_512_tmem_columns");
    return;
  }

  int active = 0;
  auto kernel = tma_mma_ld_kernel<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume,
                                  KernelMode, BChunkSlices>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads,
                                                           kDynamicSmemBytes));
  const float ms = run_timed_case<MmaN, ConsumerWarps, ProducerWarps, AccumulatesPerConsume,
                                  KernelMode, BChunkSlices>(args, a_map, b_map, sink);
  const double seconds = static_cast<double>(ms) * 1.0e-3;
  const double tflops = total_flops / seconds / 1.0e12;
  const double read_tbps = tbps_from_bytes(logical_read_bytes, ms);
  const double tma_tbps = tbps_from_bytes(a_tma_bytes + b_tma_bytes, ms);
  const double utilization = tflops / static_cast<double>(kPeakTflops);

  write_result_row(csv, mode, b_tma_mode, BChunkSlices, BChunkSlices * 16,
                   "coupled_tma_mma", kMmaM, MmaN, kMmaK, ConsumerWarps, kStreams,
                   ProducerWarps, 0, 0, AccumulatesPerConsume, args.repeats, args.blocks,
                   threads, active, kDepth, kTmemColumns, kDynamicSmemBytes,
                   consume_pattern_name<MmaN>(), ms, total_mmas, tflops,
                   logical_read_bytes / 1.0e9, read_tbps, a_tma_bytes / 1.0e9,
                   b_tma_bytes / 1.0e9, tma_tbps, ideal_ms, utilization, "ok",
                   "a_tma_full_tile_once_b_tma_mode_then_tmem_ld_consume");
}

template <int MmaN,
          int ConsumerWarps,
          int MmaProducerWarps,
          int TmaProducerWarps,
          Mode KernelMode,
          int BChunkSlices>
float run_prefetch_timed_case(const Args& args,
                              const CUtensorMap& a_map,
                              const CUtensorMap& b_map,
                              uint32_t* sink) {
  dim3 grid(args.blocks);
  dim3 block((ConsumerWarps + MmaProducerWarps + TmaProducerWarps) * kWarpSize);
  auto kernel = tma_prefetch_mma_ld_kernel<MmaN, ConsumerWarps, MmaProducerWarps,
                                           TmaProducerWarps, KernelMode, BChunkSlices>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map, sink, args.repeats,
                                               args.repeats + 7);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    kernel<<<grid, block, kDynamicSmemBytes>>>(a_map, b_map, sink, args.repeats,
                                               args.repeats + 7);
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

template <int MmaN,
          int ConsumerWarps,
          int MmaProducerWarps,
          int TmaProducerWarps,
          Mode KernelMode,
          int BChunkSlices>
void write_prefetch_case(FILE* csv,
                         const Args& args,
                         const CUtensorMap& a_map,
                         const CUtensorMap& b_map,
                         uint32_t* sink) {
  constexpr int kStreams = ConsumerWarps / 4;
  constexpr int kAccumulatesPerConsume = 8;
  constexpr int kOutDepth = KernelMode == Mode::kSerial ? 1 : 2;
  constexpr int kTmemColumns = kStreams * kOutDepth * slot_cols<MmaN>();
  const char* mode = KernelMode == Mode::kSerial ? "serial" : "double_buffer";
  const char* b_tma_mode = b_tma_mode_name<BChunkSlices>();
  const int threads = (ConsumerWarps + MmaProducerWarps + TmaProducerWarps) * kWarpSize;
  const int groups = (args.repeats + kAccumulatesPerConsume - 1) / kAccumulatesPerConsume;
  const double total_mmas = static_cast<double>(args.blocks) * args.repeats * kStreams;
  const double total_flops = total_mmas * flops_per_mma<MmaN>();
  const double logical_read_bytes =
      static_cast<double>(args.blocks) * groups * kStreams * 128.0 * MmaN * 4.0;
  const double a_tma_bytes = static_cast<double>(args.blocks) * kTileBytes;
  const double b_tma_bytes = static_cast<double>(args.blocks) * groups * kStreams *
                             8.0 * static_cast<double>(kMmaSliceBytes);
  const double ideal_ms = elapsed_peak_ms(total_flops, static_cast<double>(kPeakTflops));

  if constexpr (kTmemColumns > kMaxTmemColumns) {
    write_result_row(csv, mode, b_tma_mode, BChunkSlices, BChunkSlices * 16,
                     "separate_tma_mma_bdouble", kMmaM, MmaN, kMmaK, ConsumerWarps,
                     kStreams, MmaProducerWarps, TmaProducerWarps, 2,
                     kAccumulatesPerConsume, args.repeats, args.blocks, threads, 0,
                     kOutDepth, kTmemColumns, kDynamicSmemBytes, consume_pattern_name<MmaN>(),
                     0.0, total_mmas, 0.0, logical_read_bytes / 1.0e9, 0.0,
                     a_tma_bytes / 1.0e9, b_tma_bytes / 1.0e9, 0.0, ideal_ms, 0.0,
                     "skipped_tmem_capacity", "requires_more_than_512_tmem_columns");
    return;
  }

  int active = 0;
  auto kernel = tma_prefetch_mma_ld_kernel<MmaN, ConsumerWarps, MmaProducerWarps,
                                           TmaProducerWarps, KernelMode, BChunkSlices>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads,
                                                           kDynamicSmemBytes));
  const float ms =
      run_prefetch_timed_case<MmaN, ConsumerWarps, MmaProducerWarps, TmaProducerWarps,
                              KernelMode, BChunkSlices>(args, a_map, b_map, sink);
  const double seconds = static_cast<double>(ms) * 1.0e-3;
  const double tflops = total_flops / seconds / 1.0e12;
  const double read_tbps = tbps_from_bytes(logical_read_bytes, ms);
  const double tma_tbps = tbps_from_bytes(a_tma_bytes + b_tma_bytes, ms);
  const double utilization = tflops / static_cast<double>(kPeakTflops);

  write_result_row(csv, mode, b_tma_mode, BChunkSlices, BChunkSlices * 16,
                   "separate_tma_mma_bdouble", kMmaM, MmaN, kMmaK, ConsumerWarps, kStreams,
                   MmaProducerWarps, TmaProducerWarps, 2, kAccumulatesPerConsume,
                   args.repeats, args.blocks, threads, active, kOutDepth, kTmemColumns,
                   kDynamicSmemBytes, consume_pattern_name<MmaN>(), ms, total_mmas, tflops,
                   logical_read_bytes / 1.0e9, read_tbps, a_tma_bytes / 1.0e9,
                   b_tma_bytes / 1.0e9, tma_tbps, ideal_ms, utilization, "ok",
                   "a_once_b_smem_double_buffer_tma_prefetch_mma_x8_ld_consume");
}

template <int MmaN, int ConsumerWarps, int ProducerWarps, int BChunkSlices>
void write_chunk_case(FILE* csv,
                      const Args& args,
                      const CUtensorMap& a_map,
                      const CUtensorMap& b_map,
                      uint32_t* sink) {
  constexpr int kAccumulatesPerConsume = 8;
  write_case<MmaN, ConsumerWarps, ProducerWarps, kAccumulatesPerConsume, Mode::kSerial,
             BChunkSlices>(csv, args, a_map, b_map, sink);
  std::fflush(csv);
  write_case<MmaN, ConsumerWarps, ProducerWarps, kAccumulatesPerConsume, Mode::kDoubleBuffer,
             BChunkSlices>(csv, args, a_map, b_map, sink);
  std::fflush(csv);
}

template <int MmaN, int ConsumerWarps, int ProducerWarps>
void write_chunk_sweep(FILE* csv,
                       const Args& args,
                       const CUtensorMap& a_map,
                       const CUtensorMap& b1_map,
                       const CUtensorMap& b2_map,
                       const CUtensorMap& b4_map,
                       const CUtensorMap& b8_map,
                       uint32_t* sink) {
  write_chunk_case<MmaN, ConsumerWarps, ProducerWarps, 1>(csv, args, a_map, b1_map, sink);
  write_chunk_case<MmaN, ConsumerWarps, ProducerWarps, 2>(csv, args, a_map, b2_map, sink);
  write_chunk_case<MmaN, ConsumerWarps, ProducerWarps, 4>(csv, args, a_map, b4_map, sink);
  write_chunk_case<MmaN, ConsumerWarps, ProducerWarps, 8>(csv, args, a_map, b8_map, sink);
}

template <int MmaN>
void write_n_sweep(FILE* csv,
                   const Args& args,
                   const CUtensorMap& a_map,
                   const CUtensorMap& b1_map,
                   const CUtensorMap& b2_map,
                   const CUtensorMap& b4_map,
                   const CUtensorMap& b8_map,
                   uint32_t* sink) {
  write_chunk_sweep<MmaN, 4, 1>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_chunk_sweep<MmaN, 8, 1>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_chunk_sweep<MmaN, 8, 2>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
}

template <int ConsumerWarps, int MmaProducerWarps, int TmaProducerWarps, int BChunkSlices>
void write_prefetch_chunk(FILE* csv,
                          const Args& args,
                          const CUtensorMap& a_map,
                          const CUtensorMap& b_map,
                          uint32_t* sink) {
  constexpr int kMmaN = 128;
  write_prefetch_case<kMmaN, ConsumerWarps, MmaProducerWarps, TmaProducerWarps, Mode::kSerial,
                      BChunkSlices>(csv, args, a_map, b_map, sink);
  std::fflush(csv);
  write_prefetch_case<kMmaN, ConsumerWarps, MmaProducerWarps, TmaProducerWarps,
                      Mode::kDoubleBuffer, BChunkSlices>(csv, args, a_map, b_map, sink);
  std::fflush(csv);
}

template <int ConsumerWarps, int MmaProducerWarps, int TmaProducerWarps>
void write_prefetch_chunk_sweep(FILE* csv,
                                const Args& args,
                                const CUtensorMap& a_map,
                                const CUtensorMap& b1_map,
                                const CUtensorMap& b2_map,
                                const CUtensorMap& b4_map,
                                const CUtensorMap& b8_map,
                                uint32_t* sink) {
  write_prefetch_chunk<ConsumerWarps, MmaProducerWarps, TmaProducerWarps, 1>(
      csv, args, a_map, b1_map, sink);
  write_prefetch_chunk<ConsumerWarps, MmaProducerWarps, TmaProducerWarps, 2>(
      csv, args, a_map, b2_map, sink);
  write_prefetch_chunk<ConsumerWarps, MmaProducerWarps, TmaProducerWarps, 4>(
      csv, args, a_map, b4_map, sink);
  write_prefetch_chunk<ConsumerWarps, MmaProducerWarps, TmaProducerWarps, 8>(
      csv, args, a_map, b8_map, sink);
}

void write_prefetch_sweep(FILE* csv,
                          const Args& args,
                          const CUtensorMap& a_map,
                          const CUtensorMap& b1_map,
                          const CUtensorMap& b2_map,
                          const CUtensorMap& b4_map,
                          const CUtensorMap& b8_map,
                          uint32_t* sink) {
  write_prefetch_chunk_sweep<4, 1, 1>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<4, 1, 2>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<4, 2, 1>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<4, 2, 2>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<8, 1, 1>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<8, 1, 2>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<8, 2, 1>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
  write_prefetch_chunk_sweep<8, 2, 2>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, sink);
}

void write_header(FILE* csv) {
  std::fprintf(csv,
               "mode,b_tma_mode,b_chunk_k16,b_chunk_k,pipeline,mma_m,mma_n,mma_k,"
               "consumer_warps,streams,mma_producer_warps,tma_producer_warps,b_smem_buffers,"
               "accumulates_per_consume,repeats,blocks,threads_per_cta,actual_ctas_per_sm,"
               "buffer_depth,tmem_columns_used,dynamic_smem_bytes,consume_pattern,elapsed_ms,"
               "total_mmas,mma_TFLOP_per_s,logical_read_GB,logical_read_TBps,"
               "a_tma_GB,b_tma_GB,total_tma_TBps,ideal_mma_ms_2200,utilization_vs_2200,"
               "status,notes\n");
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

  const size_t a_words = static_cast<size_t>(args.blocks) * kTileWords;
  const int b_stride_repeats = args.repeats + 7;
  const size_t b_words = static_cast<size_t>(b_stride_repeats) * kMaxStreams * kMmaSliceWords;
  uint32_t *d_a = nullptr, *d_b = nullptr, *d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_b, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink, args.blocks * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_a, 0x3f, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_b, 0x11, b_words * sizeof(uint32_t)));

  CUtensorMap a_map{}, b1_map{}, b2_map{}, b4_map{}, b8_map{};
  encode_tma_map(&a_map, d_a, 128, static_cast<uint64_t>(args.blocks) * 64, 128, 64);
  encode_tma_map(&b1_map, d_b, 128,
                 static_cast<uint64_t>(b_stride_repeats) * kMaxStreams * 8, 128, 8);
  encode_tma_map(&b2_map, d_b, 128,
                 static_cast<uint64_t>(b_stride_repeats) * kMaxStreams * 8, 128, 16);
  encode_tma_map(&b4_map, d_b, 128,
                 static_cast<uint64_t>(b_stride_repeats) * kMaxStreams * 8, 128, 32);
  encode_tma_map(&b8_map, d_b, 128,
                 static_cast<uint64_t>(b_stride_repeats) * kMaxStreams * 8, 128, 64);

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    return 1;
  }
  write_header(csv);

  write_n_sweep<128>(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, d_sink);
  write_prefetch_sweep(csv, args, a_map, b1_map, b2_map, b4_map, b8_map, d_sink);

  std::fclose(csv);
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
