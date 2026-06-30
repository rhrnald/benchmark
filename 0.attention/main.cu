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

#ifndef ATTENTION_CLOCK_TRACE
#define ATTENTION_CLOCK_TRACE 0
#endif

#ifndef ATTENTION_CLOCK_TRACE_PACK_GROUP
#define ATTENTION_CLOCK_TRACE_PACK_GROUP -1
#endif

#ifndef ATTENTION_ROW_MAX_ONLY
#define ATTENTION_ROW_MAX_ONLY 0
#endif

#ifndef ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#define ATTENTION_FIRST_ITER_ROW_MAX_SHIFT 1
#endif

#ifndef ATTENTION_FIRST_ITER_APPLY_SHIFT
#define ATTENTION_FIRST_ITER_APPLY_SHIFT ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#endif

#ifndef ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
#define ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE ATTENTION_FIRST_ITER_APPLY_SHIFT
#endif

#ifndef ATTENTION_FIRST_ITER_COMPUTE_MAX
#define ATTENTION_FIRST_ITER_COMPUTE_MAX ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
#endif

#ifndef ATTENTION_ROW_SUM_RARE_UPDATE
#define ATTENTION_ROW_SUM_RARE_UPDATE 1
#endif

#ifndef ATTENTION_ROW_SUM_UPDATE_LIMIT
#define ATTENTION_ROW_SUM_UPDATE_LIMIT 256.0f
#endif

#ifndef ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS
#define ATTENTION_ROW_SUM_PREFIX_UPDATE_CHECKS 1
#endif

#ifndef ATTENTION_SPLIT_V_TMA
#define ATTENTION_SPLIT_V_TMA 1
#endif

#ifndef ATTENTION_OUTPUT_TMA_SWIZZLE_128B
#define ATTENTION_OUTPUT_TMA_SWIZZLE_128B 1
#endif

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
static constexpr int kPipeCount = 2;
static constexpr int kActivePipeStride = kPipeCount;
static constexpr int kConsumerWarpsPerPipe = 4;
static constexpr int kTraceConsumerLanesPerPipe = kConsumerWarpsPerPipe * 2;
static constexpr int kClockTraceSyncCount = 4;
static constexpr int kActiveConsumerPipeCount = kPipeCount;
static constexpr int kProducerWarpCount = 4;
static constexpr int kConsumerBaseWarp = 4;
static constexpr int kMainWarps = kProducerWarpCount + kPipeCount * kConsumerWarpsPerPipe;
static constexpr int kMainThreads = kMainWarps * kWarpSize;
static constexpr int kTileM = 128;
static constexpr int kTileN = 128;
static constexpr int kMmaK = 16;
static constexpr int kMmasPerTile = 8;
static constexpr int kTileBf16Elems = kTileM * kTileN;
static constexpr int kTileWords = kTileBf16Elems / 2;
static constexpr int kTileBytes = kTileWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kKBufferCount = 1;
static constexpr int kKBufferTileCount = kPipeCount * kKBufferCount;
static constexpr int kVBufferCount = kPipeCount;
static constexpr int kSBufferCount = kPipeCount;
static constexpr int kFixedBenchmarkRepeats = 256;
static constexpr int kFixedBenchmarkKTiles = 256;
#ifndef ATTENTION_EX2_EMU_FREQ
#define ATTENTION_EX2_EMU_FREQ 10
#endif
#ifndef ATTENTION_EX2_EMU_RES
#define ATTENTION_EX2_EMU_RES 1
#endif
static constexpr int kEx2EmuFreq = ATTENTION_EX2_EMU_FREQ;
static constexpr int kEx2EmuRes = ATTENTION_EX2_EMU_RES;
static constexpr int kDynamicSmemBytes =
    (1 + kKBufferTileCount + kVBufferCount + kSBufferCount) * kTileBytes + 1024;
static constexpr int kTmemAllocCols = 512;
static constexpr int kTmemUsedCols = 512;
static constexpr int kPTmemCols = 256;
static constexpr int kOTmemCols = 256;
static constexpr float kLog2E = 1.44269504088896340736f;
static constexpr double kFlopsPerMma =
    2.0 * static_cast<double>(kTileM) * static_cast<double>(kTileN) *
    static_cast<double>(kMmaK);


template <int kFixedKTiles>
__device__ __forceinline__ int kv_tile_base_for_block(int block_idx,
                                                      int loop_k_tiles) {
  if constexpr (kFixedKTiles == kFixedBenchmarkKTiles) {
    return block_idx & ~(kFixedBenchmarkKTiles - 1);
  } else {
    return (block_idx / loop_k_tiles) * loop_k_tiles;
  }
}

template <int kFixedKTiles>
__device__ __forceinline__ int local_k_tile_for_iter(int iter,
                                                     int loop_k_tiles) {
  if constexpr (kFixedKTiles == kFixedBenchmarkKTiles) {
    return iter;
  } else {
    return iter % loop_k_tiles;
  }
}

struct Args {
  int blocks = 4096;
  int repeats = 256;
  int k_tiles = 256;
  bool k_tiles_set = false;
  int warmup = 3;
  int iters = 10;
  bool clock_trace = false;
  int clock_trace_start = 0;
  int clock_trace_iters = 8;
  std::string stage = "benchmark";
  std::string pattern = "constant";
  const char* csv = "0.attention/attention_compare_default_b1_h16_s32k.csv";

  // Real attention path.  This is the correctness-oriented implementation:
  // contiguous BF16 Q/K/V/O with shape [B, H, S, D], D fixed to 128.
  bool validation_suite = false;
  int checksum_repeats = 2;
  int B = 1;
  int Hq = 1;
  int Hkv = 1;
  int Sq = 128;
  int Skv = 128;
  bool skv_set = false;
  int D = 128;
  bool causal = false;
  float softmax_scale = -1.0f;  // < 0 means 1 / sqrt(D).
};

struct RunResult {
  float ms = 0.0f;
  cudaError_t error = cudaSuccess;
  const char* status = "ok";
};

struct ClockTraceRecord {
  int stage = 0;
  int iter = 0;
  int pipe = 0;
  int warp_id = 0;
  int consumer_warp = 0;
  int half = 0;
  unsigned long long start = 0;
  unsigned long long end = 0;
};

struct TraceRecord {
  unsigned long long tma_start = 0;
  unsigned long long tma_issue_end = 0;
  unsigned long long tma_end = 0;
  unsigned long long mma_start = 0xffffffffffffffffull;
  unsigned long long mma_issue_end = 0;
  unsigned long long mma_end = 0xffffffffffffffffull;
  unsigned long long ld_start = 0xffffffffffffffffull;
  unsigned long long ld_end = 0;
  unsigned long long ld_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long ld_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long pack_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_detail_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long pack_detail_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long st_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long st_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long sum_warp_start[kTraceConsumerLanesPerPipe];
  unsigned long long sum_warp_end[kTraceConsumerLanesPerPipe];
  unsigned long long pack_start = 0xffffffffffffffffull;
  unsigned long long pack_end = 0;
  unsigned long long st_start = 0xffffffffffffffffull;
  unsigned long long st_end = 0;
  unsigned long long v_tma_start = 0;
  unsigned long long v_tma_issue_end = 0;
  unsigned long long v_tma_end = 0;
  unsigned long long v_tma_h0_start = 0;
  unsigned long long v_tma_h0_issue_end = 0;
  unsigned long long v_tma_h0_end = 0;
  unsigned long long v_tma_h1_start = 0;
  unsigned long long v_tma_h1_issue_end = 0;
  unsigned long long v_tma_h1_end = 0;
  unsigned long long pv_start = 0xffffffffffffffffull;
  unsigned long long pv_issue_end = 0;
  unsigned long long pv_end = 0;
  unsigned long long pv_h0_start = 0xffffffffffffffffull;
  unsigned long long pv_h0_end = 0;
  unsigned long long pv_h1_start = 0xffffffffffffffffull;
  unsigned long long pv_h1_end = 0;
  int tma_warp_id = -1;
  int pv_warp_id = -1;
  int pv_h0_warp_id = -1;
  int pv_h1_warp_id = -1;
  unsigned long long sync_start[kClockTraceSyncCount];
  unsigned long long sync_end[kClockTraceSyncCount];
  unsigned int iter = 0;
  unsigned int pipe = 0;
  unsigned int warp_id = 0;
};

enum ClockTraceStage {
  kClockTraceQTma = 1,
  kClockTraceKTma = 2,
  kClockTraceQkMma = 3,
  kClockTraceVTma = 4,
  kClockTracePvMma = 5,
  kClockTraceLd = 6,
  kClockTracePack = 7,
  kClockTraceRowSum = 8,
  kClockTraceTailWait = 9,
  kClockTraceTmemDrain = 10,
  kClockTracePackNorm = 11,
  kClockTraceGlobalStore = 12,
  kClockTraceTailTotal = 13,
  kClockTraceStore = 14,
  kClockTracePvMmaH0 = 15,
  kClockTracePvMmaH1 = 16,
  kClockTraceSync = 17,
  kClockTracePackDetail = 18,
  kClockTraceQkMmaIssue = 19,
  kClockTracePvMmaIssue = 20,
  kClockTraceKTmaIssue = 21,
  kClockTraceVTmaIssue = 22,
};

static constexpr int kClockTraceSlotsPerIter = 64;
static constexpr int kClockTraceLdBase = 8;
static constexpr int kClockTracePackStoreBase = 16;
static constexpr int kClockTraceRowSumBase = 24;
static constexpr int kClockTraceStoreBase = 32;
static constexpr int kClockTraceSyncBase = 40;
static constexpr int kClockTraceQkMmaIssueSlot = 44;
static constexpr int kClockTracePvMmaIssueSlot = 45;
static constexpr int kClockTraceKTmaIssueSlot = 46;
static constexpr int kClockTraceVTmaIssueSlot = 47;
static constexpr int kClockTracePackDetailBase = 52;
static constexpr int kClockTraceExtraSlots = 32;

#include "ptx_wrappers.cuh"
#include "attention.cu"

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

void encode_qk_contiguous_tma_map(CUtensorMap* map, void* base, uint64_t tiles) {
  const cuuint64_t global_dim[5] = {
      4,
      8,
      2,
      static_cast<cuuint64_t>(16) * tiles,
      8};
  const cuuint64_t global_stride[4] = {
      64ull * sizeof(uint32_t),
      4ull * sizeof(uint32_t),
      512ull * sizeof(uint32_t),
      8ull * sizeof(uint32_t)};
  const cuuint32_t box_dim[5] = {4, 8, 2, 16, 8};
  const cuuint32_t elem_stride[5] = {1, 1, 1, 1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_UINT32,
                                      5,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(qk_contiguous)");
}

void encode_qk_contiguous_k16_split_tma_map(CUtensorMap* map, void* base,
                                            uint64_t tiles) {
  const cuuint64_t global_dim[4] = {
      kTileN / 2,
      8,
      2,
      static_cast<cuuint64_t>(16) * tiles};
  const cuuint64_t global_stride[3] = {
      64ull * sizeof(uint32_t),
      4ull * sizeof(uint32_t),
      512ull * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {4, 8, 2, 16};
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
               "cuTensorMapEncodeTiled(qk_contiguous_k16_split)");
}

void encode_qk_contiguous_sw128_tma_map(CUtensorMap* map, void* base,
                                        uint64_t tiles) {
  const cuuint64_t global_dim[2] = {
      kTileN / 2,
      static_cast<cuuint64_t>(kTileM) * tiles};
  const cuuint64_t global_stride[1] = {
      static_cast<cuuint64_t>(kTileN / 2) * sizeof(uint32_t)};
  const cuuint32_t box_dim[2] = {kTileN / 4, kTileM};
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
               "cuTensorMapEncodeTiled(qk_contiguous_sw128)");
}

void encode_contiguous_sw128_k16_tma_map(CUtensorMap* map, void* base,
                                         uint64_t tiles) {
  const cuuint64_t global_dim[4] = {
      kTileN / 4,
      16,
      2,
      static_cast<cuuint64_t>(8) * tiles};
  const cuuint64_t global_stride[3] = {
      static_cast<cuuint64_t>(kTileN / 2) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kTileN / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(16 * kTileN / 2) * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kTileN / 4, 16, 2, 8};
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
               "cuTensorMapEncodeTiled(contiguous_sw128_k16)");
}

void encode_contiguous_sw128_k16_half_tma_map(CUtensorMap* map, void* base,
                                              uint64_t tiles) {
  const cuuint64_t global_dim[4] = {
      kTileN / 4,
      16,
      2,
      static_cast<cuuint64_t>(8) * tiles};
  const cuuint64_t global_stride[3] = {
      static_cast<cuuint64_t>(kTileN / 2) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kTileN / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(16 * kTileN / 2) * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kTileN / 4, 16, 2, 4};
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
               "cuTensorMapEncodeTiled(contiguous_sw128_k16_half)");
}

void encode_bf16_output_tma_map(CUtensorMap* map, void* base, uint64_t tiles) {
#if ATTENTION_OUTPUT_TMA_SWIZZLE_128B
  const cuuint64_t global_dim[4] = {kTileN / 2, kTileM, 2, tiles};
  const cuuint64_t global_stride[3] = {
      kTileN * sizeof(uint16_t),
      (kTileN / 2) * sizeof(uint16_t),
      static_cast<cuuint64_t>(kTileN) * kTileM * sizeof(uint16_t)};
  const cuuint32_t box_dim[4] = {kTileN / 2, kTileM, 2, 1};
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
               "cuTensorMapEncodeTiled(output_bf16_sw128_single)");
#else
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
#endif
}

double tbps_from_bytes(double bytes, double ms) {
  return ms > 0.0 ? bytes / (ms * 1.0e-3) / 1.0e12 : 0.0;
}

double tflops_from_flops(double flops, double ms) {
  return ms > 0.0 ? flops / (ms * 1.0e-3) / 1.0e12 : 0.0;
}

RunResult run_kernel(const Args& args,
                     const CUtensorMap& q_map,
                     const CUtensorMap& k_map,
                     const CUtensorMap& v_map,
                     const CUtensorMap& o_map,
                     float score_to_exp2_scale,
                     void* output,
                     int* active_ctas_per_sm
#if ATTENTION_CLOCK_TRACE
                     ,
                     ClockTraceRecord* clock_trace,
                     int clock_trace_iters
#endif
                     ) {
  RunResult result{};
  auto kernel = args.repeats == kFixedBenchmarkRepeats &&
                        args.k_tiles == kFixedBenchmarkKTiles
                    ? qk_tma_mma_ld_kernel<kFixedBenchmarkRepeats,
                                            kFixedBenchmarkKTiles>
                    : qk_tma_mma_ld_kernel<0, 0>;
  CUDA_CHECK(cudaFuncSetAttribute(kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_ctas_per_sm, kernel, kMainThreads, kDynamicSmemBytes));

  for (int i = 0; i < args.warmup; ++i) {
    kernel<<<args.blocks, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, args.repeats, args.k_tiles,
        score_to_exp2_scale, output
#if ATTENTION_CLOCK_TRACE
        ,
        clock_trace, clock_trace_iters, args.clock_trace_start
#endif
        );
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

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < args.iters; ++i) {
    kernel<<<args.blocks, kMainThreads, kDynamicSmemBytes>>>(
        q_map, k_map, v_map, o_map, args.repeats, args.k_tiles,
        score_to_exp2_scale, output
#if ATTENTION_CLOCK_TRACE
        ,
        clock_trace, clock_trace_iters, args.clock_trace_start
#endif
        );
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

void write_benchmark_csv(const Args& args, int active, const RunResult& result) {
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
               "status,cuda_error,notes\n");

  const double groups = static_cast<double>(args.blocks) * args.repeats;
  const double total_mmas = groups * kMmasPerTile * 2.0;
  const double q_tma_bytes = static_cast<double>(args.blocks) * kTileBytes;
  const double k_tma_bytes = groups * kTileBytes;
  const double v_tma_bytes = groups * kTileBytes;
  const double p_read_bytes = groups * 2.0 * kTileBytes;
  const double s_store_bytes = groups * kTileBytes;
  const double qk_flops = groups * kMmasPerTile * kFlopsPerMma;
  const double pv_flops = qk_flops;
  const int kv_tile_sets = (args.blocks + args.k_tiles - 1) / args.k_tiles;
  const int kv_total_tiles = kv_tile_sets * args.k_tiles;
  const char* mode =
#if ATTENTION_ROW_SUM_RARE_UPDATE
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_row_sum_rare_update";
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT && !ATTENTION_FIRST_ITER_COMPUTE_MAX
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_first_iter_branch_only";
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT && ATTENTION_FIRST_ITER_APPLY_SHIFT && ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_pipe_shift_epilogue_scale";
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT && ATTENTION_FIRST_ITER_APPLY_SHIFT
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_first_iter_shift_no_epilogue_scale";
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_first_iter_max_only";
#elif ATTENTION_ROW_MAX_ONLY
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_row_max_only";
#else
      "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output";
#endif
  char notes[256];
  std::snprintf(
      notes, sizeof(notes),
      "fixed_best_path_qk_contiguous_row_major_2d_sw128_tma_major_k_"
      "v_contiguous_k16_sw128_mn_major_dep_masked_"
      "no_s_ready_bf16_output"
#if ATTENTION_ROW_SUM_RARE_UPDATE
      "_row_sum_rare_update"
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT && !ATTENTION_FIRST_ITER_COMPUTE_MAX
      "_first_iter_branch_only"
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT && ATTENTION_FIRST_ITER_APPLY_SHIFT && ATTENTION_PIPE_SHIFT_EPILOGUE_SCALE
      "_pipe_shift_epilogue_scale"
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT && ATTENTION_FIRST_ITER_APPLY_SHIFT
      "_first_iter_shift_no_epilogue_scale"
#elif ATTENTION_FIRST_ITER_ROW_MAX_SHIFT
      "_first_iter_max_only"
#endif
#if ATTENTION_ROW_MAX_ONLY
      "_row_max_only"
#endif
      "_dep_default_mask_0x%03x"
      ,
      0x000
  );
  char output_shape[64];
  std::snprintf(output_shape, sizeof(output_shape), "O[%d,128,128]_bf16", args.blocks);

  std::fprintf(csv,
               "%s,Q[%d,128,128]_bf16,K[%d,128,128]_bf16,V[%d,128,128]_bf16,%s,%d,%d,%d,%d,%d,%d,%d,"
               "%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%.6f,%.0f,%.0f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
               "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%s,%s,%s\n",
               mode, args.blocks, kv_total_tiles, kv_total_tiles, output_shape, kTileM, kTileN, kMmaK,
               kMmasPerTile, args.blocks, args.repeats, args.k_tiles, args.warmup, args.iters,
               kMainThreads, active, kDynamicSmemBytes, kTmemAllocCols, kTmemUsedCols, kPTmemCols, kOTmemCols,
               "bf16_direct", result.ms, groups, total_mmas,
               q_tma_bytes / 1.0e9, k_tma_bytes / 1.0e9, v_tma_bytes / 1.0e9,
               (q_tma_bytes + k_tma_bytes + v_tma_bytes) / 1.0e9,
               p_read_bytes / 1.0e9, s_store_bytes / 1.0e9,
               tbps_from_bytes(q_tma_bytes, result.ms), tbps_from_bytes(k_tma_bytes, result.ms),
               tbps_from_bytes(v_tma_bytes, result.ms),
               tbps_from_bytes(q_tma_bytes + k_tma_bytes + v_tma_bytes, result.ms),
               tbps_from_bytes(p_read_bytes, result.ms), tbps_from_bytes(s_store_bytes, result.ms),
               tflops_from_flops(qk_flops, result.ms), tflops_from_flops(pv_flops, result.ms),
               tflops_from_flops(qk_flops + pv_flops, result.ms),
               result.status, cudaGetErrorString(result.error), notes);
  std::fclose(csv);
}

int clock_trace_record_count(const Args& args) {
  return args.clock_trace_iters * kClockTraceSlotsPerIter + kClockTraceExtraSlots;
}

void init_trace_record(TraceRecord* r, int iter) {
  *r = TraceRecord{};
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    r->ld_warp_start[w] = 0xffffffffffffffffull;
    r->ld_warp_end[w] = 0;
    r->pack_warp_start[w] = 0xffffffffffffffffull;
    r->pack_warp_end[w] = 0;
    r->pack_detail_warp_start[w] = 0xffffffffffffffffull;
    r->pack_detail_warp_end[w] = 0;
    r->st_warp_start[w] = 0xffffffffffffffffull;
    r->st_warp_end[w] = 0;
    r->sum_warp_start[w] = 0xffffffffffffffffull;
    r->sum_warp_end[w] = 0;
  }
  for (int i = 0; i < kClockTraceSyncCount; ++i) {
    r->sync_start[i] = 0xffffffffffffffffull;
    r->sync_end[i] = 0;
  }
  r->pack_start = 0xffffffffffffffffull;
  r->st_start = 0xffffffffffffffffull;
  r->pv_start = 0xffffffffffffffffull;
  r->mma_start = 0xffffffffffffffffull;
  r->ld_start = 0xffffffffffffffffull;
  r->iter = static_cast<unsigned int>(iter);
  r->pipe = static_cast<unsigned int>(iter & 1);
  r->warp_id = r->pipe;
}

const char* clock_trace_stage_name(int stage) {
  switch (stage) {
    case kClockTraceQTma:
      return "q_tma";
    case kClockTraceKTma:
      return "k_tma";
    case kClockTraceKTmaIssue:
      return "k_tma_issue";
    case kClockTraceQkMma:
      return "qk_mma";
    case kClockTraceQkMmaIssue:
      return "qk_mma_issue";
    case kClockTraceVTma:
      return "v_tma";
    case kClockTraceVTmaIssue:
      return "v_tma_issue";
    case kClockTracePvMma:
      return "pv_mma";
    case kClockTracePvMmaIssue:
      return "pv_mma_issue";
    case kClockTracePvMmaH0:
      return "pv_mma_h0";
    case kClockTracePvMmaH1:
      return "pv_mma_h1";
    case kClockTraceLd:
      return "ld_x64";
    case kClockTracePack:
      return "pack";
    case kClockTraceRowSum:
      return "row_sum";
    case kClockTraceTailWait:
      return "tail_wait_pv_done";
    case kClockTraceTmemDrain:
      return "tmem_drain";
    case kClockTracePackNorm:
      return "pack_norm";
    case kClockTraceGlobalStore:
      return "global_store";
    case kClockTraceTailTotal:
      return "tail_total";
    case kClockTraceStore:
      return "st";
    case kClockTraceSync:
      return "sync";
    case kClockTracePackDetail:
      return "pack_detail";
    default:
      return "unknown";
  }
}

void merge_trace_range(unsigned long long* start,
                       unsigned long long* end,
                       unsigned long long s,
                       unsigned long long e) {
  if (e <= s) return;
  if (*start == 0xffffffffffffffffull || s < *start) *start = s;
  if (e > *end) *end = e;
}

unsigned long long trace_cycles(unsigned long long start, unsigned long long end) {
  return end > start && start != 0xffffffffffffffffull &&
                 end != 0xffffffffffffffffull
             ? end - start
             : 0ull;
}

unsigned long long trace_start_or_zero(unsigned long long start) {
  return start == 0xffffffffffffffffull ? 0ull : start;
}

unsigned long long trace_end_or_zero(unsigned long long start, unsigned long long end) {
  return trace_cycles(start, end) > 0 ? end : 0ull;
}

std::string epilogue_trace_path(const char* csv_path) {
  std::string path(csv_path ? csv_path : "attention_trace.csv");
  const std::string suffix = ".csv";
  if (path.size() >= suffix.size() &&
      path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0) {
    path.insert(path.size() - suffix.size(), "_epilogue");
  } else {
    path += ".epilogue.csv";
  }
  return path;
}

void write_clock_trace_csv(const Args& args,
                           const RunResult& result,
                           const std::vector<ClockTraceRecord>& records) {
  std::vector<TraceRecord> rows(args.clock_trace_iters);
  for (int i = 0; i < args.clock_trace_iters; ++i) {
    init_trace_record(&rows[i], args.clock_trace_start + i);
  }
  unsigned long long tail_wait_start = 0xffffffffffffffffull;
  unsigned long long tail_wait_end = 0;
  unsigned long long tmem_drain_start[kConsumerWarpsPerPipe];
  unsigned long long tmem_drain_end[kConsumerWarpsPerPipe];
  unsigned long long pack_norm_start[kConsumerWarpsPerPipe];
  unsigned long long pack_norm_end[kConsumerWarpsPerPipe];
  for (int i = 0; i < kConsumerWarpsPerPipe; ++i) {
    tmem_drain_start[i] = 0xffffffffffffffffull;
    tmem_drain_end[i] = 0;
    pack_norm_start[i] = 0xffffffffffffffffull;
    pack_norm_end[i] = 0;
  }
  unsigned long long global_store_start = 0xffffffffffffffffull;
  unsigned long long global_store_end = 0;
  unsigned long long tail_total_start = 0xffffffffffffffffull;
  unsigned long long tail_total_end = 0;

  for (const ClockTraceRecord& r : records) {
    if (r.stage == 0 || r.end <= r.start) continue;
    if (r.iter == args.k_tiles) {
      switch (r.stage) {
        case kClockTraceTailWait:
          merge_trace_range(&tail_wait_start, &tail_wait_end, r.start, r.end);
          break;
        case kClockTraceTmemDrain:
          if (r.consumer_warp >= 0 && r.consumer_warp < kConsumerWarpsPerPipe) {
            merge_trace_range(&tmem_drain_start[r.consumer_warp],
                              &tmem_drain_end[r.consumer_warp], r.start, r.end);
          }
          break;
        case kClockTracePackNorm:
          if (r.consumer_warp >= 0 && r.consumer_warp < kConsumerWarpsPerPipe) {
            merge_trace_range(&pack_norm_start[r.consumer_warp],
                              &pack_norm_end[r.consumer_warp], r.start, r.end);
          }
          break;
        case kClockTraceGlobalStore:
          merge_trace_range(&global_store_start, &global_store_end, r.start, r.end);
          break;
        case kClockTraceTailTotal:
          merge_trace_range(&tail_total_start, &tail_total_end, r.start, r.end);
          break;
        default:
          break;
      }
      continue;
    }
    if (r.iter < args.clock_trace_start ||
        r.iter >= args.clock_trace_start + args.clock_trace_iters) {
      continue;
    }
    TraceRecord& out = rows[r.iter - args.clock_trace_start];
    switch (r.stage) {
      case kClockTraceKTma:
        out.pipe = static_cast<unsigned int>(r.pipe);
        out.tma_warp_id = r.warp_id;
        out.tma_start = r.start;
        out.tma_end = r.end;
        break;
      case kClockTraceKTmaIssue:
        out.pipe = static_cast<unsigned int>(r.pipe);
        out.tma_warp_id = r.warp_id;
        out.tma_start = r.start;
        out.tma_issue_end = r.end;
        break;
      case kClockTraceQkMma:
        out.pipe = static_cast<unsigned int>(r.pipe);
        out.warp_id = static_cast<unsigned int>(r.warp_id);
        out.mma_start = r.start;
        out.mma_end = r.end;
        break;
      case kClockTraceQkMmaIssue:
        out.pipe = static_cast<unsigned int>(r.pipe);
        out.warp_id = static_cast<unsigned int>(r.warp_id);
        out.mma_start = r.start;
        out.mma_issue_end = r.end;
        break;
      case kClockTraceVTma:
        out.pipe = static_cast<unsigned int>(r.pipe);
        if (r.half == 0) {
          out.v_tma_h0_start = r.start;
          out.v_tma_h0_end = r.end;
        } else if (r.half == 1) {
          out.v_tma_h1_start = r.start;
          out.v_tma_h1_end = r.end;
        } else {
          out.v_tma_start = r.start;
          out.v_tma_end = r.end;
        }
        break;
      case kClockTraceVTmaIssue:
        out.pipe = static_cast<unsigned int>(r.pipe);
        if (r.half == 0) {
          out.v_tma_h0_start = r.start;
          out.v_tma_h0_issue_end = r.end;
        } else if (r.half == 1) {
          out.v_tma_h1_start = r.start;
          out.v_tma_h1_issue_end = r.end;
        } else {
          out.v_tma_start = r.start;
          out.v_tma_issue_end = r.end;
        }
        break;
      case kClockTracePvMma:
        out.pv_start = r.start;
        out.pv_end = r.end;
        out.pv_warp_id = r.warp_id;
        break;
      case kClockTracePvMmaIssue:
        out.pv_start = r.start;
        out.pv_issue_end = r.end;
        out.pv_warp_id = r.warp_id;
        break;
      case kClockTracePvMmaH0:
        out.pv_h0_start = r.start;
        out.pv_h0_end = r.end;
        out.pv_h0_warp_id = r.warp_id;
        break;
      case kClockTracePvMmaH1:
        out.pv_h1_start = r.start;
        out.pv_h1_end = r.end;
        out.pv_h1_warp_id = r.warp_id;
        break;
      case kClockTraceSync:
        if (r.half >= 0 && r.half < kClockTraceSyncCount) {
          out.sync_start[r.half] = r.start;
          out.sync_end[r.half] = r.end;
        }
        break;
      case kClockTraceLd:
      case kClockTracePack:
      case kClockTracePackDetail:
      case kClockTraceStore:
      case kClockTraceRowSum: {
        const int lane = r.consumer_warp * 2 + r.half;
        if (lane < 0 || lane >= kTraceConsumerLanesPerPipe) break;
        if (r.stage == kClockTraceLd) {
          out.ld_warp_start[lane] = r.start;
          out.ld_warp_end[lane] = r.end;
          merge_trace_range(&out.ld_start, &out.ld_end, r.start, r.end);
        } else if (r.stage == kClockTracePack) {
          out.pack_warp_start[lane] = r.start;
          out.pack_warp_end[lane] = r.end;
          merge_trace_range(&out.pack_start, &out.pack_end, r.start, r.end);
        } else if (r.stage == kClockTracePackDetail) {
          out.pack_detail_warp_start[lane] = r.start;
          out.pack_detail_warp_end[lane] = r.end;
        } else if (r.stage == kClockTraceStore) {
          out.st_warp_start[lane] = r.start;
          out.st_warp_end[lane] = r.end;
          merge_trace_range(&out.st_start, &out.st_end, r.start, r.end);
        } else {
          out.sum_warp_start[lane] = r.start;
          out.sum_warp_end[lane] = r.end;
        }
        break;
      }
      default:
        break;
    }
  }

  FILE* csv = std::fopen(args.csv, "w");
  if (!csv) {
    std::perror(args.csv);
    std::exit(1);
  }
  std::fprintf(csv,
               "mode,elapsed_ms,iter,pipe,warp_id,tma_warp_id,tma_start,tma_end,tma_cycles,mma_start,mma_end,"
               "mma_cycles,tma_issue_end,tma_issue_cycles,tma_wait_cycles,"
               "mma_issue_end,mma_issue_cycles,mma_wait_cycles,"
               "ld_start,ld_end,ld_cycles,pack_start,pack_end,pack_cycles,"
               "st_start,st_end,st_cycles,v_tma_start,v_tma_end,v_tma_cycles,"
               "v_tma_issue_end,v_tma_issue_cycles,v_tma_wait_cycles,"
               "v_tma_h0_start,v_tma_h0_end,v_tma_h0_cycles,"
               "v_tma_h0_issue_end,v_tma_h0_issue_cycles,v_tma_h0_wait_cycles,"
               "v_tma_h1_start,v_tma_h1_end,v_tma_h1_cycles,"
               "v_tma_h1_issue_end,v_tma_h1_issue_cycles,v_tma_h1_wait_cycles,"
               "pv_start,pv_end,pv_cycles,pv_issue_end,pv_issue_cycles,pv_wait_cycles,"
               "pv_warp_id,pv_h0_start,pv_h0_end,"
               "pv_h0_cycles,pv_h0_warp_id,pv_h1_start,pv_h1_end,pv_h1_cycles,"
               "pv_h1_warp_id,total_start,total_end,total_cycles");
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",ld_warp%d_start,ld_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",pack_warp%d_start,pack_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",pack_detail_warp%d_start,pack_detail_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",st_warp%d_start,st_warp%d_end", w, w);
  }
  for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
    std::fprintf(csv, ",sum_warp%d_start,sum_warp%d_end", w, w);
  }
  for (int i = 0; i < kClockTraceSyncCount; ++i) {
    std::fprintf(csv, ",sync%d_start,sync%d_end,sync%d_cycles", i, i, i);
  }
  std::fprintf(csv, ",status,cuda_error,notes\n");

  const char* mode = "contiguous_qkv_sw128_2d_tma_pv_2pipe_bf16_output_trace";
  const char* notes =
      "fixed_best_path_qk_contiguous_row_major_2d_sw128_tma_"
      "v_contiguous_k16_sw128_mn_major_no_cross_pipe_qk_dep_"
      "no_s_ready_trace_schema"
      ;
  for (const TraceRecord& r : rows) {
    const unsigned long long ld_start = trace_start_or_zero(r.ld_start);
    const unsigned long long pack_start = trace_start_or_zero(r.pack_start);
    const unsigned long long st_start = trace_start_or_zero(r.st_start);
    const unsigned long long mma_start = trace_start_or_zero(r.mma_start);
    const unsigned long long pv_start = trace_start_or_zero(r.pv_start);
    const unsigned long long pv_h0_start = trace_start_or_zero(r.pv_h0_start);
    const unsigned long long pv_h1_start = trace_start_or_zero(r.pv_h1_start);
    const unsigned long long total_start = r.tma_start;
    const unsigned long long total_end =
        std::max(std::max(r.ld_end, r.st_end),
                 std::max(trace_end_or_zero(r.pv_start, r.pv_end),
                          trace_end_or_zero(r.mma_start, r.mma_end)));
    const unsigned long long mma_issue_end =
        trace_cycles(r.mma_start, r.mma_issue_end) > 0 ? r.mma_issue_end : 0ull;
    const unsigned long long pv_issue_end =
        trace_cycles(r.pv_start, r.pv_issue_end) > 0 ? r.pv_issue_end : 0ull;
    const unsigned long long tma_issue_end =
        trace_cycles(r.tma_start, r.tma_issue_end) > 0 ? r.tma_issue_end : 0ull;
    const unsigned long long v_tma_issue_end =
        trace_cycles(r.v_tma_start, r.v_tma_issue_end) > 0 ? r.v_tma_issue_end : 0ull;
    const unsigned long long v_tma_h0_issue_end =
        trace_cycles(r.v_tma_h0_start, r.v_tma_h0_issue_end) > 0 ? r.v_tma_h0_issue_end : 0ull;
    const unsigned long long v_tma_h1_issue_end =
        trace_cycles(r.v_tma_h1_start, r.v_tma_h1_issue_end) > 0 ? r.v_tma_h1_issue_end : 0ull;
    std::fprintf(csv,
                 "%s,%.6f,%u,%u,%u,%d,%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%llu,"
                 "%llu,%llu,%llu,"
                 "%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%llu,"
                 "%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%llu,%llu,%llu,%llu,"
                 "%llu,%llu,%llu,%llu,%llu,%llu,%d,%llu,%llu,%llu,%d,%llu,%llu,%llu,%d,%llu,%llu,%llu",
                 mode, result.ms, r.iter, r.pipe, r.warp_id, r.tma_warp_id,
                 r.tma_start, r.tma_end, trace_cycles(r.tma_start, r.tma_end),
                 mma_start,
                 trace_end_or_zero(r.mma_start, r.mma_end),
                 trace_cycles(r.mma_start, r.mma_end), tma_issue_end,
                 trace_cycles(r.tma_start, r.tma_issue_end),
                 tma_issue_end ? trace_cycles(r.tma_issue_end, r.tma_end) : 0ull,
                 mma_issue_end,
                 trace_cycles(r.mma_start, r.mma_issue_end),
                 mma_issue_end ? trace_cycles(r.mma_issue_end, r.mma_end) : 0ull,
                 ld_start, r.ld_end,
                 trace_cycles(r.ld_start, r.ld_end), pack_start, r.pack_end,
                 trace_cycles(r.pack_start, r.pack_end), st_start, r.st_end,
                 trace_cycles(r.st_start, r.st_end), r.v_tma_start, r.v_tma_end,
                 trace_cycles(r.v_tma_start, r.v_tma_end), v_tma_issue_end,
                 trace_cycles(r.v_tma_start, r.v_tma_issue_end),
                 v_tma_issue_end ? trace_cycles(r.v_tma_issue_end, r.v_tma_end) : 0ull,
                 r.v_tma_h0_start, r.v_tma_h0_end,
                 trace_cycles(r.v_tma_h0_start, r.v_tma_h0_end),
                 v_tma_h0_issue_end,
                 trace_cycles(r.v_tma_h0_start, r.v_tma_h0_issue_end),
                 v_tma_h0_issue_end ? trace_cycles(r.v_tma_h0_issue_end, r.v_tma_h0_end) : 0ull,
                 r.v_tma_h1_start, r.v_tma_h1_end,
                 trace_cycles(r.v_tma_h1_start, r.v_tma_h1_end),
                 v_tma_h1_issue_end,
                 trace_cycles(r.v_tma_h1_start, r.v_tma_h1_issue_end),
                 v_tma_h1_issue_end ? trace_cycles(r.v_tma_h1_issue_end, r.v_tma_h1_end) : 0ull,
                 pv_start, r.pv_end,
                 trace_cycles(r.pv_start, r.pv_end), pv_issue_end,
                 trace_cycles(r.pv_start, r.pv_issue_end),
                 pv_issue_end ? trace_cycles(r.pv_issue_end, r.pv_end) : 0ull,
                 r.pv_warp_id, pv_h0_start,
                 r.pv_h0_end, trace_cycles(r.pv_h0_start, r.pv_h0_end),
                 r.pv_h0_warp_id, pv_h1_start, r.pv_h1_end,
                 trace_cycles(r.pv_h1_start, r.pv_h1_end), r.pv_h1_warp_id,
                 total_start, total_end,
                 trace_cycles(total_start, total_end));
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.ld_warp_start[w]),
                   r.ld_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.pack_warp_start[w]),
                   r.pack_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu",
                   trace_start_or_zero(r.pack_detail_warp_start[w]),
                   r.pack_detail_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.st_warp_start[w]),
                   r.st_warp_end[w]);
    }
    for (int w = 0; w < kTraceConsumerLanesPerPipe; ++w) {
      std::fprintf(csv, ",%llu,%llu", trace_start_or_zero(r.sum_warp_start[w]),
                   r.sum_warp_end[w]);
    }
    for (int i = 0; i < kClockTraceSyncCount; ++i) {
      std::fprintf(csv, ",%llu,%llu,%llu", trace_start_or_zero(r.sync_start[i]),
                   trace_end_or_zero(r.sync_start[i], r.sync_end[i]),
                   trace_cycles(r.sync_start[i], r.sync_end[i]));
    }
    std::fprintf(csv, ",%s,%s,%s\n", result.status,
                 cudaGetErrorString(result.error), notes);
  }
  std::fclose(csv);

  const std::string epi_path = epilogue_trace_path(args.csv);
  FILE* epi = std::fopen(epi_path.c_str(), "w");
  if (!epi) {
    std::perror(epi_path.c_str());
    std::exit(1);
  }
  std::fprintf(epi, "stage,consumer_warp,start,end,cycles\n");
  std::fprintf(epi, "tail_wait_pv_done,-1,%llu,%llu,%llu\n",
               trace_start_or_zero(tail_wait_start),
               trace_end_or_zero(tail_wait_start, tail_wait_end),
               trace_cycles(tail_wait_start, tail_wait_end));
  for (int w = 0; w < kConsumerWarpsPerPipe; ++w) {
    std::fprintf(epi, "tmem_drain,%d,%llu,%llu,%llu\n", w,
                 trace_start_or_zero(tmem_drain_start[w]),
                 trace_end_or_zero(tmem_drain_start[w], tmem_drain_end[w]),
                 trace_cycles(tmem_drain_start[w], tmem_drain_end[w]));
  }
  for (int w = 0; w < kConsumerWarpsPerPipe; ++w) {
    std::fprintf(epi, "pack_norm,%d,%llu,%llu,%llu\n", w,
                 trace_start_or_zero(pack_norm_start[w]),
                 trace_end_or_zero(pack_norm_start[w], pack_norm_end[w]),
                 trace_cycles(pack_norm_start[w], pack_norm_end[w]));
  }
  std::fprintf(epi, "global_store,-1,%llu,%llu,%llu\n",
               trace_start_or_zero(global_store_start),
               trace_end_or_zero(global_store_start, global_store_end),
               trace_cycles(global_store_start, global_store_end));
  std::fprintf(epi, "tail_total,-1,%llu,%llu,%llu\n",
               trace_start_or_zero(tail_total_start),
               trace_end_or_zero(tail_total_start, tail_total_end),
               trace_cycles(tail_total_start, tail_total_end));
  std::fclose(epi);
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

uint32_t pack_bf16_bits_pair(uint16_t lo, uint16_t hi) {
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

int logical_word_offset(int row, int col_pair) {
  return atom_major_k_word_offset(row, col_pair);
}

void pack_real_row_major_words(const std::vector<uint16_t>& src,
                               int rows,
                               std::vector<uint32_t>* words) {
  words->assign(static_cast<size_t>(rows) * (kTileN / 2), 0);
  for (int row = 0; row < rows; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const int col = col_pair * 2;
      (*words)[static_cast<size_t>(row) * (kTileN / 2) + col_pair] =
          pack_bf16_bits_pair(src[static_cast<size_t>(row) * kTileN + col],
                              src[static_cast<size_t>(row) * kTileN + col + 1]);
    }
  }
}

void pack_real_q_internal(const std::vector<uint16_t>& q,
                          std::vector<uint32_t>* q_words) {
  q_words->assign(kTileWords, 0);
  for (int row = 0; row < kTileM; ++row) {
    for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
      const int col = col_pair * 2;
      (*q_words)[logical_word_offset(row, col_pair)] =
          pack_bf16_bits_pair(q[row * kTileN + col], q[row * kTileN + col + 1]);
    }
  }
}

void pack_real_k_internal(const std::vector<uint16_t>& k,
                          int k_tiles,
                          std::vector<uint32_t>* k_words) {
  k_words->assign(static_cast<size_t>(k_tiles) * kTileWords, 0);
  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int row = 0; row < kTileM; ++row) {
      const int k_idx = tile * kTileM + row;
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int col = col_pair * 2;
        (*k_words)[static_cast<size_t>(tile) * kTileWords +
                   logical_word_offset(row, col_pair)] =
            pack_bf16_bits_pair(k[k_idx * kTileN + col],
                                k[k_idx * kTileN + col + 1]);
      }
    }
  }
}

void pack_real_v_internal(const std::vector<uint16_t>& v,
                          int k_tiles,
                          std::vector<uint32_t>* v_words) {
  v_words->assign(static_cast<size_t>(k_tiles) * kTileWords, 0);
  for (int tile = 0; tile < k_tiles; ++tile) {
    for (int d = 0; d < kTileN; ++d) {
      for (int col_pair = 0; col_pair < kTileN / 2; ++col_pair) {
        const int n = col_pair * 2;
        const int k0 = tile * kTileN + n;
        const int k1 = k0 + 1;
        (*v_words)[static_cast<size_t>(tile) * kTileWords +
                   logical_word_offset(d, col_pair)] =
            pack_bf16_bits_pair(v[k0 * kTileN + d], v[k1 * kTileN + d]);
      }
    }
  }
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

uint32_t mix_hash_u32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

float centered_hash_value(uint32_t x) {
  const uint32_t h = mix_hash_u32(x);
  const float u = static_cast<float>(h & 0xffffu) * (1.0f / 65535.0f);
  return u * 2.0f - 1.0f;
}

float real_pattern_value(const std::string& pattern,
                         char matrix,
                         int b,
                         int h,
                         int s,
                         int d) {
  if (pattern == "constant") {
    return matrix == 'v' ? 0.125f : 0.0625f;
  }
  if (pattern == "onehot") {
    if (matrix == 'q') return d == (s & 127) ? 1.0f : 0.0f;
    if (matrix == 'k') return d == (s & 127) ? 1.0f : 0.0f;
    return d == ((s + h) & 127) ? 0.5f : 0.0f;
  }
  if (pattern == "rank1") {
    const float s_scale = static_cast<float>((s % 17) + 1) * 0.0078125f;
    const float d_scale = static_cast<float>((d % 19) + 1) * 0.00390625f;
    const float h_scale = static_cast<float>((h % 7) + 1) * 0.0009765625f;
    if (matrix == 'v') return 0.5f * s_scale + d_scale + h_scale;
    return s_scale + d_scale + h_scale;
  }
  const uint32_t tag = matrix == 'q' ? 0x1234u : (matrix == 'k' ? 0x5678u : 0x9abcu);
  const uint32_t x = tag ^ static_cast<uint32_t>(b * 131071 + h * 8191 + s * 257 + d);
  const float amp = matrix == 'v' ? 0.5f : 0.25f;
  return amp * centered_hash_value(x);
}

void fill_real_bf16_matrix(std::vector<uint16_t>& dst,
                           const std::string& pattern,
                           char matrix,
                           int B,
                           int H,
                           int S,
                           int D) {
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < H; ++h) {
      for (int s_idx = 0; s_idx < S; ++s_idx) {
        for (int d = 0; d < D; ++d) {
          const size_t idx = ((static_cast<size_t>(b) * H + h) * S + s_idx) * D + d;
          dst[idx] = float_to_bf16_bits(real_pattern_value(pattern, matrix, b, h, s_idx, d));
        }
      }
    }
  }
}

std::vector<float> unpack_bf16_vector_to_float(const std::vector<uint16_t>& src) {
  std::vector<float> dst(src.size(), 0.0f);
  for (size_t i = 0; i < src.size(); ++i) dst[i] = bf16_to_float(src[i]);
  return dst;
}

void build_real_attention_reference(const std::vector<uint16_t>& q,
                                    const std::vector<uint16_t>& k,
                                    const std::vector<uint16_t>& v,
                                    int B,
                                    int Hq,
                                    int Hkv,
                                    int Sq,
                                    int Skv,
                                    int D,
                                    float softmax_scale,
                                    bool causal,
                                    std::vector<float>* ref) {
  ref->assign(static_cast<size_t>(B) * Hq * Sq * D, 0.0f);
  std::vector<float> scores(static_cast<size_t>(Skv), 0.0f);
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < Hq; ++hq) {
      const int hkv = real_attention_hkv_for_hq(hq, Hq, Hkv);
      for (int q_idx = 0; q_idx < Sq; ++q_idx) {
        float row_max = -std::numeric_limits<float>::infinity();
        const size_t q_base = ((static_cast<size_t>(b) * Hq + hq) * Sq + q_idx) * D;
        const size_t kv_base = (static_cast<size_t>(b) * Hkv + hkv) * Skv * D;
        for (int k_idx = 0; k_idx < Skv; ++k_idx) {
          if (!real_attention_key_is_valid(q_idx, k_idx, Sq, Skv, causal ? 1 : 0)) {
            scores[k_idx] = -std::numeric_limits<float>::infinity();
            continue;
          }
          float dot = 0.0f;
          const size_t k_base = kv_base + static_cast<size_t>(k_idx) * D;
          for (int d = 0; d < D; ++d) {
            dot += bf16_to_float(q[q_base + d]) * bf16_to_float(k[k_base + d]);
          }
          const float score = dot * softmax_scale;
          scores[k_idx] = score;
          row_max = std::max(row_max, score);
        }
        if (!std::isfinite(row_max)) continue;
        float denom = 0.0f;
        for (int k_idx = 0; k_idx < Skv; ++k_idx) {
          if (!std::isfinite(scores[k_idx])) continue;
          denom += std::exp(scores[k_idx] - row_max);
        }
        const size_t o_base = ((static_cast<size_t>(b) * Hq + hq) * Sq + q_idx) * D;
        for (int d = 0; d < D; ++d) {
          float acc = 0.0f;
          for (int k_idx = 0; k_idx < Skv; ++k_idx) {
            if (!std::isfinite(scores[k_idx])) continue;
            const float w = std::exp(scores[k_idx] - row_max);
            const size_t v_idx = kv_base + static_cast<size_t>(k_idx) * D + d;
            acc += w * bf16_to_float(v[v_idx]);
          }
          (*ref)[o_base + d] = denom > 0.0f ? acc / denom : 0.0f;
        }
      }
    }
  }
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
  std::string stage;
  bool ok;
  float max_abs;
  float max_rel;
  size_t bad_count;
  uint64_t checksum;
  uint64_t repeat_checksum;
  bool checksum_stable;
};

uint64_t validation_checksum_bytes(const void* data, size_t bytes) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= static_cast<uint64_t>(p[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

template <typename T>
uint64_t validation_checksum_vector(const std::vector<T>& values) {
  return values.empty()
             ? validation_checksum_bytes(nullptr, 0)
             : validation_checksum_bytes(values.data(), values.size() * sizeof(T));
}

void set_validation_checksum(CompareResult* r, uint64_t checksum) {
  r->checksum = checksum;
  r->repeat_checksum = checksum;
  r->checksum_stable = true;
}

template <typename Fn>
CompareResult run_validation_repeated(Fn run_once, int repeats) {
  const int count = std::max(1, repeats);
  CompareResult out = run_once();
  for (int i = 1; i < count; ++i) {
    CompareResult next = run_once();
    out.ok = out.ok && next.ok;
    out.max_abs = std::max(out.max_abs, next.max_abs);
    out.max_rel = std::max(out.max_rel, next.max_rel);
    out.bad_count += next.bad_count;
    out.repeat_checksum = next.checksum;
    out.checksum_stable = out.checksum_stable && next.checksum_stable &&
                          out.checksum == next.checksum;
  }
  out.ok = out.ok && out.checksum_stable;
  return out;
}

CompareResult compare_float(const char* stage,
                            const std::vector<float>& got,
                            const std::vector<float>& expected,
                            float abs_tol,
                            float rel_tol) {
  CompareResult r{std::string(stage), true, 0.0f, 0.0f, 0, 0, 0, true};
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
  CompareResult r{std::string(stage), true, 0.0f, 0.0f, 0, 0, 0, true};
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expected[i]) {
      r.ok = false;
      ++r.bad_count;
    }
  }
  set_validation_checksum(&r, validation_checksum_vector(got));
  return r;
}

void write_validation_csv(const char* path, const std::vector<CompareResult>& results) {
  FILE* csv = std::fopen(path, "w");
  if (!csv) {
    std::perror(path);
    std::exit(1);
  }
  std::fprintf(csv,
               "stage,status,max_abs,max_rel,bad_count,checksum,repeat_checksum,checksum_stable\n");
  for (const CompareResult& r : results) {
    std::fprintf(csv, "%s,%s,%g,%g,%zu,%016llx,%016llx,%s\n",
                 r.stage.c_str(), r.ok ? "ok" : "fail", r.max_abs, r.max_rel,
                 r.bad_count, static_cast<unsigned long long>(r.checksum),
                 static_cast<unsigned long long>(r.repeat_checksum),
                 r.checksum_stable ? "yes" : "no");
  }
  std::fclose(csv);
}

void parse_args(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--blocks") && i + 1 < argc) {
      args->blocks = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--k-tiles") && i + 1 < argc) {
      args->k_tiles = std::atoi(argv[++i]);
      args->k_tiles_set = true;
    } else if (!std::strcmp(argv[i], "--warmup") && i + 1 < argc) {
      args->warmup = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) {
      args->iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--clock-trace")) {
      args->clock_trace = true;
    } else if (!std::strcmp(argv[i], "--clock-trace-start") && i + 1 < argc) {
      args->clock_trace_start = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--clock-trace-iters") && i + 1 < argc) {
      args->clock_trace_iters = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--validate") ||
               !std::strcmp(argv[i], "--real-validate") ||
               !std::strcmp(argv[i], "--fused-validate")) {
      args->stage = "fused_validate";
    } else if (!std::strcmp(argv[i], "--scalar-validate")) {
      args->stage = "scalar_validate";
    } else if (!std::strcmp(argv[i], "--scalar-validate-suite")) {
      args->stage = "scalar_validate";
      args->validation_suite = true;
    } else if (!std::strcmp(argv[i], "--validate-suite")) {
      args->stage = "fused_validate";
      args->validation_suite = true;
    } else if (!std::strcmp(argv[i], "--checksum-repeats") && i + 1 < argc) {
      args->checksum_repeats = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--pattern") && i + 1 < argc) {
      args->pattern = argv[++i];
    } else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) {
      args->csv = argv[++i];
    } else if ((!std::strcmp(argv[i], "--b") || !std::strcmp(argv[i], "--B")) && i + 1 < argc) {
      args->B = std::atoi(argv[++i]);
    } else if ((!std::strcmp(argv[i], "--h") || !std::strcmp(argv[i], "--H")) && i + 1 < argc) {
      args->Hq = std::atoi(argv[++i]);
      args->Hkv = args->Hq;
    } else if (!std::strcmp(argv[i], "--hq") && i + 1 < argc) {
      args->Hq = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--hkv") && i + 1 < argc) {
      args->Hkv = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--sq") && i + 1 < argc) {
      args->Sq = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--skv") && i + 1 < argc) {
      args->Skv = std::atoi(argv[++i]);
      args->skv_set = true;
    } else if ((!std::strcmp(argv[i], "--d") || !std::strcmp(argv[i], "--D")) && i + 1 < argc) {
      args->D = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--causal")) {
      args->causal = true;
    } else if (!std::strcmp(argv[i], "--no-causal")) {
      args->causal = false;
    } else if (!std::strcmp(argv[i], "--scale") && i + 1 < argc) {
      args->softmax_scale = std::atof(argv[++i]);
    } else if (!std::strcmp(argv[i], "--help")) {
      std::printf(
          "Usage: %s [benchmark args] [validation args]\n"
          "\n"
          "Benchmark path:\n"
          "  --blocks N --k-tiles N --warmup N --iters N\n"
          "  fixed path: contiguous Q/K 2D SW128 TMA, contiguous V k16 SW128 MN-major,\n"
          "              split S-ready, QK/PV pingpong dep, BF16 output store\n"
          "  --clock-trace --clock-trace-start N --clock-trace-iters N   write output-path clock trace CSV; requires -DATTENTION_CLOCK_TRACE=1\n"
          "\n"
          "Fused real-attention validation path:\n"
          "  --validate                     validate qk_tma_mma_ld_kernel for B=H=1,Sq=128,D=128\n"
          "  --validate-suite               run fused validation cases for k_tiles 1/4/8\n"
          "  --scalar-validate              run the separate scalar correctness kernel\n"
          "  --b N --h N | --hq N --hkv N   batch/head counts; Hq must be divisible by Hkv\n"
          "  --sq N --skv N --d 128         sequence sizes and head dimension\n"
          "  --causal                       bottom-right aligned causal mask\n"
          "  --scale F                      softmax scale; default is 1/sqrt(D)\n"
          "  --pattern constant|rank1|onehot|random\n"
          "  --checksum-repeats N           rerun each validation case and require matching output checksum; default 2\n"
          "  --csv PATH\n",
          argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (args->blocks < 1) args->blocks = 1;
  if (args->repeats < 1) args->repeats = 1;
  if (args->k_tiles < 1) args->k_tiles = 1;
  if (args->B < 1) args->B = 1;
  if (args->Hq < 1) args->Hq = 1;
  if (args->Hkv < 1) args->Hkv = 1;
  if (args->Sq < 1) args->Sq = 1;
  if (args->Skv < 1) args->Skv = 1;
  if (args->D < 1) args->D = 1;
  if (args->checksum_repeats < 1) args->checksum_repeats = 1;
  if (args->clock_trace_start < 0) args->clock_trace_start = 0;
  if (args->clock_trace_iters < 1) args->clock_trace_iters = 1;
  if (args->clock_trace_start >= args->k_tiles) args->clock_trace_start = args->k_tiles - 1;
  if (args->clock_trace_start + args->clock_trace_iters > args->k_tiles) {
    args->clock_trace_iters = args->k_tiles - args->clock_trace_start;
  }
}

bool validate_real_attention_args(const Args& args) {
  if (args.D != kRealAttentionD) {
    std::fprintf(stderr, "real attention path currently requires D=128; got D=%d\n", args.D);
    return false;
  }
  if (args.Hq < 1 || args.Hkv < 1 || args.Hq % args.Hkv != 0) {
    std::fprintf(stderr, "real attention path requires Hq %% Hkv == 0; got Hq=%d Hkv=%d\n",
                 args.Hq, args.Hkv);
    return false;
  }
  if (args.B < 1 || args.Sq < 1 || args.Skv < 1) {
    std::fprintf(stderr, "invalid shape: B=%d Sq=%d Skv=%d\n", args.B, args.Sq, args.Skv);
    return false;
  }
  return true;
}

float resolved_softmax_scale(const Args& args) {
  return args.softmax_scale >= 0.0f ? args.softmax_scale
                                    : 1.0f / std::sqrt(static_cast<float>(args.D));
}

float resolved_score_to_exp2_scale(const Args& args) {
  return resolved_softmax_scale(args) * kLog2E;
}

CompareResult run_real_attention_case(const Args& args, const std::string& label) {
  const float scale = resolved_softmax_scale(args);
  const size_t q_elems = static_cast<size_t>(args.B) * args.Hq * args.Sq * args.D;
  const size_t kv_elems = static_cast<size_t>(args.B) * args.Hkv * args.Skv * args.D;
  const size_t o_elems = q_elems;

  std::vector<uint16_t> h_q(q_elems);
  std::vector<uint16_t> h_k(kv_elems);
  std::vector<uint16_t> h_v(kv_elems);
  fill_real_bf16_matrix(h_q, args.pattern, 'q', args.B, args.Hq, args.Sq, args.D);
  fill_real_bf16_matrix(h_k, args.pattern, 'k', args.B, args.Hkv, args.Skv, args.D);
  fill_real_bf16_matrix(h_v, args.pattern, 'v', args.B, args.Hkv, args.Skv, args.D);

  std::vector<float> ref;
  build_real_attention_reference(h_q, h_k, h_v, args.B, args.Hq, args.Hkv, args.Sq,
                                 args.Skv, args.D, scale, args.causal, &ref);

  uint16_t* d_q = nullptr;
  uint16_t* d_k = nullptr;
  uint16_t* d_v = nullptr;
  uint16_t* d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, q_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_k, kv_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_v, kv_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMalloc(&d_o, o_elems * sizeof(uint16_t)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_o, 0, o_elems * sizeof(uint16_t)));

  RealAttentionParams params{};
  params.q = d_q;
  params.k = d_k;
  params.v = d_v;
  params.o = d_o;
  params.B = args.B;
  params.Hq = args.Hq;
  params.Hkv = args.Hkv;
  params.Sq = args.Sq;
  params.Skv = args.Skv;
  params.D = args.D;
  params.causal = args.causal ? 1 : 0;
  params.softmax_scale = scale;

  const dim3 grid(args.Sq, args.B * args.Hq, 1);
  real_attention_bf16_d128_kernel<<<grid, kRealAttentionThreads>>>(params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint16_t> h_o(o_elems);
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, o_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  std::vector<float> got = unpack_bf16_vector_to_float(h_o);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  char stage[256];
  std::snprintf(stage, sizeof(stage),
                "%s_B%d_Hq%d_Hkv%d_Sq%d_Skv%d_D%d_%s_scale%.6g_pattern_%s",
                label.c_str(), args.B, args.Hq, args.Hkv, args.Sq, args.Skv, args.D,
                args.causal ? "causal" : "noncausal", scale, args.pattern.c_str());

  // BF16 output plus different CPU/GPU reduction orders makes bit-exact comparison
  // inappropriate.  These tolerances catch real algorithmic errors while allowing
  // expected BF16 quantization and exp/dot ordering differences.
  CompareResult result = compare_float(stage, got, ref, 6.0e-2f, 8.0e-2f);
  set_validation_checksum(&result, validation_checksum_vector(h_o));
  return result;
}

bool prepare_fused_real_attention_args(const Args& args, Args* out) {
  *out = args;
  if (out->Sq != kTileM || out->D != kRealAttentionD) {
    std::fprintf(stderr, "fused validation V1 requires Sq=128,D=128; got Sq=%d D=%d\n",
                 out->Sq, out->D);
    return false;
  }
  if (!out->k_tiles_set) {
    if (out->skv_set) {
      if (out->Skv % kTileN != 0) {
        std::fprintf(stderr, "fused validation requires Skv %% 128 == 0; got Skv=%d\n",
                     out->Skv);
        return false;
      }
      out->k_tiles = out->Skv / kTileN;
    } else {
      out->k_tiles = 1;
      out->Skv = kTileN;
    }
  } else {
    if (out->skv_set && out->Skv != out->k_tiles * kTileN) {
      std::fprintf(stderr,
                   "fused validation requires Skv == 128 * k_tiles; got Skv=%d k_tiles=%d\n",
                   out->Skv, out->k_tiles);
      return false;
    }
    out->Skv = out->k_tiles * kTileN;
  }
  if (out->B != 1 || out->Hq != 1 || out->Hkv != 1) {
    std::fprintf(stderr,
                 "fused validation V1 requires B=1,Hq=1,Hkv=1; got B=%d Hq=%d Hkv=%d\n",
                 out->B, out->Hq, out->Hkv);
    return false;
  }
  if (out->causal) {
    std::fprintf(stderr, "fused validation V1 is non-causal only\n");
    return false;
  }
  if (out->k_tiles < 1) {
    std::fprintf(stderr, "fused validation requires k_tiles >= 1\n");
    return false;
  }
  return true;
}

CompareResult run_fused_real_attention_case(const Args& args, const std::string& label) {
  const float scale = resolved_softmax_scale(args);
  const size_t q_elems = kTileBf16Elems;
  const size_t kv_elems = static_cast<size_t>(args.k_tiles) * kTileBf16Elems;

  std::vector<uint16_t> h_q(q_elems);
  std::vector<uint16_t> h_k(kv_elems);
  std::vector<uint16_t> h_v(kv_elems);
  fill_real_bf16_matrix(h_q, args.pattern, 'q', 1, 1, kTileM, kRealAttentionD);
  fill_real_bf16_matrix(h_k, args.pattern, 'k', 1, 1, args.Skv, kRealAttentionD);
  fill_real_bf16_matrix(h_v, args.pattern, 'v', 1, 1, args.Skv, kRealAttentionD);

  std::vector<float> ref;
  build_real_attention_reference(h_q, h_k, h_v, 1, 1, 1, kTileM, args.Skv,
                                 kRealAttentionD, scale, false, &ref);

  std::vector<uint32_t> h_q_contiguous;
  std::vector<uint32_t> h_k_contiguous;
  std::vector<uint32_t> h_v_words;
  pack_real_row_major_words(h_q, kTileM, &h_q_contiguous);
  pack_real_row_major_words(h_k, args.Skv, &h_k_contiguous);
  pack_real_row_major_words(h_v, args.Skv, &h_v_words);

  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, h_q_contiguous.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, h_k_contiguous.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, h_v_words.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_o, kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_q, h_q_contiguous.data(), h_q_contiguous.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, h_k_contiguous.data(), h_k_contiguous.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, h_v_words.data(), h_v_words.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_o, 0, kTileWords * sizeof(uint32_t)));

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  encode_qk_contiguous_sw128_tma_map(&q_map, d_q, 1);
  encode_qk_contiguous_sw128_tma_map(&k_map, d_k,
                                     static_cast<uint64_t>(args.k_tiles));
#if ATTENTION_SPLIT_V_TMA
  encode_contiguous_sw128_k16_half_tma_map(&v_map, d_v,
                                           static_cast<uint64_t>(args.k_tiles));
#else
  encode_contiguous_sw128_k16_tma_map(&v_map, d_v,
                                      static_cast<uint64_t>(args.k_tiles));
#endif
  encode_bf16_output_tma_map(&o_map, d_o, 1);

  CUDA_CHECK(cudaFuncSetAttribute(qk_tma_mma_ld_kernel<0, 0>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  kDynamicSmemBytes));
  qk_tma_mma_ld_kernel<0, 0><<<1, kMainThreads, kDynamicSmemBytes>>>(
      q_map, k_map, v_map, o_map, args.k_tiles, args.k_tiles,
      resolved_score_to_exp2_scale(args), d_o
#if ATTENTION_CLOCK_TRACE
      ,
      nullptr, 0, 0
#endif
      );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> h_o(kTileWords);
  CUDA_CHECK(cudaMemcpy(h_o.data(), d_o, kTileWords * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  const std::vector<float> got = unpack_row_major_bf16_words(h_o);

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  char stage[256];
  std::snprintf(stage, sizeof(stage),
                "%s_fused_B1_H1_Sq128_Skv%d_D128_noncausal_scale%.6g_pattern_%s",
                label.c_str(), args.Skv, scale, args.pattern.c_str());
  CompareResult result = compare_float(stage, got, ref, 6.0e-2f, 8.0e-2f);
  set_validation_checksum(&result, validation_checksum_vector(h_o));
  return result;
}

int run_fused_real_validation(const Args& args) {
  std::vector<CompareResult> results;
  if (args.validation_suite) {
    struct CaseSpec {
      int k_tiles;
      const char* pattern;
      const char* label;
    };
    const CaseSpec cases[] = {
        {1, "constant", "fused_real_attention_constant_k1"},
        {4, "rank1", "fused_real_attention_rank1_k4"},
        {4, "random", "fused_real_attention_random_k4"},
        {8, "random", "fused_real_attention_random_k8"},
    };
    for (const CaseSpec& c : cases) {
      Args case_args = args;
      case_args.B = 1;
      case_args.Hq = 1;
      case_args.Hkv = 1;
      case_args.Sq = kTileM;
      case_args.k_tiles = c.k_tiles;
      case_args.k_tiles_set = true;
      case_args.Skv = c.k_tiles * kTileN;
      case_args.skv_set = false;
      case_args.D = kRealAttentionD;
      case_args.causal = false;
      case_args.pattern = c.pattern;
      Args prepared;
      if (!prepare_fused_real_attention_args(case_args, &prepared)) return 2;
      results.push_back(run_validation_repeated(
          [&]() { return run_fused_real_attention_case(prepared, c.label); },
          args.checksum_repeats));
    }
  } else {
    Args prepared;
    if (!prepare_fused_real_attention_args(args, &prepared)) return 2;
    results.push_back(run_validation_repeated(
        [&]() { return run_fused_real_attention_case(prepared, "real_attention"); },
        args.checksum_repeats));
  }

  write_validation_csv(args.csv, results);
  for (const CompareResult& r : results) {
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu checksum=%016llx "
                "repeat_checksum=%016llx checksum_stable=%s\n",
                r.stage.c_str(), r.ok ? "ok" : "fail", r.max_abs, r.max_rel,
                r.bad_count, static_cast<unsigned long long>(r.checksum),
                static_cast<unsigned long long>(r.repeat_checksum),
                r.checksum_stable ? "yes" : "no");
  }
  const bool ok = std::all_of(results.begin(), results.end(),
                              [](const CompareResult& r) { return r.ok; });
  return ok ? 0 : 1;
}

int run_scalar_real_validation(const Args& args) {
  std::vector<CompareResult> results;
  if (args.validation_suite) {
    struct CaseSpec {
      int B, Hq, Hkv, Sq, Skv;
      bool causal;
      const char* pattern;
      const char* label;
    };
    const CaseSpec cases[] = {
        {1, 1, 1, 128, 128, false, "constant", "real_attention_constant_square"},
        {1, 1, 1, 128, 256, false, "rank1", "real_attention_rank1_multi_k"},
        {1, 1, 1, 128, 384, true, "rank1", "real_attention_causal_k_longer"},
        {1, 1, 1, 77, 193, false, "random", "real_attention_random_tail"},
        {2, 4, 4, 128, 256, false, "random", "real_attention_b2_h4"},
        {1, 8, 2, 64, 128, false, "random", "real_attention_gqa"},
    };
    for (const CaseSpec& c : cases) {
      Args case_args = args;
      case_args.B = c.B;
      case_args.Hq = c.Hq;
      case_args.Hkv = c.Hkv;
      case_args.Sq = c.Sq;
      case_args.Skv = c.Skv;
      case_args.D = kRealAttentionD;
      case_args.causal = c.causal;
      case_args.pattern = c.pattern;
      if (!validate_real_attention_args(case_args)) return 2;
      results.push_back(run_validation_repeated(
          [&]() { return run_real_attention_case(case_args, c.label); },
          args.checksum_repeats));
    }
  } else {
    if (!validate_real_attention_args(args)) return 2;
    results.push_back(run_validation_repeated(
        [&]() { return run_real_attention_case(args, "real_attention"); },
        args.checksum_repeats));
  }

  write_validation_csv(args.csv, results);
  for (const CompareResult& r : results) {
    std::printf("%s status=%s max_abs=%g max_rel=%g bad=%zu checksum=%016llx "
                "repeat_checksum=%016llx checksum_stable=%s\n",
                r.stage.c_str(), r.ok ? "ok" : "fail", r.max_abs, r.max_rel,
                r.bad_count, static_cast<unsigned long long>(r.checksum),
                static_cast<unsigned long long>(r.repeat_checksum),
                r.checksum_stable ? "yes" : "no");
  }
  const bool ok = std::all_of(results.begin(), results.end(),
                              [](const CompareResult& r) { return r.ok; });
  return ok ? 0 : 1;
}

int run_benchmark(const Args& args_in) {
  Args args = args_in;
#if !ATTENTION_CLOCK_TRACE
  if (args.clock_trace) {
    std::fprintf(stderr,
                 "--clock-trace requires compiling with -DATTENTION_CLOCK_TRACE=1.\n");
    return 2;
  }
#endif
  args.repeats = args.k_tiles;
  uint32_t* d_q = nullptr;
  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;
  uint32_t* d_o = nullptr;
  ClockTraceRecord* d_clock_trace = nullptr;
  const int kv_tile_sets = (args.blocks + args.k_tiles - 1) / args.k_tiles;
  const int kv_total_tiles = kv_tile_sets * args.k_tiles;
  const size_t q_words = static_cast<size_t>(args.blocks) * kTileWords;
  const size_t k_words = static_cast<size_t>(kv_total_tiles) * kTileWords;
  const size_t v_words = static_cast<size_t>(kv_total_tiles) * kTileWords;
  CUDA_CHECK(cudaMalloc(&d_q, q_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_k, k_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_v, v_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_o, static_cast<size_t>(args.blocks) * kTileWords * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_o, 0, static_cast<size_t>(args.blocks) * kTileWords * sizeof(uint32_t)));

  const int fill_threads = 256;
  fill_packed_bf16<<<static_cast<int>((q_words + fill_threads - 1) / fill_threads), fill_threads>>>(d_q, q_words, 3);
  CUDA_CHECK(cudaGetLastError());
  fill_packed_bf16<<<static_cast<int>((k_words + fill_threads - 1) / fill_threads), fill_threads>>>(d_k, k_words, 11);
  CUDA_CHECK(cudaGetLastError());
  fill_packed_bf16<<<static_cast<int>((v_words + fill_threads - 1) / fill_threads), fill_threads>>>(d_v, v_words, 17);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUtensorMap q_map{}, k_map{}, v_map{}, o_map{};
  encode_qk_contiguous_sw128_tma_map(&q_map, d_q,
                                     static_cast<uint64_t>(args.blocks));
  encode_qk_contiguous_sw128_tma_map(&k_map, d_k,
                                     static_cast<uint64_t>(kv_total_tiles));
#if ATTENTION_SPLIT_V_TMA
  encode_contiguous_sw128_k16_half_tma_map(&v_map, d_v,
                                           static_cast<uint64_t>(kv_total_tiles));
#else
  encode_contiguous_sw128_k16_tma_map(&v_map, d_v,
                                      static_cast<uint64_t>(kv_total_tiles));
#endif
  encode_bf16_output_tma_map(&o_map, d_o, static_cast<uint64_t>(args.blocks));

#if ATTENTION_CLOCK_TRACE
  const int clock_record_count = clock_trace_record_count(args);
  if (args.clock_trace) {
    CUDA_CHECK(cudaMalloc(&d_clock_trace,
                          static_cast<size_t>(clock_record_count) *
                              sizeof(ClockTraceRecord)));
    CUDA_CHECK(cudaMemset(d_clock_trace, 0,
                          static_cast<size_t>(clock_record_count) *
                              sizeof(ClockTraceRecord)));
  }
#endif

  int active = 0;
  const float score_to_exp2_scale = resolved_score_to_exp2_scale(args);
  RunResult result = run_kernel(args, q_map, k_map, v_map, o_map,
                                score_to_exp2_scale, d_o, &active
#if ATTENTION_CLOCK_TRACE
                                ,
                                d_clock_trace, args.clock_trace_iters
#endif
                                );
#if ATTENTION_CLOCK_TRACE
  if (args.clock_trace && result.error == cudaSuccess) {
    std::vector<ClockTraceRecord> h_clock_trace(clock_record_count);
    CUDA_CHECK(cudaMemcpy(h_clock_trace.data(), d_clock_trace,
                          static_cast<size_t>(clock_record_count) *
                              sizeof(ClockTraceRecord),
                          cudaMemcpyDeviceToHost));
    write_clock_trace_csv(args, result, h_clock_trace);
  } else
#endif
  {
    write_benchmark_csv(args, active, result);
  }

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  if (d_o) CUDA_CHECK(cudaFree(d_o));
  if (d_clock_trace) CUDA_CHECK(cudaFree(d_clock_trace));
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
  if (args.stage != "scalar_validate" && prop.major < 10) {
    std::fprintf(stderr, "The tcgen05 benchmark/toy path requires SM100+; got sm_%d%d\n",
                 prop.major, prop.minor);
    return 77;
  }
  driver_check(cuInit(0), "cuInit");
  if (args.stage == "fused_validate") return run_fused_real_validation(args);
  if (args.stage == "scalar_validate") return run_scalar_real_validation(args);
  return run_benchmark(args);
}
