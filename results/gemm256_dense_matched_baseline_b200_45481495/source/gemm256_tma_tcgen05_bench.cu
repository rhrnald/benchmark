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

#ifndef GEMM_REPEAT_INPUT
#define GEMM_REPEAT_INPUT 0
#endif

// Keep address selection independent from the pipeline/epilogue tuning.  This
// lets the dense-address kernel use the settings established by the
// same-address ceiling experiment.
#ifndef GEMM_REPEAT_TUNING
#define GEMM_REPEAT_TUNING GEMM_REPEAT_INPUT
#endif

#ifndef GEMM_PERSISTENT_CTA
#define GEMM_PERSISTENT_CTA 0
#endif

// Optional persistent/epilogue ablations.  Keep these independent so each
// fixed-cost optimization can be measured against the recovered baseline.
#ifndef GEMM_PERSISTENT_STATIC_SCHEDULER
#define GEMM_PERSISTENT_STATIC_SCHEDULER 0
#endif

#ifndef GEMM_FUSED_CSTORE_STAGING
#define GEMM_FUSED_CSTORE_STAGING 0
#endif

#ifndef GEMM_ELIDE_DENSE_SINK
#define GEMM_ELIDE_DENSE_SINK 0
#endif

#ifndef GEMM_DENSE_L2_TUNING
#define GEMM_DENSE_L2_TUNING 0
#endif

#ifndef GEMM_PERSISTENT_MACRO_M
#define GEMM_PERSISTENT_MACRO_M 16
#endif

#ifndef GEMM_PERSISTENT_MACRO_N
#define GEMM_PERSISTENT_MACRO_N 16
#endif

#ifndef GEMM_PERSISTENT_32K_MACRO_M
#define GEMM_PERSISTENT_32K_MACRO_M 8
#endif

#ifndef GEMM_PERSISTENT_32K_MACRO_N
#define GEMM_PERSISTENT_32K_MACRO_N 18
#endif

#ifndef GEMM_PERSISTENT_8K_MACRO_M
#define GEMM_PERSISTENT_8K_MACRO_M 16
#endif

#ifndef GEMM_PERSISTENT_8K_MACRO_N
#define GEMM_PERSISTENT_8K_MACRO_N 16
#endif

#ifndef GEMM_PERSISTENT_LOCAL_M_FAST
#define GEMM_PERSISTENT_LOCAL_M_FAST 1
#endif

#ifndef GEMM_PERSISTENT_MACRO_N_FAST
#define GEMM_PERSISTENT_MACRO_N_FAST 1
#endif

#ifndef GEMM_SINGLE_PIPELINE
#define GEMM_SINGLE_PIPELINE 0
#endif

#ifndef GEMM_SINGLE_PIPELINE_NTILE_128
#define GEMM_SINGLE_PIPELINE_NTILE_128 0
#endif

#ifndef GEMM_SINGLE_PIPELINE_WIDE_MMA
#define GEMM_SINGLE_PIPELINE_WIDE_MMA 1
#endif

#ifndef GEMM_SINGLE_PIPELINE_SPLIT_TMA
#define GEMM_SINGLE_PIPELINE_SPLIT_TMA 1
#endif

#ifndef GEMM_SINGLE_PIPELINE_INTERLEAVE_PIPES
#define GEMM_SINGLE_PIPELINE_INTERLEAVE_PIPES 1
#endif

#ifndef GEMM_WIDE_B_TMA
#define GEMM_WIDE_B_TMA 0
#endif

// Repeated-tile ceiling control: load one 128x128 B panel and let both
// N-direction issuer warps consume it.  This matches the historical 1797
// kernel's implicit tiled-B GEMM and reduces a K=128 stage from 96 to 64 KiB,
// allowing three stages.  It is intentionally invalid for dense addresses.
#ifndef GEMM_REPEAT_B_BROADCAST
#define GEMM_REPEAT_B_BROADCAST 0
#endif

#ifndef GEMM_CTA_M
#define GEMM_CTA_M 256
#endif

#ifndef GEMM_STAGE_K
#define GEMM_STAGE_K 64
#endif

#ifndef GEMM_STAGES
#define GEMM_STAGES 3
#endif

// Epilogue ablation:
//   0 = existing serialized SMEM -> TMA store
//   1 = serialized TMEM -> global direct store
//   2 = persistent TMEM double buffer with dedicated direct-store warp(s)
#ifndef GEMM_EPILOGUE_MODE
#define GEMM_EPILOGUE_MODE 0
#endif

#ifndef GEMM_EPILOGUE_WARPS
#define GEMM_EPILOGUE_WARPS 2
#endif

#ifndef GEMM_PIPE1_PHASE_SHIFT_CYCLES
#define GEMM_PIPE1_PHASE_SHIFT_CYCLES 512
#endif

#ifndef GEMM_PIPE1_PHASE_SHIFT_CYCLES_8K
#define GEMM_PIPE1_PHASE_SHIFT_CYCLES_8K 96
#endif

#ifndef GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES
#define GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES GEMM_PIPE1_PHASE_SHIFT_CYCLES
#endif

#ifndef GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES
#define GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES GEMM_PIPE1_PHASE_SHIFT_CYCLES
#endif

#ifndef GEMM_GRID_B_REUSE
#define GEMM_GRID_B_REUSE 0
#endif

#ifndef GEMM_GRID_SWIZZLE
#define GEMM_GRID_SWIZZLE 1
#endif

#ifndef GEMM_GRID_SWIZZLE_M
#define GEMM_GRID_SWIZZLE_M 12
#endif

#ifndef GEMM_GRID_SWIZZLE_N
#define GEMM_GRID_SWIZZLE_N 1
#endif

#ifndef GEMM_GRID_SWIZZLE_MIN_TILES
#define GEMM_GRID_SWIZZLE_MIN_TILES 32
#endif

#ifndef GEMM_CSTORE_CHUNK_N
#define GEMM_CSTORE_CHUNK_N 128
#endif

#ifndef GEMM_CSTORE_VECTORIZE_SMEM
#define GEMM_CSTORE_VECTORIZE_SMEM 1
#endif

#ifndef GEMM_CSTORE_SWIZZLE_128B
#define GEMM_CSTORE_SWIZZLE_128B (GEMM_REPEAT_TUNING ? 1 : 0)
#endif

#ifndef GEMM_TMA_A_L2_PROMOTION
#define GEMM_TMA_A_L2_PROMOTION CU_TENSOR_MAP_L2_PROMOTION_NONE
#endif

#ifndef GEMM_TMA_B_L2_PROMOTION
#define GEMM_TMA_B_L2_PROMOTION CU_TENSOR_MAP_L2_PROMOTION_NONE
#endif

#ifndef GEMM_TMA_C_L2_PROMOTION
#define GEMM_TMA_C_L2_PROMOTION CU_TENSOR_MAP_L2_PROMOTION_NONE
#endif

#ifndef GEMM_TUNED_4K_GRID_SWIZZLE
#define GEMM_TUNED_4K_GRID_SWIZZLE GEMM_GRID_SWIZZLE
#endif

#ifndef GEMM_TUNED_4K_GRID_SWIZZLE_M
#define GEMM_TUNED_4K_GRID_SWIZZLE_M 16
#endif

#ifndef GEMM_TUNED_4K_GRID_SWIZZLE_N
#define GEMM_TUNED_4K_GRID_SWIZZLE_N 1
#endif

#ifndef GEMM_TUNED_4K_PIPE1_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_4K_PIPE1_PHASE_SHIFT_CYCLES 96
#endif

#ifndef GEMM_TUNED_4K_PIPE1_TMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_4K_PIPE1_TMA_PHASE_SHIFT_CYCLES \
  GEMM_TUNED_4K_PIPE1_PHASE_SHIFT_CYCLES
#endif

#ifndef GEMM_TUNED_4K_PIPE1_MMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_4K_PIPE1_MMA_PHASE_SHIFT_CYCLES \
  GEMM_TUNED_4K_PIPE1_PHASE_SHIFT_CYCLES
#endif

#ifndef GEMM_TUNED_4K_TMA_A_L2_PROMOTION
#define GEMM_TUNED_4K_TMA_A_L2_PROMOTION GEMM_TMA_A_L2_PROMOTION
#endif

#ifndef GEMM_TUNED_4K_TMA_B_L2_PROMOTION
#define GEMM_TUNED_4K_TMA_B_L2_PROMOTION GEMM_TMA_B_L2_PROMOTION
#endif

#ifndef GEMM_TUNED_4K_TMA_C_L2_PROMOTION
#define GEMM_TUNED_4K_TMA_C_L2_PROMOTION GEMM_TMA_C_L2_PROMOTION
#endif

#ifndef GEMM_TUNED_4K_CSTORE_SWIZZLE_128B
#define GEMM_TUNED_4K_CSTORE_SWIZZLE_128B 1
#endif

#ifndef GEMM_TUNED_8K_GRID_SWIZZLE_M
#define GEMM_TUNED_8K_GRID_SWIZZLE_M (GEMM_DENSE_L2_TUNING ? 12 : 16)
#endif

#ifndef GEMM_TUNED_8K_GRID_SWIZZLE_N
#define GEMM_TUNED_8K_GRID_SWIZZLE_N 1
#endif

#ifndef GEMM_TUNED_8K_PIPE1_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_8K_PIPE1_PHASE_SHIFT_CYCLES 128
#endif

#ifndef GEMM_TUNED_8K_PIPE1_TMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_8K_PIPE1_TMA_PHASE_SHIFT_CYCLES \
  (GEMM_DENSE_L2_TUNING ? 0 : (GEMM_REPEAT_TUNING ? 96 : 128))
#endif

#ifndef GEMM_TUNED_8K_PIPE1_MMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_8K_PIPE1_MMA_PHASE_SHIFT_CYCLES 0
#endif

#ifndef GEMM_TUNED_8K_TMA_A_L2_PROMOTION
#define GEMM_TUNED_8K_TMA_A_L2_PROMOTION \
  (GEMM_DENSE_L2_TUNING ? CU_TENSOR_MAP_L2_PROMOTION_NONE \
                        : CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
#endif

#ifndef GEMM_TUNED_8K_TMA_B_L2_PROMOTION
#define GEMM_TUNED_8K_TMA_B_L2_PROMOTION \
  (GEMM_DENSE_L2_TUNING ? CU_TENSOR_MAP_L2_PROMOTION_NONE \
                        : CU_TENSOR_MAP_L2_PROMOTION_L2_256B)
#endif

#ifndef GEMM_TUNED_8K_TMA_C_L2_PROMOTION
#define GEMM_TUNED_8K_TMA_C_L2_PROMOTION CU_TENSOR_MAP_L2_PROMOTION_NONE
#endif

#ifndef GEMM_TUNED_8K_CSTORE_SWIZZLE_128B
#define GEMM_TUNED_8K_CSTORE_SWIZZLE_128B 1
#endif

#ifndef GEMM_TUNED_16K_GRID_SWIZZLE_M
#define GEMM_TUNED_16K_GRID_SWIZZLE_M (GEMM_DENSE_L2_TUNING ? 10 : 16)
#endif

#ifndef GEMM_TUNED_16K_GRID_SWIZZLE_N
#define GEMM_TUNED_16K_GRID_SWIZZLE_N 1
#endif

#ifndef GEMM_TUNED_16K_PIPE1_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_16K_PIPE1_PHASE_SHIFT_CYCLES 128
#endif

#ifndef GEMM_TUNED_16K_PIPE1_TMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_16K_PIPE1_TMA_PHASE_SHIFT_CYCLES \
  (GEMM_DENSE_L2_TUNING ? 0 : 128)
#endif

#ifndef GEMM_TUNED_16K_PIPE1_MMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_16K_PIPE1_MMA_PHASE_SHIFT_CYCLES 0
#endif

#ifndef GEMM_TUNED_16K_CSTORE_SWIZZLE_128B
#define GEMM_TUNED_16K_CSTORE_SWIZZLE_128B GEMM_CSTORE_SWIZZLE_128B
#endif

#ifndef GEMM_TUNED_32K_GRID_SWIZZLE_M
#define GEMM_TUNED_32K_GRID_SWIZZLE_M (GEMM_DENSE_L2_TUNING ? 3 : 12)
#endif

#ifndef GEMM_TUNED_32K_GRID_SWIZZLE_N
#define GEMM_TUNED_32K_GRID_SWIZZLE_N 1
#endif

#ifndef GEMM_TUNED_32K_PIPE1_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_32K_PIPE1_PHASE_SHIFT_CYCLES 128
#endif

#ifndef GEMM_TUNED_32K_PIPE1_TMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_32K_PIPE1_TMA_PHASE_SHIFT_CYCLES \
  (GEMM_DENSE_L2_TUNING ? 0 : (GEMM_REPEAT_TUNING ? 128 : 768))
#endif

#ifndef GEMM_TUNED_32K_PIPE1_MMA_PHASE_SHIFT_CYCLES
#define GEMM_TUNED_32K_PIPE1_MMA_PHASE_SHIFT_CYCLES \
  (GEMM_REPEAT_TUNING ? 0 : 1536)
#endif

#ifndef GEMM_TUNED_32K_TMA_A_L2_PROMOTION
#define GEMM_TUNED_32K_TMA_A_L2_PROMOTION GEMM_TMA_A_L2_PROMOTION
#endif

#ifndef GEMM_TUNED_32K_TMA_B_L2_PROMOTION
#define GEMM_TUNED_32K_TMA_B_L2_PROMOTION GEMM_TMA_B_L2_PROMOTION
#endif

#ifndef GEMM_TUNED_32K_TMA_C_L2_PROMOTION
#define GEMM_TUNED_32K_TMA_C_L2_PROMOTION GEMM_TMA_C_L2_PROMOTION
#endif

#ifndef GEMM_TUNED_32K_CSTORE_SWIZZLE_128B
#define GEMM_TUNED_32K_CSTORE_SWIZZLE_128B GEMM_CSTORE_SWIZZLE_128B
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
static constexpr int kCoreWarps = 4;
static constexpr int kEpilogueWarps =
    GEMM_EPILOGUE_MODE == 2 ? GEMM_EPILOGUE_WARPS : 0;
static constexpr int kWarps = kCoreWarps + kEpilogueWarps;
static constexpr int kThreads = kWarps * kThreadsPerWarp;
static constexpr int kSinglePipeline = GEMM_SINGLE_PIPELINE;
static constexpr int kSinglePipelineNtile128 = GEMM_SINGLE_PIPELINE_NTILE_128;
static constexpr int kSinglePipelineWideMma =
    kSinglePipeline && GEMM_SINGLE_PIPELINE_WIDE_MMA &&
    !kSinglePipelineNtile128;
static constexpr int kSinglePipelineSplitTma = GEMM_SINGLE_PIPELINE_SPLIT_TMA;
static constexpr int kSinglePipelineInterleavePipes =
    GEMM_SINGLE_PIPELINE_INTERLEAVE_PIPES;
static constexpr int kCtaM = GEMM_CTA_M;
static constexpr int kStageK = GEMM_STAGE_K;
static constexpr int kMmaM = 128;
static constexpr int kMmaN = 128;
static constexpr int kMmaK = 16;
static constexpr int kCtaN =
    (kSinglePipeline && kSinglePipelineNtile128) ? kMmaN : 256;
static constexpr int kSingleWideMmaM = kMmaM;
static constexpr int kSingleWideMmaN = kCtaN;
static constexpr int kBTmaN =
    (kSinglePipelineWideMma || GEMM_WIDE_B_TMA) ? kCtaN : kMmaN;
static constexpr int kBTmaNSubtiles = kBTmaN / 64;
static constexpr int kStages = GEMM_STAGES;
static constexpr int kPipes = kCtaN / kMmaN;
static constexpr int kMBlocks = kCtaM / kMmaM;
static constexpr bool kRepeatBBroadcast = GEMM_REPEAT_B_BROADCAST != 0;
static_assert(GEMM_EPILOGUE_MODE >= 0 && GEMM_EPILOGUE_MODE <= 2,
              "GEMM_EPILOGUE_MODE must be 0, 1, or 2");
static_assert(GEMM_EPILOGUE_MODE != 2 || GEMM_EPILOGUE_WARPS == 4,
              "A dedicated TMEM epilogue needs one full 4-warp warpgroup: "
              "warpgroup-local warp IDs 0..3 access TMEM lanes 0..127");
static_assert(GEMM_EPILOGUE_MODE != 2 || kCtaM == 128,
              "TMEM double-buffer epilogue currently requires CTA_M=128");
static_assert(GEMM_EPILOGUE_MODE != 2 || GEMM_PERSISTENT_CTA,
              "TMEM double-buffer epilogue requires persistent CTAs");
static constexpr int kAStageWords = kCtaM * kStageK / 2;
static constexpr int kBPipeWords = kStageK * kMmaN / 2;
static constexpr int kBStageWords =
    kRepeatBBroadcast ? kBPipeWords : kStageK * kCtaN / 2;
static constexpr int kStageWords = kAStageWords + kBStageWords;
static constexpr int kAStageBytes = kAStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kBStageBytes = kBStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kBPipeBytes = kBPipeWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kStageBytes = kStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kMainloopSmemBytes = kStages * kStageBytes;
static constexpr int kCStoreChunkM = 128;
static constexpr int kCStoreChunkN = GEMM_CSTORE_CHUNK_N;
static_assert(kCStoreChunkN % 32 == 0,
              "GEMM_CSTORE_CHUNK_N must be a multiple of 32");
static constexpr int kCStoreWarps = kCStoreChunkM / 32;
static constexpr int kCStoreStageWords = kCStoreChunkM * kCStoreChunkN;
static constexpr int kCStoreStageBytes =
    kCStoreStageWords * static_cast<int>(sizeof(uint32_t));
static constexpr int kCStoreChunksM = kCtaM / kCStoreChunkM;
static constexpr int kCStoreChunksN = kCtaN / kCStoreChunkN;
static constexpr int kCStoreChunkCount = kCStoreChunksM * kCStoreChunksN;
static constexpr int kCStoreTilesPerChunkN = kCStoreChunkN / kMmaN;
static constexpr int kCStoreBuffers = kCStoreTilesPerChunkN == 1 ? 2 : 1;
static constexpr int kCStoreTotalBytes = kCStoreBuffers * kCStoreStageBytes;
static constexpr int kDynamicSmemPayloadBytes =
    kMainloopSmemBytes > kCStoreTotalBytes ? kMainloopSmemBytes
                                           : kCStoreTotalBytes;
static constexpr int kDynamicSmemBytes =
    kDynamicSmemPayloadBytes + 1024;
static constexpr int kHalfTileWords = kMmaM * kStageK / 2;
static constexpr int kAK64TileWords = kMmaM * 64 / 2;
static constexpr int kTmemTileStride = 128;
[[maybe_unused]] static constexpr int kTraceSlotsPerIter = 8;
[[maybe_unused]] static constexpr int kPipe1PhaseShiftCycles =
    GEMM_PIPE1_PHASE_SHIFT_CYCLES;
[[maybe_unused]] static constexpr int kPipe1PhaseShiftCycles8K =
    GEMM_PIPE1_PHASE_SHIFT_CYCLES_8K;
[[maybe_unused]] static constexpr int kPipe1TmaPhaseShiftCycles =
    GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES;
[[maybe_unused]] static constexpr int kPipe1MmaPhaseShiftCycles =
    GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES;

static constexpr int kTuned4KGridSwizzle = GEMM_TUNED_4K_GRID_SWIZZLE;
static constexpr int kTuned4KGroupM = GEMM_TUNED_4K_GRID_SWIZZLE_M;
static constexpr int kTuned4KGroupN = GEMM_TUNED_4K_GRID_SWIZZLE_N;
static constexpr int kTuned4KPhaseCycles =
    GEMM_TUNED_4K_PIPE1_PHASE_SHIFT_CYCLES;
static constexpr int kTuned4KTmaPhaseCycles =
    GEMM_TUNED_4K_PIPE1_TMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned4KMmaPhaseCycles =
    GEMM_TUNED_4K_PIPE1_MMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned4KCStoreSwizzle128B =
    GEMM_TUNED_4K_CSTORE_SWIZZLE_128B;
static constexpr int kTuned8KGroupM = GEMM_TUNED_8K_GRID_SWIZZLE_M;
static constexpr int kTuned8KGroupN = GEMM_TUNED_8K_GRID_SWIZZLE_N;
static constexpr int kTuned8KPhaseCycles =
    GEMM_TUNED_8K_PIPE1_PHASE_SHIFT_CYCLES;
static constexpr int kTuned8KTmaPhaseCycles =
    GEMM_TUNED_8K_PIPE1_TMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned8KMmaPhaseCycles =
    GEMM_TUNED_8K_PIPE1_MMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned8KCStoreSwizzle128B =
    GEMM_TUNED_8K_CSTORE_SWIZZLE_128B;
static constexpr int kTuned16KGroupM = GEMM_TUNED_16K_GRID_SWIZZLE_M;
static constexpr int kTuned16KGroupN = GEMM_TUNED_16K_GRID_SWIZZLE_N;
static constexpr int kTuned16KPhaseCycles =
    GEMM_TUNED_16K_PIPE1_PHASE_SHIFT_CYCLES;
static constexpr int kTuned16KTmaPhaseCycles =
    GEMM_TUNED_16K_PIPE1_TMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned16KMmaPhaseCycles =
    GEMM_TUNED_16K_PIPE1_MMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned16KCStoreSwizzle128B =
    GEMM_TUNED_16K_CSTORE_SWIZZLE_128B;
static constexpr int kTuned32KGroupM = GEMM_TUNED_32K_GRID_SWIZZLE_M;
static constexpr int kTuned32KGroupN = GEMM_TUNED_32K_GRID_SWIZZLE_N;
static constexpr int kTuned32KPhaseCycles =
    GEMM_TUNED_32K_PIPE1_PHASE_SHIFT_CYCLES;
static constexpr int kTuned32KTmaPhaseCycles =
    GEMM_TUNED_32K_PIPE1_TMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned32KMmaPhaseCycles =
    GEMM_TUNED_32K_PIPE1_MMA_PHASE_SHIFT_CYCLES;
static constexpr int kTuned32KCStoreSwizzle128B =
    GEMM_TUNED_32K_CSTORE_SWIZZLE_128B;
static constexpr int kTuningTagGeneric = 0;
static constexpr int kTuningTag4K = 4;
static constexpr int kTuningTag8K = 8;
static constexpr int kTuningTag16K = 16;
static constexpr int kTuningTag32K = 32;

static_assert(kCtaM == 128 || kCtaM == 256);
static_assert(kCtaM % kMmaM == 0);
static_assert(kRepeatBBroadcast || kPipes * kBPipeWords == kBStageWords);
static_assert(!kRepeatBBroadcast || GEMM_REPEAT_INPUT,
              "B broadcast is only valid for the repeated-input control");
static_assert(!kRepeatBBroadcast ||
                  (kCtaM == 128 && kCtaN == 256 && kStageK == 128),
              "B broadcast currently requires CTA=128x256 and stage K=128");
static_assert(kMmaM % kCStoreChunkM == 0);
static_assert(kCtaM % kCStoreChunkM == 0);
static_assert(kCStoreChunkN % kMmaN == 0);
static_assert(kCtaN % kCStoreChunkN == 0);
static_assert(kCStoreChunkN % 64 == 0);
static_assert(kCStoreWarps * (kMmaM / kCStoreChunkM) <= kWarps);
static_assert(kCStoreChunkCount % kCStoreBuffers == 0);
static_assert(kCStoreChunkN == 128 || kCStoreChunkN == 256);
static_assert(kBTmaN == 128 || kBTmaN == 256);
static_assert(kBTmaN % 64 == 0);
static_assert(kSingleWideMmaM == 128);
static_assert(kSingleWideMmaN == 128 || kSingleWideMmaN == 256);

enum TraceStage {
  kTraceNone = 0,
  kTraceTmaIssue = 1,
  kTraceTmaWait = 2,
  kTraceMmaIssue = 3,
  kTraceMmaWait = 4,
  kTraceDrain = 5,
};

enum StoreMode : int {
  kStoreNone = 0,
  kStoreScalar = 1,
  kStoreTma = 2,
};

enum InputInitMode : int {
  kInputInitMemset = 0,
  kInputInitFormula = 1,
  kInputInitRandom = 2,
  kInputInitRandomSigned8 = 3,
};

struct Args {
  int device = 0;
  int warmup = 1;
  int iters = 5;
  std::vector<int> sizes = {4096, 8192, 16384, 32768};
  const char* csv = "gemm256_tma_tcgen05_bench.csv";
  int input_init_mode = kInputInitMemset;
  // 0 launches one CTA per output tile.  A positive value launches at most
  // this many CTAs and lets each CTA process a strided tile stream.
  int persistent_ctas = 0;
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

template <int PhaseCycles>
__device__ __forceinline__ void wait_pipe1_phase_shift_tuned() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if constexpr (PhaseCycles > 0) {
    const unsigned long long start = clock64();
    while (clock64() - start <
           static_cast<unsigned long long>(PhaseCycles)) {
    }
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

__device__ __forceinline__ uint64_t make_stage_a_smem_desc(
    uint32_t* a_smem,
    int mblock,
    int mma) {
  uint32_t* matrix = a_smem + mblock * kHalfTileWords;
  if constexpr (kStageK <= 64) {
    return make_sw128_major_k_smem_desc(smem_ptr_u32(matrix), mma);
  } else {
    // Wider stages are stored as independent 128x64 SW128 matrices.  Keeping
    // the K64 chunks planar preserves the 128-byte row stride encoded by the
    // normal major-K descriptor.
    constexpr int kMmasPerK64 = 64 / kMmaK;
    const int chunk = mma / kMmasPerK64;
    const int mma_in_chunk = mma - chunk * kMmasPerK64;
    matrix += chunk * kAK64TileWords;
    return make_sw128_major_k_smem_desc(smem_ptr_u32(matrix), mma_in_chunk);
  }
}

template <int MmaN>
__host__ __device__ __forceinline__ uint64_t
make_sw128_major_mn_smem_desc_shape(uint32_t matrix_start_addr, int mma) {
  constexpr uint64_t desc_base = (static_cast<uint64_t>(128u) << 16) |
                                 (static_cast<uint64_t>(64u) << 32) |
                                 (static_cast<uint64_t>(1u) << 46) |
                                 (static_cast<uint64_t>(2u) << 61);
  constexpr uint32_t kSliceBytes =
      static_cast<uint32_t>(kMmaK * MmaN / 2 * sizeof(uint32_t));
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (kSliceBytes >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__host__ __device__ __forceinline__ uint64_t make_sw128_major_mn_smem_desc(
    uint32_t matrix_start_addr,
    int mma) {
  return make_sw128_major_mn_smem_desc_shape<kMmaN>(matrix_start_addr, mma);
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

template <int MmaM, int MmaN>
__host__ __device__ __forceinline__ uint32_t make_bf16_idesc_shape() {
  uint32_t desc = 0;
  desc |= 1u << 4;   // C format: F32.
  desc |= 1u << 7;   // A format: BF16.
  desc |= 1u << 10;  // B format: BF16.
  desc |= static_cast<uint32_t>(MmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(MmaM >> 4) << 24;
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

__device__ __forceinline__ void tma_load_3d(const CUtensorMap* map,
                                            uint32_t dst_smem,
                                            uint64_t* barrier,
                                            int c0,
                                            int c1,
                                            int c2) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t bar = smem_ptr_u32(barrier);
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4, %5}], [%2];"
      :
      : "r"(dst_smem), "l"(map), "r"(bar), "r"(c0), "r"(c1), "r"(c2)
      : "memory");
#else
  (void)map;
  (void)dst_smem;
  (void)barrier;
  (void)c0;
  (void)c1;
  (void)c2;
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

__device__ __forceinline__ void tma_store_fence_shared() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_2d(const CUtensorMap* map,
                                             uint32_t src_smem,
                                             int c,
                                             int r) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group"
      " [%0, {%2, %3}], [%1];"
      :
      : "l"(map), "r"(src_smem), "r"(c), "r"(r)
      : "memory");
#else
  (void)map;
  (void)src_smem;
  (void)c;
  (void)r;
#endif
}

__device__ __forceinline__ void tma_store_4d(const CUtensorMap* map,
                                             uint32_t src_smem,
                                             int c0,
                                             int c1,
                                             int c2,
                                             int c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
      " [%0, {%2, %3, %4, %5}], [%1];"
      :
      : "l"(map), "r"(src_smem), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
      : "memory");
#else
  (void)map;
  (void)src_smem;
  (void)c0;
  (void)c1;
  (void)c2;
  (void)c3;
#endif
}

__device__ __forceinline__ void tma_store_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void tma_store_wait_group_0() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
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

__device__ __forceinline__ void tcgen05_fence_before_thread_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
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

__device__ __forceinline__ uint32_t consume_128x256(uint32_t taddr) {
  uint32_t acc = consume_128x128(taddr);
  acc ^= tcgen05_ld_32x32b_x64_acc(taddr + 128u);
  acc ^= tcgen05_ld_32x32b_x64_acc(taddr + 192u);
  tcgen05_wait_ld();
  return acc;
}

__host__ __device__ __forceinline__ int cstore_sw128_float_word_offset(
    int row,
    int col) {
  const int col_block = col >> 5;
  const int in_block = col & 31;
  return col_block * (kCStoreChunkM * 32) + row * 32 +
         (in_block ^ ((row & 7) << 2));
}

__device__ __forceinline__ void store_u32x4_smem(uint32_t* smem,
                                                 int word_offset,
                                                 uint32_t p0,
                                                 uint32_t p1,
                                                 uint32_t p2,
                                                 uint32_t p3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  reinterpret_cast<uint4*>(smem + word_offset)[0] =
      make_uint4(p0, p1, p2, p3);
#else
  smem[word_offset + 0] = p0;
  smem[word_offset + 1] = p1;
  smem[word_offset + 2] = p2;
  smem[word_offset + 3] = p3;
#endif
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

// Each warp can access only the corresponding 32-lane TMEM partition within
// its warpgroup.  A complete 128-row drain therefore requires four epilogue
// warps (one full warpgroup), even though the register-to-global part of each
// 32-row fragment is otherwise independent.
__device__ __forceinline__ void store_128x128_float_tile_epilogue_warp(
    uint32_t src_taddr,
    float* out,
    int out_ld,
    int row_offset,
    int col_offset,
    int row_group) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int row = row_offset + row_group * 32 + lane;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint32_t r[64];
    const uint32_t row_taddr =
        src_taddr + (static_cast<uint32_t>(row_group * 32) << 16) +
        static_cast<uint32_t>(half * 64);
    tcgen05_ld_32x32b_x64(r, row_taddr);
    tcgen05_wait_ld();
    uint32_t* dst = reinterpret_cast<uint32_t*>(
        out + static_cast<size_t>(row) * out_ld + col_offset + half * 64);
#pragma unroll
    for (int i = 0; i < 64; i += 4) {
      reinterpret_cast<uint4*>(dst + i)[0] =
          make_uint4(r[i + 0], r[i + 1], r[i + 2], r[i + 3]);
    }
  }
#else
  (void)src_taddr;
  (void)out;
  (void)out_ld;
  (void)row_offset;
  (void)col_offset;
  (void)row_group;
#endif
}

__device__ __forceinline__ void store_128x256_float_tile_epilogue_warps(
    const uint32_t (&c_taddr)[2],
    float* out,
    int out_ld,
    int row_offset,
    int col_offset,
    int warp_id) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int epilogue_warp = warp_id - kCoreWarps;
  if (epilogue_warp >= 0 && epilogue_warp < 4) {
    tcgen05_fence_after_thread_sync();
    store_128x128_float_tile_epilogue_warp(
        c_taddr[0], out, out_ld, row_offset, col_offset, epilogue_warp);
    store_128x128_float_tile_epilogue_warp(
        c_taddr[1], out, out_ld, row_offset, col_offset + 128,
        epilogue_warp);
    tcgen05_fence_before_thread_sync();
  }
#else
  (void)c_taddr;
  (void)out;
  (void)out_ld;
  (void)row_offset;
  (void)col_offset;
  (void)warp_id;
#endif
}

template <bool CStoreSwizzle128B, bool SingleWideMma>
__device__ __forceinline__ void stage_float_c_chunk(
    const uint32_t (&c_taddr)[4],
    uint32_t* c_smem,
    int chunk_m,
    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int selected_warp_base = 0;
  if (warp_id >= selected_warp_base &&
      warp_id < selected_warp_base + kCStoreWarps) {
    uint32_t r[64];
    const int local_warp = warp_id - selected_warp_base;
    const uint32_t row_base = static_cast<uint32_t>(local_warp * 32);
    const int local_row = local_warp * 32 + lane;
#pragma unroll
    for (int tile_n_part = 0; tile_n_part < kCStoreTilesPerChunkN;
         ++tile_n_part) {
      const int pipe = chunk_n * kCStoreTilesPerChunkN + tile_n_part;
      const int tile = SingleWideMma ? chunk_m * 2 : chunk_m * 2 + pipe;
#pragma unroll
      for (int load = 0; load < kMmaN / 64; ++load) {
        const uint32_t col_base =
            static_cast<uint32_t>(
                (SingleWideMma ? chunk_n * kCStoreChunkN : 0) +
                tile_n_part * kMmaN + load * 64);
        const uint32_t row_taddr =
            c_taddr[tile] + (row_base << 16) + col_base;
        tcgen05_ld_32x32b_x64(r, row_taddr);
        tcgen05_wait_ld();
        const int col_offset = tile_n_part * kMmaN + load * 64;
        if constexpr (CStoreSwizzle128B) {
#pragma unroll
          for (int i = 0; i < 64; i += 4) {
            store_u32x4_smem(c_smem,
                             cstore_sw128_float_word_offset(local_row,
                                                            col_offset + i),
                             r[i + 0], r[i + 1], r[i + 2], r[i + 3]);
          }
        } else {
        uint32_t* dst_words = c_smem + local_row * kCStoreChunkN;
#if GEMM_CSTORE_VECTORIZE_SMEM
#pragma unroll
        for (int i = 0; i < 64; i += 4) {
          store_u32x4_smem(dst_words, col_offset + i, r[i + 0], r[i + 1],
                           r[i + 2], r[i + 3]);
        }
#else
        float* dst_part = reinterpret_cast<float*>(dst_words + col_offset);
#pragma unroll
        for (int i = 0; i < 64; ++i) {
          dst_part[i] = __uint_as_float(r[i]);
        }
#endif
        }
      }
    }
  }
#else
  (void)c_taddr;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
#endif
}

template <bool CStoreSwizzle128B, bool SingleWideMma>
__device__ __forceinline__ void issue_float_c_chunk_tma(
    const uint32_t (&c_taddr)[4],
    const CUtensorMap* c_map,
    uint32_t* c_smem,
    int chunk_m,
    int chunk_n,
    int row_offset,
    int col_offset) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  stage_float_c_chunk<CStoreSwizzle128B, SingleWideMma>(
      c_taddr, c_smem, chunk_m, chunk_n);
  __syncthreads();
  tma_store_fence_shared();
  __syncthreads();
  if (threadIdx.x == 0) {
    if constexpr (CStoreSwizzle128B) {
      tma_store_4d(c_map, smem_ptr_u32(c_smem), 0, row_offset,
                   col_offset / 32, 0);
    } else {
      tma_store_2d(c_map, smem_ptr_u32(c_smem), col_offset, row_offset);
    }
  }
#else
  (void)c_taddr;
  (void)c_map;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
  (void)row_offset;
  (void)col_offset;
#endif
}

template <bool CStoreSwizzle128B, bool SingleWideMma>
__device__ __forceinline__ void store_256x256_float_tile_tma(
    const uint32_t (&c_taddr)[4],
    const CUtensorMap* c_map,
    uint32_t* c_smem,
    int row_offset,
    int col_offset) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
#pragma unroll
  for (int group = 0; group < kCStoreChunkCount; group += kCStoreBuffers) {
    if constexpr (GEMM_FUSED_CSTORE_STAGING != 0) {
      // Both buffers are independent.  Fill them first, then publish all
      // shared writes with one CTA synchronization/fence pair before issuing
      // the two TMA stores as one group.
#pragma unroll
      for (int i = 0; i < kCStoreBuffers; ++i) {
        const int chunk = group + i;
        uint32_t* tile_smem = c_smem + i * kCStoreStageWords;
        const int chunk_m = chunk / kCStoreChunksN;
        const int chunk_n = chunk - chunk_m * kCStoreChunksN;
        stage_float_c_chunk<CStoreSwizzle128B, SingleWideMma>(
            c_taddr, tile_smem, chunk_m, chunk_n);
      }
      __syncthreads();
      tma_store_fence_shared();
      __syncthreads();
      if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < kCStoreBuffers; ++i) {
          const int chunk = group + i;
          uint32_t* tile_smem = c_smem + i * kCStoreStageWords;
          const int chunk_m = chunk / kCStoreChunksN;
          const int chunk_n = chunk - chunk_m * kCStoreChunksN;
          const int tile_row = row_offset + chunk_m * kCStoreChunkM;
          const int tile_col = col_offset + chunk_n * kCStoreChunkN;
          if constexpr (CStoreSwizzle128B) {
            tma_store_4d(c_map, smem_ptr_u32(tile_smem), 0, tile_row,
                         tile_col / 32, 0);
          } else {
            tma_store_2d(c_map, smem_ptr_u32(tile_smem), tile_col, tile_row);
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < kCStoreBuffers; ++i) {
        const int chunk = group + i;
        uint32_t* tile_smem = c_smem + i * kCStoreStageWords;
        const int chunk_m = chunk / kCStoreChunksN;
        const int chunk_n = chunk - chunk_m * kCStoreChunksN;
        const int tile_row = row_offset + chunk_m * kCStoreChunkM;
        const int tile_col = col_offset + chunk_n * kCStoreChunkN;
        issue_float_c_chunk_tma<CStoreSwizzle128B, SingleWideMma>(
            c_taddr, c_map, tile_smem, chunk_m, chunk_n, tile_row, tile_col);
      }
    }
    if (threadIdx.x == 0) {
      tma_store_commit_group();
      tma_store_wait_group_0();
    }
    __syncthreads();
  }
#else
  (void)c_taddr;
  (void)c_map;
  (void)c_smem;
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
  const int a_row = tile_m * kCtaM;
  if constexpr (kStageK <= 64) {
    const int a_col_words = ktile * (kStageK / 2);
    tma_load_2d(a_map, smem_ptr_u32(a_smem), ready, a_col_words, a_row);
  } else {
    // A 128B-swizzled tensor map may expose at most 32 uint32 words in its
    // contiguous dimension.  Represent wider K stages as 64-BF16 subtiles.
    const int a_k64 = ktile * (kStageK / 64);
    tma_load_3d(a_map, smem_ptr_u32(a_smem), ready, 0, a_row, a_k64);
  }
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

__device__ __forceinline__ void issue_b_stage_tma(
    const CUtensorMap* b_map,
    uint32_t* b_smem,
    uint64_t* ready,
    int tile_n,
    int ktile,
    ClockTraceRecord* clock_trace,
    int clock_trace_start,
    int clock_trace_iters,
    unsigned long long trace_base,
    int trace_slot,
    int trace_warp) {
  const unsigned long long trace_start =
      clock_trace != nullptr ? clock64() : 0ull;
  mbarrier_expect_tx(ready, kBStageBytes);
  const int b_col_words = tile_n * (kCtaN / 2);
  const int b_k16 = ktile * (kStageK / kMmaK);
  tma_load_4d(b_map, smem_ptr_u32(b_smem), ready, b_col_words, 0, 0, b_k16);
  const unsigned long long trace_end =
      clock_trace != nullptr ? clock64() : 0ull;
  write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                     trace_base, kTraceTmaIssue, ktile, trace_slot, trace_warp,
                     trace_start, trace_end);
}

template <int GridSwizzle,
          int GroupM,
          int GroupN,
          int Pipe1TmaPhaseCycles,
          int Pipe1MmaPhaseCycles,
          int CStoreSwizzle128B,
          int SinglePipeline,
          int TuningTag>
__global__ __launch_bounds__(kThreads, 1)
void gemm256_tma_tcgen05_kernel(const __grid_constant__ CUtensorMap a_map,
                                const __grid_constant__ CUtensorMap b_map,
                                const __grid_constant__ CUtensorMap c_map,
                                uint32_t* __restrict__ sink,
                                float* __restrict__ out,
                                int out_ld,
                                int store_mode,
                                int ktiles,
                                int mtile_count,
                                int ntile_count,
                                ClockTraceRecord* __restrict__ clock_trace,
                                int clock_trace_start,
                                int clock_trace_iters) {
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ < 1000)
  (void)a_map;
  (void)b_map;
  (void)c_map;
  (void)sink;
  (void)out;
  (void)out_ld;
  (void)store_mode;
  (void)ktiles;
  (void)mtile_count;
  (void)ntile_count;
  (void)clock_trace;
  (void)clock_trace_start;
  (void)clock_trace_iters;
  (void)TuningTag;
#else
  extern __shared__ uint32_t smem_raw[];
  const uintptr_t smem_addr =
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023u) & ~static_cast<uintptr_t>(1023u);
  uint32_t* smem = reinterpret_cast<uint32_t*>(smem_addr);
  uint32_t* c_store_smem = smem;

  __shared__ uint64_t a_ready[kStages];
  __shared__ uint64_t b_ready[kPipes][kStages];
  __shared__ uint64_t mma_done[kPipes][kStages];
  __shared__ uint32_t tmem_smem;
  __shared__ uint32_t tmem_base_shared;
  __shared__ uint32_t warp_sinks[kWarps];
  __shared__ unsigned long long trace_base_shared;
  __shared__ int persistent_task_shared;

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
#pragma unroll
      for (int s = 0; s < kStages; ++s) {
        mbarrier_init(&mma_done[p][s], 1);
      }
    }
    trace_base_shared = clock64();
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane0 = lane == 0;
  if (warp_id == 0) {
    const uint32_t taddr = tcgen05_alloc_512cols(&tmem_smem);
    if (lane0) tmem_base_shared = taddr;
  }
  __syncthreads();

  const uint32_t tmem_base = tmem_base_shared;
  const uint32_t tmem_tile_addr[4] = {
      tmem_base + 0u * kTmemTileStride,
      tmem_base + 1u * kTmemTileStride,
      tmem_base + 2u * kTmemTileStride,
      tmem_base + 3u * kTmemTileStride,
  };
  const uint32_t idesc = make_bf16_idesc() | (1u << 16);
  const uint32_t single_wide_idesc =
      make_bf16_idesc_shape<kSingleWideMmaM, kSingleWideMmaN>() | (1u << 16);

  // Ordinary mode maps one output tile per CTA.  Persistent mode instead uses
  // sink[total_tiles] as a launch-local work counter.  Work is handed out in
  // 16x16 macroblocks, with M varying fastest so consecutive workers share B.
  // This preserves locality even when persistent CTAs progress at different
  // rates; the former blockIdx + iteration * gridDim schedule did not.
  const int total_tiles = mtile_count * ntile_count;
  const int linear_block = static_cast<int>(blockIdx.y) *
                               static_cast<int>(gridDim.x) +
                           static_cast<int>(blockIdx.x);
  const bool persistent_8k =
      GEMM_DENSE_L2_TUNING && mtile_count == 8192 / kCtaM &&
      ntile_count == 8192 / kCtaN;
  const bool persistent_32k =
      GEMM_DENSE_L2_TUNING && mtile_count == 32768 / kCtaM &&
      ntile_count == 32768 / kCtaN;
  const int selected_macro_m =
      persistent_8k    ? GEMM_PERSISTENT_8K_MACRO_M
      : persistent_32k ? GEMM_PERSISTENT_32K_MACRO_M
                       : GEMM_PERSISTENT_MACRO_M;
  const int selected_macro_n =
      persistent_8k    ? GEMM_PERSISTENT_8K_MACRO_N
      : persistent_32k ? GEMM_PERSISTENT_32K_MACRO_N
                       : GEMM_PERSISTENT_MACRO_N;
  const int persistent_macro_m =
      mtile_count < selected_macro_m ? mtile_count : selected_macro_m;
  const int persistent_macro_n =
      ntile_count < selected_macro_n ? ntile_count : selected_macro_n;
  const int persistent_groups_m =
      (mtile_count + persistent_macro_m - 1) / persistent_macro_m;
  const int persistent_groups_n =
      (ntile_count + persistent_macro_n - 1) / persistent_macro_n;
  const int persistent_macro_tiles = persistent_macro_m * persistent_macro_n;
  const int persistent_task_count =
      persistent_groups_m * persistent_groups_n * persistent_macro_tiles;
  int tile_iter = 0;
  int task_iter = 0;
  int previous_tile_m = 0;
  int previous_tile_n = 0;
  bool have_previous_tile = false;
  while (true) {
    int linear_tile = linear_block;
    if constexpr (GEMM_PERSISTENT_CTA) {
      if constexpr (GEMM_PERSISTENT_STATIC_SCHEDULER != 0) {
        // Dense square GEMM tiles have identical K and therefore identical
        // work.  A fixed grid-stride assignment removes the per-tile global
        // atomic and CTA-wide handoff barrier.  task_iter counts padded
        // macroblock positions; tile_iter counts only valid barrier epochs.
        linear_tile = linear_block +
                      task_iter * static_cast<int>(gridDim.x * gridDim.y);
        ++task_iter;
      } else {
        if (threadIdx.x == 0) {
          persistent_task_shared =
              static_cast<int>(atomicAdd(sink + total_tiles, 1u));
        }
        __syncthreads();
        linear_tile = persistent_task_shared;
      }
      if (linear_tile >= persistent_task_count) break;
    } else if (tile_iter != 0) {
      break;
    }

    int tile_m = 0;
    int tile_n = 0;
    if constexpr (GEMM_PERSISTENT_CTA) {
      const int macro_id = linear_tile / persistent_macro_tiles;
      const int local = linear_tile - macro_id * persistent_macro_tiles;
      const int macro_n = GEMM_PERSISTENT_MACRO_N_FAST
                              ? macro_id % persistent_groups_n
                              : macro_id / persistent_groups_m;
      const int macro_m = GEMM_PERSISTENT_MACRO_N_FAST
                              ? macro_id / persistent_groups_n
                              : macro_id % persistent_groups_m;
      const int local_m = GEMM_PERSISTENT_LOCAL_M_FAST
                              ? local % persistent_macro_m
                              : local / persistent_macro_n;
      const int local_n = GEMM_PERSISTENT_LOCAL_M_FAST
                              ? local / persistent_macro_m
                              : local % persistent_macro_n;
      tile_m = macro_m * persistent_macro_m + local_m;
      tile_n = macro_n * persistent_macro_n + local_n;
      // Only edge macroblocks can contain padded tasks.  They consume no
      // barrier epochs, so the next valid tile keeps the expected parity.
      if (tile_m >= mtile_count || tile_n >= ntile_count) {
        __syncthreads();
        continue;
      }
    } else if constexpr (GridSwizzle > 0) {
      const int groups_n = (ntile_count + GroupN - 1) / GroupN;
      const int group_tiles = GroupM * GroupN;
      const int group_id = linear_block / group_tiles;
      const int local = linear_block - group_id * group_tiles;
      const int group_tile_n = group_id % groups_n;
      const int group_tile_m = group_id / groups_n;
      tile_n = group_tile_n * GroupN + local % GroupN;
      tile_m = group_tile_m * GroupM + local / GroupN;
      if (tile_m >= mtile_count || tile_n >= ntile_count) break;
    } else {
#if GEMM_GRID_B_REUSE
      tile_m = static_cast<int>(blockIdx.x);
      tile_n = static_cast<int>(blockIdx.y);
#else
      tile_n = static_cast<int>(blockIdx.x);
      tile_m = static_cast<int>(blockIdx.y);
#endif
    }
    const int ntile = ntile_count;
    const int stage_epoch_base = GEMM_PERSISTENT_CTA ? tile_iter * ktiles : 0;
    const uint32_t tmem_bank_offset =
        GEMM_EPILOGUE_MODE == 2
            ? static_cast<uint32_t>((tile_iter & 1) * 2) * kTmemTileStride
            : 0u;
    const uint32_t c_taddr[4] = {
        tmem_tile_addr[0] + tmem_bank_offset,
        tmem_tile_addr[1] + tmem_bank_offset,
        tmem_tile_addr[2] + tmem_bank_offset,
        tmem_tile_addr[3] + tmem_bank_offset,
    };

    // Tile N drains while tile N+1 computes into the other 256-column TMEM
    // bank.  The iteration-end CTA rendezvous below prevents bank reuse until
    // both paths have finished.
    if constexpr (GEMM_EPILOGUE_MODE == 2) {
      if (have_previous_tile && out != nullptr) {
        const uint32_t previous_bank_offset =
            static_cast<uint32_t>(((tile_iter - 1) & 1) * 2) *
            kTmemTileStride;
        const uint32_t previous_c_taddr[2] = {
            tmem_tile_addr[0] + previous_bank_offset,
            tmem_tile_addr[1] + previous_bank_offset,
        };
        store_128x256_float_tile_epilogue_warps(
            previous_c_taddr, out, out_ld, previous_tile_m * kCtaM,
            previous_tile_n * kCtaN, warp_id);
      }
      if (warp_id == 2 || warp_id == 3) {
        // Orders this bank's MMA operations after the epilogue warp's prior
        // tcgen05.ld + wait + before_thread_sync sequence at the CTA barrier.
        tcgen05_fence_after_thread_sync();
      }
    }

  if (warp_id == 0 && lane0) {
    for (int kt = 0; kt < ktiles; ++kt) {
      const int stage_epoch = stage_epoch_base + kt;
      const int stage = stage_epoch % kStages;
      uint32_t* stage_smem = smem + stage * kStageWords;
      uint32_t* a_smem = stage_smem;
      uint32_t* b_smem = stage_smem + kAStageWords;
      if (stage_epoch >= kStages) {
        const uint32_t reuse_phase = static_cast<uint32_t>(
            ((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
        for (int p = 0; p < kPipes; ++p) {
          mbarrier_wait(&mma_done[p][stage], reuse_phase);
        }
      }
      const unsigned long long trace_start =
          clock_trace != nullptr ? clock64() : 0ull;
      const int source_m = GEMM_REPEAT_INPUT ? 0 : tile_m;
      const int source_n = GEMM_REPEAT_INPUT ? 0 : tile_n;
      const int source_kt = GEMM_REPEAT_INPUT ? 0 : kt;
      issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], source_m, source_kt);
      if constexpr ((SinglePipeline != 0 && kSinglePipelineWideMma != 0) ||
                    GEMM_WIDE_B_TMA != 0) {
        issue_b_stage_tma(&b_map, b_smem, &b_ready[0][stage], source_n,
                          source_kt,
                          nullptr, 0, 0, trace_base_shared, 0, 0);
      } else {
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], source_n,
                               source_kt, 0, nullptr, 0, 0,
                               trace_base_shared, 0, 0);
      }
      if constexpr (SinglePipeline != 0 && kPipes > 1 &&
                    kSinglePipelineSplitTma == 0 &&
                    kSinglePipelineWideMma == 0) {
        uint32_t* b1_smem = b_smem + kBPipeWords;
        issue_b_pipe_stage_tma(&b_map, b1_smem, &b_ready[1][stage], source_n,
                               source_kt, 1, nullptr, 0, 0,
                               trace_base_shared, 1, 0);
      }
      const unsigned long long trace_end =
          clock_trace != nullptr ? clock64() : 0ull;
      write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                         trace_base_shared, kTraceTmaIssue, kt, 0, 0,
                         trace_start, trace_end);
    }
  }

  if constexpr (kPipes > 1 &&
                (SinglePipeline == 0 ||
                 (kSinglePipelineSplitTma != 0 &&
                  kSinglePipelineWideMma == 0)) &&
                GEMM_WIDE_B_TMA == 0 && !kRepeatBBroadcast) {
    if (warp_id == 1 && lane0) {
      if constexpr (SinglePipeline == 0) {
        wait_pipe1_phase_shift_tuned<Pipe1TmaPhaseCycles>();
      }
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        uint32_t* stage_smem = smem + stage * kStageWords;
        uint32_t* b_smem = stage_smem + kAStageWords + kBPipeWords;
        if (stage_epoch >= kStages) {
          mbarrier_wait(&mma_done[1][stage], static_cast<uint32_t>(
                                                    ((stage_epoch - kStages) /
                                                     kStages) &
                                                    1));
        }
        const int source_n = GEMM_REPEAT_INPUT ? 0 : tile_n;
        const int source_kt = GEMM_REPEAT_INPUT ? 0 : kt;
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], source_n,
                               source_kt, 1, clock_trace, clock_trace_start,
                               clock_trace_iters, trace_base_shared, 1, 1);
      }
    }
  }

  if constexpr (SinglePipeline != 0) {
    if (warp_id == 2 && lane0) {
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        uint32_t* stage_smem = smem + stage * kStageWords;
        uint32_t* a_smem = stage_smem;
        mbarrier_wait(&a_ready[stage], tma_phase);
        if constexpr (kSinglePipelineWideMma != 0) {
          uint32_t* b_smem = stage_smem + kAStageWords;
          const unsigned long long tma_wait_start =
              clock_trace != nullptr ? clock64() : 0ull;
          mbarrier_wait(&b_ready[0][stage], tma_phase);
          const unsigned long long tma_wait_end =
              clock_trace != nullptr ? clock64() : 0ull;
          write_trace_record(clock_trace, clock_trace_start,
                             clock_trace_iters, trace_base_shared,
                             kTraceTmaWait, kt, 2, warp_id, tma_wait_start,
                             tma_wait_end);
          const unsigned long long mma_issue_start =
              clock_trace != nullptr ? clock64() : 0ull;
#pragma unroll
          for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
            const uint32_t b0 = smem_ptr_u32(b_smem);
            const uint64_t a0_desc = make_stage_a_smem_desc(a_smem, 0, kk);
            const uint64_t a1_desc = make_stage_a_smem_desc(a_smem, 1, kk);
            const uint64_t b0_desc =
                make_sw128_major_mn_smem_desc_shape<kSingleWideMmaN>(b0, kk);
            const bool input_d = (kt != 0) || (kk != 0);
            tcgen05_mma_bf16_ss(c_taddr[0], a0_desc, b0_desc,
                                single_wide_idesc, input_d);
            tcgen05_mma_bf16_ss(c_taddr[2], a1_desc, b0_desc,
                                single_wide_idesc, input_d);
          }
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            tcgen05_commit(&mma_done[pipe][stage]);
          }
          const unsigned long long mma_issue_end =
              clock_trace != nullptr ? clock64() : 0ull;
          write_trace_record(clock_trace, clock_trace_start,
                             clock_trace_iters, trace_base_shared,
                             kTraceMmaIssue, kt, 4, warp_id, mma_issue_start,
                             mma_issue_end);
        } else if constexpr (kPipes > 1 &&
                             kSinglePipelineInterleavePipes != 0) {
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            const unsigned long long tma_wait_start =
                clock_trace != nullptr ? clock64() : 0ull;
            mbarrier_wait(&b_ready[pipe][stage], tma_phase);
            const unsigned long long tma_wait_end =
                clock_trace != nullptr ? clock64() : 0ull;
            write_trace_record(clock_trace, clock_trace_start,
                               clock_trace_iters, trace_base_shared,
                               kTraceTmaWait, kt, 2 + pipe, warp_id,
                               tma_wait_start, tma_wait_end);
          }

          const unsigned long long mma_issue_start =
              clock_trace != nullptr ? clock64() : 0ull;
#pragma unroll
          for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
            const uint64_t a0_desc = make_stage_a_smem_desc(a_smem, 0, kk);
            const uint64_t a1_desc = make_stage_a_smem_desc(a_smem, 1, kk);
            const bool input_d = (kt != 0) || (kk != 0);
#pragma unroll
            for (int pipe = 0; pipe < kPipes; ++pipe) {
              uint32_t* b_smem =
                  stage_smem + kAStageWords + pipe * kBPipeWords;
              const uint32_t b0 = smem_ptr_u32(b_smem);
              const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
              tcgen05_mma_bf16_ss(c_taddr[pipe], a0_desc, b0_desc, idesc,
                                  input_d);
              tcgen05_mma_bf16_ss(c_taddr[pipe + 2], a1_desc, b0_desc,
                                  idesc, input_d);
            }
          }
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            tcgen05_commit(&mma_done[pipe][stage]);
          }
          const unsigned long long mma_issue_end =
              clock_trace != nullptr ? clock64() : 0ull;
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            write_trace_record(clock_trace, clock_trace_start,
                               clock_trace_iters, trace_base_shared,
                               kTraceMmaIssue, kt, 4 + pipe, warp_id,
                               mma_issue_start, mma_issue_end);
          }
        } else {
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            uint32_t* b_smem = stage_smem + kAStageWords + pipe * kBPipeWords;
            const int top_c = pipe;
            const int bottom_c = pipe + 2;

            const unsigned long long tma_wait_start =
                clock_trace != nullptr ? clock64() : 0ull;
            mbarrier_wait(&b_ready[pipe][stage], tma_phase);
            const unsigned long long tma_wait_end =
                clock_trace != nullptr ? clock64() : 0ull;
            write_trace_record(clock_trace, clock_trace_start,
                               clock_trace_iters, trace_base_shared,
                               kTraceTmaWait, kt, 2 + pipe, warp_id,
                               tma_wait_start, tma_wait_end);

            const unsigned long long mma_issue_start =
                clock_trace != nullptr ? clock64() : 0ull;
#pragma unroll
            for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
              const uint32_t b0 = smem_ptr_u32(b_smem);
              const uint64_t a0_desc = make_stage_a_smem_desc(a_smem, 0, kk);
              const uint64_t a1_desc = make_stage_a_smem_desc(a_smem, 1, kk);
              const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
              const bool input_d = (kt != 0) || (kk != 0);

              tcgen05_mma_bf16_ss(c_taddr[top_c], a0_desc, b0_desc, idesc,
                                  input_d);
              tcgen05_mma_bf16_ss(c_taddr[bottom_c], a1_desc, b0_desc, idesc,
                                  input_d);
            }
            tcgen05_commit(&mma_done[pipe][stage]);
            const unsigned long long mma_issue_end =
                clock_trace != nullptr ? clock64() : 0ull;
            write_trace_record(clock_trace, clock_trace_start,
                               clock_trace_iters, trace_base_shared,
                               kTraceMmaIssue, kt, 4 + pipe, warp_id,
                               mma_issue_start, mma_issue_end);
          }
        }

#pragma unroll
        for (int pipe = 0; pipe < kPipes; ++pipe) {
          const unsigned long long mma_wait_start =
              clock_trace != nullptr ? clock64() : 0ull;
          mbarrier_wait(&mma_done[pipe][stage], tma_phase);
          const unsigned long long mma_wait_end =
              clock_trace != nullptr ? clock64() : 0ull;
          write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                             trace_base_shared, kTraceMmaWait, kt, 6 + pipe,
                             warp_id, mma_wait_start, mma_wait_end);
        }
      }
    }
  } else {
    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
      if (pipe == 1) wait_pipe1_phase_shift_tuned<Pipe1MmaPhaseCycles>();
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        uint32_t* stage_smem = smem + stage * kStageWords;
        uint32_t* a_smem = stage_smem;
        uint32_t* b_smem =
            stage_smem + kAStageWords +
            (kRepeatBBroadcast ? 0 : pipe * kBPipeWords);

        const unsigned long long tma_wait_start =
            clock_trace != nullptr ? clock64() : 0ull;
        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(
            &b_ready[(GEMM_WIDE_B_TMA || kRepeatBBroadcast) ? 0 : pipe][stage],
            tma_phase);
        const unsigned long long tma_wait_end =
            clock_trace != nullptr ? clock64() : 0ull;
        write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                           trace_base_shared, kTraceTmaWait, kt, 2 + pipe,
                           warp_id, tma_wait_start, tma_wait_end);

        const unsigned long long mma_issue_start =
            clock_trace != nullptr ? clock64() : 0ull;
#pragma unroll
        for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
          uint64_t b0_desc = 0;
          if constexpr (GEMM_WIDE_B_TMA != 0) {
            // A 64x256 TMA interleaves both N pipes inside each K=16 slice:
            // [K16 pipe0][K16 pipe1], repeated four times.  Point each
            // 128-wide MMA descriptor at its half of the wide slice.
            constexpr int kWideBSliceWords = kMmaK * kCtaN / 2;
            constexpr int kPipeBSliceWords = kMmaK * kMmaN / 2;
            uint32_t* wide_b_smem = stage_smem + kAStageWords;
            const uint32_t b0 = smem_ptr_u32(
                wide_b_smem + kk * kWideBSliceWords +
                pipe * kPipeBSliceWords);
            b0_desc = make_sw128_major_mn_smem_desc(b0, 0);
          } else {
            const uint32_t b0 = smem_ptr_u32(b_smem);
            b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
          }
          const bool input_d = (kt != 0) || (kk != 0);
#pragma unroll
          for (int mblock = 0; mblock < kMBlocks; ++mblock) {
            const uint64_t a_desc =
                make_stage_a_smem_desc(a_smem, mblock, kk);
            const int c_tile = mblock * 2 + pipe;
            tcgen05_mma_bf16_ss(c_taddr[c_tile], a_desc, b0_desc, idesc,
                                input_d);
          }
        }
        tcgen05_commit(&mma_done[pipe][stage]);
        const unsigned long long mma_issue_end =
            clock_trace != nullptr ? clock64() : 0ull;
        write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                           trace_base_shared, kTraceMmaIssue, kt, 4 + pipe,
                           warp_id, mma_issue_start, mma_issue_end);

      }
      const int last_stage_epoch = stage_epoch_base + ktiles - 1;
      const int last_stage = last_stage_epoch % kStages;
      const uint32_t last_phase =
          static_cast<uint32_t>((last_stage_epoch / kStages) & 1);
      const unsigned long long mma_wait_start =
          clock_trace != nullptr ? clock64() : 0ull;
      mbarrier_wait(&mma_done[pipe][last_stage], last_phase);
      const unsigned long long mma_wait_end =
          clock_trace != nullptr ? clock64() : 0ull;
      write_trace_record(clock_trace, clock_trace_start, clock_trace_iters,
                         trace_base_shared, kTraceMmaWait, ktiles - 1,
                         6 + pipe, warp_id, mma_wait_start, mma_wait_end);
    }
  }
  __syncthreads();

  uint32_t acc = static_cast<uint32_t>(threadIdx.x + 0x9e3779b9u);
  if (warp_id < kWarps) {
    if (store_mode == kStoreNone) {
      const unsigned long long drain_start =
          lane0 && clock_trace != nullptr ? clock64() : 0ull;
      if constexpr (SinglePipeline != 0 && kSinglePipelineWideMma != 0) {
        if (warp_id == 0) acc ^= consume_128x256(c_taddr[0]);
        if constexpr (kMBlocks > 1) {
          if (warp_id == 2) acc ^= consume_128x256(c_taddr[2]);
        }
      } else {
#pragma unroll
        for (int mblock = 0; mblock < kMBlocks; ++mblock) {
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            const int c_tile = mblock * 2 + pipe;
            if (warp_id == c_tile) acc ^= consume_128x128(c_taddr[c_tile]);
          }
        }
      }
      const unsigned long long drain_end =
          lane0 && clock_trace != nullptr ? clock64() : 0ull;
      if (lane0) {
        write_trace_extra_record(clock_trace, clock_trace_iters,
                                 trace_base_shared, kTraceDrain, ktiles,
                                 warp_id, warp_id, drain_start, drain_end);
      }
    }
    if (lane0) warp_sinks[warp_id] = acc;
  }
  __syncthreads();

  if constexpr (GEMM_EPILOGUE_MODE != 2) {
    if (out != nullptr &&
        (store_mode == kStoreScalar || GEMM_EPILOGUE_MODE == 1) &&
        warp_id < kCoreWarps) {
      const int global_row_base = tile_m * kCtaM;
      const int global_col_base = tile_n * kCtaN;
      if constexpr (SinglePipeline != 0 && kSinglePipelineWideMma != 0) {
        store_128x128_float_tile(c_taddr[0], out, out_ld, global_row_base,
                                 global_col_base);
        store_128x128_float_tile(c_taddr[0] + 128u, out, out_ld,
                                 global_row_base, global_col_base + 128);
        store_128x128_float_tile(c_taddr[2], out, out_ld,
                                 global_row_base + 128, global_col_base);
        store_128x128_float_tile(c_taddr[2] + 128u, out, out_ld,
                                 global_row_base + 128,
                                 global_col_base + 128);
      } else {
#pragma unroll
        for (int mblock = 0; mblock < kMBlocks; ++mblock) {
#pragma unroll
          for (int pipe = 0; pipe < kPipes; ++pipe) {
            const int c_tile = mblock * 2 + pipe;
            store_128x128_float_tile(c_taddr[c_tile], out, out_ld,
                                     global_row_base + mblock * kMmaM,
                                     global_col_base + pipe * kMmaN);
          }
        }
      }
    } else if (out != nullptr && store_mode == kStoreTma) {
      const int global_row_base = tile_m * kCtaM;
      const int global_col_base = tile_n * kCtaN;
      store_256x256_float_tile_tma<
          CStoreSwizzle128B != 0,
          (SinglePipeline != 0 && kSinglePipelineWideMma != 0)>(
          c_taddr, &c_map, c_store_smem, global_row_base, global_col_base);
    }
  }
  if constexpr (GEMM_FUSED_CSTORE_STAGING == 0) {
    __syncthreads();
  } else if (store_mode != kStoreTma) {
    // The fused TMA-store helper already leaves the CTA synchronized after
    // its final wait.  Scalar/no-store paths still need this rendezvous.
    __syncthreads();
  }

  if constexpr (GEMM_ELIDE_DENSE_SINK == 0) {
    if (threadIdx.x == 0) {
      if constexpr (!GEMM_PERSISTENT_CTA) {
        tcgen05_fence_after_thread_sync();
      }
      uint32_t out = tmem_base ^ static_cast<uint32_t>(ktiles);
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        out ^= warp_sinks[w];
      }
      sink[tile_m * ntile + tile_n] = out;
    }
    __syncthreads();
  } else if (store_mode != kStoreTma) {
    if (threadIdx.x == 0) {
      if constexpr (!GEMM_PERSISTENT_CTA) {
        tcgen05_fence_after_thread_sync();
      }
      uint32_t out = tmem_base ^ static_cast<uint32_t>(ktiles);
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        out ^= warp_sinks[w];
      }
      sink[tile_m * ntile + tile_n] = out;
    }
    __syncthreads();
  }

  previous_tile_m = tile_m;
  previous_tile_n = tile_n;
  have_previous_tile = true;
  ++tile_iter;
  }  // persistent output-tile loop

  if constexpr (GEMM_EPILOGUE_MODE == 2) {
    // Final drain: no next tile exists to hide this last epilogue behind.
    if (have_previous_tile && out != nullptr) {
      const uint32_t previous_bank_offset =
          static_cast<uint32_t>(((tile_iter - 1) & 1) * 2) *
          kTmemTileStride;
      const uint32_t previous_c_taddr[2] = {
          tmem_tile_addr[0] + previous_bank_offset,
          tmem_tile_addr[1] + previous_bank_offset,
      };
      store_128x256_float_tile_epilogue_warps(
          previous_c_taddr, out, out_ld, previous_tile_m * kCtaM,
          previous_tile_n * kCtaN, warp_id);
    }
    __syncthreads();
  }

  if constexpr (GEMM_PERSISTENT_CTA) {
    if (threadIdx.x == 0) tcgen05_fence_after_thread_sync();
    __syncthreads();
  }

  if (warp_id == 0) tcgen05_dealloc_512cols(tmem_base);
  __syncthreads();
  if (warp_id == 0) tcgen05_relinquish_alloc_permit();
#endif
}

void encode_a_row_major_sw128_tma_map(CUtensorMap* map,
                                      void* base,
                                      uint64_t rows,
                                      uint64_t cols_bf16,
                                      CUtensorMapL2promotion l2_promotion) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  if constexpr (kStageK <= 64) {
    const cuuint64_t global_dim[2] = {cols_words, rows};
    const cuuint64_t global_stride[1] = {cols_words * sizeof(uint32_t)};
    const cuuint32_t box_dim[2] = {kStageK / 2, kCtaM};
    const cuuint32_t elem_stride[2] = {1, 1};
    driver_check(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 2,
                                        base, global_dim, global_stride,
                                        box_dim, elem_stride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_128B,
                                        l2_promotion,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled(a_row_major_sw128)");
  } else {
    static_assert(kStageK % 64 == 0,
                  "GEMM_STAGE_K above 64 must be a multiple of 64");
    const cuuint64_t global_dim[3] = {32, rows, cols_words / 32};
    const cuuint64_t global_stride[2] = {
        cols_words * sizeof(uint32_t), 32 * sizeof(uint32_t)};
    const cuuint32_t box_dim[3] = {32, kCtaM, kStageK / 64};
    const cuuint32_t elem_stride[3] = {1, 1, 1};
    driver_check(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT32, 3,
                                        base, global_dim, global_stride,
                                        box_dim, elem_stride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_128B,
                                        l2_promotion,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled(a_row_major_sw128_k64)");
  }
}

void encode_b_row_major_sw128_k16_tma_map(CUtensorMap* map,
                                          void* base,
                                          uint64_t rows,
                                          uint64_t cols_bf16,
                                          CUtensorMapL2promotion l2_promotion) {
  const cuuint64_t cols_words = cols_bf16 / 2;
  const cuuint64_t global_dim[4] = {cols_words, kMmaK, kBTmaNSubtiles,
                                    rows / kMmaK};
  const cuuint64_t global_stride[3] = {
      cols_words * sizeof(uint32_t),
      static_cast<cuuint64_t>(kMmaN / 4) * sizeof(uint32_t),
      static_cast<cuuint64_t>(kMmaK) * cols_words * sizeof(uint32_t)};
  const cuuint32_t box_dim[4] = {kMmaN / 4, kMmaK, kBTmaNSubtiles,
                                 kStageK / kMmaK};
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
                                      l2_promotion,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(b_row_major_sw128_k16)");
}

void encode_c_row_major_float_tma_map(CUtensorMap* map,
                                      void* base,
                                      uint64_t rows,
                                      uint64_t cols,
                                      CUtensorMapL2promotion l2_promotion,
                                      bool swizzle_128b) {
  if (swizzle_128b) {
    const cuuint64_t global_dim[4] = {32, rows, cols / 32, 1};
    const cuuint64_t global_stride[3] = {
        cols * sizeof(float),
        static_cast<cuuint64_t>(32) * sizeof(float),
        rows * cols * sizeof(float)};
    const cuuint32_t box_dim[4] = {32, kCStoreChunkM, kCStoreChunkN / 32, 1};
    const cuuint32_t elem_stride[4] = {1, 1, 1, 1};
    driver_check(cuTensorMapEncodeTiled(map,
                                        CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                        4,
                                        base,
                                        global_dim,
                                        global_stride,
                                        box_dim,
                                        elem_stride,
                                        CU_TENSOR_MAP_INTERLEAVE_NONE,
                                        CU_TENSOR_MAP_SWIZZLE_128B,
                                        l2_promotion,
                                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 "cuTensorMapEncodeTiled(c_row_major_float_sw128)");
    return;
  }

  const cuuint64_t global_dim[2] = {cols, rows};
  const cuuint64_t global_stride[1] = {cols * sizeof(float)};
  const cuuint32_t box_dim[2] = {kCStoreChunkN, kCStoreChunkM};
  const cuuint32_t elem_stride[2] = {1, 1};
  driver_check(cuTensorMapEncodeTiled(map,
                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                                      2,
                                      base,
                                      global_dim,
                                      global_stride,
                                      box_dim,
                                      elem_stride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_NONE,
                                      l2_promotion,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(c_row_major_float)");
}

CUtensorMapL2promotion tma_a_l2_promotion_for_size(int size) {
  if (size == 4096) return GEMM_TUNED_4K_TMA_A_L2_PROMOTION;
  if (size == 8192) return GEMM_TUNED_8K_TMA_A_L2_PROMOTION;
  if (size == 32768) return GEMM_TUNED_32K_TMA_A_L2_PROMOTION;
  return GEMM_TMA_A_L2_PROMOTION;
}

CUtensorMapL2promotion tma_b_l2_promotion_for_size(int size) {
  if (size == 4096) return GEMM_TUNED_4K_TMA_B_L2_PROMOTION;
  if (size == 8192) return GEMM_TUNED_8K_TMA_B_L2_PROMOTION;
  if (size == 32768) return GEMM_TUNED_32K_TMA_B_L2_PROMOTION;
  return GEMM_TMA_B_L2_PROMOTION;
}

CUtensorMapL2promotion tma_c_l2_promotion_for_size(int size) {
  if (size == 4096) return GEMM_TUNED_4K_TMA_C_L2_PROMOTION;
  if (size == 8192) return GEMM_TUNED_8K_TMA_C_L2_PROMOTION;
  if (size == 32768) return GEMM_TUNED_32K_TMA_C_L2_PROMOTION;
  return GEMM_TMA_C_L2_PROMOTION;
}

const char* store_mode_name(int store_mode) {
  switch (store_mode) {
    case kStoreNone:
      return "none";
    case kStoreScalar:
      return "scalar";
    case kStoreTma:
      return "tma";
    default:
      return "unknown";
  }
}

const char* input_init_mode_name(int mode) {
  switch (mode) {
    case kInputInitMemset:
      return "memset";
    case kInputInitFormula:
      return "formula";
    case kInputInitRandom:
      return "random";
    case kInputInitRandomSigned8:
      return "random-signed8";
    default:
      return "unknown";
  }
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
              "[--input-init memset|formula|random|random-signed8] "
              "[--persistent-ctas N] "
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
    } else if (std::strcmp(argv[i], "--input-init") == 0) {
      const char* mode = need_arg("--input-init");
      if (std::strcmp(mode, "memset") == 0) {
        args.input_init_mode = kInputInitMemset;
      } else if (std::strcmp(mode, "formula") == 0) {
        args.input_init_mode = kInputInitFormula;
      } else if (std::strcmp(mode, "random") == 0) {
        args.input_init_mode = kInputInitRandom;
      } else if (std::strcmp(mode, "random-signed8") == 0) {
        args.input_init_mode = kInputInitRandomSigned8;
      } else {
        std::fprintf(stderr,
                     "input init must be 'memset', 'formula', 'random', or "
                     "'random-signed8'\n");
        std::exit(EXIT_FAILURE);
      }
    } else if (std::strcmp(argv[i], "--persistent-ctas") == 0) {
      args.persistent_ctas = std::atoi(need_arg("--persistent-ctas"));
      if (args.persistent_ctas < 0) {
        std::fprintf(stderr, "persistent CTA count must be non-negative\n");
        std::exit(EXIT_FAILURE);
      }
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
      args.validate_size % kCtaN != 0 || args.validate_size % kStageK != 0) {
    std::fprintf(stderr,
                 "validate size must be a positive multiple of cta_m=%d, "
                 "cta_n=%d, and stage_k=%d\n",
                 kCtaM, kCtaN, kStageK);
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

static constexpr float kFormulaInitScale = 1.0f / 128.0f;

__device__ __forceinline__ uint16_t float_to_bf16_bits_device(float value) {
  uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7fffu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

__global__ void init_formula_bf16_words(uint32_t* words,
                                        size_t word_count,
                                        uint64_t index_offset,
                                        float scale) {
  const size_t word_idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_idx >= word_count) return;

  const uint64_t lo_idx = index_offset + static_cast<uint64_t>(word_idx) * 2u;
  const uint64_t hi_idx = lo_idx + 1u;
  const float lo_value =
      (static_cast<float>(static_cast<int>(lo_idx % 251u)) - 125.0f) * scale;
  const float hi_value =
      (static_cast<float>(static_cast<int>(hi_idx % 251u)) - 125.0f) * scale;
  const uint32_t lo = float_to_bf16_bits_device(lo_value);
  const uint32_t hi = float_to_bf16_bits_device(hi_value);
  words[word_idx] = lo | (hi << 16);
}

__device__ __forceinline__ uint32_t random_mix32(uint32_t x) {
  x += 0x9e3779b9u;
  x = (x ^ (x >> 16)) * 0x85ebca6bu;
  x = (x ^ (x >> 13)) * 0xc2b2ae35u;
  return x ^ (x >> 16);
}

__global__ void init_random_bf16_words(uint32_t* words,
                                       size_t word_count,
                                       uint32_t seed,
                                       float scale,
                                       float bias) {
  const size_t word_idx =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (word_idx >= word_count) return;
  const uint32_t index = static_cast<uint32_t>(word_idx);
  const uint32_t lo24 = random_mix32(seed ^ index ^ 0x9e3779b9u) >> 8;
  const uint32_t hi24 = random_mix32(seed ^ index ^ 0x243f6a88u) >> 8;
  const float lo = static_cast<float>(lo24) * 0x1.0p-24f * scale + bias;
  const float hi = static_cast<float>(hi24) * 0x1.0p-24f * scale + bias;
  words[word_idx] = static_cast<uint32_t>(float_to_bf16_bits_device(lo)) |
                    (static_cast<uint32_t>(float_to_bf16_bits_device(hi))
                     << 16);
}

void initialize_bf16_inputs(uint32_t* d_a,
                            size_t a_words,
                            uint32_t* d_b,
                            size_t b_words,
                            int input_init_mode) {
  if (input_init_mode == kInputInitMemset) {
    CUDA_CHECK(cudaMemset(d_a, 0x3f, a_words * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_b, 0x11, b_words * sizeof(uint32_t)));
    return;
  }
  if (input_init_mode == kInputInitRandom ||
      input_init_mode == kInputInitRandomSigned8) {
    constexpr int kInitThreads = 256;
    const int a_blocks =
        static_cast<int>((a_words + kInitThreads - 1) / kInitThreads);
    const int b_blocks =
        static_cast<int>((b_words + kInitThreads - 1) / kInitThreads);
    const float scale = input_init_mode == kInputInitRandom ? 1.0f : 16.0f;
    const float bias = input_init_mode == kInputInitRandom ? 0.0f : -8.0f;
    init_random_bf16_words<<<a_blocks, kInitThreads>>>(
        d_a, a_words, 20260719u ^ 0xa511e9b3u, scale, bias);
    CUDA_CHECK(cudaGetLastError());
    init_random_bf16_words<<<b_blocks, kInitThreads>>>(
        d_b, b_words, 20260719u ^ 0x63d83595u, scale, bias);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  if (input_init_mode != kInputInitFormula) {
    std::fprintf(stderr, "Unknown input init mode: %d\n", input_init_mode);
    std::exit(EXIT_FAILURE);
  }

  constexpr int kInitThreads = 256;
  const int a_blocks =
      static_cast<int>((a_words + kInitThreads - 1) / kInitThreads);
  const int b_blocks =
      static_cast<int>((b_words + kInitThreads - 1) / kInitThreads);
  init_formula_bf16_words<<<a_blocks, kInitThreads>>>(d_a, a_words, 0,
                                                      kFormulaInitScale);
  CUDA_CHECK(cudaGetLastError());
  init_formula_bf16_words<<<b_blocks, kInitThreads>>>(
      d_b, b_words, static_cast<uint64_t>(a_words) * 2u, kFormulaInitScale);
  CUDA_CHECK(cudaGetLastError());
}

struct GemmTuning {
  int grid_swizzle = 0;
  int group_m = 1;
  int group_n = 1;
  int pipe1_phase_cycles = GEMM_PIPE1_PHASE_SHIFT_CYCLES;
  int pipe1_tma_phase_cycles = GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES;
  int pipe1_mma_phase_cycles = GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES;
  int cstore_swizzle_128b = GEMM_CSTORE_SWIZZLE_128B;
  int tag = kTuningTagGeneric;
};

int select_grid_swizzle(int mtile, int ntile) {
#if GEMM_GRID_SWIZZLE > 0
  if (mtile >= GEMM_GRID_SWIZZLE_MIN_TILES &&
      ntile >= GEMM_GRID_SWIZZLE_MIN_TILES) {
    return GEMM_GRID_SWIZZLE;
  }
#endif
  return 0;
}

GemmTuning select_gemm_tuning(int mtile, int ntile, int ktiles) {
  GemmTuning tuning;
  tuning.grid_swizzle = select_grid_swizzle(mtile, ntile);
  tuning.group_m = GEMM_GRID_SWIZZLE_M;
  tuning.group_n = GEMM_GRID_SWIZZLE_N;
  tuning.pipe1_phase_cycles = GEMM_PIPE1_PHASE_SHIFT_CYCLES;
  tuning.pipe1_tma_phase_cycles = GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES;
  tuning.pipe1_mma_phase_cycles = GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES;
  tuning.cstore_swizzle_128b = GEMM_CSTORE_SWIZZLE_128B;
  if (mtile == 4096 / kCtaM && ntile == 4096 / kCtaN &&
      ktiles == 4096 / kStageK && kTuned4KGridSwizzle > 0) {
    tuning.grid_swizzle = kTuned4KGridSwizzle;
    tuning.group_m = kTuned4KGroupM;
    tuning.group_n = kTuned4KGroupN;
    tuning.pipe1_phase_cycles = kTuned4KPhaseCycles;
    tuning.pipe1_tma_phase_cycles = kTuned4KTmaPhaseCycles;
    tuning.pipe1_mma_phase_cycles = kTuned4KMmaPhaseCycles;
    tuning.cstore_swizzle_128b = kTuned4KCStoreSwizzle128B;
    tuning.tag = kTuningTag4K;
  }
  if (tuning.grid_swizzle > 0) {
    if (mtile == 8192 / kCtaM && ntile == 8192 / kCtaN &&
        ktiles == 8192 / kStageK) {
      tuning.group_m = kTuned8KGroupM;
      tuning.group_n = kTuned8KGroupN;
      tuning.pipe1_phase_cycles = kTuned8KPhaseCycles;
      tuning.pipe1_tma_phase_cycles = kTuned8KTmaPhaseCycles;
      tuning.pipe1_mma_phase_cycles = kTuned8KMmaPhaseCycles;
      tuning.cstore_swizzle_128b = kTuned8KCStoreSwizzle128B;
      tuning.tag = kTuningTag8K;
    } else if (mtile == 16384 / kCtaM && ntile == 16384 / kCtaN &&
               ktiles == 16384 / kStageK) {
      tuning.group_m = kTuned16KGroupM;
      tuning.group_n = kTuned16KGroupN;
      tuning.pipe1_phase_cycles = kTuned16KPhaseCycles;
      tuning.pipe1_tma_phase_cycles = kTuned16KTmaPhaseCycles;
      tuning.pipe1_mma_phase_cycles = kTuned16KMmaPhaseCycles;
      tuning.cstore_swizzle_128b = kTuned16KCStoreSwizzle128B;
      tuning.tag = kTuningTag16K;
    } else if (mtile == 32768 / kCtaM && ntile == 32768 / kCtaN &&
               ktiles == 32768 / kStageK) {
      tuning.group_m = kTuned32KGroupM;
      tuning.group_n = kTuned32KGroupN;
      tuning.pipe1_phase_cycles = kTuned32KPhaseCycles;
      tuning.pipe1_tma_phase_cycles = kTuned32KTmaPhaseCycles;
      tuning.pipe1_mma_phase_cycles = kTuned32KMmaPhaseCycles;
      tuning.cstore_swizzle_128b = kTuned32KCStoreSwizzle128B;
      tuning.tag = kTuningTag32K;
    }
  }
  return tuning;
}

dim3 launch_grid(int mtile, int ntile, const GemmTuning& tuning) {
  if (tuning.grid_swizzle > 0) {
    const int groups_m = (mtile + tuning.group_m - 1) / tuning.group_m;
    const int groups_n = (ntile + tuning.group_n - 1) / tuning.group_n;
    return dim3(groups_m * groups_n * tuning.group_m * tuning.group_n, 1, 1);
  }
#if GEMM_GRID_B_REUSE
  return dim3(mtile, ntile, 1);
#else
  return dim3(ntile, mtile, 1);
#endif
}

template <int GridSwizzle,
          int GroupM,
          int GroupN,
          int Pipe1TmaPhaseCycles,
          int Pipe1MmaPhaseCycles,
          int CStoreSwizzle128B,
          int SinglePipeline,
          int TuningTag>
void set_one_gemm_kernel_attribute() {
  CUDA_CHECK(cudaFuncSetAttribute(
      gemm256_tma_tcgen05_kernel<GridSwizzle,
                                 GroupM,
                                 GroupN,
                                 Pipe1TmaPhaseCycles,
                                 Pipe1MmaPhaseCycles,
                                 CStoreSwizzle128B,
                                 SinglePipeline,
                                 TuningTag>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      kDynamicSmemBytes));
}

void set_gemm_kernel_attributes() {
  set_one_gemm_kernel_attribute<0,
                                1,
                                1,
                                GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES,
                                GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES,
                                GEMM_CSTORE_SWIZZLE_128B,
                                kSinglePipeline,
                                kTuningTagGeneric>();
#if GEMM_GRID_SWIZZLE > 0
  set_one_gemm_kernel_attribute<GEMM_GRID_SWIZZLE,
                                GEMM_GRID_SWIZZLE_M,
                                GEMM_GRID_SWIZZLE_N,
                                GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES,
                                GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES,
                                GEMM_CSTORE_SWIZZLE_128B,
                                kSinglePipeline,
                                kTuningTagGeneric>();
  set_one_gemm_kernel_attribute<kTuned4KGridSwizzle,
                                kTuned4KGroupM,
                                kTuned4KGroupN,
                                kTuned4KTmaPhaseCycles,
                                kTuned4KMmaPhaseCycles,
                                kTuned4KCStoreSwizzle128B,
                                kSinglePipeline,
                                kTuningTag4K>();
  set_one_gemm_kernel_attribute<GEMM_GRID_SWIZZLE,
                                kTuned8KGroupM,
                                kTuned8KGroupN,
                                kTuned8KTmaPhaseCycles,
                                kTuned8KMmaPhaseCycles,
                                kTuned8KCStoreSwizzle128B,
                                kSinglePipeline,
                                kTuningTag8K>();
  set_one_gemm_kernel_attribute<GEMM_GRID_SWIZZLE,
                                kTuned16KGroupM,
                                kTuned16KGroupN,
                                kTuned16KTmaPhaseCycles,
                                kTuned16KMmaPhaseCycles,
                                kTuned16KCStoreSwizzle128B,
                                kSinglePipeline,
                                kTuningTag16K>();
  set_one_gemm_kernel_attribute<GEMM_GRID_SWIZZLE,
                                kTuned32KGroupM,
                                kTuned32KGroupN,
                                kTuned32KTmaPhaseCycles,
                                kTuned32KMmaPhaseCycles,
                                kTuned32KCStoreSwizzle128B,
                                kSinglePipeline,
                                kTuningTag32K>();
#endif
}

void launch_gemm_kernel(const GemmTuning& tuning,
                        dim3 grid,
                        dim3 block,
                        const CUtensorMap& a_map,
                        const CUtensorMap& b_map,
                        const CUtensorMap& c_map,
                        uint32_t* d_sink,
                        float* d_c,
                        int out_ld,
                        int store_mode,
                        int ktiles,
                        int mtile,
                        int ntile,
                        ClockTraceRecord* clock_trace,
                        int clock_trace_start,
                        int clock_trace_iters) {
#if GEMM_GRID_SWIZZLE > 0
  if (tuning.grid_swizzle > 0) {
    if (tuning.tag == kTuningTag4K) {
      gemm256_tma_tcgen05_kernel<kTuned4KGridSwizzle,
                                 kTuned4KGroupM,
                                 kTuned4KGroupN,
                                 kTuned4KTmaPhaseCycles,
                                 kTuned4KMmaPhaseCycles,
                                 kTuned4KCStoreSwizzle128B,
                                 kSinglePipeline,
                                 kTuningTag4K>
          <<<grid, block, kDynamicSmemBytes>>>(
              a_map, b_map, c_map, d_sink, d_c, out_ld, store_mode, ktiles,
              mtile, ntile, clock_trace, clock_trace_start, clock_trace_iters);
      return;
    }
    if (tuning.tag == kTuningTag8K) {
      gemm256_tma_tcgen05_kernel<GEMM_GRID_SWIZZLE,
                                 kTuned8KGroupM,
                                 kTuned8KGroupN,
                                 kTuned8KTmaPhaseCycles,
                                 kTuned8KMmaPhaseCycles,
                                 kTuned8KCStoreSwizzle128B,
                                 kSinglePipeline,
                                 kTuningTag8K>
          <<<grid, block, kDynamicSmemBytes>>>(
              a_map, b_map, c_map, d_sink, d_c, out_ld, store_mode, ktiles,
              mtile, ntile, clock_trace, clock_trace_start, clock_trace_iters);
      return;
    }
    if (tuning.tag == kTuningTag16K) {
      gemm256_tma_tcgen05_kernel<GEMM_GRID_SWIZZLE,
                                 kTuned16KGroupM,
                                 kTuned16KGroupN,
                                 kTuned16KTmaPhaseCycles,
                                 kTuned16KMmaPhaseCycles,
                                 kTuned16KCStoreSwizzle128B,
                                 kSinglePipeline,
                                 kTuningTag16K>
          <<<grid, block, kDynamicSmemBytes>>>(
              a_map, b_map, c_map, d_sink, d_c, out_ld, store_mode, ktiles,
              mtile, ntile, clock_trace, clock_trace_start, clock_trace_iters);
      return;
    }
    if (tuning.tag == kTuningTag32K) {
      gemm256_tma_tcgen05_kernel<GEMM_GRID_SWIZZLE,
                                 kTuned32KGroupM,
                                 kTuned32KGroupN,
                                 kTuned32KTmaPhaseCycles,
                                 kTuned32KMmaPhaseCycles,
                                 kTuned32KCStoreSwizzle128B,
                                 kSinglePipeline,
                                 kTuningTag32K>
          <<<grid, block, kDynamicSmemBytes>>>(
              a_map, b_map, c_map, d_sink, d_c, out_ld, store_mode, ktiles,
              mtile, ntile, clock_trace, clock_trace_start, clock_trace_iters);
      return;
    }
    gemm256_tma_tcgen05_kernel<GEMM_GRID_SWIZZLE,
                               GEMM_GRID_SWIZZLE_M,
                               GEMM_GRID_SWIZZLE_N,
                               GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES,
                               GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES,
                               GEMM_CSTORE_SWIZZLE_128B,
                               kSinglePipeline,
                               kTuningTagGeneric>
        <<<grid, block, kDynamicSmemBytes>>>(
            a_map, b_map, c_map, d_sink, d_c, out_ld, store_mode, ktiles,
            mtile, ntile, clock_trace, clock_trace_start, clock_trace_iters);
    return;
  }
#endif
  gemm256_tma_tcgen05_kernel<0,
                             1,
                             1,
                             GEMM_PIPE1_TMA_PHASE_SHIFT_CYCLES,
                             GEMM_PIPE1_MMA_PHASE_SHIFT_CYCLES,
                             GEMM_CSTORE_SWIZZLE_128B,
                             kSinglePipeline,
                             kTuningTagGeneric>
      <<<grid, block, kDynamicSmemBytes>>>(
          a_map, b_map, c_map, d_sink, d_c, out_ld, store_mode, ktiles,
          mtile, ntile, clock_trace, clock_trace_start, clock_trace_iters);
}

struct CaseResult {
  int size = 0;
  int mtile = 0;
  int ntile = 0;
  int ktiles = 0;
  int ctas = 0;
  int launch_ctas = 0;
  int grid_swizzle = 0;
  int group_m = 1;
  int group_n = 1;
  int pipe1_phase_cycles = 0;
  int pipe1_tma_phase_cycles = 0;
  int pipe1_mma_phase_cycles = 0;
  int cstore_swizzle_128b = 0;
  int tma_a_l2_promotion = 0;
  int tma_b_l2_promotion = 0;
  int tma_c_l2_promotion = 0;
  int input_init_mode = kInputInitMemset;
  int store_mode = kStoreNone;
  float event_ms = 0.0f;
  double wall_ms = 0.0;
  double event_tflops = 0.0;
  double wall_tflops = 0.0;
  uint32_t checksum = 0;
};

CaseResult run_case(int size,
                    int warmup,
                    int iters,
                    int store_mode,
                    int input_init_mode,
                    int persistent_ctas) {
  if (size % kCtaM != 0 || size % kCtaN != 0 || size % kStageK != 0) {
    std::fprintf(stderr,
                 "size must be a multiple of cta_m=%d, cta_n=%d, and "
                 "stage_k=%d; got %d\n",
                 kCtaM, kCtaN, kStageK, size);
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
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_b, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_sink,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  if (store_mode != kStoreNone) {
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_c, 0, static_cast<size_t>(m) * n * sizeof(float)));
  }
  initialize_bf16_inputs(d_a, a_words, d_b, b_words, input_init_mode);
  CUDA_CHECK(cudaMemset(d_sink, 0,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaDeviceSynchronize());

  const GemmTuning tuning = select_gemm_tuning(mtile, ntile, ktiles);
  CUtensorMap a_map{}, b_map{}, c_map{};
  const CUtensorMapL2promotion a_l2_promotion =
      tma_a_l2_promotion_for_size(size);
  const CUtensorMapL2promotion b_l2_promotion =
      tma_b_l2_promotion_for_size(size);
  const CUtensorMapL2promotion c_l2_promotion =
      tma_c_l2_promotion_for_size(size);
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k, a_l2_promotion);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n, b_l2_promotion);
  if (store_mode == kStoreTma) {
    encode_c_row_major_float_tma_map(&c_map, d_c, m, n, c_l2_promotion,
                                     tuning.cstore_swizzle_128b != 0);
  }

  set_gemm_kernel_attributes();

  dim3 grid = launch_grid(mtile, ntile, tuning);
  if (GEMM_PERSISTENT_CTA && persistent_ctas > 0) {
    grid = dim3(std::min(persistent_ctas, ctas), 1, 1);
  }
  dim3 block(kThreads, 1, 1);
  auto launch_gemm = [&]() {
    if (GEMM_PERSISTENT_CTA && !GEMM_PERSISTENT_STATIC_SCHEDULER) {
      CUDA_CHECK(cudaMemsetAsync(d_sink + ctas, 0, sizeof(uint32_t)));
    }
    launch_gemm_kernel(tuning, grid, block, a_map, b_map, c_map, d_sink,
                       d_c, n, store_mode, ktiles, mtile, ntile, nullptr, 0,
                       0);
  };

  for (int i = 0; i < warmup; ++i) {
    launch_gemm();
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start{}, stop{};
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  const auto wall_start = std::chrono::steady_clock::now();
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    launch_gemm();
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
  result.launch_ctas = static_cast<int>(grid.x * grid.y * grid.z);
  result.grid_swizzle = tuning.grid_swizzle;
  result.group_m = tuning.group_m;
  result.group_n = tuning.group_n;
  result.pipe1_phase_cycles = tuning.pipe1_phase_cycles;
  result.pipe1_tma_phase_cycles = tuning.pipe1_tma_phase_cycles;
  result.pipe1_mma_phase_cycles = tuning.pipe1_mma_phase_cycles;
  result.cstore_swizzle_128b = tuning.cstore_swizzle_128b;
  result.tma_a_l2_promotion = static_cast<int>(a_l2_promotion);
  result.tma_b_l2_promotion = static_cast<int>(b_l2_promotion);
  result.tma_c_l2_promotion = static_cast<int>(c_l2_promotion);
  result.input_init_mode = input_init_mode;
  result.store_mode = store_mode;
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
  if (d_c != nullptr) {
    CUDA_CHECK(cudaFree(d_c));
  }
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

ValidateResult run_validation(int size,
                              const char* pattern,
                              int store_mode,
                              int persistent_ctas) {
  if (store_mode == kStoreNone) {
    store_mode = kStoreScalar;
  }
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
  CUDA_CHECK(cudaMalloc(&d_sink,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m) * n * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_sink, 0,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_c, 0, static_cast<size_t>(m) * n * sizeof(float)));

  const GemmTuning tuning = select_gemm_tuning(mtile, ntile, ktiles);
  CUtensorMap a_map{}, b_map{}, c_map{};
  const CUtensorMapL2promotion a_l2_promotion =
      tma_a_l2_promotion_for_size(size);
  const CUtensorMapL2promotion b_l2_promotion =
      tma_b_l2_promotion_for_size(size);
  const CUtensorMapL2promotion c_l2_promotion =
      tma_c_l2_promotion_for_size(size);
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k, a_l2_promotion);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n, b_l2_promotion);
  if (store_mode == kStoreTma) {
    encode_c_row_major_float_tma_map(&c_map, d_c, m, n, c_l2_promotion,
                                     tuning.cstore_swizzle_128b != 0);
  }
  set_gemm_kernel_attributes();

  dim3 grid = launch_grid(mtile, ntile, tuning);
  if (GEMM_PERSISTENT_CTA && persistent_ctas > 0) {
    grid = dim3(std::min(persistent_ctas, ctas), 1, 1);
  }
  dim3 block(kThreads, 1, 1);
  launch_gemm_kernel(tuning, grid, block, a_map, b_map, c_map, d_sink, d_c,
                     n, store_mode, ktiles, mtile, ntile, nullptr, 0, 0);
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
        const int source_row = GEMM_REPEAT_INPUT ? row % kCtaM : row;
        const int source_col =
            GEMM_REPEAT_INPUT
                ? col % (kRepeatBBroadcast ? kMmaN : kCtaN)
                : col;
        const int source_k = GEMM_REPEAT_INPUT ? kk % kStageK : kk;
        ref += static_cast<double>(
                   a_ref[static_cast<size_t>(source_row) * k + source_k]) *
               static_cast<double>(
                   b_ref[static_cast<size_t>(source_k) * n + source_col]);
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
    std::fprintf(stderr,
                 "trace size must be a multiple of cta_m=%d, cta_n=%d, "
                 "and stage_k=%d; got %d\n",
                 kCtaM, kCtaN, kStageK, size);
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
  CUDA_CHECK(cudaMalloc(&d_sink,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_trace,
                        static_cast<size_t>(records_count) *
                            sizeof(ClockTraceRecord)));
  CUDA_CHECK(cudaMemset(d_a, 0x3f, a_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_b, 0x11, b_words * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_sink, 0,
                        (static_cast<size_t>(ctas) + 1) * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemset(d_trace, 0,
                        static_cast<size_t>(records_count) *
                            sizeof(ClockTraceRecord)));

  CUtensorMap a_map{}, b_map{}, c_map{};
  const CUtensorMapL2promotion a_l2_promotion =
      tma_a_l2_promotion_for_size(size);
  const CUtensorMapL2promotion b_l2_promotion =
      tma_b_l2_promotion_for_size(size);
  encode_a_row_major_sw128_tma_map(&a_map, d_a, m, k, a_l2_promotion);
  encode_b_row_major_sw128_k16_tma_map(&b_map, d_b, k, n, b_l2_promotion);
  set_gemm_kernel_attributes();

  const GemmTuning tuning = select_gemm_tuning(mtile, ntile, ktiles);
  dim3 grid = launch_grid(mtile, ntile, tuning);
  dim3 block(kThreads, 1, 1);
  auto launch_trace_gemm = [&](ClockTraceRecord* trace, int trace_start,
                               int trace_iters_arg) {
    if (GEMM_PERSISTENT_CTA && !GEMM_PERSISTENT_STATIC_SCHEDULER) {
      CUDA_CHECK(cudaMemsetAsync(d_sink + ctas, 0, sizeof(uint32_t)));
    }
    launch_gemm_kernel(tuning, grid, block, a_map, b_map, c_map, d_sink,
                       nullptr, 0, kStoreNone, ktiles, mtile, ntile, trace,
                       trace_start, trace_iters_arg);
  };
  for (int i = 0; i < args.warmup; ++i) {
    launch_trace_gemm(nullptr, 0, 0);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  launch_trace_gemm(d_trace, args.clock_trace_start, trace_iters);
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
    ValidateResult r =
        run_validation(args.validate_size, args.validate_pattern,
                       kStoreTma, args.persistent_ctas);
    std::printf("validation size=%d pattern=%s store_mode=%s status=%s "
                "max_abs=%g max_rel=%g bad=%zu\n",
                args.validate_size, args.validate_pattern,
                store_mode_name(kStoreTma), r.ok ? "ok" : "fail",
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
               "size,m,n,k,cta_m,cta_n,stage_k,mtile,ntile,ktiles,ctas,launch_ctas,"
               "warmup,iters,grid_swizzle,group_m,group_n,pipe1_phase_cycles,"
               "pipe1_tma_phase_cycles,pipe1_mma_phase_cycles,"
               "cstore_swizzle_128b,single_pipeline,single_pipeline_ntile_128,"
               "single_pipeline_wide_mma,single_pipeline_split_tma,"
               "single_pipeline_interleave_pipes,"
               "tma_a_l2_promotion,tma_b_l2_promotion,tma_c_l2_promotion,"
               "input_init,store_mode,dynamic_smem_bytes,event_ms,"
               "wall_ms,event_TFLOPS,wall_TFLOPS,checksum,device\n");

  std::printf("device=%d name=\"%s\" cc=%d.%d dynamic_smem=%d bytes\n",
              args.device, prop.name, prop.major, prop.minor, kDynamicSmemBytes);
  std::printf("mode=bf16_tcgen05_compute_sink layout=row_major_sw128 "
              "cta_tile=%dx%d stage_k=%d stages=%d pipes=%d "
              "pipe1_phase_shift_cycles=%d pipe1_phase_shift_cycles_8k=%d "
              "pipe1_tma_phase_shift_cycles=%d "
              "pipe1_mma_phase_shift_cycles=%d "
              "grid_swizzle=%d "
              "grid_swizzle_m=%d grid_swizzle_n=%d "
              "grid_swizzle_min_tiles=%d "
              "tuned4k=%dx%d:%d/%d tuned8k=%dx%d:%d/%d "
              "tuned16k=%dx%d:%d/%d "
              "tuned32k=%dx%d:%d/%d "
              "tuned_cstore_swizzle_128b=%d/%d/%d/%d "
              "tma_l2_promotion_a=%d tma_l2_promotion_b=%d "
              "tma_l2_promotion_c=%d tuned4k_tma_l2=%d/%d/%d "
              "tuned8k_tma_l2=%d/%d/%d "
              "tuned32k_tma_l2=%d/%d/%d "
              "cstore_swizzle_128b=%d cstore_vectorize_smem=%d "
              "single_pipeline=%d single_pipeline_ntile_128=%d "
              "single_pipeline_wide_mma=%d "
              "single_pipeline_split_tma=%d single_pipeline_interleave_pipes=%d "
              "wide_b_tma=%d repeat_b_broadcast=%d "
              "input_init=%s formula_scale=%g "
              "persistent_ctas=%d "
              "persistent_kernel=%d "
              "persistent_macro=%dx%d persistent_8k_macro=%dx%d "
              "persistent_32k_macro=%dx%d "
              "persistent_order_local_m_fast=%d macro_n_fast=%d "
              "persistent_static_scheduler=%d fused_cstore_staging=%d "
              "elide_dense_sink=%d epilogue_mode=%d epilogue_warps=%d "
              "store_mode=%s c_type=%s\n",
              kCtaM, kCtaN, kStageK, kStages, kPipes,
              kPipe1PhaseShiftCycles, kPipe1PhaseShiftCycles8K,
              kPipe1TmaPhaseShiftCycles, kPipe1MmaPhaseShiftCycles,
              GEMM_GRID_SWIZZLE,
              GEMM_GRID_SWIZZLE_M, GEMM_GRID_SWIZZLE_N,
              GEMM_GRID_SWIZZLE_MIN_TILES,
              kTuned4KGroupM, kTuned4KGroupN, kTuned4KTmaPhaseCycles,
              kTuned4KMmaPhaseCycles,
              kTuned8KGroupM, kTuned8KGroupN, kTuned8KTmaPhaseCycles,
              kTuned8KMmaPhaseCycles,
              kTuned16KGroupM, kTuned16KGroupN, kTuned16KTmaPhaseCycles,
              kTuned16KMmaPhaseCycles,
              kTuned32KGroupM, kTuned32KGroupN, kTuned32KTmaPhaseCycles,
              kTuned32KMmaPhaseCycles,
              kTuned4KCStoreSwizzle128B,
              kTuned8KCStoreSwizzle128B, kTuned16KCStoreSwizzle128B,
              kTuned32KCStoreSwizzle128B,
              static_cast<int>(GEMM_TMA_A_L2_PROMOTION),
              static_cast<int>(GEMM_TMA_B_L2_PROMOTION),
              static_cast<int>(GEMM_TMA_C_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_4K_TMA_A_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_4K_TMA_B_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_4K_TMA_C_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_8K_TMA_A_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_8K_TMA_B_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_8K_TMA_C_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_32K_TMA_A_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_32K_TMA_B_L2_PROMOTION),
              static_cast<int>(GEMM_TUNED_32K_TMA_C_L2_PROMOTION),
              GEMM_CSTORE_SWIZZLE_128B, GEMM_CSTORE_VECTORIZE_SMEM,
              kSinglePipeline, kSinglePipelineNtile128,
              kSinglePipelineWideMma, kSinglePipelineSplitTma,
              kSinglePipelineInterleavePipes,
              GEMM_WIDE_B_TMA, static_cast<int>(kRepeatBBroadcast),
              input_init_mode_name(args.input_init_mode), kFormulaInitScale,
              args.persistent_ctas,
              GEMM_PERSISTENT_CTA,
              GEMM_PERSISTENT_MACRO_M, GEMM_PERSISTENT_MACRO_N,
              GEMM_PERSISTENT_8K_MACRO_M, GEMM_PERSISTENT_8K_MACRO_N,
              GEMM_PERSISTENT_32K_MACRO_M, GEMM_PERSISTENT_32K_MACRO_N,
              GEMM_PERSISTENT_LOCAL_M_FAST, GEMM_PERSISTENT_MACRO_N_FAST,
              GEMM_PERSISTENT_STATIC_SCHEDULER,
              GEMM_FUSED_CSTORE_STAGING, GEMM_ELIDE_DENSE_SINK,
              GEMM_EPILOGUE_MODE, kEpilogueWarps,
              store_mode_name(kStoreTma),
              "fp32");

  for (int size : args.sizes) {
    CaseResult r =
        run_case(size, args.warmup, args.iters, kStoreTma,
                 args.input_init_mode, args.persistent_ctas);
    std::printf("size=%d mtile=%d ntile=%d ktiles=%d ctas=%d launch_ctas=%d "
                "grid_swizzle=%d group=%dx%d pipe1_phase=%d "
                "pipe1_tma_phase=%d pipe1_mma_phase=%d "
                "cstore_swizzle_128b=%d single_pipeline=%d "
                "single_pipeline_ntile_128=%d single_pipeline_wide_mma=%d "
                "single_pipeline_split_tma=%d single_pipeline_interleave_pipes=%d "
                "tma_l2=%d/%d/%d input_init=%s "
                "store_mode=%s event_ms=%.6f wall_ms=%.6f "
                "event_TFLOPS=%.3f wall_TFLOPS=%.3f checksum=%08x\n",
                r.size, r.mtile, r.ntile, r.ktiles, r.ctas, r.launch_ctas,
                r.grid_swizzle,
                r.group_m, r.group_n, r.pipe1_phase_cycles,
                r.pipe1_tma_phase_cycles, r.pipe1_mma_phase_cycles,
                r.cstore_swizzle_128b, kSinglePipeline,
                kSinglePipelineNtile128, kSinglePipelineWideMma,
                kSinglePipelineSplitTma,
                kSinglePipelineInterleavePipes,
                r.tma_a_l2_promotion, r.tma_b_l2_promotion,
                r.tma_c_l2_promotion,
                input_init_mode_name(r.input_init_mode),
                store_mode_name(r.store_mode), r.event_ms, r.wall_ms,
                r.event_tflops, r.wall_tflops, r.checksum);
    std::fprintf(csv,
                 "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%d,%.6f,"
                 "%.6f,%.3f,%.3f,%08x,%s\n",
                 r.size, r.size, r.size, r.size, kCtaM, kCtaN, kStageK,
                 r.mtile, r.ntile, r.ktiles, r.ctas, r.launch_ctas,
                 args.warmup, args.iters,
                 r.grid_swizzle, r.group_m, r.group_n, r.pipe1_phase_cycles,
                 r.pipe1_tma_phase_cycles, r.pipe1_mma_phase_cycles,
                 r.cstore_swizzle_128b, kSinglePipeline,
                 kSinglePipelineNtile128, kSinglePipelineWideMma,
                 kSinglePipelineSplitTma,
                 kSinglePipelineInterleavePipes,
                 r.tma_a_l2_promotion, r.tma_b_l2_promotion,
                 r.tma_c_l2_promotion,
                 input_init_mode_name(r.input_init_mode),
                 store_mode_name(r.store_mode),
                 kDynamicSmemBytes,
                 r.event_ms, r.wall_ms, r.event_tflops, r.wall_tflops,
                 r.checksum, prop.name);
    std::fflush(csv);
  }

  std::fclose(csv);
  return 0;
}
