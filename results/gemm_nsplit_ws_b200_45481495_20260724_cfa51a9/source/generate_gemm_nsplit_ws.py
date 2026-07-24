#!/usr/bin/env python3
"""Generate hash-gated static-consumer N-split controls and WS candidates."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca"
)

OPCODE_HELPERS_ANCHOR = """__host__ __device__ __forceinline__ int
cstore_sw128_float_word_offset"""

OPCODE_HELPERS = r"""__device__ __forceinline__ void
tcgen05_mma_ws_b0_fill(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc,
                       uint32_t idesc, bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  asm volatile("{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
               "tcgen05.mma.ws.cta_group::1.kind::f16."
               "collector::b0::fill [%0], %1, %2, %3, pred; }" ::
                   "r"(d_taddr),
               "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p)
               : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

__device__ __forceinline__ void
tcgen05_mma_ws_b0_lastuse(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc,
                          uint32_t idesc, bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  asm volatile("{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
               "tcgen05.mma.ws.cta_group::1.kind::f16."
               "collector::b0::lastuse [%0], %1, %2, %3, pred; }" ::
                   "r"(d_taddr),
               "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p)
               : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

__device__ __forceinline__ void
tcgen05_mma_ws_b1_fill(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc,
                       uint32_t idesc, bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  asm volatile("{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
               "tcgen05.mma.ws.cta_group::1.kind::f16."
               "collector::b1::fill [%0], %1, %2, %3, pred; }" ::
                   "r"(d_taddr),
               "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p)
               : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

__device__ __forceinline__ void
tcgen05_mma_ws_b1_lastuse(uint32_t d_taddr, uint64_t a_desc, uint64_t b_desc,
                          uint32_t idesc, bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  asm volatile("{ .reg .pred pred; setp.ne.u32 pred, %4, 0; "
               "tcgen05.mma.ws.cta_group::1.kind::f16."
               "collector::b1::lastuse [%0], %1, %2, %3, pred; }" ::
                   "r"(d_taddr),
               "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(p)
               : "memory");
#else
  (void)d_taddr;
  (void)a_desc;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

"""

CONSUMER_HELPER_ANCHOR = """__global__ __launch_bounds__(kThreads, 1) void gemm256_bf16_16k_kernel"""

CONSUMER_HELPER = r"""template <int Pipe, bool UseWs>
__device__ __forceinline__ void run_nsplit_static_consumer(
    uint32_t *smem, uint64_t *a_ready, uint64_t *b_ready,
    uint64_t *mma_done, uint32_t tmem_base, uint32_t idesc, int ktiles,
    int stage_epoch_base) {
  static_assert(Pipe == 0 || Pipe == 1);
#pragma unroll 1
  for (int kt = 0; kt < ktiles; ++kt) {
    const int stage_epoch = stage_epoch_base + kt;
    const int stage = stage_epoch % kStages;
    const uint32_t tma_phase =
        static_cast<uint32_t>((stage_epoch / kStages) & 1);
    uint32_t *stage_smem = smem + stage * kStageWords;
    uint32_t *a_smem = stage_smem;
    uint32_t *b_smem =
        stage_smem + kAStageWords + Pipe * kBPipeWords;

    mbarrier_wait(&a_ready[stage], tma_phase);
    mbarrier_wait(&b_ready[Pipe * kStages + stage], tma_phase);

#pragma unroll
    for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
      const uint32_t b0 = smem_ptr_u32(b_smem);
      const uint64_t b_desc = make_sw128_major_mn_smem_desc(b0, kk);
      const bool input_d = (kt != 0) || (kk != 0);
      const uint64_t a_top = make_stage_a_smem_desc(a_smem, 0, kk);
      const uint64_t a_bottom = make_stage_a_smem_desc(a_smem, 1, kk);
      const uint32_t c_top =
          tmem_base + Pipe * kTmemTileStride;
      const uint32_t c_bottom =
          tmem_base + (2 + Pipe) * kTmemTileStride;
      if constexpr (UseWs) {
        if constexpr (Pipe == 0) {
          tcgen05_mma_ws_b0_fill(c_top, a_top, b_desc, idesc, input_d);
          tcgen05_mma_ws_b0_lastuse(c_bottom, a_bottom, b_desc, idesc,
                                   input_d);
        } else {
          tcgen05_mma_ws_b1_fill(c_top, a_top, b_desc, idesc, input_d);
          tcgen05_mma_ws_b1_lastuse(c_bottom, a_bottom, b_desc, idesc,
                                   input_d);
        }
      } else {
        tcgen05_mma_bf16_ss(c_top, a_top, b_desc, idesc, input_d);
        tcgen05_mma_bf16_ss(c_bottom, a_bottom, b_desc, idesc, input_d);
      }
    }
    tcgen05_commit(&mma_done[Pipe * kStages + stage]);
  }
  const int last_stage_epoch = stage_epoch_base + ktiles - 1;
  const int last_stage = last_stage_epoch % kStages;
  const uint32_t last_phase =
      static_cast<uint32_t>((last_stage_epoch / kStages) & 1);
  mbarrier_wait(&mma_done[Pipe * kStages + last_stage], last_phase);
}

"""

CONSUMER_BLOCK = """    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
        uint32_t *stage_smem = smem + stage * kStageWords;
        uint32_t *a_smem = stage_smem;
        uint32_t *b_smem = stage_smem + kAStageWords + pipe * kBPipeWords;

        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);

#pragma unroll
        for (int kk = 0; kk < kStageK / kMmaK; ++kk) {
          const uint32_t b0 = smem_ptr_u32(b_smem);
          const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
          const bool input_d = (kt != 0) || (kk != 0);
#pragma unroll
          for (int mblock = 0; mblock < kMBlocks; ++mblock) {
            const uint64_t a_desc = make_stage_a_smem_desc(a_smem, mblock, kk);
            const int c_tile = mblock * 2 + pipe;
            tcgen05_mma_bf16_ss(tmem_base + c_tile * kTmemTileStride, a_desc,
                                b0_desc, idesc, input_d);
          }
        }
        tcgen05_commit(&mma_done[pipe][stage]);
      }
      const int last_stage_epoch = stage_epoch_base + ktiles - 1;
      const int last_stage = last_stage_epoch % kStages;
      const uint32_t last_phase =
          static_cast<uint32_t>((last_stage_epoch / kStages) & 1);
      mbarrier_wait(&mma_done[pipe][last_stage], last_phase);
    }
"""

CONSUMER_REPLACEMENT = """    if (warp_id == 2 && lane0) {
      run_nsplit_static_consumer<0, @USE_WS@>(
          smem, &a_ready[0], &b_ready[0][0], &mma_done[0][0], tmem_base,
          idesc, ktiles, stage_epoch_base);
    }
    if (warp_id == 3 && lane0) {
      run_nsplit_static_consumer<1, @USE_WS@>(
          smem, &a_ready[0], &b_ready[0][0], &mma_done[0][0], tmem_base,
          idesc, ktiles, stage_epoch_base);
    }
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def generate(source: str, variant: str) -> str:
    use_ws = variant == "ws-b01"
    replacement = CONSUMER_REPLACEMENT.replace(
        "@USE_WS@", "true" if use_ws else "false"
    )
    text = replace_once(
        source,
        OPCODE_HELPERS_ANCHOR,
        OPCODE_HELPERS + OPCODE_HELPERS_ANCHOR,
        "WS opcode helper insertion",
    )
    text = replace_once(
        text,
        CONSUMER_HELPER_ANCHOR,
        CONSUMER_HELPER + CONSUMER_HELPER_ANCHOR,
        "WS consumer helper insertion",
    )
    text = replace_once(
        text, CONSUMER_BLOCK, replacement, "consumer mainloop"
    )
    label = "ws_b01" if use_ws else "static_control"
    return replace_once(
        text,
        '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "',
        f'"nsplit_variant={label} phase=0/0 '
        'c_store=tma_fp32_sw128 l2_promotion=none "',
        "host configuration label",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent
        / "baseline"
        / "gemm256_bf16_16k.cu",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--variant",
        choices=("static-control", "ws-b01"),
        default="ws-b01",
    )
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited N-split source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )

    generated = generate(source_bytes.decode("utf-8"), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(
        f"variant=nsplit_{args.variant.replace('-', '_')} "
        f"source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
