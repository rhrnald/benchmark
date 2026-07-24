#!/usr/bin/env python3
"""Generate the hash-gated N-split C-transpose MMA prototype.

The generated kernel preserves the canonical A/B TMA traffic and the compact
runtime-pipe consumer.  It computes each logical 256x128 output half as its
transpose with one m128n256k16 operation per K16, then transposes the FP32
TMEM result while staging C for the existing TMA store.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca"
)

CONSTANT_ANCHOR = """static constexpr int kTmemTileStride = 128;
"""

TRANSPOSE_CONSTANTS = """static constexpr int kTmemTileStride = 128;
static constexpr int kTransposeMmaM = 128;
static constexpr int kTransposeMmaN = 256;
static constexpr int kTransposeTmemPipeStride = 256;

static_assert(kTransposeMmaM == kMmaM);
static_assert(kTransposeMmaN == kCtaM);
static_assert(kPipes * kTransposeTmemPipeStride == 512);
"""

IDESC_HELPER_ANCHOR = """__device__ __forceinline__ void mbarrier_init"""

IDESC_HELPER = r"""// Compute C_half^T = B_half^T * A^T.  The physical B panel is
// MN-major when consumed as MMA A, while the physical A tile is K-major when
// consumed as MMA B.
__host__ __device__ __forceinline__ uint32_t
make_bf16_transpose_idesc() {
  uint32_t desc = 0;
  desc |= 1u << 4;  // C format: F32.
  desc |= 1u << 7;  // A format: BF16.
  desc |= 1u << 10; // B format: BF16.
  desc |= 1u << 15; // MMA A (physical B): MN-major.
  desc |= static_cast<uint32_t>(kTransposeMmaN >> 3) << 17;
  desc |= static_cast<uint32_t>(kTransposeMmaM >> 4) << 24;
  return desc;      // MMA B (physical A) remains K-major: bit 16 is zero.
}

"""

IDESC_USE = """  const uint32_t idesc = make_bf16_idesc() | (1u << 16);
"""

TRANSPOSE_IDESC_USE = (
    "  const uint32_t idesc = make_bf16_transpose_idesc();\n"
)

STAGE_C_START = """__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
"""

STAGE_C_END = """__device__ __forceinline__ void
issue_float_c_chunk_tma"""

TRANSPOSE_STAGE_C = r"""// TMEM holds two C-half transposes:
//   pipe 0: rows N[0:128],   columns M[0:256], TMEM columns [0,256)
//   pipe 1: rows N[128:256], columns M[0:256], TMEM columns [256,512)
//
// One warp owns 32 TMEM rows.  For each output M row, its 32 lanes write 32
// adjacent logical N values.  cstore_sw128_float_word_offset applies only a
// lane permutation within that 128-byte row segment, so every scalar warp
// store is conflict-free and covers one complete 128-byte shared-memory line.
__device__ __forceinline__ void
stage_float_c_chunk(uint32_t tmem_base, uint32_t *c_smem, int chunk_m,
                    int chunk_n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  if (warp_id < kCStoreWarps) {
    uint32_t r[64];
    const int n_band = warp_id * 32;
    const uint32_t pipe_tmem =
        tmem_base + chunk_n * kTransposeTmemPipeStride;
#pragma unroll
    for (int load = 0; load < kCStoreChunkM / 64; ++load) {
      const uint32_t global_m_base =
          static_cast<uint32_t>(chunk_m * kCStoreChunkM + load * 64);
      const uint32_t row_taddr =
          pipe_tmem + (static_cast<uint32_t>(n_band) << 16) + global_m_base;
      tcgen05_ld_32x32b_x64(r, row_taddr);
      tcgen05_wait_ld();
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        const int local_m = load * 64 + i;
        const int local_n = n_band + lane;
        c_smem[cstore_sw128_float_word_offset(local_m, local_n)] = r[i];
      }
    }
  }
#else
  (void)tmem_base;
  (void)c_smem;
  (void)chunk_m;
  (void)chunk_n;
#endif
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

TRANSPOSE_CONSUMER_BLOCK = """    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
#pragma unroll 1
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
          // B_pipe^T is the m128k16 MMA A operand.  A^T is the k16n256
          // MMA B operand.  Both descriptors address the original TMA
          // payloads; only their operand roles and instruction majors change.
          const uint64_t bt_desc = make_sw128_major_mn_smem_desc(
              smem_ptr_u32(b_smem), kk);
          const uint64_t at_desc = make_sw128_major_k_smem_desc(
              smem_ptr_u32(a_smem), kk);
          const bool input_d = (kt != 0) || (kk != 0);
          tcgen05_mma_bf16_ss(
              tmem_base + pipe * kTransposeTmemPipeStride, bt_desc, at_desc,
              idesc, input_d);
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

BANNER_ANCHOR = (
    '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'
)

TRANSPOSE_BANNER = (
    '"nsplit_variant=transpose_mma compute=c_transpose "'
    '\n      "mma=m128n256k16 epilogue=scalar_coalesced_transpose "'
    '\n      "phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_region_once(
    text: str, start: str, end: str, replacement: str, label: str
) -> str:
    start_count = text.count(start)
    if start_count != 1:
        raise RuntimeError(
            f"{label}: expected one start anchor, found {start_count}"
        )
    start_pos = text.index(start)
    end_pos = text.find(end, start_pos + len(start))
    if end_pos < 0:
        raise RuntimeError(f"{label}: end anchor not found")
    return text[:start_pos] + replacement + text[end_pos:]


def audit_generated(text: str) -> None:
    required = (
        "make_bf16_transpose_idesc()",
        "desc |= 1u << 15;",
        "kTransposeTmemPipeStride = 256",
        "pipe_tmem + (static_cast<uint32_t>(n_band) << 16) + global_m_base",
        "make_sw128_major_mn_smem_desc(\n              smem_ptr_u32(b_smem), kk)",
        "make_sw128_major_k_smem_desc(\n              smem_ptr_u32(a_smem), kk)",
        "tmem_base + pipe * kTransposeTmemPipeStride",
        "const int pipe = warp_id - 2;\n#pragma unroll 1\n"
        "      for (int kt = 0; kt < ktiles; ++kt)",
        "nsplit_variant=transpose_mma",
    )
    for fragment in required:
        if fragment not in text:
            raise RuntimeError(f"generated audit: missing {fragment!r}")
    if IDESC_USE in text:
        raise RuntimeError("generated audit: old B-major-MN idesc use remains")
    if text.count("if ((warp_id == 2 || warp_id == 3) && lane0)") != 1:
        raise RuntimeError("generated audit: runtime-pipe consumer was duplicated")


def generate(source: str) -> str:
    text = replace_once(
        source,
        CONSTANT_ANCHOR,
        TRANSPOSE_CONSTANTS,
        "transpose constants",
    )
    text = replace_once(
        text,
        IDESC_HELPER_ANCHOR,
        IDESC_HELPER + IDESC_HELPER_ANCHOR,
        "transpose idesc helper",
    )
    text = replace_region_once(
        text,
        STAGE_C_START,
        STAGE_C_END,
        TRANSPOSE_STAGE_C,
        "transpose C staging",
    )
    text = replace_once(
        text,
        IDESC_USE,
        TRANSPOSE_IDESC_USE,
        "transpose idesc use",
    )
    text = replace_once(
        text,
        CONSUMER_BLOCK,
        TRANSPOSE_CONSUMER_BLOCK,
        "transpose consumer mainloop",
    )
    text = replace_once(
        text,
        BANNER_ANCHOR,
        TRANSPOSE_BANNER,
        "host configuration label",
    )
    audit_generated(text)
    return text


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
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited N-split source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )

    generated = generate(source_bytes.decode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(
        f"variant=nsplit_transpose source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
