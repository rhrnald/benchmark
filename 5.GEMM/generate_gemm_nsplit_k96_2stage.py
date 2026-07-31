#!/usr/bin/env python3
"""Generate the two-stage K96 N-split GEMM ablation."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "c0395009e00fcdd2f0a5266ee3cc79e308b51e35e7510ad671683ec7dcd67107"
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str, start: str, end: str, replacement: str, label: str
) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise RuntimeError(
            f"{label}: start={text.count(start)} end={text.count(end)}"
        )
    begin = text.index(start)
    finish = text.index(end, begin)
    return text[:begin] + replacement + text[finish:]


def generate(source: str) -> str:
    text = source
    text = replace_once(
        text,
        """static constexpr int kStageK = 64;
static constexpr int kStages = 3;""",
        """static constexpr int kStageK = 96;
static constexpr int kStages = 2;
static constexpr int kASlabK = 32;
static constexpr int kASlabs = kStageK / kASlabK;""",
        "K96 two-stage constants",
    )

    descriptor_start = (
        """__device__ __forceinline__ uint64_t make_stage_a_smem_desc"""
    )
    descriptor_end = (
        """__host__ __device__ __forceinline__ uint64_t
make_sw128_major_mn_smem_desc"""
    )
    descriptor_new = """__host__ __device__ __forceinline__ uint64_t
make_sw64_major_k_smem_desc(uint32_t matrix_start_addr, int mma) {
  constexpr uint64_t desc_base =
      (static_cast<uint64_t>(1u) << 16) |
      (static_cast<uint64_t>(32u) << 32) |
      (static_cast<uint64_t>(1u) << 46) |
      (static_cast<uint64_t>(4u) << 61);
  const uint32_t addr16 = ((matrix_start_addr & ~0xFu) >> 4) +
                          static_cast<uint32_t>(mma) * (32u >> 4);
  return desc_base | static_cast<uint64_t>(addr16 & 0x3fffu);
}

__device__ __forceinline__ uint64_t
make_stage_a_smem_desc(uint32_t *a_smem, int mblock, int mma) {
  constexpr int kMmasPerSlab = kASlabK / kMmaK;
  constexpr int kSlabWords = kCtaM * kASlabK / 2;
  constexpr int kMBlockWords = kMmaM * kASlabK / 2;
  const int slab = mma / kMmasPerSlab;
  const int mma_in_slab = mma - slab * kMmasPerSlab;
  uint32_t *matrix =
      a_smem + slab * kSlabWords + mblock * kMBlockWords;
  return make_sw64_major_k_smem_desc(
      smem_ptr_u32(matrix), mma_in_slab);
}

"""
    text = replace_between(
        text,
        descriptor_start,
        descriptor_end,
        descriptor_new,
        "K32 A slab descriptors",
    )

    issue_start = """__device__ __forceinline__ void issue_a_stage_tma"""
    issue_end = """__device__ __forceinline__ void
issue_b_pipe_stage_tma"""
    issue_new = """__device__ __forceinline__ void issue_a_stage_tma(
    const CUtensorMap *a_map, uint32_t *a_smem,
    uint64_t (*ready)[kStages], int stage, int tile_m, int ktile) {
  constexpr int kSlabWords = kCtaM * kASlabK / 2;
  constexpr int kSlabBytes =
      kSlabWords * static_cast<int>(sizeof(uint32_t));
  const int a_row = tile_m * kCtaM;
  const int a_col_words = ktile * (kStageK / 2);
#pragma unroll
  for (int slab = 0; slab < kASlabs; ++slab) {
    mbarrier_expect_tx(&ready[slab][stage], kSlabBytes);
    tma_load_2d(a_map, smem_ptr_u32(a_smem + slab * kSlabWords),
                &ready[slab][stage],
                a_col_words + slab * (kASlabK / 2), a_row);
  }
}

"""
    text = replace_between(
        text, issue_start, issue_end, issue_new, "three A slab TMA"
    )

    text = replace_once(
        text,
        """  __shared__ uint64_t a_ready[kStages];""",
        """  __shared__ uint64_t a_ready[kASlabs][kStages];""",
        "A ready barriers",
    )
    text = replace_once(
        text,
        """    for (int s = 0; s < kStages; ++s) {
      mbarrier_init(&a_ready[s], 1);
#pragma unroll
      for (int p = 0; p < kPipes; ++p) {""",
        """    for (int s = 0; s < kStages; ++s) {
#pragma unroll
      for (int slab = 0; slab < kASlabs; ++slab)
        mbarrier_init(&a_ready[slab][s], 1);
#pragma unroll
      for (int p = 0; p < kPipes; ++p) {""",
        "A barrier init",
    )
    text = replace_once(
        text,
        """        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);""",
        """        issue_a_stage_tma(
            &a_map, a_smem, a_ready, stage, tile_m, kt);""",
        "A producer call",
    )
    text = replace_once(
        text,
        """        mbarrier_wait(&b_ready[pipe][stage], tma_phase);
        mbarrier_wait(&a_ready[stage], tma_phase);

#pragma unroll""",
        """        mbarrier_wait(&b_ready[pipe][stage], tma_phase);
#pragma unroll
        for (int slab = 0; slab < kASlabs; ++slab)
          mbarrier_wait(&a_ready[slab][stage], tma_phase);

#pragma unroll""",
        "A consumer waits",
    )

    text = replace_once(
        text,
        """  const cuuint32_t box_dim[2] = {kStageK / 2, kCtaM};""",
        """  const cuuint32_t box_dim[2] = {kASlabK / 2, kCtaM};""",
        "A K32 tensor map",
    )
    text = replace_once(
        text,
        """                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(a_row_major_sw128)");""",
        """                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_64B,
                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled(a_row_major_sw64_k32)");""",
        "A SW64 tensor map",
    )

    text = replace_once(
        text,
        """      args.validate_size % kCtaM != 0 || args.validate_size % kCtaN != 0 ||
      args.validate_size % kStageK != 0) {""",
        """      args.validate_size % kCtaM != 0 || args.validate_size % kCtaN != 0 ||
      args.validate_size % kMmaK != 0) {""",
        "validation divisibility",
    )
    text = replace_once(
        text,
        """"validate size must be <= %d and a positive multiple of "
                 "cta_m=%d, cta_n=%d, and stage_k=%d\\n",
                 kMaxValidationSize, kCtaM, kCtaN, kStageK);""",
        """"validate size must be <= %d and a positive multiple of "
                 "cta_m=%d, cta_n=%d, and mma_k=%d\\n",
                 kMaxValidationSize, kCtaM, kCtaN, kMmaK);""",
        "validation message",
    )

    # Ceil-div stage counts: 16384 -> 171, 512 -> 6, 256 -> 3.
    for old, new in (
        ("<256, 64, 64>", "<171, 64, 64>"),
        ("<8, 2, 2>", "<6, 2, 2>"),
        ("<4, 1, 1>", "<3, 1, 1>"),
        ("ktiles == 256", "ktiles == 171"),
        ("ktiles == 8", "ktiles == 6"),
        ("ktiles == 4", "ktiles == 3"),
    ):
        text = text.replace(old, new)
    if text.count("const int ktiles = k / kStageK;") != 2:
        raise RuntimeError("expected two runtime ktiles calculations")
    text = text.replace(
        "const int ktiles = k / kStageK;",
        "const int ktiles = (k + kStageK - 1) / kStageK;",
    )
    text = replace_once(
        text,
        """"phase=0/0 consumer_wait=b_then_a c_store=tma_fp32_sw128 l2_promotion=none """,
        """"ablation=k96_2stage phase=0/0 consumer_wait=b_then_a c_store=tma_fp32_sw128 l2_promotion=none """,
        "banner",
    )
    text = replace_once(
        text,
        """"device=%d name=\\"%s\\" cc=%d.%d cta=256x256 stage_k=64 "
      "stages=3 pipes=2 persistent_ctas=%d""",
        """"device=%d name=\\"%s\\" cc=%d.%d cta=256x256 stage_k=96 "
      "stages=2 pipes=2 persistent_ctas=%d""",
        "banner stage shape",
    )
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
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            f"refusing unaudited source: expected {EXPECTED_SOURCE_SHA256}, "
            f"got {source_hash}"
        )
    generated = generate(source_bytes.decode())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    print(
        f"source_sha256={source_hash} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
