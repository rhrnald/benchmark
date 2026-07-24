#!/usr/bin/env python3
"""Generate a hash-gated N-split candidate with paired ordinary MMA PTX."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca"
)

HELPER_ANCHOR = """__host__ __device__ __forceinline__ int
cstore_sw128_float_word_offset"""

PAIR_HELPER = r"""__device__ __forceinline__ void
tcgen05_mma_bf16_ss_pair(uint32_t d_top, uint32_t d_bottom,
                         uint64_t a_top, uint64_t a_bottom,
                         uint64_t b_desc, uint32_t idesc, bool input_d) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t p = input_d ? 1u : 0u;
  uint32_t mask[4] = {0, 0, 0, 0};
  asm volatile(
      "{ .reg .pred pred; setp.ne.u32 pred, %6, 0; "
      "tcgen05.mma.cta_group::1.kind::f16 "
      "[%0], %2, %4, %5, {%7, %8, %9, %10}, pred; "
      "tcgen05.mma.cta_group::1.kind::f16 "
      "[%1], %3, %4, %5, {%7, %8, %9, %10}, pred; }" ::
          "r"(d_top),
      "r"(d_bottom), "l"(a_top), "l"(a_bottom), "l"(b_desc), "r"(idesc),
      "r"(p), "r"(mask[0]), "r"(mask[1]), "r"(mask[2]), "r"(mask[3])
      : "memory");
#else
  (void)d_top;
  (void)d_bottom;
  (void)a_top;
  (void)a_bottom;
  (void)b_desc;
  (void)idesc;
  (void)input_d;
#endif
}

"""

CONSUMER_PAIR = """          const uint32_t b0 = smem_ptr_u32(b_smem);
          const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
          const bool input_d = (kt != 0) || (kk != 0);
          const uint64_t a_top = make_stage_a_smem_desc(a_smem, 0, kk);
          const uint64_t a_bottom = make_stage_a_smem_desc(a_smem, 1, kk);
          const uint32_t c_top =
              tmem_base + pipe * kTmemTileStride;
          const uint32_t c_bottom =
              tmem_base + (2 + pipe) * kTmemTileStride;
          tcgen05_mma_bf16_ss_pair(c_top, c_bottom, a_top, a_bottom,
                                   b0_desc, idesc, input_d);
"""

CONSUMER_LOOP = """          const uint32_t b0 = smem_ptr_u32(b_smem);
          const uint64_t b0_desc = make_sw128_major_mn_smem_desc(b0, kk);
          const bool input_d = (kt != 0) || (kk != 0);
#pragma unroll
          for (int mblock = 0; mblock < kMBlocks; ++mblock) {
            const uint64_t a_desc = make_stage_a_smem_desc(a_smem, mblock, kk);
            const int c_tile = mblock * 2 + pipe;
            tcgen05_mma_bf16_ss(tmem_base + c_tile * kTmemTileStride, a_desc,
                                b0_desc, idesc, input_d);
          }
"""

CONSUMER_OUTER = """    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
      for (int kt = 0; kt < ktiles; ++kt) {
"""

CONSUMER_OUTER_U1 = """    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
#pragma unroll 1
      for (int kt = 0; kt < ktiles; ++kt) {
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def generate(source: str) -> str:
    text = replace_once(
        source,
        HELPER_ANCHOR,
        PAIR_HELPER + HELPER_ANCHOR,
        "paired MMA helper insertion",
    )
    text = replace_once(
        text,
        CONSUMER_LOOP,
        CONSUMER_PAIR,
        "consumer M-block pair",
    )
    text = replace_once(
        text,
        CONSUMER_OUTER,
        CONSUMER_OUTER_U1,
        "consumer outer K64 loop policy",
    )
    return replace_once(
        text,
        '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "',
        '"nsplit_variant=mma_pair phase=0/0 '
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
        f"variant=nsplit_mma_pair source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
