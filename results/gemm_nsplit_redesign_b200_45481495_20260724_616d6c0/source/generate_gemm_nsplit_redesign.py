#!/usr/bin/env python3
"""Generate hash-gated first-step variants of the clean N-split GEMM."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca"
)

CONSUMER_LOOP_ANCHOR = """    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
      for (int kt = 0; kt < ktiles; ++kt) {
"""

SUSPEND_HELPER = r"""
__device__ __forceinline__ void mbarrier_wait_suspend(uint64_t *barrier,
                                                      uint32_t phase) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const uint32_t addr = smem_ptr_u32(barrier);
  // Use the long suspend hint only for producer waits on stage reuse.
  constexpr uint32_t kSuspendTicks = 0x989680u;
  asm volatile("{ .reg .pred p; "
               "L_wait_%=: "
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2; "
               "@p bra.uni L_done_%=; "
               "bra.uni L_wait_%=; "
               "L_done_%=: }" ::"r"(addr),
               "r"(phase), "r"(kSuspendTicks)
               : "memory");
#else
  (void)barrier;
  (void)phase;
#endif
}

"""

HELPER_ANCHOR = """__device__ __forceinline__ void mbarrier_expect_tx"""

W0_REUSE_ANCHOR = """#pragma unroll
          for (int p = 0; p < kPipes; ++p) {
            mbarrier_wait(&mma_done[p][stage], reuse_phase);
          }
"""

W1_REUSE_ANCHOR = """          mbarrier_wait(
              &mma_done[1][stage],
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def add_consumer_u1(text: str) -> str:
    replacement = """    if ((warp_id == 2 || warp_id == 3) && lane0) {
      const int pipe = warp_id - 2;
#pragma unroll 1
      for (int kt = 0; kt < ktiles; ++kt) {
"""
    return replace_once(
        text, CONSUMER_LOOP_ANCHOR, replacement, "consumer outer K loop"
    )


def add_producer_suspend(text: str) -> str:
    text = replace_once(
        text,
        HELPER_ANCHOR,
        SUSPEND_HELPER + HELPER_ANCHOR,
        "suspend helper",
    )
    text = replace_once(
        text,
        W0_REUSE_ANCHOR,
        """#pragma unroll
          for (int p = 0; p < kPipes; ++p) {
            mbarrier_wait_suspend(&mma_done[p][stage], reuse_phase);
          }
""",
        "warp 0 producer reuse wait",
    )
    return replace_once(
        text,
        W1_REUSE_ANCHOR,
        """          mbarrier_wait_suspend(
              &mma_done[1][stage],
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
""",
        "warp 1 producer reuse wait",
    )


def generate(source: str, variant: str) -> str:
    text = add_consumer_u1(source)
    if variant == "nsplit_u1_suspend_prod":
        text = add_producer_suspend(text)
    elif variant != "nsplit_consumer_u1":
        raise ValueError(f"unknown variant: {variant}")

    return replace_once(
        text,
        '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "',
        f'"nsplit_variant={variant} phase=0/0 '
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
    parser.add_argument(
        "--variant",
        required=True,
        choices=("nsplit_consumer_u1", "nsplit_u1_suspend_prod"),
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

    generated = generate(source_bytes.decode("utf-8"), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(
        f"variant={args.variant} source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
