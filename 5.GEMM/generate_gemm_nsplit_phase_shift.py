#!/usr/bin/env python3
"""Generate hash-gated phase-shift variants from the direct N-split source."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca"
)

VARIANTS = (
    "cta4",
    "cta8",
    "b1_gap0",
    "b1_gap32",
    "b1_gap64",
    "pipe1_gap0",
    "pipe1_gap32",
    "pipe1_gap64",
)

SPIN_HELPER = r"""
__device__ __forceinline__ void nsplit_phase_spin_cycles(
    unsigned long long cycles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const unsigned long long start = clock64();
  while (clock64() - start < cycles) {
  }
#else
  (void)cycles;
#endif
}

"""

HELPER_ANCHOR = """__device__ __forceinline__ void mbarrier_expect_tx"""

CTA_ANCHOR = """    if (lane0)
      tmem_base_shared = taddr;
"""

B1_ISSUE_ANCHOR = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
"""

PIPE_READY_ANCHOR = """        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);

#pragma unroll
"""

BANNER_ANCHOR = '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def add_helper(text: str) -> str:
    return replace_once(
        text, HELPER_ANCHOR, SPIN_HELPER + HELPER_ANCHOR, "phase helper"
    )


def generate(source: str, variant: str) -> str:
    text = source
    if variant in {"cta4", "cta8"}:
        groups, step = (4, 256) if variant == "cta4" else (8, 128)
        text = add_helper(text)
        replacement = CTA_ANCHOR + (
            "    // One-time CTA startup staggering.  The existing\n"
            "    // post-allocation CTA barrier publishes this delay.\n"
            "    if (lane0) {\n"
            f"      const unsigned int cohort = blockIdx.x % {groups}u;\n"
            f"      nsplit_phase_spin_cycles(cohort * {step}ull);\n"
            "    }\n"
        )
        text = replace_once(text, CTA_ANCHOR, replacement, variant)
    elif variant in {"b1_gap0", "b1_gap32", "b1_gap64"}:
        cycles = {"b1_gap0": 0, "b1_gap32": 32, "b1_gap64": 64}[variant]
        replacement = (
            f'        asm volatile("nanosleep.u32 {cycles};");\n'
            + B1_ISSUE_ANCHOR
        )
        text = replace_once(text, B1_ISSUE_ANCHOR, replacement, variant)
    elif variant in {"pipe1_gap0", "pipe1_gap32", "pipe1_gap64"}:
        cycles = {
            "pipe1_gap0": 0,
            "pipe1_gap32": 32,
            "pipe1_gap64": 64,
        }[variant]
        replacement = (
            "        mbarrier_wait(&a_ready[stage], tma_phase);\n"
            "        mbarrier_wait(&b_ready[pipe][stage], tma_phase);\n"
            "        if (pipe == 1)\n"
            f'          asm volatile("nanosleep.u32 {cycles};");\n'
            "\n"
            "#pragma unroll\n"
        )
        text = replace_once(text, PIPE_READY_ANCHOR, replacement, variant)
    else:
        raise ValueError(f"unknown variant: {variant}")

    text = replace_once(
        text,
        BANNER_ANCHOR,
        f'"nsplit_phase_shift={variant} c_store=tma_fp32_sw128 '
        'l2_promotion=none "',
        "host configuration label",
    )
    audit(text, variant)
    return text


def audit(text: str, variant: str) -> None:
    required = [f"nsplit_phase_shift={variant}"]
    if variant.startswith("cta"):
        required += ["nsplit_phase_spin_cycles(", "blockIdx.x %"]
    if variant.startswith("b1_gap") or variant.startswith("pipe1_gap"):
        required += ["nanosleep.u32"]
    if variant.startswith("pipe1_gap"):
        required += ["if (pipe == 1)"]
    for fragment in required:
        if fragment not in text:
            raise RuntimeError(f"generated audit: missing {fragment!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent
        / "baseline"
        / "gemm256_bf16_16k.cu",
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
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
