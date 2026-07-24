#!/usr/bin/env python3
"""Generate hash-gated phase-shift variants from the clean E7a source.

The clean source stays macro-free.  This generator makes experiment-only
sources whose changes are limited to one precisely identified scheduling
mechanism:

* cta4 / cta8: one-time whole-CTA startup staggering across persistent CTAs;
* b1_gap0 / b1_gap32 / b1_gap64: codegen control and per-stage spacing
  between A and late-B1 TMA issue;
* b1_cross: move W3's existing B1 wait before its first MMA half, so W2 and
  W3 cross phases without adding a wait or changing accumulation order.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a"
)

SPIN_HELPER = r"""
__device__ __forceinline__ void e7a_phase_spin_cycles(
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

B1_GAP_ANCHOR = """        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_producer_part_tma(&b_map, b_smem, &b_ready[1][stage], tile_n,
                                  kt, 1);
"""

CONSUMER_READY_ANCHOR = """        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[0][stage], tma_phase);
#pragma unroll
"""

CONSUMER_B1_ANCHOR = """        mbarrier_wait(&b_ready[1][stage], tma_phase);
#pragma unroll
        for (int kk = kBK16PerProducerPart;
"""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def add_helper(text: str, helper: str) -> str:
    return replace_once(
        text,
        HELPER_ANCHOR,
        helper + HELPER_ANCHOR,
        "phase helper",
    )


def generate(source: str, variant: str) -> str:
    text = source

    if variant in {"cta4", "cta8"}:
        groups, step = (4, 256) if variant == "cta4" else (8, 128)
        text = add_helper(text, SPIN_HELPER)
        replacement = CTA_ANCHOR + (
            "    // One-time whole-CTA staggering.  This uses the existing\n"
            "    // post-allocation CTA barrier, so no steady-state barrier or\n"
            "    // per-output-tile delay is added.\n"
            "    if (lane0) {\n"
            f"      const unsigned int cohort = blockIdx.x % {groups}u;\n"
            f"      e7a_phase_spin_cycles(cohort * {step}ull);\n"
            "    }\n"
        )
        text = replace_once(text, CTA_ANCHOR, replacement, variant)

    elif variant in {"b1_gap0", "b1_gap32", "b1_gap64"}:
        cycles = {"b1_gap0": 0, "b1_gap32": 32, "b1_gap64": 64}[variant]
        replacement = (
            "        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], "
            "tile_m, kt);\n"
            "        // A is the observed early critical dependency.  B1 is\n"
            "        // consumed only after the first two wide MMAs, so space\n"
            "        // this late transaction without delaying A itself.\n"
            f'        asm volatile("nanosleep.u32 {cycles};");\n'
            "        issue_b_producer_part_tma(&b_map, b_smem, "
            "&b_ready[1][stage], tile_n,\n"
            "                                  kt, 1);\n"
        )
        text = replace_once(text, B1_GAP_ANCHOR, replacement, variant)

    elif variant == "b1_cross":
        ready_replacement = (
            "        mbarrier_wait(&a_ready[stage], tma_phase);\n"
            "        mbarrier_wait(&b_ready[0][stage], tma_phase);\n"
            "        // W3 moves its existing B1 wait ahead of the first MMA\n"
            "        // half.  W2 keeps the original order.  Both warps still\n"
            "        // perform exactly one B1 wait and issue K in 0..3 order.\n"
            "        if (mblock == 1)\n"
            "          mbarrier_wait(&b_ready[1][stage], tma_phase);\n"
            "#pragma unroll\n"
        )
        text = replace_once(
            text,
            CONSUMER_READY_ANCHOR,
            ready_replacement,
            "b1_cross early wait",
        )
        late_replacement = (
            "        if (mblock == 0)\n"
            "          mbarrier_wait(&b_ready[1][stage], tma_phase);\n"
            "#pragma unroll\n"
            "        for (int kk = kBK16PerProducerPart;\n"
        )
        text = replace_once(
            text,
            CONSUMER_B1_ANCHOR,
            late_replacement,
            "b1_cross late wait",
        )
    else:
        raise ValueError(f"unknown variant: {variant}")

    text = replace_once(
        text,
        '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "',
        f'"phase_redesign={variant} c_store=tma_fp32_sw128 '
        'l2_promotion=none "',
        "host configuration label",
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
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "cta4",
            "cta8",
            "b1_gap0",
            "b1_gap32",
            "b1_gap64",
            "b1_cross",
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited E7a source: "
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
