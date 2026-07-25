#!/usr/bin/env python3
"""Generate hash-gated A-locality scheduler variants for clean N-split GEMM."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca"
)

VARIANTS = (
    "baseline",
    "nfast_16x16",
    "nfast_8x32",
    "nfast_4x64",
)

MACRO_M_ANCHOR = "static constexpr int kPersistentMacroM = 16;\n"
MACRO_N_ANCHOR = "static constexpr int kPersistentMacroN = 16;\n"

ORDER_BLOCK = """    const int macro_n = macro_id % persistent_groups_n;
    const int macro_m = macro_id / persistent_groups_n;
    const int local_m = local % persistent_macro_m;
    const int local_n = local / persistent_macro_m;
    const int tile_m = macro_m * persistent_macro_m + local_m;
    const int tile_n = macro_n * persistent_macro_n + local_n;
"""

BANNER_ANCHOR = '"phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def generate(source: str, variant: str) -> str:
    text = source
    if variant == "baseline":
        label = "baseline"
    else:
        macro = {
            "nfast_16x16": (16, 16),
            "nfast_8x32": (8, 32),
            "nfast_4x64": (4, 64),
        }[variant]
        macro_m, macro_n = macro
        text = replace_once(
            text,
            MACRO_M_ANCHOR,
            f"static constexpr int kPersistentMacroM = {macro_m};\n",
            f"{variant} macro M",
        )
        text = replace_once(
            text,
            MACRO_N_ANCHOR,
            f"static constexpr int kPersistentMacroN = {macro_n};\n",
            f"{variant} macro N",
        )
        nfast_order = """    const int macro_m = macro_id % persistent_groups_m;
    const int macro_n = macro_id / persistent_groups_m;
    const int local_n = local % persistent_macro_n;
    const int local_m = local / persistent_macro_n;
    const int tile_m = macro_m * persistent_macro_m + local_m;
    const int tile_n = macro_n * persistent_macro_n + local_n;
"""
        text = replace_once(text, ORDER_BLOCK, nfast_order, f"{variant} order")
        label = f"a_locality={variant}"

    text = replace_once(
        text,
        BANNER_ANCHOR,
        f'"{label} phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "',
        "host configuration label",
    )
    if variant != "baseline":
        for fragment in (
            "local_n = local % persistent_macro_n",
            "macro_m = macro_id % persistent_groups_m",
            f"kPersistentMacroM = {macro_m}",
            f"kPersistentMacroN = {macro_n}",
        ):
            if fragment not in text:
                raise RuntimeError(f"generated audit: missing {fragment!r}")
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
