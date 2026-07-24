#!/usr/bin/env python3
"""Generate hash-gated no-C-store controls for N-split mainloop timing."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXPECTED_SHA256 = {
    "exact": "cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca",
    "transpose": "a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc",
}

STORE_BLOCK = """    const int global_row_base = tile_m * kCtaM;
    const int global_col_base = tile_n * kCtaN;
    store_256x256_float_tile_tma(tmem_base, &c_map, c_store_smem,
                                 global_row_base,
                                 global_col_base);
"""

NO_STORE_BLOCK = """    // Mainloop-only timing control: retain the final MMA drain and CTA
    // synchronization, but do not load TMEM or stage/store FP32 C.
"""

BANNER_ANCHOR = "c_store=tma_fp32_sw128 l2_promotion=none "
NO_STORE_BANNER = "c_store=none mainloop_only=1 l2_promotion=none "


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def generate(source: str) -> str:
    text = replace_once(
        source,
        STORE_BLOCK,
        NO_STORE_BLOCK,
        "FP32 C-store call",
    )
    return replace_once(
        text,
        BANNER_ANCHOR,
        NO_STORE_BANNER,
        "host configuration label",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=tuple(EXPECTED_SHA256))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    expected = EXPECTED_SHA256[args.variant]
    if source_sha256 != expected:
        raise SystemExit(
            f"refusing to patch unaudited {args.variant} source: "
            f"expected {expected}, got {source_sha256}"
        )

    generated = generate(source_bytes.decode("utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated, encoding="utf-8")
    print(
        f"variant=nsplit_{args.variant}_no_cstore "
        f"source_sha256={source_sha256} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
