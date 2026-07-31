#!/usr/bin/env python3
"""Generate audited Morton/Hilbert spatial-order variants of N-split GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from generate_gemm_nsplit_l2_phase import (
    EXPECTED_SOURCE_SHA256,
    TASK_COUNT,
    format_device_table,
    replace_once,
    task_map_metrics,
)


GRID = 64
VARIANTS = (
    "table_identity",
    "morton",
    "hilbert",
    "hilbert_transpose",
    "hilbert_reverse",
)


def canonical_task(tile_m: int, tile_n: int) -> int:
    """Encode a 64x64 tile coordinate in canonical static-8x16 order."""
    if not (0 <= tile_m < GRID and 0 <= tile_n < GRID):
        raise ValueError((tile_m, tile_n))
    return (tile_m // 8) * 512 + tile_n * 8 + tile_m % 8


def morton_xy(distance: int) -> tuple[int, int]:
    x = 0
    y = 0
    for bit in range(6):
        x |= ((distance >> (2 * bit)) & 1) << bit
        y |= ((distance >> (2 * bit + 1)) & 1) << bit
    return x, y


def hilbert_rotate(
    side: int, x: int, y: int, rx: int, ry: int
) -> tuple[int, int]:
    if ry == 0:
        if rx == 1:
            x = side - 1 - x
            y = side - 1 - y
        x, y = y, x
    return x, y


def hilbert_xy(distance: int) -> tuple[int, int]:
    x = 0
    y = 0
    remaining = distance
    side = 1
    while side < GRID:
        rx = (remaining // 2) & 1
        ry = (remaining ^ rx) & 1
        x, y = hilbert_rotate(side, x, y, rx, ry)
        x += side * rx
        y += side * ry
        remaining //= 4
        side *= 2
    return x, y


def build_spatial_map(variant: str) -> list[int]:
    if variant == "table_identity":
        return list(range(TASK_COUNT))
    mapped: list[int] = []
    for position in range(TASK_COUNT):
        distance = TASK_COUNT - 1 - position if variant == "hilbert_reverse" else position
        if variant == "morton":
            tile_m, tile_n = morton_xy(distance)
        else:
            tile_m, tile_n = hilbert_xy(distance)
            if variant == "hilbert_transpose":
                tile_m, tile_n = tile_n, tile_m
        mapped.append(canonical_task(tile_m, tile_n))
    if sorted(mapped) != list(range(TASK_COUNT)):
        raise RuntimeError(f"{variant}: map is not a permutation")
    return mapped


def extended_metrics(task_map: list[int], variant: str) -> dict[str, object]:
    metrics = task_map_metrics(task_map)
    waves = metrics["wave_panels"]
    assert isinstance(waves, list)
    totals = [int(w["unique_a"]) + int(w["unique_b"]) for w in waves]
    metrics["variant"] = variant
    metrics["wave_count"] = len(waves)
    metrics["mean_unique_a"] = sum(int(w["unique_a"]) for w in waves) / len(waves)
    metrics["mean_unique_b"] = sum(int(w["unique_b"]) for w in waves) / len(waves)
    metrics["mean_unique_panels"] = sum(totals) / len(totals)
    metrics["max_unique_panels"] = max(totals)
    return metrics


def generate(source: str, variant: str) -> tuple[str, dict[str, object]]:
    task_map = build_spatial_map(variant)
    metrics = extended_metrics(task_map, variant)
    constants_anchor = "static_assert(kBTmaN == 128);\n\n"
    table = format_device_table(task_map, "kPersistentSpatialMap16K")
    text = replace_once(
        source,
        constants_anchor,
        constants_anchor + table,
        "spatial-order task table",
    )
    mapping_anchor = """    const int macro_id = linear_tile / persistent_macro_tiles;
    const int local = linear_tile - macro_id * persistent_macro_tiles;
"""
    mapping_new = """    int mapped_linear_tile = linear_tile;
    if constexpr (mtile_count == 64 && ntile_count == 64)
      mapped_linear_tile =
          static_cast<int>(kPersistentSpatialMap16K[linear_tile]);
    const int macro_id = mapped_linear_tile / persistent_macro_tiles;
    const int local = mapped_linear_tile - macro_id * persistent_macro_tiles;
"""
    text = replace_once(text, mapping_anchor, mapping_new, "spatial-order lookup")
    banner_anchor = '"phase=0/0 consumer_wait=b_then_a c_store=tma_fp32_sw128 l2_promotion=none "'
    banner_new = (
        f'"spatial_order={variant} phase=0/0 consumer_wait=b_then_a '
        'c_store=tma_fp32_sw128 l2_promotion=none "'
    )
    text = replace_once(text, banner_anchor, banner_new, "configuration banner")
    return text, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent / "baseline" / "gemm256_bf16_16k.cu",
    )
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", required=True, type=Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited N-split source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_hash}"
        )
    generated, metrics = generate(source_bytes.decode(), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"variant={args.variant} source_sha256={source_hash} "
        f"output_sha256={hashlib.sha256(generated.encode()).hexdigest()} "
        f"mean_unique_panels={metrics['mean_unique_panels']:.3f} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
