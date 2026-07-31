#!/usr/bin/env python3
"""Generate audited L2/scheduler/phase variants of the canonical N-split GEMM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SOURCE_SHA256 = (
    "eb90e11322c3af9adba08ed6262fe585bd0b37c9ba2fb1750c802e393c6b6dc2"
)
TASK_COUNT = 64 * 64
PERSISTENT_CTAS = 148
VARIANTS = (
    "c_evict_first",
    "table_identity",
    "wave_a",
    "wave_b",
    "wave_balanced",
    "issue_b0_first",
    "early_b0",
    "stage_ring_state",
    "wait_b_first",
    "b1_delay0",
    "b1_delay48",
    "b1_delay64",
    "b1_delay80",
    "b1_delay96",
    "b1_delay128",
    "b1_delay192",
    "b1_delay256",
    "b1_delay384",
    "b1_delay448",
    "b1_delay512",
    "b1_delay576",
    "b1_delay640",
    "b1_delay768",
    "b1_delay1024",
)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def tile_coords(task: int) -> tuple[int, int]:
    """Canonical 16K static-8x16 mapping, simplified for the 64x64 grid."""
    return 8 * (task // 512) + task % 8, (task % 512) // 8


def hungarian(cost: list[list[int]]) -> list[int]:
    """Return row->column minimum-cost assignment for a square integer matrix."""
    n = len(cost)
    if n == 0 or any(len(row) != n for row in cost):
        raise ValueError("hungarian expects a non-empty square matrix")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        minv = [10**30] * (n + 1)
        used = [False] * (n + 1)
        j0 = 0
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = 10**30
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, n + 1):
        assignment[p[j] - 1] = j - 1
    if sorted(assignment) != list(range(n)):
        raise RuntimeError("invalid Hungarian assignment")
    return assignment


def edge_cost(previous: int, current: int, mode: str) -> int:
    pm, pn = tile_coords(previous)
    cm, cn = tile_coords(current)
    same_a = int(pm == cm)
    same_b = int(pn == cn)
    manhattan = abs(pm - cm) + abs(pn - cn)
    # Large integer tiers make the optimization objective explicit.  The final
    # current-task term only breaks otherwise identical assignments.
    if mode == "wave_a":
        return -same_a * 1_000_000 + abs(pn - cn) * 1_000 + manhattan * 10 + current
    if mode == "wave_b":
        return -same_b * 1_000_000 + abs(pm - cm) * 1_000 + manhattan * 10 + current
    if mode == "wave_balanced":
        return -(same_a + same_b) * 1_000_000 + manhattan * 1_000 + current
    raise ValueError(mode)


def build_task_map(mode: str) -> list[int]:
    if mode == "table_identity":
        return list(range(TASK_COUNT))
    task_map = [-1] * TASK_COUNT
    previous_by_cta: list[int | None] = [None] * PERSISTENT_CTAS
    for wave_start in range(0, TASK_COUNT, PERSISTENT_CTAS):
        tasks = list(range(wave_start, min(wave_start + PERSISTENT_CTAS, TASK_COUNT)))
        active = len(tasks)
        if wave_start == 0:
            assigned = tasks
        else:
            # Preserve baseline tail ownership: only CTA [0, active) receives a
            # task, so each CTA still executes exactly 27 or 28 output tiles.
            cost = [
                [edge_cost(previous_by_cta[cta], task, mode) for task in tasks]
                for cta in range(active)
            ]
            row_to_col = hungarian(cost)
            assigned = [tasks[col] for col in row_to_col]
        for cta, task in enumerate(assigned):
            task_map[wave_start + cta] = task
            previous_by_cta[cta] = task
    if sorted(task_map) != list(range(TASK_COUNT)):
        raise RuntimeError(f"{mode}: task map is not a permutation")
    return task_map


def task_map_metrics(task_map: list[int]) -> dict[str, object]:
    same_a = 0
    same_b = 0
    transitions = 0
    distances: list[int] = []
    for cta in range(PERSISTENT_CTAS):
        previous: int | None = None
        for slot in range(cta, TASK_COUNT, PERSISTENT_CTAS):
            task = task_map[slot]
            if previous is not None:
                pm, pn = tile_coords(previous)
                cm, cn = tile_coords(task)
                same_a += int(pm == cm)
                same_b += int(pn == cn)
                distances.append(abs(pm - cm) + abs(pn - cn))
                transitions += 1
            previous = task
    wave_panels = []
    for start in range(0, TASK_COUNT, PERSISTENT_CTAS):
        tasks = task_map[start : min(start + PERSISTENT_CTAS, TASK_COUNT)]
        coords = [tile_coords(task) for task in tasks]
        wave_panels.append(
            {
                "tasks": len(tasks),
                "unique_a": len({m for m, _ in coords}),
                "unique_b": len({n for _, n in coords}),
            }
        )
    return {
        "tasks": len(task_map),
        "transitions": transitions,
        "same_a": same_a,
        "same_b": same_b,
        "same_a_fraction": same_a / transitions,
        "same_b_fraction": same_b / transitions,
        "mean_manhattan": sum(distances) / len(distances),
        "max_manhattan": max(distances),
        "wave_panels": wave_panels,
    }


def format_device_table(task_map: list[int], name: str) -> str:
    rows = []
    for start in range(0, len(task_map), 16):
        values = ", ".join(str(value) for value in task_map[start : start + 16])
        rows.append(f"  {values},")
    return (
        f"__device__ __constant__ uint16_t {name}[{len(task_map)}] = {{\n"
        + "\n".join(rows)
        + "\n};\n\n"
    )


def apply_task_map(text: str, variant: str) -> tuple[str, dict[str, object]]:
    task_map = build_task_map(variant)
    metrics = task_map_metrics(task_map)
    constants_anchor = "static_assert(kBTmaN == 128);\n\n"
    table = format_device_table(task_map, "kPersistentTaskMap16K")
    text = replace_once(
        text,
        constants_anchor,
        constants_anchor + table,
        "persistent task table",
    )
    mapping_anchor = """    const int macro_id = linear_tile / persistent_macro_tiles;
    const int local = linear_tile - macro_id * persistent_macro_tiles;
"""
    mapping_new = """    int mapped_linear_tile = linear_tile;
    if constexpr (mtile_count == 64 && ntile_count == 64)
      mapped_linear_tile = static_cast<int>(kPersistentTaskMap16K[linear_tile]);
    const int macro_id = mapped_linear_tile / persistent_macro_tiles;
    const int local = mapped_linear_tile - macro_id * persistent_macro_tiles;
"""
    text = replace_once(text, mapping_anchor, mapping_new, "mapped task lookup")
    return text, metrics


def apply_c_evict_first(text: str) -> str:
    helper_anchor = """__device__ __forceinline__ void tma_store_4d(const CUtensorMap *map,
                                             uint32_t src_smem, int c0, int c1,
                                             int c2, int c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group"
               " [%0, {%2, %3, %4, %5}], [%1];"
               :
               : "l"(map), "r"(src_smem), "r"(c0), "r"(c1), "r"(c2), "r"(c3)
               : "memory");
"""
    helper_new = """__device__ __forceinline__ void tma_store_4d(const CUtensorMap *map,
                                             uint32_t src_smem, int c0, int c1,
                                             int c2, int c3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint64_t policy;
  asm volatile(
      "createpolicy.fractional.L2::evict_first.b64 %0, 1.0;"
      : "=l"(policy));
  asm volatile(
      "cp.async.bulk.tensor.4d.global.shared::cta.bulk_group.L2::cache_hint"
      " [%0, {%2, %3, %4, %5}], [%1], %6;"
      :
      : "l"(map), "r"(src_smem), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
        "l"(policy)
      : "memory");
"""
    return replace_once(text, helper_anchor, helper_new, "C-store evict-first")


def apply_issue_b0_first(text: str) -> str:
    issue_anchor = """        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
"""
    issue_new = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
"""
    return replace_once(text, issue_anchor, issue_new, "B0/A issue order")


def apply_early_b0(text: str) -> str:
    loop_anchor = """        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
#pragma unroll
          for (int p = 0; p < kPipes; ++p) {
            mbarrier_wait(&mma_done[p][stage], reuse_phase);
          }
        }
        issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
"""
    loop_new = """        if (stage_epoch >= kStages) {
          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
          // B0 is consumed only by pipe 0, so release and refill it as soon as
          // pipe 0 completes.  Shared A still waits for both consumers.
          mbarrier_wait(&mma_done[0][stage], reuse_phase);
          issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                                 0);
          mbarrier_wait(&mma_done[1][stage], reuse_phase);
          issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        } else {
          issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                                 0);
          issue_a_stage_tma(&a_map, a_smem, &a_ready[stage], tile_m, kt);
        }
"""
    return replace_once(text, loop_anchor, loop_new, "early B0 dependency split")


def apply_stage_ring_state(text: str) -> str:
    loop_start = """      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = stage_epoch % kStages;
"""
    producer_start = """      int ring_stage = stage_epoch_base % kStages;
      uint32_t ring_phase =
          static_cast<uint32_t>((stage_epoch_base / kStages) & 1);
      for (int kt = 0; kt < ktiles; ++kt) {
        const int stage_epoch = stage_epoch_base + kt;
        const int stage = ring_stage;
"""
    # The same loop prologue appears once in each producer and once in the
    # consumer block.
    if text.count(loop_start) != 3:
        raise RuntimeError(
            f"stage ring loop prologue: expected three anchors, "
            f"found {text.count(loop_start)}"
        )
    text = text.replace(loop_start, producer_start)
    reuse_anchor = """          const uint32_t reuse_phase =
              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1);
"""
    if text.count(reuse_anchor) != 1:
        raise RuntimeError("stage ring warp-0 reuse phase anchor mismatch")
    text = text.replace(
        reuse_anchor,
        """          const uint32_t reuse_phase = ring_phase ^ 1u;
""",
        1,
    )
    warp1_reuse_anchor = """              static_cast<uint32_t>(((stage_epoch - kStages) / kStages) & 1));
"""
    text = replace_once(
        text,
        warp1_reuse_anchor,
        """              ring_phase ^ 1u);
""",
        "stage ring warp-1 reuse phase",
    )
    consumer_phase_anchor = """        const uint32_t tma_phase =
            static_cast<uint32_t>((stage_epoch / kStages) & 1);
"""
    text = replace_once(
        text,
        consumer_phase_anchor,
        """        const uint32_t tma_phase = ring_phase;
""",
        "stage ring consumer phase",
    )

    producer0_end = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
      }
    }

    if (warp_id == 1 && lane0) {"""
    producer0_end_new = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[0][stage], tile_n, kt,
                               0);
        if (++ring_stage == kStages) {
          ring_stage = 0;
          ring_phase ^= 1u;
        }
      }
    }

    if (warp_id == 1 && lane0) {"""
    text = replace_once(
        text, producer0_end, producer0_end_new, "stage ring producer 0 update"
    )
    producer1_end = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {"""
    producer1_end_new = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
        if (++ring_stage == kStages) {
          ring_stage = 0;
          ring_phase ^= 1u;
        }
      }
    }

    if ((warp_id == 2 || warp_id == 3) && lane0) {"""
    text = replace_once(
        text, producer1_end, producer1_end_new, "stage ring producer 1 update"
    )
    consumer_end = """        tcgen05_commit(&mma_done[pipe][stage]);
      }
      const int last_stage_epoch"""
    consumer_end_new = """        tcgen05_commit(&mma_done[pipe][stage]);
        if (++ring_stage == kStages) {
          ring_stage = 0;
          ring_phase ^= 1u;
        }
      }
      const int last_stage_epoch"""
    text = replace_once(
        text, consumer_end, consumer_end_new, "stage ring consumer update"
    )
    return text


def apply_b1_delay(text: str, cycles: int) -> str:
    issue_anchor = """        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
"""
    issue_new = f"""        // Focused B1-only phase shift.  This changes neither the B1
        // address nor its dependency; it delays only the producer issue site.
        asm volatile("nanosleep.u32 {cycles};");
        issue_b_pipe_stage_tma(&b_map, b_smem, &b_ready[1][stage], tile_n, kt,
                               1);
"""
    return replace_once(text, issue_anchor, issue_new, "focused B1 delay")


def apply_wait_b_first(text: str) -> str:
    wait_anchor = """        mbarrier_wait(&a_ready[stage], tma_phase);
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);
"""
    wait_new = """        // B is half the A transfer size. Probe it first so the second
        // wait reveals whether A, rather than B, is the consumer dependency.
        mbarrier_wait(&b_ready[pipe][stage], tma_phase);
        mbarrier_wait(&a_ready[stage], tma_phase);
"""
    return replace_once(text, wait_anchor, wait_new, "B-first consumer wait")


def set_banner(text: str, variant: str) -> str:
    anchor = (
        '"stages=3 pipes=2 persistent_ctas=%d scheduler=static_%dx%d_mfast '
        'overhead=fixed_sink "\n'
        '      "phase=0/0 c_store=tma_fp32_sw128 l2_promotion=none "'
    )
    replacement = (
        '"stages=3 pipes=2 persistent_ctas=%d scheduler=static_%dx%d_mfast '
        'overhead=fixed_sink "\n'
        f'      "l2_phase_variant={variant} c_store=tma_fp32_sw128 '
        'l2_promotion=none "'
    )
    return replace_once(text, anchor, replacement, "configuration banner")


def generate(source: str, variant: str) -> tuple[str, dict[str, object] | None]:
    text = source
    metrics = None
    if variant == "c_evict_first":
        text = apply_c_evict_first(text)
    elif variant in {"table_identity", "wave_a", "wave_b", "wave_balanced"}:
        text, metrics = apply_task_map(text, variant)
    elif variant == "issue_b0_first":
        text = apply_issue_b0_first(text)
    elif variant == "early_b0":
        text = apply_early_b0(text)
    elif variant == "stage_ring_state":
        text = apply_stage_ring_state(text)
    elif variant == "wait_b_first":
        text = apply_wait_b_first(text)
    elif variant.startswith("b1_delay"):
        text = apply_b1_delay(text, int(variant.removeprefix("b1_delay")))
    else:
        raise ValueError(variant)
    text = set_banner(text, variant)
    return text, metrics


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
    parser.add_argument("--metrics", type=Path)
    args = parser.parse_args()

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise SystemExit(
            "refusing to patch an unaudited N-split source: "
            f"expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    generated, metrics = generate(source_bytes.decode(), args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(generated)
    output_sha256 = hashlib.sha256(generated.encode()).hexdigest()
    if args.metrics:
        if metrics is None:
            raise SystemExit("--metrics is valid only for task-map variants")
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        args.metrics.write_text(json.dumps(metrics, indent=2) + "\n")
    print(
        f"variant={args.variant} source_sha256={source_sha256} "
        f"output_sha256={output_sha256} output={args.output}"
    )
    if metrics is not None:
        print(
            "transitions={transitions} same_a={same_a} same_b={same_b} "
            "mean_manhattan={mean_manhattan:.6f}".format(**metrics)
        )


if __name__ == "__main__":
    main()
