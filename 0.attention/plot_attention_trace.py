#!/usr/bin/env python3
import argparse
import csv
import html


K_BYTES = 128 * 128 * 2
LD_X64_BYTES_PER_WARP = 32 * 64 * 4
LD_X64X2_BYTES_PER_WARP = 2 * LD_X64_BYTES_PER_WARP
MMA8_FLOPS = 8 * 2 * 128 * 128 * 16


def read_rows(path, iter_start, num_iters):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            it = int(row["iter"])
            if iter_start <= it < iter_start + num_iters:
                rows.append(row)
    rows.sort(key=lambda r: (int(r["iter"]), int(r["pipe"])))
    return rows


def rate_text(stage, cycles, clock_mhz, sms, ld_warps):
    hz = clock_mhz * 1.0e6
    if stage == "TMA":
        value = K_BYTES * hz / cycles / 1.0e12 * sms
        return f"{value:.3f} TB/s"
    if stage == "MMA":
        value = MMA8_FLOPS * hz / cycles / 1.0e12 * sms
        return f"{value:.1f} TFLOP/s"
    if stage == "MMA12":
        value = (MMA8_FLOPS * 3 // 2) * hz / cycles / 1.0e12 * sms
        return f"{value:.1f} TFLOP/s"
    if stage == "MMA4":
        value = (MMA8_FLOPS / 2) * hz / cycles / 1.0e12 * sms
        return f"{value:.1f} TFLOP/s"
    if stage in ("PACK", "SUM"):
        return "reg"
    if stage == "PVTAIL":
        return "async"
    if stage == "WAIT":
        return "wait"
    bytes_per_stage = K_BYTES if stage == "ST" else (LD_X64X2_BYTES_PER_WARP * ld_warps)
    value = bytes_per_stage * hz / cycles / 1.0e12 * sms
    return f"{value:.3f} TB/s"


def byte_rate_text(bytes_count, cycles, clock_mhz, sms):
    hz = clock_mhz * 1.0e6
    value = bytes_count * hz / cycles / 1.0e12 * sms
    return f"{value:.3f} TB/s"


def has_consumer_warp_stamps(row):
    return "pack_warp0_start" in row and "st_warp0_start" in row


def has_pv_activity(row):
    keys = (
        "v_tma_start",
        "v_tma_end",
        "pv_start",
        "pv_end",
        "pv_h0_start",
        "pv_h0_end",
        "pv_h1_start",
        "pv_h1_end",
    )
    return any(row.get(key, "0") not in ("", "0") for key in keys)


def consumer_warp_count(row):
    n = 0
    while f"ld_warp{n}_start" in row and f"st_warp{n}_start" in row:
        n += 1
    return n


def consumer_stage_items(row, base, clock_mhz, sms, consumer_warp):
    count = max(1, consumer_warp_count(row))
    ld_bytes = LD_X64_BYTES_PER_WARP if count >= 8 else LD_X64X2_BYTES_PER_WARP
    st_bytes = K_BYTES // count
    specs = [
        ("LD", f"ld_warp{consumer_warp}_start", f"ld_warp{consumer_warp}_end", "#1f4f7a", ld_bytes),
        ("PACK", f"pack_warp{consumer_warp}_start", f"pack_warp{consumer_warp}_end", "#e19b2e", None),
        ("ST", f"st_warp{consumer_warp}_start", f"st_warp{consumer_warp}_end", "#9a6a2f", st_bytes),
        ("SUM", f"sum_warp{consumer_warp}_start", f"sum_warp{consumer_warp}_end", "#c04c84", None),
    ]
    out = []
    for name, start_key, end_key, color, bytes_count in specs:
        if start_key not in row or end_key not in row:
            continue
        if row[start_key] in ("", "0") or row[end_key] in ("", "0"):
            continue
        start = int(row[start_key]) - base
        end = int(row[end_key]) - base
        if end <= start:
            continue
        cycles = end - start
        rate = "reg" if bytes_count is None else byte_rate_text(bytes_count, cycles, clock_mhz, sms)
        out.append((name, start, end, cycles, color, rate))
    return out


def sync_items(row, base):
    out = []
    idx = 0
    while f"sync{idx}_start" in row and f"sync{idx}_end" in row:
        start_key = f"sync{idx}_start"
        end_key = f"sync{idx}_end"
        if row[start_key] not in ("", "0") and row[end_key] not in ("", "0"):
            start = int(row[start_key]) - base
            end = int(row[end_key]) - base
            if end > start:
                out.append((idx, start, end, end - start))
        idx += 1
    return out


def stage_items(row, base, clock_mhz, sms, ld_warps):
    has_split_ld = (
        row.get("pack_start", "0") not in ("", "0")
        and row.get("pack_end", "0") not in ("", "0")
        and row.get("st_start", "0") not in ("", "0")
        and row.get("st_end", "0") not in ("", "0")
    )
    stages = []
    tma_issue_end = _int_field(row, "tma_issue_end", 0)
    tma_start_raw = _int_field(row, "tma_start", 0)
    tma_end_raw = _int_field(row, "tma_end", 0)
    if tma_start_raw and tma_issue_end and tma_end_raw and tma_start_raw < tma_issue_end < tma_end_raw:
        stages += [
            ("K TMA issue", "tma_start", "tma_issue_end", "#08b557"),
            ("K TMA wait", "tma_issue_end", "tma_end", "#9aa0a6"),
        ]
    else:
        stages.append(("K TMA", "tma_start", "tma_end", "#08b557"))
    mma_issue_end = _int_field(row, "mma_issue_end", 0)
    mma_start_raw = _int_field(row, "mma_start", 0)
    mma_end_raw = _int_field(row, "mma_end", 0)
    pipe = int(row["pipe"])
    iter_id = int(row["iter"])
    qk_label = "QK + PV h0 MMA" if iter_id >= 2 else "QK MMA"
    if mma_start_raw and mma_issue_end and mma_end_raw and mma_start_raw < mma_issue_end < mma_end_raw:
        stages += [
            (f"{qk_label} issue", "mma_start", "mma_issue_end", "#4777c3"),
            (f"{qk_label} tail", "mma_issue_end", "mma_end", "#9aa0a6"),
        ]
    else:
        stages.append((qk_label, "mma_start", "mma_end", "#4777c3"))
    if has_split_ld:
        stages += [
            ("LD", "ld_start", "ld_end", "#1f4f7a"),
            ("PACK", "pack_start", "pack_end", "#e19b2e"),
            ("ST", "st_start", "st_end", "#9a6a2f"),
        ]
    else:
        stages.append(("PACK", "ld_start", "ld_end", "#1f4f7a"))
    v_tma_issue_end = _int_field(row, "v_tma_issue_end", 0)
    v_tma_start_raw = _int_field(row, "v_tma_start", 0)
    v_tma_end_raw = _int_field(row, "v_tma_end", 0)
    if v_tma_start_raw and v_tma_issue_end and v_tma_end_raw and v_tma_start_raw < v_tma_issue_end < v_tma_end_raw:
        stages += [
            ("V TMA issue", "v_tma_start", "v_tma_issue_end", "#00a6a6"),
            ("V TMA wait", "v_tma_issue_end", "v_tma_end", "#9aa0a6"),
        ]
    else:
        stages.append(("V TMA", "v_tma_start", "v_tma_end", "#00a6a6"))
    has_pv_h0 = row.get("pv_h0_start", "0") not in ("", "0")
    has_pv_h1 = row.get("pv_h1_start", "0") not in ("", "0")
    if has_pv_h0 or has_pv_h1:
        if has_pv_h0:
            stages.append(("PV MMA h0", "pv_h0_start", "pv_h0_end", "#8c5fbf"))
        if has_pv_h1:
            stages.append(("PV MMA h1", "pv_h1_start", "pv_h1_end", "#6f47a3"))
        pv_end_raw = _int_field(row, "pv_end", 0)
        pv_h1_end_raw = _int_field(row, "pv_h1_end", 0)
        if pv_h1_end_raw and pv_end_raw and pv_h1_end_raw < pv_end_raw:
            stages.append(("PV MMA tail", "pv_h1_end", "pv_end", "#9aa0a6"))
    else:
        pv_issue_end = _int_field(row, "pv_issue_end", 0)
        pv_start_raw = _int_field(row, "pv_start", 0)
        pv_end_raw = _int_field(row, "pv_end", 0)
        if pv_start_raw and pv_issue_end and pv_end_raw and pv_start_raw < pv_issue_end < pv_end_raw:
            stages += [
                ("PV MMA issue", "pv_start", "pv_issue_end", "#8c5fbf"),
                ("PV MMA tail", "pv_issue_end", "pv_end", "#9aa0a6"),
            ]
        else:
            stages.append(("PV MMA", "pv_start", "pv_end", "#8c5fbf"))
    out = []
    prev_split_end = None
    for name, start_key, end_key, color in stages:
        if start_key not in row or end_key not in row:
            continue
        if row[start_key] == "" or row[end_key] == "":
            continue
        start = int(row[start_key]) - base
        end = int(row[end_key]) - base
        if has_split_ld and name in ("PACK", "ST") and prev_split_end is not None:
            start = max(start, prev_split_end)
        if end <= start:
            continue
        cycles = end - start
        if name.endswith("wait") or name.endswith("tail"):
            rate_kind = "WAIT"
        elif name.startswith("PV MMA h"):
            rate_kind = "MMA4"
        elif name == "PV async tail":
            rate_kind = "PVTAIL"
        elif name.startswith("QK + PV h0 MMA"):
            rate_kind = "MMA12"
        elif "MMA" in name:
            rate_kind = "MMA"
        elif "TMA" in name:
            rate_kind = "TMA"
        else:
            rate_kind = name
        out.append((name, start, end, cycles, color,
                    rate_text(rate_kind, cycles, clock_mhz, sms, ld_warps)))
        if has_split_ld and name in ("LD", "PACK", "ST"):
            prev_split_end = end
    return out


def _int_field(row, key, default=None):
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def pv_stage_warp(row, name):
    pipe = int(row["pipe"])
    if name.startswith("V TMA"):
        return 2 + pipe
    if name == "PV MMA h0":
        return _int_field(row, "pv_h0_warp_id",
                          _int_field(row, "pv_warp_id", 2 + pipe))
    if name == "PV MMA h1" or name == "PV async tail" or name == "PV MMA tail":
        return _int_field(row, "pv_h1_warp_id",
                          _int_field(row, "pv_warp_id", 2 + pipe))
    if name.startswith("PV "):
        return _int_field(row, "pv_warp_id", 2 + pipe)
    return int(row["warp_id"])


def k_tma_stage_warp(row):
    return _int_field(row, "tma_warp_id", int(row["warp_id"]))


def pv_consumer_lane(row, name, consumer_base, consumer_count, consumer_lane_mode):
    if not name.startswith("PV "):
        return None
    pipe = int(row["pipe"])
    warp = pv_stage_warp(row, name)
    if consumer_count == 8:
        first_warp = consumer_base + pipe * 4
        if first_warp <= warp < first_warp + 4:
            half = 1 if name in ("PV MMA h1", "PV async tail", "PV MMA tail") else 0
            lane = (warp - first_warp) * 2 + half
            if consumer_lane_mode == "warp":
                lane //= 2
            return ("ld", (pipe, lane))
    else:
        first_warp = consumer_base + pipe * consumer_count
        if first_warp <= warp < first_warp + consumer_count:
            return ("ld", (pipe, warp - first_warp))
    return None


def write_summary(path, rows, base, clock_mhz, sms, ld_warps, consumer_lane_mode):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "iter",
                "warp",
                "pipe",
                "stage",
                "start_norm",
                "end_norm",
                "cycles",
                "rate_148sm",
            ]
        )
        for row in rows:
            pipe = int(row["pipe"])
            consumer_base = 4 if has_pv_activity(row) else 8
            for name, start, end, cycles, _, rate in stage_items(
                    row, base, clock_mhz, sms, ld_warps):
                if has_consumer_warp_stamps(row) and name in ("LD", "PACK", "ST"):
                    continue
                if name.startswith("K TMA"):
                    warp = k_tma_stage_warp(row)
                elif name.startswith("V TMA") or name.startswith("PV "):
                    warp = pv_stage_warp(row, name)
                else:
                    warp = row["warp_id"]
                w.writerow([row["iter"], warp, row["pipe"], name, start, end, cycles, rate])
            if has_consumer_warp_stamps(row):
                count = consumer_warp_count(row)
                for consumer_warp in range(count):
                    if count == 8:
                        warp = consumer_base + pipe * 4 + consumer_warp // 2
                        suffix = f" h{consumer_warp & 1}"
                        if consumer_lane_mode == "half":
                            warp = f"{warp}.h{consumer_warp & 1}"
                            suffix = ""
                    else:
                        warp = consumer_base + pipe * count + consumer_warp
                        suffix = ""
                    for name, start, end, cycles, _, rate in consumer_stage_items(
                            row, base, clock_mhz, sms, consumer_warp):
                        w.writerow([row["iter"], warp, row["pipe"], name + suffix, start, end, cycles, rate])
            for sync_idx, start, end, cycles in sync_items(row, base):
                w.writerow([row["iter"], "CTA", row["pipe"], f"SYNC{sync_idx}", start, end, cycles, "barrier"])


def write_svg(path, rows, base, clock_mhz, sms, ld_warps, title, consumer_lane_mode):
    max_end = max(
        [end for row in rows
         for _, _, end, _, _, _ in stage_items(row, base, clock_mhz, sms, ld_warps)]
        + [end for row in rows for _, _, end, _ in sync_items(row, base)]
    )
    axis_end = ((max_end + 999) // 1000) * 1000
    scale = 0.24
    left = 132
    label_left = 24
    top = 64
    row_h = 64
    bar_h = 42
    right = 40

    producer_warps = sorted({int(row["warp_id"]) for row in rows})
    pipes = sorted({int(row["pipe"]) for row in rows})
    lanes = [("producer", warp, f"QK/PV warp{warp}") for warp in producer_warps]
    has_pv = any(has_pv_activity(row) for row in rows)
    consumer_base = 4 if has_pv else 8
    per_consumer = any(has_consumer_warp_stamps(row) for row in rows)
    consumer_count = max((consumer_warp_count(row) for row in rows), default=4)
    if has_pv:
        lanes += [("v_tma", pipe, f"K/V TMA warp{2 + pipe}") for pipe in pipes]
    if per_consumer:
        for pipe in pipes:
            lane_count = consumer_count // 2 if consumer_count == 8 and consumer_lane_mode == "warp" else consumer_count
            for consumer_warp in range(lane_count):
                if consumer_count == 8 and consumer_lane_mode == "warp":
                    warp = consumer_base + pipe * 4 + consumer_warp
                    label = f"LD/PACK/ST warp{warp}"
                elif consumer_count == 8:
                    warp = consumer_base + pipe * 4 + consumer_warp // 2
                    label = f"LD/PACK/ST warp{warp}.h{consumer_warp & 1}"
                else:
                    warp = consumer_base + pipe * consumer_count + consumer_warp
                    label = f"LD/PACK/ST warp{warp}"
                lanes.append(("ld", (pipe, consumer_warp), label))
    else:
        lanes += [
            ("ld", pipe, f"LD/PACK/ST warps{consumer_base + pipe * 4}-{consumer_base + pipe * 4 + 3}")
            for pipe in pipes
        ]
    if has_pv:
        pv_issue_warps = sorted({
            pv_stage_warp(row, name)
            for row in rows
            for name, _, _, _, _, _ in stage_items(row, base, clock_mhz, sms, ld_warps)
            if name.startswith("PV ")
            and pv_stage_warp(row, name) not in producer_warps
            and pv_stage_warp(row, name) != 2 + int(row["pipe"])
            and not (
                per_consumer and pv_consumer_lane(
                    row, name, consumer_base, consumer_count, consumer_lane_mode))
        })
        lanes += [("pv_issue", warp, f"PV issue warp{warp}") for warp in pv_issue_warps]
    lane_index = {(kind, ident): idx for idx, (kind, ident, _) in enumerate(lanes)}

    width = int(left + axis_end * scale + right)
    height = top + len(lanes) * row_h + 36

    def x(cyc):
        return left + cyc * scale

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{label_left}" y="26" font-family="Arial, sans-serif" font-size="17" fill="#222">{html.escape(title)}</text>',
    ]

    for grid in range(0, axis_end + 1, 1000):
        gx = x(grid)
        parts.append(
            f'<line x1="{gx:.1f}" y1="42" x2="{gx:.1f}" y2="{height - 22}" stroke="#e3e8ef" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="43" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#667">{grid}</text>'
        )

    for idx, (_, _, label) in enumerate(lanes):
        y = top + idx * row_h
        parts.append(
            f'<text x="{label_left}" y="{y + 18}" font-family="Arial, sans-serif" font-size="13" fill="#333">{html.escape(label)}</text>'
        )
        parts.append(
            f'<line x1="{left}" y1="{y + bar_h + 8}" x2="{x(axis_end):.1f}" y2="{y + bar_h + 8}" stroke="#ccd5e0" stroke-width="1"/>'
        )

    blocks = []
    for row in rows:
        pipe = int(row["pipe"])
        warp = int(row["warp_id"])
        for name, start, end, cycles, color, rate in stage_items(
                row, base, clock_mhz, sms, ld_warps):
            if per_consumer and name in ("LD", "PACK", "ST"):
                continue
            if name in ("LD", "PACK", "ST"):
                lane = ("ld", pipe)
            elif name.startswith("K TMA"):
                k_warp = k_tma_stage_warp(row)
                if k_warp == 2 + pipe:
                    lane = ("v_tma", pipe)
                else:
                    lane = ("producer", k_warp)
            elif name.startswith("V TMA"):
                lane = ("v_tma", pipe)
            elif name.startswith("PV "):
                pv_warp = pv_stage_warp(row, name)
                if pv_warp in producer_warps:
                    lane = ("producer", pv_warp)
                elif pv_warp == 2 + pipe:
                    lane = ("v_tma", pipe)
                else:
                    lane = pv_consumer_lane(
                        row, name, consumer_base, consumer_count, consumer_lane_mode
                    ) if per_consumer else None
                    if lane is None:
                        lane = ("pv_issue", pv_warp)
            else:
                lane = ("producer", warp)
            blocks.append((start, lane_index[lane], row, name, end, cycles, color, rate))
        if per_consumer:
            for consumer_warp in range(consumer_count):
                for name, start, end, cycles, color, rate in consumer_stage_items(
                        row, base, clock_mhz, sms, consumer_warp):
                    if consumer_count == 8 and consumer_lane_mode == "warp":
                        lane = ("ld", (pipe, consumer_warp // 2))
                        name = f"{name} h{consumer_warp & 1}"
                    else:
                        lane = ("ld", (pipe, consumer_warp))
                    blocks.append((start, lane_index[lane], row, name, end, cycles, color, rate))

    for start, idx, row, name, end, cycles, color, rate in sorted(
            blocks, key=lambda b: (b[0], b[1], int(b[2]["iter"]), b[3])):
        y = top + idx * row_h
        pipe = int(row["pipe"])
        iteration = int(row["iter"])
        bx = x(start)
        bw = max(1.0, (end - start) * scale)
        parts.append(
            f'<rect x="{bx:.1f}" y="{y}" width="{bw:.1f}" height="{bar_h}" fill="{color}" stroke="#0f3768" stroke-width="1"/>'
        )
        cx = bx + bw / 2
        if bw > 82:
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 16}" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="white">{name} i{iteration} p{pipe}</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 30}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="white">{cycles} cyc</text>'
            )
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 40}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="white">{html.escape(rate)}</text>'
            )
        elif bw > 44:
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 18}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="white">{name}</text>'
            )
            bx = x(start)
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 33}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="white">i{iteration} p{pipe}</text>'
            )
        else:
            parts.append(
                f'<text x="{cx:.1f}" y="{y + 25}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="white">{name}</text>'
            )
    sync_marks = []
    for row in rows:
        iteration = int(row["iter"])
        pipe = int(row["pipe"])
        for sync_idx, start, end, cycles in sync_items(row, base):
            sync_marks.append((end, start, sync_idx, iteration, pipe, cycles))
    for end, start, sync_idx, iteration, pipe, cycles in sorted(sync_marks):
        ex = x(end)
        parts.append(
            f'<line x1="{ex:.1f}" y1="46" x2="{ex:.1f}" y2="{height - 22}" stroke="#d62728" stroke-width="2" stroke-dasharray="4 3"/>'
        )
        parts.append(
            f'<text x="{ex:.1f}" y="{height - 7}" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="#b32020">S{sync_idx}</text>'
        )

    parts.append("</svg>\n")
    with open(path, "w") as f:
        f.write("\n".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", required=True)
    p.add_argument("--iter-start", type=int, default=24)
    p.add_argument("--num-iters", type=int, default=2)
    p.add_argument("--clock-mhz", type=float, default=1155.0)
    p.add_argument("--sms", type=int, default=148)
    p.add_argument("--ld-warps", type=int, default=1)
    p.add_argument("--summary-csv", required=True)
    p.add_argument("--svg", required=True)
    p.add_argument("--title", default="QK trace timeline")
    p.add_argument("--consumer-lane-mode", choices=["warp", "half"], default="warp")
    args = p.parse_args()

    rows = read_rows(args.trace, args.iter_start, args.num_iters)
    if not rows:
        raise SystemExit("no rows selected")
    base = min(int(row["tma_start"]) for row in rows)
    write_summary(args.summary_csv, rows, base, args.clock_mhz, args.sms, args.ld_warps,
                  args.consumer_lane_mode)
    write_svg(args.svg, rows, base, args.clock_mhz, args.sms, args.ld_warps, args.title,
              args.consumer_lane_mode)


if __name__ == "__main__":
    main()
