#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def f(row, name):
    value = row.get(name, "")
    return float(value) if value not in ("", None) else 0.0


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize(name, values):
    return {
        "name": name,
        "n": len(values),
        "min": min(values) if values else 0.0,
        "p05": pct(values, 0.05),
        "mean": mean(values) if values else 0.0,
        "p95": pct(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def probability(v_values, e_values, alpha):
    n = min(len(v_values), len(e_values))
    if n == 0:
        return 0.0
    ok = sum(1 for i in range(n) if v_values[i] < e_values[i] + alpha)
    return ok / n


def group_race(rows):
    grouped = {}
    for row in rows:
        delay = int(f(row, "delay_cycles"))
        grouped.setdefault(delay, []).append(row)
    return grouped


def parse_alphas(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start, stop, step = [int(x) for x in part.split(":")]
            values.extend(range(start, stop + 1, step))
        else:
            values.append(int(part))
    return sorted(set(values))


def write_rows(path, rows, fieldnames):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-csv", required=True)
    parser.add_argument("--race-detail-csv")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--compare-csv")
    parser.add_argument("--out-md")
    parser.add_argument("--alphas", default="0:256:8")
    args = parser.parse_args()

    model = read_csv(args.model_csv)
    alphas = parse_alphas(args.alphas)

    e = [f(row, "early_wait_end") for row in model]
    w = [f(row, "full_wait_end") for row in model]
    t_late = [f(row, "t_late_wait") for row in model]
    t_ready_start = [f(row, "t_ready_from_commit_start") for row in model]
    t_ready_end = [f(row, "t_ready_from_commit_end") for row in model]
    v_late = [f(row, "v_est_late_wait") for row in model]
    v_ready_start = [f(row, "v_est_ready_commit_start") for row in model]
    v_ready_end = [f(row, "v_est_ready_commit_end") for row in model]

    overlap_rows = []
    for alpha in alphas:
        overlap_rows.append(
            {
                "alpha": alpha,
                "p_v_late_lt_e_alpha": f"{probability(v_late, e, alpha):.6f}",
                "p_v_ready_start_lt_e_alpha": f"{probability(v_ready_start, e, alpha):.6f}",
                "p_v_ready_end_lt_e_alpha": f"{probability(v_ready_end, e, alpha):.6f}",
            }
        )
    write_rows(
        args.out_csv,
        overlap_rows,
        [
            "alpha",
            "p_v_late_lt_e_alpha",
            "p_v_ready_start_lt_e_alpha",
            "p_v_ready_end_lt_e_alpha",
        ],
    )

    compare_rows = []
    if args.race_detail_csv:
        race_rows = read_csv(args.race_detail_csv)
        for delay, rows in sorted(group_race(race_rows).items()):
            safe = sum(1 for row in rows if int(f(row, "safe")) == 1)
            alpha_values = [f(row, "early_wait_to_ld_start") for row in rows]
            alpha_mean = mean(alpha_values) if alpha_values else 0.0
            compare_rows.append(
                {
                    "delay_cycles": delay,
                    "n": len(rows),
                    "safe_rate": f"{safe / len(rows):.6f}" if rows else "0.000000",
                    "alpha_min": f"{min(alpha_values):.3f}" if alpha_values else "0.000",
                    "alpha_mean": f"{alpha_mean:.3f}",
                    "alpha_max": f"{max(alpha_values):.3f}" if alpha_values else "0.000",
                    "pred_late_at_alpha_mean": f"{probability(v_late, e, alpha_mean):.6f}",
                    "pred_ready_start_at_alpha_mean": f"{probability(v_ready_start, e, alpha_mean):.6f}",
                    "pred_ready_end_at_alpha_mean": f"{probability(v_ready_end, e, alpha_mean):.6f}",
                    "ld_start_after_target_issue_end_min": f"{min(f(row, 'ld_start_after_target_issue_end') for row in rows):.3f}",
                    "ld_start_after_target_issue_end_mean": f"{mean(f(row, 'ld_start_after_target_issue_end') for row in rows):.3f}",
                    "ld_start_after_target_issue_end_max": f"{max(f(row, 'ld_start_after_target_issue_end') for row in rows):.3f}",
                }
            )
        if args.compare_csv:
            write_rows(
                args.compare_csv,
                compare_rows,
                [
                    "delay_cycles",
                    "n",
                    "safe_rate",
                    "alpha_min",
                    "alpha_mean",
                    "alpha_max",
                    "pred_late_at_alpha_mean",
                    "pred_ready_start_at_alpha_mean",
                    "pred_ready_end_at_alpha_mean",
                    "ld_start_after_target_issue_end_min",
                    "ld_start_after_target_issue_end_mean",
                    "ld_start_after_target_issue_end_max",
                ],
            )

    if args.out_md:
        summaries = [
            summarize("E early_wait_end", e),
            summarize("W full_wait_end", w),
            summarize("W-E", [wi - ei for wi, ei in zip(w, e)]),
            summarize("T_late_wait", t_late),
            summarize("T_ready_from_commit_start", t_ready_start),
            summarize("T_ready_from_commit_end", t_ready_end),
            summarize("V_est_late_wait", v_late),
            summarize("V_est_ready_commit_start", v_ready_start),
            summarize("V_est_ready_commit_end", v_ready_end),
        ]

        def first_full(column):
            for row in overlap_rows:
                if float(row[column]) >= 1.0:
                    return row["alpha"]
            return ""

        lines = [
            "# Early Commit Model Analysis",
            "",
            f"model_csv: `{Path(args.model_csv).resolve()}`",
            f"model_rows: {len(model)}",
            "",
            "## Distributions",
            "",
            "| quantity | n | min | p05 | mean | p95 | max |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for item in summaries:
            lines.append(
                f"| {item['name']} | {item['n']} | {item['min']:.3f} | "
                f"{item['p05']:.3f} | {item['mean']:.3f} | {item['p95']:.3f} | "
                f"{item['max']:.3f} |"
            )
        lines.extend(
            [
                "",
                "## Alpha Thresholds",
                "",
                "| V estimate | first sampled alpha with P(V < E + alpha)=1.0 |",
                "|---|---:|",
                f"| W - T_late_wait | {first_full('p_v_late_lt_e_alpha')} |",
                f"| W - T_ready_from_commit_start | {first_full('p_v_ready_start_lt_e_alpha')} |",
                f"| W - T_ready_from_commit_end | {first_full('p_v_ready_end_lt_e_alpha')} |",
            ]
        )
        if compare_rows:
            lines.extend(
                [
                    "",
                    "## Race Comparison",
                    "",
                    "| delay | actual safe | alpha mean | pred late | pred ready-start | pred ready-end | LD-target mean |",
                    "|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in compare_rows:
                lines.append(
                    f"| {row['delay_cycles']} | {row['safe_rate']} | {row['alpha_mean']} | "
                    f"{row['pred_late_at_alpha_mean']} | "
                    f"{row['pred_ready_start_at_alpha_mean']} | "
                    f"{row['pred_ready_end_at_alpha_mean']} | "
                    f"{row['ld_start_after_target_issue_end_mean']} |"
                )
        Path(args.out_md).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
