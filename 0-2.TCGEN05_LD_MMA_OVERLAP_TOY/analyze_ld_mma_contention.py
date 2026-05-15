#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def f(row, key):
    return float(row[key])


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the tcgen05.ld vs tcgen05.mma contention claim."
    )
    parser.add_argument(
        "--claim-csv",
        default="0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_contention_claim.csv",
    )
    parser.add_argument(
        "--report-md",
        default="0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_contention_report.md",
    )
    args = parser.parse_args()

    claim_path = Path(args.claim_csv)
    with claim_path.open() as fp:
        rows = list(csv.DictReader(fp))
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one claim row in {claim_path}, got {len(rows)}")
    row = rows[0]

    lines = [
        "# tcgen05.ld vs tcgen05.mma Contention",
        "",
        "## Reproduction",
        "",
        "```bash",
        "cd /home/snu_avq1/workspace/chaewon/benchmark",
        "make tcgen05_ld_mma_overlap_toy",
        "./build/0-2.TCGEN05_LD_MMA_OVERLAP_TOY/tcgen05_ld_mma_overlap_toy \\",
        "  --peak-only --blocks 4096 --repeats 8192 --ld-repeats 81920 \\",
        "  --warmup 2 --iters 5",
        "python3 0-2.TCGEN05_LD_MMA_OVERLAP_TOY/analyze_ld_mma_contention.py",
        "```",
        "",
        "## Result",
        "",
        f"- LD-only peak: `{f(row, 'ld_peak_TBps'):.3f} TB/s`",
        f"- LD while MMA overlaps: `{f(row, 'overlap_ld_TBps'):.3f} TB/s`",
        f"- Observed LD drop: `{f(row, 'observed_ld_drop_TBps'):.3f} TB/s` "
        f"= `{f(row, 'observed_ld_drop_B_per_cycle_per_SM'):.1f} B/cyc/SM`",
        f"- Reference MMA peak: `{f(row, 'reference_mma_TFLOP_per_s'):.3f} TFLOP/s`",
        f"- Acc32 C tile size: `{f(row, 'acc32_tile_bytes'):.0f} B`",
        f"- Reference MMA C-read demand: `{f(row, 'reference_mma_C_read_TBps'):.3f} TB/s` "
        f"= `{f(row, 'reference_mma_C_read_B_per_cycle_per_SM'):.1f} B/cyc/SM`",
        f"- Ratio, observed drop / C-read demand: `{f(row, 'drop_to_C_read_ratio'):.3f}`",
        "",
        "## Claim",
        "",
        "The observed loss in tcgen05.ld bandwidth is approximately equal to the "
        "C-read bandwidth demanded by peak tcgen05.mma. This supports the claim "
        "that tcgen05.mma accumulator C-side traffic competes with tcgen05.ld for "
        "the same TMEM read/fabric resource.",
        "",
    ]

    report_path = Path(args.report_md)
    report_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
