#!/usr/bin/env python3
"""Summarize one or more dual-wide phase-trace CSV files."""

import csv
import statistics
import sys
from collections import defaultdict


def main(paths):
    if not paths:
        raise SystemExit("usage: decode_trace.py TRACE.csv [TRACE.csv ...]")

    grouped = defaultdict(list)
    for path in paths:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                key = (
                    int(row["slot"]),
                    row["event"],
                    int(row["warp"]),
                    row["kind"],
                )
                grouped[key].append(
                    (
                        int(row["envelope_cycles"]),
                        int(row["total_cycles"]),
                        float(row["avg_cycles_per_sample"]),
                    )
                )

    print(
        "slot,event,warp,kind,n,envelope_median,envelope_min,envelope_max,"
        "total_median,total_min,total_max,avg_sample_median"
    )
    for (slot, event, warp, kind), values in sorted(grouped.items()):
        envelope = [value[0] for value in values]
        total = [value[1] for value in values]
        average = [value[2] for value in values]
        print(
            f"{slot},{event},{warp},{kind},{len(values)},"
            f"{statistics.median(envelope):g},{min(envelope)},{max(envelope)},"
            f"{statistics.median(total):g},{min(total)},{max(total)},"
            f"{statistics.median(average):.6f}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
