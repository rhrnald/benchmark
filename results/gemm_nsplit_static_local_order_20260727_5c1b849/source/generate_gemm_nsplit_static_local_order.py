#!/usr/bin/env python3
import argparse
from pathlib import Path


MFAST = """    const int macro_n = macro_id % persistent_groups_n;
    const int macro_m = macro_id / persistent_groups_n;
    const int local_m = local % persistent_macro_m;
    const int local_n = local / persistent_macro_m;
"""

NFAST = """    const int macro_n = macro_id % persistent_groups_n;
    const int macro_m = macro_id / persistent_groups_n;
    const int local_n = local % persistent_macro_n;
    const int local_m = local / persistent_macro_n;
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--order", choices=("mfast", "nfast"), required=True)
    args = parser.parse_args()

    text = Path(args.base).read_text()
    if text.count(MFAST) != 1:
        raise RuntimeError("base is not the expected static 8x16 M-fast source")
    if args.order == "nfast":
        text = text.replace(MFAST, NFAST)
        text = text.replace(
            "scheduler=static_%dx%d_mfast",
            "scheduler=static_%dx%d_nfast",
        )
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
