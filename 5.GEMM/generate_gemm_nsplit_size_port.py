#!/usr/bin/env python3
import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"expected exactly one occurrence of {old!r}, got {count}")
    return text.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", required=True, type=int)
    parser.add_argument("--macro-m", required=True, type=int)
    parser.add_argument("--macro-n", required=True, type=int)
    args = parser.parse_args()

    if args.size not in (8192, 16384, 32768):
        raise ValueError("size must be 8192, 16384, or 32768")
    if args.macro_m <= 0 or args.macro_n <= 0:
        raise ValueError("macro dimensions must be positive")

    text = Path(args.base).read_text()
    text = replace_once(
        text,
        "static constexpr int kBenchmarkSize = 16384;",
        f"static constexpr int kBenchmarkSize = {args.size};",
    )
    text = replace_once(
        text,
        "static constexpr int kPersistentMacroM = 16;",
        f"static constexpr int kPersistentMacroM = {args.macro_m};",
    )
    text = replace_once(
        text,
        "static constexpr int kPersistentMacroN = 16;",
        f"static constexpr int kPersistentMacroN = {args.macro_n};",
    )
    Path(args.output).write_text(text)


if __name__ == "__main__":
    main()
