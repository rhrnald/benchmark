#!/usr/bin/env python3
"""Host-only coverage checks for the persistent snake/N-strip mappings."""

from __future__ import annotations


def mapped_tiles(grid_m: int, grid_n: int, macro_m_arg: int,
                 macro_n_arg: int, snake: int, strip: int):
    macro_m = min(grid_m, macro_m_arg)
    macro_n = min(grid_n, macro_n_arg)
    groups_m = (grid_m + macro_m - 1) // macro_m
    groups_n = (grid_n + macro_n - 1) // macro_n
    strips_n = (macro_n + strip - 1) // strip
    items_per_macro = macro_m * strips_n

    emitted = []
    for work in range(groups_m * groups_n * items_per_macro):
        macro_id, local = divmod(work, items_per_macro)
        macro_n_sequence = macro_id % groups_n
        macro_m_id = macro_id // groups_n
        reverse_n = snake != 0 and macro_m_id % 2 == 1
        macro_n_id = groups_n - 1 - macro_n_sequence if reverse_n \
            else macro_n_sequence
        local_m = local % macro_m
        strip_id = local // macro_m
        for offset in range(strip):
            local_n = strip_id * strip + offset
            if snake == 2 and reverse_n:
                local_n = macro_n - 1 - local_n
            tile_m = macro_m_id * macro_m + local_m
            tile_n = macro_n_id * macro_n + local_n
            if (0 <= local_n < macro_n and tile_m < grid_m and
                    0 <= tile_n < grid_n):
                emitted.append((tile_m, tile_n))
    return emitted


def check_coverage(grid_m: int, grid_n: int, macro_m: int, macro_n: int,
                   snake: int, strip: int):
    tiles = mapped_tiles(grid_m, grid_n, macro_m, macro_n, snake, strip)
    expected = {(m, n) for m in range(grid_m) for n in range(grid_n)}
    actual = set(tiles)
    assert len(tiles) == len(expected), (
        grid_m, grid_n, macro_m, macro_n, snake, strip,
        len(tiles), len(expected))
    assert actual == expected, (
        grid_m, grid_n, macro_m, macro_n, snake, strip,
        expected - actual, actual - expected)


def main():
    grids = (1, 2, 3, 15, 16, 17, 63, 64, 65)
    for macro_m, macro_n in ((16, 16), (8, 18), (12, 12)):
        for grid_m in grids:
            for grid_n in grids:
                for snake in (0, 1, 2):
                    for strip in (1, 2, 4, 8):
                        if snake and strip != 1:
                            continue
                        check_coverage(grid_m, grid_n, macro_m, macro_n,
                                       snake, strip)

    baseline = mapped_tiles(64, 64, 16, 16, 0, 1)
    macro_snake = mapped_tiles(64, 64, 16, 16, 1, 1)
    full_snake = mapped_tiles(64, 64, 16, 16, 2, 1)
    assert baseline[1023:1025] == [(15, 63), (16, 0)]
    assert macro_snake[1023:1025] == [(15, 63), (16, 48)]
    assert full_snake[1023:1025] == [(15, 63), (16, 63)]
    print("persistent mapping coverage: ok")
    print("baseline boundary:", baseline[1023:1025])
    print("macro snake boundary:", macro_snake[1023:1025])
    print("full snake boundary:", full_snake[1023:1025])


if __name__ == "__main__":
    main()
