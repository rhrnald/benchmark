# Early Commit Race Microbenchmark

This folder measures how early a consumer `tcgen05.ld` can start relative to a
full `tcgen05.commit`/wait point while still reading stable TMEM data.

The benchmark is intentionally independent from `0.attention`:

1. A producer warp issues `target_mmas` BF16 MMAs into a target TMEM tile.
2. It can issue only `early_target_mmas` of those target MMAs before an early
   `tcgen05.commit`.
3. It can also issue `early_extra_mmas` unrelated MMAs into a different TMEM
   tile before the early commit.
4. The consumer warp waits on that early commit, then optionally executes
   `delay_cycles` dependent dummy ALU instructions in the same inline PTX block
   as the timestamp and `tcgen05.ld.x64`. The delay uses a runtime-counted loop
   with an `add.u32` in the loop body and carries the loop result into the TMEM
   load address through a runtime zero mask, so the effective address is
   unchanged while the load still depends on the dummy operations.
5. After the producer's full commit completes, the consumer loads the same TMEM
   address again and compares all 32 lanes x 64 registers.

If the early load differs from the post-full-commit load, the sample is counted
as unsafe.

## Build

```bash
cd /path/to/benchmark/early_commit
make build
```

`TARGET_MMAS` and `FULL_EXTRA_MMAS` are compile-time defines. The generated
binary name includes those values, for example `early_commit_race_t8_f0`.

`early_target_mmas` and `early_extra_mmas` are selected by host-side template
dispatch. One binary can sweep multiple values, but each kernel launch is a
separate specialization with constant producer loop bounds. `delay_cycles` is a
runtime loop count so the dummy ALU loop cannot be folded away as a compile-time
constant. The host uses a raw no-delay kernel for `delay_cycles=0` and a
dummy-delay kernel for `delay_cycles>0` so the baseline path is not perturbed by
the delayed-load code.

## Smoke

```bash
make smoke
```

Outputs:

- `log/smoke_detail.csv`
- `log/smoke_summary.csv`

## Main Sweep

The default main sweep uses only the target MMA sequence:

- `target_mmas=8`
- `full_extra_mmas=0`
- `early_target_mmas=8`
- `early_extra_mmas=0`
- `delay_cycles=0..128` as runtime dummy ALU loop counts.

```bash
make run
```

To test committing before all target MMAs have been issued:

```bash
make run EARLY_TARGETS=0,1,2,3,4,5,6,7,8 EARLY_EXTRAS=0 DELAYS=0,1,2,3,4,5,6,7,8,10,12,16,24,32,48,64,96,128
```

To change the compile-time target/full sequence, rebuild with Make variables:

```bash
make run TARGET_MMAS=8 FULL_EXTRA_MMAS=0 EARLY_TARGETS=7 EARLY_EXTRAS=0
```

## Summary Columns

- `safe_rate`: fraction of CTAs where early LD matched the full-commit reference.
- `avg_ld_start_ahead`: average `full_done_end - ld_start` cycles.
  Positive means LD started before full commit completion.
- `avg_ld_end_ahead`: average `full_done_end - ld_end` cycles.
  Positive means LD finished before full commit completion.
- `max_safe_ld_start_ahead`: largest observed safe head start.
- `min_unsafe_ld_start_ahead`: smallest observed unsafe head start.
- `avg_early_wait_to_ld_start`: average cycles from early wait completion to LD
  start.
- `avg_commit_issue_cycles`: cycles spent issuing the early commit instruction.
- `avg_remaining_target_issue_delay`: gap from early commit issue completion to
  the next target MMA issue start.
- `avg_remaining_target_issue_cycles`: cycles to issue the target MMAs that were
  intentionally left after early commit.
- `avg_ld_start_after_remaining_target_issue_start`: positive means LD started
  after the remaining target MMA issue sequence began.
- `avg_ld_start_after_target_issue_end`: positive means LD started after the
  remaining target MMA issue sequence finished.
- `avg_mismatch_words`: average mismatched register count across 32 lanes x 64 regs.

The detail CSV keeps per-CTA timings and signatures for debugging.
Use `early_wait_to_ld_start` for the actual measured delay in cycles; the
`delay_cycles` argument is only the requested dummy ALU loop count. The
early LD is placed after that inline PTX loop, so the requested delay is attached
to the load instead of only to a standalone timestamp; the loop result is kept
live through the load address dependency.
Any non-negative delay count accepted by `--delays` can be used.
