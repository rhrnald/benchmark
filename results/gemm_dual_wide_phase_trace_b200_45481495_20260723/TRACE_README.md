# Dual-wide u1 phase trace

This diagnostic is based on clean E7a dual-wide u1 commit `735c6e0`, whose
source SHA-256 is
`37deebeee43b426c92331d8deda6bd4d627ed71ae931a7f519dfc97109e4f33a`.
The trace source is definition commit `87b908b`, SHA-256
`d742bc0331e5faa3b6bbebe4686cdd7222ab97c9a6a416322f69da54c5f4645d`.
It samples block 0 at valid persistent `tile_iter=8`.

## Build

```bash
/usr/local/cuda-12.9/bin/nvcc -std=c++17 -O3 \
  -gencode arch=compute_100a,code=sm_100a -lineinfo -Xptxas=-v \
  gemm256_bf16_16k_dual_wide_phase_trace.cu \
  -o gemm_dual_wide_phase_trace -lcuda
```

CUDA 12.9.86 compiled the trace with 186 registers, zero stack bytes, zero
spills, and one barrier.  `ptxas` reports 2256 bytes of static shared storage
versus 160 bytes for uninstrumented u1; `cuobjdump --dump-resource-usage`
reports `SHARED:3280` versus `SHARED:1184` after including the 1024-byte
dynamic-alignment reservation.  The requested dynamic shared allocation
remains 197632 bytes in both files.

## Run

The trace path always performs one complete untraced warmup launch followed by
one traced launch.  Do not use its runtime as a performance number.

```bash
./gemm_dual_wide_phase_trace \
  --input-init random --trace-csv trace_random.csv
./gemm_dual_wide_phase_trace \
  --input-init random-signed8 --trace-csv trace_random_signed8.csv
```

Before collecting traces on B200, validate the normal null-trace path:

```bash
./gemm_dual_wide_phase_trace --validate --validate-size 512 \
  --validate-pattern pattern
./gemm_dual_wide_phase_trace --validate --validate-size 512 \
  --validate-pattern ones
```

## CSV semantics

All times are device `clock64` cycles and are meaningful only within the sampled
CTA/SM.  `start_rel` and `end_rel` use the first recorded timestamp as zero.

- `kind=interval`: `envelope_cycles == total_cycles`, and `sample_count=1`.
- `kind=aggregate`: `envelope_cycles` spans the first through last sample,
  while `total_cycles` sums only individually timed calls.  Divide by
  `sample_count` for the average timed cost.
- `consumer_wait_a`, `consumer_wait_b0`, and `consumer_wait_b1` each contain
  256 wait samples for a 16K K dimension.
- Each `consumer_mma_b*_issue` row contains 512 issued m128n256k16 MMA
  instructions.  Its timed regions include descriptor setup and issue latency,
  not asynchronous tensor-core execution in isolation.
- `consumer_mma_commit` has 256 samples.  `consumer_final_mma_drain` is the
  last completion-barrier wait after all K stages have been issued.
- Producer rows cover their complete stage-reuse-wait plus TMA-issue loops.
  Their end timestamps mean the producer warp returned from its issue loop,
  not that every outstanding TMA transaction completed globally.
- Epilogue chunks are ordered `(M0,N0)`, `(M0,N1)`, `(M1,N0)`, `(M1,N1)`,
  where each coordinate is a 128x128 C chunk.  A chunk row covers TMEM load,
  shared-memory staging/barriers, and TMA-store issue.  Group 0 drains chunks
  0/1 and group 1 drains chunks 2/3 with commit, wait-group-0, and CTA barrier.
- Back-to-back `clock_overhead` rows estimate timestamp-read overhead.

Blackwell synchronization can defer where a warp actually blocks.  Therefore,
compare warp arrival/end envelopes and repeated distributions; do not interpret
one barrier instruction duration as the complete dependency cost.

To summarize one or more trace CSVs:

```bash
python3 decode_trace.py trace_random_*.csv
```

The decoder prints medians and ranges by event and warp.  Collect independent
processes for each input distribution before drawing conclusions.
