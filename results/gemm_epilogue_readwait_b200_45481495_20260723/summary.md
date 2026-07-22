# TMA store read-wait ablation

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Parent E3.1 definition/result: `1c81f68` / `312ce3d`
- Candidate E3.2 definition: `750057b`
- Parent binary SHA-256:
  `d9254acc4b99176584e64d87629ccb081634bb026181e09c3442edbf8d550568`
- Candidate source SHA-256:
  `efcec0af21845013f2c52af9742ba4080d28013116114e08d714816dc7f978cb`
- Candidate binary SHA-256:
  `cbbf46a9c8aa12d25b54f31e1a99bb477cb9123e270c9ea79195d97455fe6a27`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

The candidate changes only the C-store wait from
`cp.async.bulk.wait_group 0` to `cp.async.bulk.wait_group.read 0`.  The latter
waits until TMA has consumed the source shared-memory buffer, which is the
required condition before this kernel overwrites that buffer.  Every output
tile is unique and the kernel never reads C, so destination-global completion
is not required at this point.  Per-writer proxy fences, the pre-issue CTA
barrier, commit, source-read wait, and the post-wait CTA barrier remain.

Both 512 pattern and ones full-C validations passed with `max_abs=0` and
`max_rel=0`.

## Codegen

| metric | E3.1 full wait | E3.2 read wait |
|---|---:|---:|
| registers / spills | 178 / 0 | 178 / 0 |
| effective instructions | 1920 | 1918 |
| `CCTL.IVALL` | 2 | 0 |
| `UTMASTG` / commit / `DEPBAR` | 4 / 2 / 2 | 4 / 2 / 2 |
| CTA `BAR` instructions | 25 | 25 |

The only opcode-count change is removal of the two `CCTL.IVALL` instructions,
one after each unrolled store group.  Resource usage is unchanged.

## Performance

All values are event TFLOP/s.  Delta is E3.2 relative to the same-pass E3.1
parent.

| input | E3.1 samples | E3.1 mean | E3.2 samples | E3.2 mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1738.242, 1741.279, 1740.648 | **1740.056** | 1738.277, 1741.844, 1736.788 | **1738.970** | -0.0625% | -0.0624% |
| BF16 uniform `[-8,8)` | 1516.364, 1510.403, 1508.690 | **1511.819** | 1506.829, 1509.606, 1507.697 | **1508.044** | -0.2497% | -0.2491% |

Decision: no measurable throughput benefit.  Both deltas are below the 0.5%
noise gate, although their direction is negative.  The weaker wait is
correctness-safe and removes real instructions, but it is not adopted as a
standalone optimization.  Keep it only temporarily as the parent for the
separate final duplicate-barrier ablation E3.3, then revert the epilogue chain
unless the combined result crosses the adoption gate.
