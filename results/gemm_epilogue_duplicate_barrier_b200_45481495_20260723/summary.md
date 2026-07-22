# Duplicate epilogue barrier ablation

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Parent E3.2 definition/result: `750057b` / `8c1a0f2`
- Candidate E3.3 definition: `d79fc0c`
- Parent binary SHA-256:
  `cbbf46a9c8aa12d25b54f31e1a99bb477cb9123e270c9ea79195d97455fe6a27`
- Candidate source SHA-256:
  `8bae50c3fabda57abc44f5e6a076d0f1a59bd1b350ca8fcb632f8d8e68e3a762`
- Candidate binary SHA-256:
  `e5278590c1c494953d145495be07356ac782853690c53a54ebd54b9ba5731303`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

The helper always completes both C-store groups and ends with a CTA barrier
after `wait_group.read 0`.  The candidate removes only the immediately
following caller CTA barrier.  The helper's source-buffer reuse barrier and
the later sink/publication barrier remain.

Both 512 pattern and ones full-C validations passed with `max_abs=0` and
`max_rel=0`.

## Codegen

| metric | E3.2 parent | E3.3 candidate |
|---|---:|---:|
| registers / spills | 178 / 0 | 178 / 0 |
| effective instructions | 1918 | 1917 |
| `BAR.SYNC` | 16 | 15 |
| TMA store / commit / source-read wait | 4 / 2 / 2 | 4 / 2 / 2 |

## Performance

All values are event TFLOP/s.  Delta is E3.3 relative to the same-pass E3.2
parent.

| input | E3.2 samples | E3.2 mean | E3.3 samples | E3.3 mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1740.500, 1738.591, 1736.311 | **1738.467** | 1739.560, 1735.923, 1737.892 | **1737.792** | -0.0389% | -0.0388% |
| BF16 uniform `[-8,8)` | 1507.737, 1512.642, 1511.190 | **1510.523** | 1512.928, 1506.695, 1506.367 | **1508.663** | -0.1231% | -0.1227% |

Decision: neutral/reject.  Both deltas are far below the 0.5% noise gate and
there is no evidence that this fixed epilogue barrier is throughput-critical
at 16K.  E3.1, E3.2, and E3.3 all failed to produce a measurable gain, so the
entire experimental epilogue chain should be reverted to the clean B0 source
before starting the next bottleneck experiment.
