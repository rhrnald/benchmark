# Epilogue pre-fence barrier ablation

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9
- Baseline definition/result: `8add902` / `ae74998`
- Candidate definition: `1c81f68`
- Baseline binary SHA-256:
  `c8cf7f415a8933c516d90725f46eceb14e66fc0168f0fffe45566d048e904667`
- Candidate source SHA-256:
  `b771f10078f984a3fa84d8b3d0be32cd47b1b32b749201446ba16a51cdc6d8f5`
- Candidate binary SHA-256:
  `d9254acc4b99176584e64d87629ccb081634bb026181e09c3442edbf8d550568`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-rotated process pairs per input distribution

The candidate removes only the CTA barrier between each C chunk's distributed
generic shared-memory stores and the per-writer
`fence.proxy.async.shared::cta`.  The required post-fence CTA barrier, TMA
issue, commit, completion wait, and buffer-reuse barrier remain unchanged.

Both 512 pattern and ones full-C validations passed with `max_abs=0` and
`max_rel=0`.

## Codegen

| metric | baseline | candidate |
|---|---:|---:|
| registers / spills | 178 / 0 | 178 / 0 |
| static instructions | 1936 | 1928 |
| CTA `BAR` instructions | 29 | 25 |

MMA, TMA load/store, async fence, commit, and wait counts are unchanged.  The
four removed static barriers correspond to one removed barrier per 128x128 C
chunk.

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass baseline.

| input | baseline samples | baseline mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1739.709, 1737.317, 1738.871 | **1738.632** | 1737.695, 1741.731, 1738.211 | **1739.212** | +0.0334% | +0.0334% |
| BF16 uniform `[-8,8)` | 1507.980, 1506.568, 1509.729 | **1508.092** | 1504.772, 1507.995, 1514.073 | **1508.947** | +0.0566% | +0.0566% |

Decision: neutral.  The source follows the canonical writer-fence-barrier-TMA
ordering and removes real instructions without a measured regression, but both
deltas are far below the 0.5% noise gate.  Do not claim a throughput gain from
this change alone.  Retain it temporarily only as the parent of the separate
`wait_group.read` and final-barrier ablations; revert the chain if those changes
also fail to produce a material result.
