# E4c aligned dynamic shared-memory declaration

- Date: 2026-07-23
- Vast.ai instance: `45481495`
- GPU: NVIDIA B200, 1000 W power limit
- Driver / toolkit: 580.126.09 / CUDA 12.9.86
- Macro-free E2a parent: `3d2d0a4`
- E4c definition: `539132a`
- Protocol: one case per process, warmup 1, timed launches 5, three
  order-balanced process pairs per input distribution
- Baseline source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Candidate source SHA-256:
  `cc022befc62f59cd5d5a3040b9a4e8de9cbe48f732f90255e321acf61923dd17`
- Baseline / candidate binary SHA-256:
  `5d4127162b83ec0584b760290d4f7542b71d7588838a012fa4201e74cd7f6023` /
  `616ac9c455177a7197f0a3679ad5afacd88ad535481f2f74369dcfb19e530ff1`

The only source change replaces manual integer round-up of the dynamic shared
pointer with an explicitly 1024-byte-aligned `extern __shared__` declaration.
This preserves the same 0x400 shared base and buffer offsets but keeps the
pointer in the compiler's shared address space.

Candidate pattern and ones full-C validations at size 512 passed exactly.

## Code generation

| metric | macro-free E2a | aligned declaration |
|---|---:|---:|
| registers / spills | 174 / 0 | 166 / 0 |
| stack | 0 B | 0 B |
| static shared | 1184 B | 2048 B |
| `ST.E.128` / `STS.128` | 128 / 0 | 0 / 128 |
| `IMAD.WIDE` in complete cubin | 77 | 69 |
| MMA / TMA load / TMA store / TMEM load | 8 / 21 / 4 / 8 | unchanged |

## Performance

All values are event TFLOP/s.  Delta is the candidate relative to the
same-pass parent.

| input | parent samples | parent mean | candidate samples | candidate mean | ratio-of-means delta | paired delta |
|---|---|---:|---|---:|---:|---:|
| BF16 uniform `[0,1)` | 1750.993, 1746.954, 1747.449 | **1748.465** | 1737.244, 1735.529, 1735.215 | **1735.996** | **-0.7132%** | **-0.7131%** |
| BF16 uniform `[-8,8)` | 1514.389, 1513.566, 1514.284 | **1514.080** | 1506.562, 1506.025, 1512.404 | **1508.330** | **-0.3797%** | **-0.3797%** |

All six paired deltas are negative.  Lower register count and replacement of
the generic-looking vector store opcode do not translate to higher throughput
on B200; instruction form alone was not a valid performance proxy.  E4c is
rejected without an extended run.  The result also rules out combining this
declaration with E4a/E4b before either component wins independently.
