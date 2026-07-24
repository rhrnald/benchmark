# N-split paired ordinary-MMA codegen result

- Definition commit:
  `8dca23bef0941554deb7e3c500075afe32a85f91`
- Exact N-split source SHA-256:
  `cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`
- Generated source SHA-256:
  `1fe1b9987066c9a3df1c8b1f4720f9126187da8912dd29b37c4407da30ef5063`
- Compiler: local CUDA 12.9, `sm_100a`

| normalized main-kernel metric | exact N-split | paired helper |
|---|---:|---:|
| SASS instruction lines | 1913 | 1917 |
| registers / stack / spill | 174 / 0 / 0 | 174 / 0 / 0 |
| static ordinary `UTCHMMA` | 8 | 8 |
| `ELECT` | 32 | 32 |
| `R2UR.BROADCAST` | 65 | 65 |
| branch sites | 154 | 154 |
| `BSSY` / `BSYNC` | 18 / 18 | 18 / 18 |
| `SYNCS.PHASECHK` | 48 | 48 |

The paired inline PTX helper does not reduce repeated B-descriptor lowering
and adds four normalized SASS instructions.  It fails the predeclared
codegen gate, so no B200 correctness or timing run was performed.
