# 16K dense TMA-only and locality ablation

Date: 2026-07-28

## 결론

현재 GEMM은 memory locality에 의미 있게 민감하다. 동일한 TMA 요청량에서
A 또는 B 주소 한쪽만 cache-resident하게 만들면 통신-only가 각각 약
4.4--4.7% 빨라지고, A/B 모두 cache-resident하게 만들면 9.3--9.5%
빨라졌다.

다만 이 결과만으로 **L2 bandwidth가 GEMM의 유일한 병목**이라고 확정할
수는 없다. Nsight Compute hardware counter는 Vast host 정책의
`ERR_NVGPUCTRPERM`으로 읽지 못했다. 실측 timing이 직접 증명하는 것은
다음 두 가지다.

1. 현재 dense TMA address stream에 약 9%의 memory-hierarchy/locality
   headroom이 있다.
2. 독립 TMA-only service time이 전체 GEMM 시간의 약 48--54%이므로 통신은
   무시할 수 없는 큰 구성요소다.

완전한 A/B reuse가 줄인 TMA-only 시간은 약 0.22 ms다. 이 감소가 실제
GEMM에 그대로 반영된다는 낙관적 상한은 `[0,1)` 약 4.6%,
`[-8,8)` 약 4.2%다. 실제 GEMM에서는 TMA와 MMA가 겹치므로 실현 가능한
이득은 이보다 작을 수 있다.

## TMA-only 정의

기존 canonical과 다음 항목을 동일하게 유지했다.

- 16K square, BF16 A/B
- 148 persistent CTA
- static `8x16`, macro N-fast / local M-fast
- 실제 output tile 순서
- K64당 A `256x64` 32 KiB TMA 1회
- K64당 B0/B1 `64x128` 16 KiB TMA 각 1회
- 3-stage SMEM ring과 최대 3-stage outstanding
- dynamic SMEM 197632 B로 SM당 CTA 하나

MMA, TMEM allocation과 FP32 C store만 제거했다. 각 stage slot은 해당
TMA completion을 기다린 뒤 재사용하고, tile을 마치기 전에 마지막 세
stage의 completion을 모두 기다린다.

한 output tile의 K64당 payload는 64 KiB이고 output tile은 4096개,
K64 stage는 256개다.

```text
logical bytes = 4096 * 256 * 65536
              = 68,719,476,736 B
              = 64 GiB
```

표의 logical TB/s는 이 중복 요청량을 시간으로 나눈 값이다. HBM의
실제 physical byte/s가 아니다.

## Dense TMA-only와 GEMM

각 cell은 독립 프로세스 4개, 프로세스당 W1/I5이며 ABBA 형태로 실행
순서를 균형화했다.

| input | GEMM TFLOP/s | GEMM ms | dense TMA-only ms | logical TB/s | TMA-only/GEMM |
|---|---:|---:|---:|---:|---:|
| `[0,1)` | 1832.607 +/- 2.784 | 4.799780 +/- 0.007292 | 2.601502 +/- 0.006145 | 26.415 | 54.20% |
| `[-8,8)` | 1619.235 +/- 2.932 | 5.432267 +/- 0.009837 | 2.604890 +/- 0.004343 | 26.381 | 47.95% |

TMA-only 시간은 입력 값에 거의 무관하다. 두 분포에서 GEMM 시간이
달라지는 현상은 memory traffic 자체보다 MMA switching activity와
전력/clock 영향이라는 기존 해석과 일치한다.

## Address-locality ablation

모든 mode가 같은 64 GiB logical payload와 같은 TMA instruction 수를
유지한다.

| mode | A tile coordinate | B tile coordinate | unique A+B footprint |
|---|---|---|---:|
| `dense` | 실제 `tile_m` | 실제 `tile_n` | 1 GiB |
| `same_a` | 0 | 실제 `tile_n` | 520 MiB |
| `same_b` | 실제 `tile_m` | 0 | 520 MiB |
| `same` | 0 | 0 | 16 MiB |

각 cell은 W1/I5 독립 프로세스 4개다. 네 mode의 네 순열을 사용해 각
mode가 실행 위치마다 정확히 한 번씩 배치됐다.

| input | address mode | ms | logical TB/s | speedup vs dense |
|---|---|---:|---:|---:|
| `[0,1)` | `dense` | 2.611839 +/- 0.004385 | 26.311 | 기준 |
| `[0,1)` | `same_a` | 2.494930 +/- 0.000541 | 27.544 | 1.0469x |
| `[0,1)` | `same_b` | 2.502640 +/- 0.015106 | 27.459 | 1.0436x |
| `[0,1)` | `same` | 2.390038 +/- 0.025342 | 28.752 | 1.0928x |
| `[-8,8)` | `dense` | 2.610193 +/- 0.003892 | 26.327 | 기준 |
| `[-8,8)` | `same_a` | 2.493986 +/- 0.000612 | 27.554 | 1.0466x |
| `[-8,8)` | `same_b` | 2.500673 +/- 0.007291 | 27.480 | 1.0438x |
| `[-8,8)` | `same` | 2.383875 +/- 0.022282 | 28.827 | 1.0949x |

A와 B 고정의 효과가 거의 대칭이고, 둘을 고정한 효과는 대략 합산된다.
현재 한 operand만 특별히 심각한 문제가 아니라 양쪽 panel locality를
같이 개선해야 한다.

## Counter 제한

Nsight Compute 2025.2.1의 `MemoryWorkloadAnalysis`를 GEMM과
TMA-only에 실행했지만 둘 다 다음 오류로 counter collection이
거부됐다.

```text
ERR_NVGPUCTRPERM
```

따라서 이번 결과에는 실제 L2 hit rate, L2 physical bytes/s, HBM
bytes/s가 없다. L2 saturation을 엄밀하게 확정하려면 counter access가
허용된 B200 host에서 같은 두 binary를 다시 profile해야 한다.

## Artifact

- TMA-only/GEMM definition: `f04052e`
- locality definition: `cfd6790`
- [TMA-only raw result](../results/gemm_nsplit_tma_only_20260728_f04052e/)
- [locality raw result](../results/gemm_nsplit_tma_locality_20260728_cfd6790/)
- TMA-only archive SHA-256:
  `2fac060df79334a15a199075208b597bb749a4fdfc1ed2bb1c21f3c7d57b3707`
- locality archive SHA-256:
  `7a7da706866f2a35651d55381e1dae677fc5fee7f3d32f723e8f0ce4c94e81d8`
