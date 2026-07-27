# Static N-split overhead ablation

Date: 2026-07-27

## 결론

측정상 아래 세 방향과 결합 후보 중 두 입력 분포에서 모두 baseline
대비 `+0.5%`를 넘은 후보는 없었다.

1. timed launch의 diagnostic sink 및 `cudaMemsetAsync` 제거
2. 16K problem shape의 compile-time 특수화
3. producer stage-reuse `mbarrier.try_wait`에 CUTLASS식 suspend hint 적용

절대 TFLOP/s는 B200 activation에 따라 달라질 수 있으므로 성능 판단에는
같은 프로세스 위치끼리 비교한 paired delta를 사용했다.

최초 측정 판정에서는 전 후보를 미채택했으나, 이후 코드 정리와 16K
전용화의 구조적 이점을 우선하기로 결정했다. 따라서 `fixed_sink`, 즉
diagnostic sink/memset 제거와 compile-time shape 특수화를 canonical에
적용했다. 이는 측정된 성능 개선에 의한 채택이 아니며, 작은 성능 손실
가능성을 감수한 engineering decision이다. Producer suspend는 명확하게
느려졌으므로 적용하지 않았다.

## 측정 조건

| 항목 | 조건 |
|---|---|
| GEMM | row-major `C[16384,16384] = A @ B` |
| datatype | BF16 A/B, FP32 accumulate/output |
| kernel | `256x256` CTA tile, K64, 3-stage |
| scheduler | 148 persistent CTAs, static `8x16`, macro N-fast / local M-fast |
| input | BF16 uniform `[0,1)`, BF16 uniform `[-8,8)` |
| timing | 한 프로세스당 warmup 1회, timed 5회 |
| replication | 각 cell 독립 프로세스 4회 |
| ordering | reverse/rotate position-balanced |
| validation | 512 pattern 및 ones full-C CPU reference, 전 후보 zero error |
| GPU | NVIDIA B200 |

## 후보 정의

| 이름 | 변경 |
|---|---|
| `baseline` | 현재 canonical 그대로 |
| `no_memset` | timed launch 전 sink용 `cudaMemsetAsync`만 제거 |
| `sink_keep` | diagnostic shared/global sink 제거, 기존 CTA sync 위치 유지 |
| `sink_trim` | sink와 그 제거로 불필요해진 tile sync까지 제거 |
| `fixed` | performance kernel을 `<256,64,64>`로 compile-time 특수화 |
| `suspend` | producer의 stage-reuse wait에 suspend hint `0x989680` 적용 |
| `fixed_sink` | `fixed + sink_trim` |
| `all` | `fixed + sink_trim + suspend` |

`fixed` 계열은 validation용 `<8,2,2>`, `<4,1,1>` 인스턴스를 별도로
생성했다. 따라서 성능 kernel은 16K shape만 대상으로 특수화되어 있다.

## 성능

### BF16 uniform `[0,1)`

| variant | TFLOP/s mean +/- sample SD | paired delta vs baseline |
|---|---:|---:|
| `baseline` | **1840.266 +/- 2.197** | 기준 |
| `no_memset` | 1838.057 +/- 1.868 | -0.120% +/- 0.180%p |
| `sink_keep` | 1837.966 +/- 3.082 | -0.125% +/- 0.201%p |
| `sink_trim` | 1838.834 +/- 1.381 | -0.078% +/- 0.174%p |
| `fixed` | 1832.372 +/- 2.655 | -0.429% +/- 0.172%p |
| `suspend` | 1823.834 +/- 1.743 | -0.893% +/- 0.046%p |
| `fixed_sink` | 1833.652 +/- 2.148 | -0.359% +/- 0.225%p |
| `all` | 1823.715 +/- 2.490 | -0.899% +/- 0.162%p |

### BF16 uniform `[-8,8)`

| variant | TFLOP/s mean +/- sample SD | paired delta vs baseline |
|---|---:|---:|
| `baseline` | 1627.704 +/- 2.642 | 기준 |
| `no_memset` | 1628.036 +/- 0.747 | +0.021% +/- 0.179%p |
| `sink_keep` | 1627.583 +/- 1.810 | -0.007% +/- 0.197%p |
| `sink_trim` | **1631.685 +/- 5.151** | +0.245% +/- 0.329%p |
| `fixed` | 1624.173 +/- 3.830 | -0.217% +/- 0.369%p |
| `suspend` | 1619.320 +/- 4.914 | -0.515% +/- 0.422%p |
| `fixed_sink` | 1622.183 +/- 3.361 | -0.339% +/- 0.075%p |
| `all` | 1623.293 +/- 3.900 | -0.271% +/- 0.390%p |

## Codegen과 해석

| variant 계열 | performance kernel registers | static SMEM | spills |
|---|---:|---:|---:|
| `baseline`, `no_memset`, `suspend` | 174 | 144 B | 0 |
| `sink_keep`, `sink_trim` | 170 | 128 B | 0 |
| `fixed` | 168 | 144 B | 0 |
| `fixed_sink`, `all` | 164 | 128 B | 0 |

- `no_memset`은 launch 밖의 작은 host-side overhead만 없애므로 kernel
  throughput에는 사실상 중립이다.
- sink 제거는 register 수를 4개 줄였지만 이 커널은 `tcgen05` 조건상
  SM당 CTA 하나이므로 occupancy 이득이 없다. `sink_trim`의 signed
  `+0.245%`는 `[0,1)`에서 음수이고 변동도 커 채택 근거가 아니다.
- compile-time 특수화는 register 수를 더 줄였지만 두 분포 모두
  느려졌다. 달라진 instruction 배치/scheduling의 손해가 작은 정수
  연산 제거 효과보다 컸다.
- producer suspend는 두 분포 모두 일관되게 느렸다. 현재 static
  grid-stride topology에서는 active polling을 줄이는 효과보다 wake-up
  latency가 더 크다.
- 결합 후보도 baseline을 넘지 못했다. 다만 후속 결정으로
  `fixed_sink`는 구조적 정리를 위해 canonical에 적용했고, suspend는
  계속 제외한다.

## 재현

실험 정의 commit은 `2e58a08`, 측정 결과 commit은 `399bdad`이다.

```bash
./run_b200_gemm_nsplit_overhead.sh
```

Raw CSV, 실행 순서, build log, SASS, GPU telemetry와 요약:

- [`results/gemm_nsplit_overhead_20260727_2e58a08`](../results/gemm_nsplit_overhead_20260727_2e58a08/)
- [`summary.md`](../results/gemm_nsplit_overhead_20260727_2e58a08/summary.md)
- archive SHA-256:
  `e8ee428f051097fe956f34aa581e26ffa3d02f0ad6475100160161860696e84f`
