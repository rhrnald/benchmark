# 16K BF16 dense GEMM working default

이 디렉터리는 16K 정방 GEMM 최적화의 현재 working default다. 실험용
전처리 분기와 다중 커널 specialization을 제거한 reconstruction에 E2a
TMEM 주소 정리와 E7a dual-wide M-split mainloop를 반영했다.

## Provenance와 현재 상태

- 보존된 성능 기준은
  `results/gemm_phase_resweep_b200_45465499/bin/p0`이다.
- `p0` SHA-256은
  `166044d6690b52dc448befefb79baabb1c9cb56950b16a0536454249d623ff72`이다.
- B200 1000 W, BF16 uniform `[0,1)`, FP32 C store, 한 프로세스당 한
  case, warmup 1회와 timed 5회, 프로세스 3회 평균으로 측정한 16K
  성능은 **1806.657 TFLOP/s**였다. 개별 값은 1807.834, 1808.010,
  1804.127 TFLOP/s였다.
- 당시 실행 소스의 SHA는 기록에 남았지만 해당 내용은 현재 Git object와
  파일시스템에 없다. 따라서 이 코드는 `p0`의 정확한 원본이라고 주장하지
  않는다. 가장 가까운 Git 이력인 `82206f8`에서 측정 구성을 직접 펼친
  reconstruction이며, `p0`를 성능 및 codegen oracle로 사용한다.
- B0 reconstruction은 로컬 SM100a에서 `REG 178`, `STACK 16 B`, static
  shared `1184 B`, local memory `0 B`였고, 실제 B200에서 `p0`와 동등했다.
- B0 교차 측정에서 clean은 `[0,1)` 1738.199 TFLOP/s, `p0`는
  1739.110 TFLOP/s로 차이가 -0.0524%였다. `[-8,8)`에서도 -0.2253%로
  0.5% noise gate 안이었다. 상세 결과는
  `../../results/gemm_clean_b0_b200_45481495_20260723/summary.md`에 있다.

그 위에 TMEM tile 주소 배열을 직접 scalar 주소식으로 바꾼 E2a를 적용했다.
6쌍 교차 측정에서 `[0,1)`은 1751.903 TFLOP/s로 B0 대비 +0.614%,
`[-8,8)`은 1518.912 TFLOP/s로 +0.393%였고, 두 분포의 모든 paired delta가
양수였다. E2a codegen은 `REG 174`, `STACK 0 B`, spill 0, static shared
`1184 B`였다. 상세 결과는
`../../results/gemm_e2a_tmem_scalar_b200_45481495_20260723/summary.md`에 있다.

현재 E7a는 두 consumer warp가 N128씩 맡던 구조를 M 방향으로 바꿨다.
warp 2는 위쪽 `128 x 256`, warp 3은 아래쪽 `128 x 256`을 각각
`m128n256k16`으로 계산한다. K64당 CTA의 동적 MMA issue 수는 16회에서
8회로 줄었고, TMA byte 수와 output store는 유지된다. 세 쌍 교차 측정에서
`[0,1)`은 **1773.523 TFLOP/s**로 E2a 대비 **+1.3366%**, `[-8,8)`은
**1531.740 TFLOP/s**로 **+1.2900%**였으며 여섯 paired delta가 모두
양수였다. pattern/ones 512 full-C validation도 정확히 통과했다. 현재
codegen은 `REG 172`, `STACK 0 B`, spill 0, static shared `1184 B`다.
상세 결과는
`../../results/gemm_dual_wide_u1_b200_45481495_20260723/summary.md`에 있다.
정의/result commit은 각각 `4e1ac27`/`f35bca3`, 측정 source/binary
SHA-256은 각각 `fdd34cec...`/`0c34e046...`이다. 이후 설명 주석과 runtime
banner만 정리했으며, 로컬 SM100a에서 측정 snapshot과 4192개 전체 SASS
instruction sequence가 동일함을 확인했다.

consumer K loop의 `#pragma unroll 1`을 제거한 compiler-auto-unroll 대조군도
별도로 측정했다. u1 대비 `[0,1)` -0.0272%, `[-8,8)` -0.2087%였고 여섯
pair 모두 느려서 명시적 u1을 유지한다. 상세 결과는
`../../results/gemm_dual_wide_auto_b200_45481495_20260723/summary.md`에 있다.

## 최신 비-L2 병목 실험

E7a dual-wide default에서 C store를 같은 binary의 runtime control로 끈
상한은 `[0,1)` +3.191%, `[-8,8)` +3.298%였다. phase trace에서는 한 tile의
95.9%가 mainloop-to-join, epilogue가 3.6%, scheduler가 0.3%였다. 두
producer는 consumer보다 약 2.9K cycle 먼저 끝났고 서로의 completion
차이도 34--35 cycle뿐이었다. 따라서 scheduler나 단순 producer workload
균형보다 mainloop readiness/wait 동작을 먼저 분리했다.

현재 E7a exact-source 계측판을 B200에서 다시 실행해 만든 standalone
`clock64` SVG는
[`gemm_e7a_clock64_trace.svg`](../../results/gemm_e7a_clock64_trace_b200_45481495_20260723/gemm_e7a_clock64_trace.svg)에
있다. fresh `[0,1)` capture의 complete tile은 279450 cycle로 기존 5-run
median 279398 cycle과 0.019% 차이였고, pattern/ones full-C validation은
모두 정확히 통과했다. SVG의 traced runtime은 성능 수치로 사용하지 않는다.

아래 delta는 매 실험의 동일 실행 clean 대비 paired 결과다. 모든 성능
case는 한 프로세스당 한 case, warmup 1회, timed 5회 평균이다.

| 실험 | `[0,1)` | `[-8,8)` | 결론 |
|---|---:|---:|---|
| shared `mma_done` barrier | -0.079% | +0.043% | neutral/reject |
| combined A+B0 readiness | +0.228% | +0.062% | 0.5% gate 아래, 미채택 |
| producer loop만 `unroll 1` | -0.491% | -0.113% | reject |
| CUTLASS식 suspend hint, 모든 wait | +0.285% | +0.241% | 6/6 pair 양수지만 gate 아래 |
| CUTLASS식 suspend hint, producer wait만 | +0.357% | +0.488% | 12/12 pair 양수지만 gate 아래 |
| A 32 KiB를 16 KiB 두 TMA로 분할 | -0.143% | -0.047% | reject |
| producer를 A 32 KiB / B 32 KiB로 재배치 | -0.108% | -0.324% | reject |

producer-only suspend 후보의 절대 평균은 1777.055/1535.769 TFLOP/s였고
일관되게 양수였지만, 사전에 정한 두 분포 모두 0.5% 이상이라는 materiality
gate를 넘지 못했으므로 default에 넣지 않았다. 가장 빠른 source-backed
canonical default는 계속 E7a의 1773.523/1531.740 TFLOP/s다. 세부 원시
결과와 source/binary hash는 다음 문서에 있다.

- [dual-wide C-store 상한](../../results/gemm_dual_wide_cstore_control_b200_45481495_20260723/summary.md)
- [dual-wide phase trace](../../results/gemm_dual_wide_phase_trace_b200_45481495_20260723/summary.md)
- [producer-only suspended wait](../../results/gemm_suspend_producer_b200_45481495_20260723/summary.md)
- [split-A](../../results/gemm_split_a_b200_45481495_20260723/summary.md)
- [balanced producers](../../results/gemm_balanced_producers_b200_45481495_20260723/summary.md)

현재 소스에는 `#define`이 하나도 없다. 64개 inline-PTX output operand도
함수 본문에 명시적으로 적었으며, macro 제거 전후의 2072개 SASS instruction
sequence가 동일함을 확인했다. 외부 `-D` 옵션도 사용하지 않는다.

성능 문제 크기 16384와 persistent CTA 수 148도 compile-time constant로
고정했다. 따라서 이 기준선은 CLI 실수로 32K scheduler나 다른 CTA 수를
같은 configuration으로 기록할 수 없다. 다른 값을 비교할 때는 해당 상수만
바꾼 별도 source commit을 만든다.

## 고정된 커널 구성

| 항목 | 값 |
|---|---|
| 문제 | `C[16384,16384] = A[16384,16384] * B[16384,16384]` |
| 자료형/layout | row-major BF16 A/B, row-major FP32 C |
| CTA output tile | `256 x 256` |
| `tcgen05.mma` | `128 x 256 x 16`, M 방향 2 consumer warp |
| K staging | `K=64`, shared-memory 3 stage |
| K64당 TMA | A `M256 x K64` 32 KiB 한 번, B `K32 x N256` 16 KiB 두 번 |
| threads | 4 warps, 128 threads |
| warp 0 | A 뒤 late B1 `K32:64 x N256` TMA issue |
| warp 1 | early B0 `K0:32 x N256` TMA issue |
| warp 2 | 위쪽 `M0:128 x N256` MMA issue |
| warp 3 | 아래쪽 `M128:256 x N256` MMA issue |
| MMA issue | CTA/K64당 8회, consumer K loop `#pragma unroll 1` |
| output | `128 x 128` 네 chunk를 SW128 shared memory에서 FP32 TMA store |
| scheduler | 148 persistent CTA, global atomic task counter |
| tile order | `16 x 16` macro, macro N-fast, macro 내부 M-fast |
| phase shift | TMA 0 cycle, MMA 0 cycle |
| L2 promotion/multicast | 없음 / 없음 |
| dynamic shared memory | 197632 B |

각 K64 stage에서 두 B TMA는 N 방향 panel이 아니라 K 방향으로 나뉜다.
consumer는 A와 early B0 완료를 기다린 뒤 K16 MMA 두 번을 issue하고, late
B1 완료를 기다린 뒤 나머지 두 번을 issue하고 commit한다. 따라서 B는
여전히 16 KiB TMA 두 번이고 A를 포함한 총 global-memory traffic도
E2a와 같다.

각 CTA는 scheduler가 지정한 실제 `(tile_m, tile_n)`의 A/B를 읽고 실제
C 위치를 저장한다. 동일한 global-memory tile을 반복해서 읽는
microbenchmark가 아니다.

## 빌드와 correctness 확인

CUDA 12.9 기준 명령은 다음과 같다. command-line macro는 필요하지 않다.

```bash
cd 5.GEMM/baseline
make build
make validate
make resources
```

직접 빌드할 때도 같은 조건을 사용한다.

```bash
/usr/local/cuda-12.9/bin/nvcc \
  -O3 -std=c++17 \
  -gencode arch=compute_100a,code=sm_100a \
  gemm256_bf16_16k.cu -lcuda -o gemm256_bf16_16k
```

`make validate`는 `512 x 512 x 512` pattern 입력의 전체 C를 CPU
reference와 비교한다. validation 크기는 CPU O(N^3) reference가 실수로
과도하게 실행되지 않도록 최대 512로 제한했다. 최적화 후보는 이 검증을
통과하기 전에는 성능 측정 대상에 넣지 않는다. performance 출력의
`checksum`은 scheduler 실행 여부를 보는 diagnostic일 뿐 C correctness
판정값은 아니다.

ones 입력도 별도로 전체 C를 확인한다.

```bash
./gemm256_bf16_16k \
  --validate --validate-size 512 \
  --validate-pattern ones
```

## 표준 성능 측정

기본 측정은 BF16 uniform `[0,1)` 입력, warmup 1회, timed 5회 평균,
한 프로세스당 한 case이다.

```bash
./gemm256_bf16_16k \
  --warmup 1 --iters 5 \
  --input-init random \
  --csv gemm256_bf16_16k_random.csv
```

`[-8,8)` 입력은 다음처럼 측정한다.

```bash
./gemm256_bf16_16k \
  --warmup 1 --iters 5 \
  --input-init random-signed8 \
  --csv gemm256_bf16_16k_random-signed8.csv
```

`run_1x5.sh`는 위 규약으로 정확히 한 프로세스만 실행하고 CSV와 log를
함께 남긴다.

```bash
./run_1x5.sh ./gemm256_bf16_16k /tmp/gemm_runs random
./run_1x5.sh ./gemm256_bf16_16k /tmp/gemm_runs random-signed8
```

기준선 동등성은 같은 B200에서 clean과 `p0`를 AB/BA 순서로 각각 3회
측정해 통과했다. 후속 후보도 GPU power limit, 온도, clock, 드라이버와
CUDA 버전을 같이 기록하고 같은 방식의 paired ratio로 판단한다.

현재 default의 canonical 세 프로세스 평균은 `[0,1)` 1773.523 TFLOP/s,
`[-8,8)` 1531.740 TFLOP/s다. 별도 auto-unroll 대조 실험에서 같은 u1
binary를 다시 측정한 평균도 각각 1773.290, 1530.367 TFLOP/s로 일치했다.

## cuBLAS 상대 성능과 환경 기준

환경 차이를 절대 TFLOP/s와 분리해 보기 위해 instance `45481495`의 한
활성화 세션에서 보존 `p0`, 현재 E7a, cuBLAS를 함께 측정했다. 16K,
동일 BF16 random stream, FP32 C, 한 프로세스당 warmup 1회와 timed 5회,
각 셀 3프로세스이며 실행 위치를 교차했다.

| input | `p0` | 현재 E7a | cuBLAS | E7a/cuBLAS |
|---|---:|---:|---:|---:|
| `[0,1)` | 1737.130 | **1770.382** | 1836.906 | **96.378%** |
| `[-8,8)` | 1512.169 | **1529.712** | 1608.169 | **95.121%** |

이번 `p0`와 E7a는 같은 인스턴스의 앞선 값에서 각각 -0.114%, -0.177%로
재현됐다. cuBLAS는 이전 별도 B200의 동일 실행 파일 1/5 결과보다
`[0,1)`에서 1.940% 낮았다. 따라서 절대 성능 차이에 환경 성분이 있다는
것은 확인되지만, 1806.657을 측정한 과거 instance `45465499`에는 같은
세션의 cuBLAS 결과가 없으므로 `p0` 하락 3.848% 전부를 환경 탓으로
확정할 수는 없다.

참고용으로 과거 `p0 1806.657`과 별도 인스턴스 cuBLAS `1873.248`의 비율은
96.445%이고, 현재 같은 세션의 E7a/cuBLAS는 96.378%다. 차이는 0.067
percentage point라 현재 source-backed E7a가 역사적 상대 효율은 거의
회복했다고 해석할 수 있다. 단, 앞 비율은 서로 다른 GPU를 나눈 값이므로
직접 증거가 아니라 보조 근거다. 원시 표본과 라이브러리 hash는
[환경 상대 비교 결과](../../results/gemm_env_cublas_reference_b200_45481495_20260723/summary.md)에
있다.

같은 GPU와 software stack에서 8K와 32K도 확장 측정했다. E7a는 8K에서
`16x16`, 32K에서 size-tuned `8x18` persistent macro를 사용한다. 아래 값은
각 행의 같은 activation 안에서 실행 순서를 균형화한 process mean이며,
모든 process가 warmup 1회와 timed 5회를 사용했다.

| input | size | `p0` | 현재 E7a | cuBLAS | E7a/cuBLAS |
|---|---:|---:|---:|---:|---:|
| `[0,1)` | 8K | 1674.848 | **1704.894** | 1737.851 | **98.104%** |
| `[0,1)` | 16K | 1737.130 | **1770.382** | 1836.906 | **96.378%** |
| `[0,1)` | 32K | 1591.566 | **1608.133** | 1625.373 | **98.939%** |
| `[-8,8)` | 8K | 1534.717 | **1555.989** | 1580.738 | **98.434%** |
| `[-8,8)` | 16K | 1512.169 | **1529.712** | 1608.169 | **95.121%** |
| `[-8,8)` | 32K | 1316.549 | **1338.924** | 1332.195 | **100.505%** |

8K signed는 첫 activation의 cuBLAS 한 표본이 다른 두 표본보다 2.489%
낮아서, 세 방법을 새 activation에서 다시 완전 회전한 confirmation 값을
표에 사용했다. 32K signed의 E7a/cuBLAS 차이는 +0.505%지만 cuBLAS
process SD가 17.252 TFLOP/s이므로 동률 범위로 해석한다. 32K에서 `8x18`은
직접 size port인 `16x16`보다 `[0,1)` +0.791%, `[-8,8)` +4.735%였다.
원시 표본, SD, 두 activation을 합치지 않은 이유와 source diff는
[8K/32K 확장 결과](../../results/gemm_env_cublas_size_extension_b200_45481495_20260723/summary.md)에
있다.

## 소스 위치

- [상수와 shared/TMEM layout](gemm256_bf16_16k.cu#L42-L110)
- [wide-B descriptor와 MMA instruction descriptor](gemm256_bf16_16k.cu#L149-L177)
- [`tcgen05.mma` PTX wrapper](gemm256_bf16_16k.cu#L402-L426)
- [C epilogue staging과 store](gemm256_bf16_16k.cu#L445-L543)
- [A/B stage TMA helper](gemm256_bf16_16k.cu#L545-L563)
- [persistent kernel과 scheduler](gemm256_bf16_16k.cu#L565-L792)
- [producer/consumer mainloop](gemm256_bf16_16k.cu#L669-L755)
- [A/B/C tensor map encoding](gemm256_bf16_16k.cu#L794-L849)
- [benchmark timing](gemm256_bf16_16k.cu#L1069-L1163)
- [full-C validation](gemm256_bf16_16k.cu#L1209-L1317)

후속 실험의 순서와 판정 규칙은 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)에
고정한다.
