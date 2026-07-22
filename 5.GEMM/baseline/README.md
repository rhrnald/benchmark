# 16K BF16 dense GEMM baseline

이 디렉터리는 16K 정방 GEMM 최적화를 다시 시작할 때 사용하는 고정
기준선이다. 실험용 `GEMM_*` 전처리 분기와 다중 커널 specialization을
제거하고, 현재까지 가장 좋았던 구성 하나만 남겼다.

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
- 로컬 SM100a 빌드 결과는 `p0`와 동일한 `REG 178`, `STACK 16 B`,
  static shared `1184 B`, local memory `0 B`이다. 실제 B200 성능 동등성은
  같은 인스턴스에서 교차 측정한 뒤 확정한다.

소스에는 `TCGEN05_LD_X64_OUTPUTS`와 `TCGEN05_LD_X64_OPERANDS` 두
`#define`만 남아 있다. 둘 다 64개 inline-PTX operand 목록을 맞춰 쓰기
위한 문법용 macro이며 커널 동작이나 tuning을 선택하지 않는다.
`GEMM_*` tuning macro와 외부 `-D` 옵션은 없다.

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
| `tcgen05.mma` | `128 x 128 x 16`, N 방향 2 pipe |
| K staging | `K=64`, shared-memory 3 stage |
| threads | 4 warps, 128 threads |
| warp 0 | A와 B의 첫 N128 TMA issue |
| warp 1 | B의 둘째 N128 TMA issue |
| warp 2/3 | 각 N128의 MMA issue |
| output | `128 x 128` 네 chunk를 SW128 shared memory에서 FP32 TMA store |
| scheduler | 148 persistent CTA, global atomic task counter |
| tile order | `16 x 16` macro, macro N-fast, macro 내부 M-fast |
| phase shift | TMA 0 cycle, MMA 0 cycle |
| L2 promotion/multicast | 없음 / 없음 |
| dynamic shared memory | 197632 B |

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

## 표준 성능 측정

기본 측정은 BF16 uniform `[0,1)` 입력, warmup 1회, timed 5회 평균,
한 프로세스당 한 case이다.

```bash
./gemm256_bf16_16k \
  --warmup 1 --iters 5 \
  --input-init random \
  --csv clean_16384_random.csv
```

`[-8,8)` 입력은 다음처럼 측정한다.

```bash
./gemm256_bf16_16k \
  --warmup 1 --iters 5 \
  --input-init random-signed8 \
  --csv clean_16384_random-signed8.csv
```

`run_1x5.sh`는 위 규약으로 정확히 한 프로세스만 실행하고 CSV와 log를
함께 남긴다.

```bash
./run_1x5.sh ./gemm256_bf16_16k /tmp/gemm_runs random
./run_1x5.sh ./gemm256_bf16_16k /tmp/gemm_runs random-signed8
```

기준선 동등성은 같은 B200에서 clean과 `p0`를 AB/BA 순서로 각각 3회
측정해 paired ratio로 판단한다. GPU power limit, 온도, clock, 드라이버와
CUDA 버전을 같이 기록한다. 평균 차이가 1%보다 크면 clean 코드를
`1806.657 TFLOP/s baseline`이라고 부르지 않고 원인을 먼저 찾는다.

후속 실험의 순서와 판정 규칙은 [OPTIMIZATION_PLAN.md](OPTIMIZATION_PLAN.md)에
고정한다.
