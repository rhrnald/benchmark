# N-split transposed-compute prototype

## 목적

현재 canonical N-split의 데이터 이동과 warp ownership은 유지하면서
CTA-group-1 `tcgen05.mma` issue 수만 줄일 수 있는지 확인한다.

고정하는 구성은 다음과 같다.

| 항목 | 고정 구성 |
|---|---|
| A TMA | CTA/K64마다 BF16 `256x64` 한 번, 32 KiB |
| B TMA | B0/B1이 각각 BF16 `64x128` 한 번, 16 KiB씩 |
| logical output | W2가 `C[:,0:128]`, W3가 `C[:,128:256]` |
| producer/barrier | canonical과 동일한 3-stage ring |
| consumer control flow | combined `(warp_id == 2 || warp_id == 3)` runtime pipe |
| scheduler/store | 148 persistent CTA와 FP32 SW128/TMA C store |

canonical source SHA-256은
`cd595ba3d6b3a0be1525e9d617083ca3841c971a0f33b514b600aeb837f307ca`다.
`generate_gemm_nsplit_transpose.py`는 이 hash가 다르면 source 생성을
거부한다.

## 계산 변환

각 warp가 담당하는 결과는 원래 다음과 같다.

```text
C_p[256x128] = A[256x64] * B_p[64x128]
```

이를 전치하면 다음과 같다.

```text
C_p^T[128x256] = B_p^T[128x64] * A^T[64x256]
```

따라서 기존 physical shared-memory payload를 다시 TMA로 옮기지 않고
operand 역할만 바꿀 수 있다.

- 기존 B panel은 N이 연속인 `KxN` SW128이다. `B_p^T`를 MMA A로 보면
  `MN-major` descriptor가 된다.
- 기존 A tile은 K가 연속인 `MxK` SW128이다. `A^T`를 MMA B로 보면
  `K-major` descriptor가 된다.
- instruction descriptor는 `M=128`, `N=256`, A-major bit 15는 1,
  B-major bit 16은 0이다.

각 warp는 K16마다 `m128n256k16` 한 번을 issue한다. CTA/K64의 동적 MMA
수는 canonical의 16회에서 8회로 줄어든다. pipe 0은 TMEM column
`[0,256)`, pipe 1은 `[256,512)`를 사용한다.

## correctness-first 전치 epilogue

MMA 결과는 TMEM에 다음처럼 놓인다.

```text
TMEM pipe p row    = 해당 B panel 안의 original N
TMEM pipe p column = original M
```

기존 epilogue는 TMEM row가 M이라고 가정하므로 그대로 사용할 수 없다.
prototype은 각 `128x128` C-store chunk를 다음과 같이 만든다.

1. 네 warp가 각각 TMEM N 32행을 맡는다.
2. warp는 `tcgen05.ld.32x32b.x64`로 N 32행과 M 64열을 읽는다.
3. register `r[i]`의 같은 `i`에 대해 32 lane이 output 한 M행의 연속
   N 32개를 scalar shared store한다.
4. 기존 `cstore_sw128_float_word_offset()`로 SW128 staging 주소를 만들고
   기존 FP32 TMA store를 그대로 사용한다.

SW128의 행 내 XOR는 32 lane에 대한 permutation이므로 한 `i` iteration의
shared store는 32개 bank를 중복 없이 사용한다. 각 warp는 서로 다른 N
32-column block을 쓰므로 warp 사이 주소 중복도 없다.

이 epilogue는 정확성 확인이 우선이다. TMEM load 수와 TMA store byte 수는
기존과 같지만, shared store instruction은 기존 lane별 `uint4` 방식보다
4배 많다. 전치 mainloop의 이득과 전치 epilogue의 손실을 end-to-end
측정으로 분리해서 판단해야 한다.

현재 prototype은 직접 비교를 위해 canonical과 같은
`mma_done wait -> CTA sync -> TMEM ld/wait` 순서를 유지한다. PTX의
different-thread canonical synchronization pattern은 TMEM 접근 전후에
`tcgen05.fence::after_thread_sync`와
`tcgen05.fence::before_thread_sync`도 제시한다. B200 full-C validation에서
문제가 보이거나 spec-correct 결과가 필요하면 fence를 candidate에만 넣지
말고, exact N-split과 transpose 양쪽에 같은 fence를 넣은 별도 대조군으로
비교한다.

## 로컬 codegen gate

CUDA 12.9, `sm_100a`, `-O3 -std=c++17 -lineinfo -Xptxas=-v`로 먼저
컴파일한다.

통과 조건은 다음과 같다.

- ordinary `UTCHMMA` static site가 4개다. combined runtime-pipe body에서
  K16 네 개가 unroll되고, 실제 CTA/K64에서는 두 warp가 총 8회 issue한다.
- `UTCBAR` commit 구조가 pipe별로 유지된다.
- TMA load 수와 producer barrier 구조가 canonical에서 바뀌지 않는다.
- stack, local-memory spill, register spill이 없다.
- TMEM allocation은 계속 512 columns다.

전치 epilogue 때문에 전체 SASS 크기가 증가하는 것은 예상한다. 그 증가는
mainloop gate 실패가 아니라 별도의 epilogue 비용으로 기록한다.

### 최초 로컬 결과

CUDA 12.9.41로 생성·컴파일한 최초 prototype은 gate를 통과했다.

- generated source SHA-256:
  `a6c31fb053aa9647d969fb4f2a565cc7dfa81bd4fb44ce9afabf16c954b211cc`
- normalized main-kernel SASS: 2229 instructions
- `UTCHMMA=4`, `UTCBAR=1`, `SYNCS.PHASECHK=48`
- global-to-shared TMA `UTMALDG=21`, TMEM `LDTM.x64=8`
- scalar shared staging `ST.E=512`
- 174 registers, stack/local/spill 0, static shared memory 1184 B

처음에는 단순화된 consumer가 K64 loop를 3-way auto-unroll하여
`UTCHMMA=12`, `UTCBAR=3`이 됐다. canonical에서 이미 SASS-identical
control로 확인된 `#pragma unroll 1`을 consumer K64 loop에 적용해
`UTCHMMA=4`, `UTCBAR=1`로 고정했다. 최종 instruction descriptor
immediate는 `0x08408490`이며 M128/N256, A MN-major, B K-major와
일치한다.

## B200 correctness 및 성능 gate

성능 전에 아래 full-C validation을 모두 통과해야 한다.

```bash
candidate --validate --validate-size 512 --validate-pattern pattern
candidate --validate --validate-size 512 --validate-pattern ones
```

두 검증 모두 `bad=0`이어야 한다. 가능하면 작은 random BF16 입력도 CPU
reference와 비교한다.

성능은 기존 규칙대로 한 프로세스당 한 case, warmup 1회, timed 5회로
측정한다. canonical N-split과 같은 activation 안에서 두 입력 분포를
position-balanced하게 교차 실행한다.

| 입력 | 비교 |
|---|---|
| BF16 uniform `[0,1)` | transpose E2E 대 exact N-split |
| BF16 uniform `[-8,8)` | transpose E2E 대 exact N-split |

두 분포 모두 exact N-split보다 0.5% 이상 빨라야 다음 전치 epilogue
최적화로 진행한다. mainloop-only 이득은 있으나 E2E가 느리면 scalar
epilogue 비용을 별도로 측정한 뒤, 32x32 register transpose 또는
shared-memory vectorization을 후속 단일 변수 실험으로 검토한다.

같은 activation에서 C-store를 제거한 matched control도 함께 측정한다.

| variant | mainloop | epilogue |
|---|---|---|
| `nsplit_exact` | canonical 16-MMA N-split | canonical SW128/TMA |
| `nsplit_transpose_scalar` | transposed 8-MMA N-split | scalar transpose + SW128/TMA |
| `nsplit_exact_nostore` | canonical과 동일 | final drain 뒤 C load/store 생략 |
| `nsplit_transpose_nostore` | transpose와 동일 | final drain 뒤 C load/store 생략 |

네 variant는 네 번의 Latin rotation에서 모든 실행 위치를 한 번씩
차지한다. 이를 통해 E2E 변화와 no-store mainloop 변화를 분리하고,
각 pass의 `E2E event_ms - no-store event_ms`로 epilogue 비용을 추정한다.

## 생성

```bash
python3 5.GEMM/generate_gemm_nsplit_transpose.py \
  --source 5.GEMM/baseline/gemm256_bf16_16k.cu \
  --output /tmp/gemm256_bf16_16k_nsplit_transpose.cu
```

generator 출력에 canonical source hash와 generated source hash가 함께
기록된다.

전체 B200 실험은 다음 runner로 재현한다.

```bash
./run_b200_gemm_nsplit_transpose.sh \
  /workspace/benchmark \
  /workspace/gemm_nsplit_transpose_b200
```
