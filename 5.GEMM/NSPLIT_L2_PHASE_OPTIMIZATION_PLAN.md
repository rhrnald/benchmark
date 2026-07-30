# N-split L2 / phase-shift optimization plan

Last updated: 2026-07-30

## 목적

현재 canonical N-split dense GEMM의 수학 연산, CTA tile, stage 수와
정확성은 유지하면서 다음 두 축을 분리해 최적화한다.

1. output tile의 실행/소유 순서를 바꿔 A/B panel의 L2 재사용률을 높인다.
2. 고정 sleep을 넣지 않고 producer dependency를 더 세밀하게 나눠
   TMA와 MMA의 자연스러운 중첩을 늘린다.

두 축을 따로 측정한 뒤 독립적인 winner만 결합한다. Nsight Compute
counter 권한이 없는 상태에서는 `L2 saturation`을 단정하지 않고,
동일 workload의 주소 locality 변화에 대한 시간 민감도로 판단한다.

## 기준 구현과 성능

- source: [`baseline/gemm256_bf16_16k.cu`](baseline/gemm256_bf16_16k.cu)
- problem: BF16 `C[16384,16384] = A @ B`, FP32 accumulate/output
- CTA: `256x256x64`, 148 persistent CTA, SM당 CTA 1개
- pipeline: shared memory 3 stages
- warp 0: A `256x64`와 B0 `64x128` TMA
- warp 1: B1 `64x128` TMA
- warp 2/3: 각각 `256x128` output MMA
- epilogue: 네 개의 `128x128` FP32 chunk를 shared memory에 staging한 뒤
  TMA store
- scheduler: static grid-stride, `8x16`, macro N-fast / local M-fast
- current codegen: 164 registers, spill/local/stack 0

2026-07-30 같은-session W1/I5 독립 프로세스 4회 평균:

| size | input | ours | cuBLAS | ours/cuBLAS |
|---:|---|---:|---:|---:|
| 8K | `[0,1)` | 1766.749 | 1799.487 | 98.181% |
| 8K | `[-8,8)` | 1590.216 | 1609.157 | 98.823% |
| 16K | `[0,1)` | 1829.322 | 1893.726 | 96.599% |
| 16K | `[-8,8)` | 1620.620 | 1680.376 | 96.444% |
| 32K | `[0,1)` | 1580.013 | 1603.139 | 98.557% |
| 32K | `[-8,8)` | 1371.745 | 1417.911 | 96.744% |

세션 간 절대 TFLOP/s가 변하므로 이 값은 상태 확인용이다. 후보 채택은
항상 같은 세션의 paired ratio로 결정한다.

## 코드 검토 결과

### 1. 현재 scheduler는 collective locality는 좋지만 per-SM locality가 없다

16K에서는 output tile grid가 `64x64`이고 현재 mapping은 다음과 같이
단순화된다.

```text
tile_m = 8 * floor(task / 512) + (task mod 8)
tile_n = floor((task mod 512) / 8)
next task of one CTA = task + 148
```

이는 16K처럼 8로 나누어 떨어지는 square grid에서 CUTLASS static
S8/AlongN과 같은 output tile 순서다. 따라서 단순히 `8x16` macro를
다시 sweep하는 것은 새로운 실험이 아니다.

반면 148개 CTA가 한 번에 처리하는 task set과 각 CTA의 다음 task
소유권을 나눠 보면 다음과 같다.

- 전체 4096 tile은 28 wave다: 27 full wave + 100-task tail.
- full wave 20개는 unique `(A tile_m, B tile_n) = (8, 19)`,
  나머지 7개는 `(16, 19)`다.
- K64 한 stage에서 full wave의 logical input은 약 9.25 MiB지만,
  unique A+B panel footprint는 각각 약 864 KiB 또는 1120 KiB다.
- 즉 같은 시간대의 CTA들 사이에는 큰 L2 reuse 기회가 있다.
- 그러나 `task += 148`인 같은 CTA의 연속 tile 3948쌍에서 동일
  `tile_m`도 0회, 동일 `tile_n`도 0회다.

현재 scheduler는 `wave 안의 집단 재사용`은 잘 만들지만, 같은 SM이
다음 output tile로 넘어갈 때 직전에 읽은 A 또는 B panel을 이어 쓰지
못한다. 이 부분은 기존 N-fast, strip, macro-shape 실험과 다른
최적화 공간이다.

CUTLASS도 last-level cache 재사용을 위해 연속 CTA를 packed 2D 영역에
배치하며, persistent kernel에서는 tile scheduler가 한 CTA의 여러 output
tile을 관리한다.

### 2. C store가 다음 tile의 A/B cache residency를 방해할 수 있다

각 output tile은 FP32 C `256x256 = 256 KiB`를 쓴다. persistent CTA가
다음 tile의 mainloop로 가기 직전에 이 streaming output이 L2를 채우므로,
직전/동시 wave의 A/B panel을 밀어낼 가능성이 있다.

현재 A/B TMA map은 `CU_TENSOR_MAP_L2_PROMOTION_NONE`이고 TMA load/store
모두 cache policy를 전달하지 않는다. 과거 A/B load의 `evict_last`는
거의 중립이었고, `evict_first`는 특히 A에서 크게 느렸다. 하지만
**C TMA store에만 `evict_first`를 주는 실험은 아직 하지 않았다.**
PTX의 shared-to-global `cp.async.bulk.tensor` store는
`.L2::cache_hint`와 cache policy operand를 지원하므로 clean ablation이
가능하다.

### 3. 현재 B1은 이미 독립적이지만 B0는 A reuse wait에 묶여 있다

현재 stage 재사용 순서는 다음과 같다.

```text
warp 0: wait mma_done[0] -> wait mma_done[1] -> issue A -> issue B0
warp 1: wait mma_done[1]                     -> issue B1
warp 2: wait A/B0 -> MMA -> commit mma_done[0]
warp 3: wait A/B1 -> MMA -> commit mma_done[1]
```

- A shared-memory 영역은 두 consumer가 모두 사용하므로 재사용 전에
  `mma_done[0]`과 `mma_done[1]`을 모두 기다려야 한다.
- B0 영역은 pipe 0만 사용하므로 `mma_done[0]`만 기다리면 된다.
- B1은 별도 warp가 `mma_done[1]`만 기다려 이미 가능한 한 독립적으로
  issue하고 있다.

따라서 유망한 phase 변화는 sleep으로 warp/CTA 시작을 늦추는 것이 아니라
warp 0의 dependency를 다음처럼 분리하는 것이다.

```text
wait mma_done[0] -> issue B0 -> wait mma_done[1] -> issue A
```

pipe 0가 pipe 1보다 먼저 끝나는 구간에서는 B0 TMA를 pipe 1 MMA tail과
겹칠 수 있다. 반대로 완료 순서가 대부분 반대라면 이득이 없으므로,
먼저 `clock64()` trace로 실제 wait 구간을 확인한다.

### 4. 임의 phase delay와 cross-tile prefetch는 우선순위가 낮다

이미 측정한 결과:

- CTA startup staggering: 중립 또는 음수
- B1 stage gap 32/64 cycles: 최대 약 `+0.09%/+0.07%`, 재현 gate 실패
- pipe-1 consumer MMA delay: 명확한 regression
- cross-tile first-K64 prefetch: `+0.0606%/-0.1353%`, 미채택
- 과거 큰 phase gain은 현재와 다른 잘못된 completion-wait topology의
  효과였으며 현재 kernel에 그대로 적용할 근거가 없다.

따라서 phase 실험은 `dependency를 풀어 실제 overlap을 늘리는 것`만
우선한다. CTA별 K 순서를 회전시키는 방식은 같은 K panel을 동시에
요청하는 L2 locality를 깨고 reduction 순서도 바꿀 수 있으므로 제외한다.

### 5. 나머지 코드 후보의 우선순위

| 후보 | 판단 |
|---|---|
| 3-stage ring의 `% 3`/division 제거 | correctness risk가 낮은 별도 instruction-overhead 실험. L2/phase 이후 진행 |
| epilogue/mainloop overlap | no-store headroom은 약 3.6--4.1%지만 현재 TMEM/SMEM 소유 구조상 큰 변경 필요 |
| C-store group 재파이프라인 | 기존 ping-pong/store wait 변경이 regression이어서 당장 반복하지 않음 |
| input A/B promotion/hint 재-sweep | 이미 중립/음수. 반복하지 않음 |
| TMA L2 prefetch instruction | 2D strided panel에 많은 issue와 중복 traffic이 필요해 후순위 |
| CUDA persisting-L2 window | 전체 16K A/B가 매우 크고 panel이 계속 이동함. C hint와 scheduler 이후 제한적으로 검토 |
| multicast | 현재 non-multicast robust peak를 넘지 못했으므로 이번 1차 범위에서 제외 |

## 실험 계획

### Phase 0 — 기준 고정과 trace

1. canonical source hash, compiler/driver/GPU clock/power limit, registers와
   spill을 기록한다.
2. 16K 두 입력에서 현재 baseline을 같은 세션에 다시 측정한다.
3. 현재 N-split 전용 `clock64()` trace variant를 만든다.
4. 한 CTA의 중간 K window에서 다음 event를 기록한다.
   - pipe 0/1 stage-reuse wait 시작/종료
   - A/B0/B1 TMA issue
   - consumer A/B ready wait 종료
   - 첫/마지막 MMA, commit, mma_done
   - epilogue staging, C TMA issue, store wait
5. trace binary는 진단 전용이며 성능 표에는 사용하지 않는다.

판정 질문은 `pipe 0 완료가 pipe 1보다 먼저 오는 빈도와 cycle 차이가
early-B0를 정당화하는가`다.

### Phase 1A — C store cache pollution ablation

변수는 C TMA store cache policy 하나만 바꾼다.

| variant | C TMA store |
|---|---|
| `c_default` | 현재 instruction |
| `c_evict_first` | `.L2::cache_hint` + fractional `evict_first` |

A/B load, tile order, arithmetic, store byte 수와 wait 순서는 동일하게
유지한다. 이 실험이 양수면 output streaming traffic이 operand
residency를 방해했다는 정황 증거가 된다.

### Phase 1B — wave-preserving persistent ownership

핵심은 **각 wave의 148개 logical task set은 그대로 유지하고, 다음
wave에서 어느 CTA/SM이 어느 task를 맡는지만 permutation**하는 것이다.
그러면 동시 실행 CTA의 A/B sharing과 full-SM occupancy는 baseline과
동일하면서 per-SM reuse만 바꿀 수 있다.

초기 구현은 host가 만든 task mapping table을 사용한다.

- 16K: 4096 entries, `uint16_t`면 8 KiB
- 32K까지도 output tile 수 16384라 `uint16_t`에 들어간다.
- `table_identity`를 반드시 넣어 table lookup 자체의 비용을 분리한다.
- 모든 output tile이 정확히 한 번 배정되고 CTA별 27/28-task load가
  유지되는지 host에서 검증한다.

| variant | 다음 wave의 CTA-task matching 목표 |
|---|---|
| `direct` | 현재 산술 mapping |
| `table_identity` | 현재 mapping을 table로만 표현 |
| `wave_A` | 같은 `tile_m` 최대화 |
| `wave_B` | 같은 `tile_n` 최대화 |
| `wave_balanced` | A/B match 수와 reuse distance를 함께 최소화 |

각 wave 사이 assignment는 bipartite matching/greedy 후보를 offline에서
만들고 다음 지표를 먼저 출력한다.

- wave별 unique A/B panel 수와 footprint
- CTA 연속 task의 same-A/same-B pair 수
- A/B reuse distance histogram
- tail load balance

`table_identity`가 유의미하게 느리면 formula 또는 작은 permutation
규칙으로 다시 표현한 뒤 GPU 후보를 비교한다.

### Phase 1C — CUTLASS-style serpentine 후보

공식 SM100 scheduler source의 swizzle/raster mapping을 정확히 옮긴
S4/S8 serpentine 후보는 offline metric이 baseline보다 Pareto 개선일
때만 측정한다. 현재 순서가 이미 static S8/AlongN에 해당하므로 단순
S2/S4/S8 raster sweep은 반복하지 않는다.

### Phase 2 — 구조적 phase shift

trace 결과를 바탕으로 아래 최소 후보만 비교한다.

| variant | warp 0 stage-reuse / issue 순서 |
|---|---|
| `phase_control` | wait p0, wait p1, A, B0 |
| `issue_B0_first` | wait p0, wait p1, B0, A |
| `early_B0` | wait p0, B0, wait p1, A |

필요할 때만 stage parity 또는 CTA parity로 A/B0 issue order를 번갈아
TMA burst를 분산하는 후보를 하나 추가한다. 고정 nanosleep과 consumer
MMA delay는 다시 측정하지 않는다.

모든 후보는 다음 조건을 만족해야 한다.

- A buffer는 p0/p1 양쪽 완료 후에만 overwrite
- B0는 p0 완료 후, B1은 p1 완료 후에만 overwrite
- 기존 A/B ready barrier epoch와 MMA accumulation 순서 유지
- full-C validation 통과

### Phase 3 — stage-control instruction overhead

L2/phase winner와 독립적으로 K loop를 stage 0/1/2 세 개씩 진행하도록
펴서 hot loop의 `% 3`와 phase division을 제거한다.

- baseline과 instruction count/SASS를 비교한다.
- code size, registers, spill이 늘면 바로 중단한다.
- 성능 gate는 다른 후보와 동일하다.

### Phase 4 — winner 결합

전체 factorial sweep 대신 독립적으로 통과한 순서대로만 결합한다.

```text
canonical
  -> C evict-first winner
  -> wave-preserving scheduler winner
  -> early-B0 winner
  -> optional stage-control winner
```

결합할 때마다 직전 winner와 paired ABBA 비교해 상호작용 regression을
확인한다. 특히 `C evict-first + per-SM A/B reuse`가 가장 직접적인
L2 조합이다.

### Phase 5 — size 확장과 library 비교

16K에서 채택된 최종 후보만 8K/32K로 포팅한다. 크기별로 scheduler
reuse metric을 다시 생성하고, 같은 세션에서 ours/cuBLAS를 두 입력
분포 모두 측정한다. CUTLASS는 동일 software stack에서 targeted config가
확보된 경우에만 최종 표에 넣는다.

## 측정 규칙과 채택 기준

- primary: 16K, BF16 uniform `[0,1)`와 `[-8,8)`
- 한 프로세스당 한 case
- warmup 1회, timed 5회
- exploratory: 각 후보 독립 프로세스 4회, position-balanced/Latin order
- confirmation: 유망 후보는 6회까지 확대
- control/candidate가 first/second position을 같은 횟수 차지하도록 구성
- 같은 세션 paired ratio와 95% CI를 우선하고 historical 절대값을
  ablation 근거로 섞지 않음
- 채택 gate: 두 입력 모두 `>= +0.5%`, correctness/resource regression 없음
- `+0.3%`~`+0.8%` 경계 후보는 confirmation 후 결정
- pattern/ones full-C 512 validation과 scheduler coverage validation
- register, spill/local/stack, binary hash, source commit, 환경 로그 보존
- GPU counter가 허용되지 않으면 `L2 병목 확정` 대신
  `locality/cache-policy sensitive`라고 기술

비용 절약 실행 순서:

```text
로컬 코드 작성 및 커밋
-> Vast instance 활성화
-> build / validate / benchmark
-> 로그와 binary/source/SASS 다운로드 및 checksum
-> instance 비활성화
-> 결과 문서 커밋
```

## 우선순위와 예상 정보 가치

| 순위 | 실험 | 이유 |
|---:|---|---|
| 1 | current N-split trace | phase 변경 전 실제 wait/issue gap 확인 |
| 2 | C-store `evict_first` | 가장 작고 아직 안 한 L2 pollution ablation |
| 3 | wave-preserving CTA permutation | collective locality를 보존하며 새 per-SM reuse 생성 |
| 4 | `early_B0` dependency split | sleep 없이 실제 producer/consumer overlap 확대 |
| 5 | winner 결합 | C pollution 감소와 operand persistence의 상승효과 확인 |
| 6 | explicit 3-stage ring | L2 외 control overhead 분리 |
| 7 | 8K/32K 확장 | 16K winner의 크기 일반성 확인 |

## 공식 자료와의 연결

- CUTLASS Efficient GEMM:
  <https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html>
- CUTLASS SM100 static scheduler:
  <https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm100_static_tile_scheduler.hpp>
- CUTLASS SM100 tile scheduler:
  <https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/kernel/sm100_tile_scheduler.hpp>
- PTX `cp.async.bulk.tensor` / L2 cache policy:
  <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>
- CUDA L2 access-policy window:
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#l2-cache>

CUDA persisting access window는 반복 영역을 L2에 우선 보존할 수 있지만,
영역이 set-aside보다 크면 `hitRatio`를 조절하지 않을 때 thrashing이
생길 수 있다. 현재 16K 전체 operand에 바로 적용하지 않고 위의
tile-order/C-store 실험 이후 제한된 panel window 후보로만 검토한다.

## 실험 ledger

| 단계 | commit | artifact | 결과 | 결정 |
|---|---|---|---|---|
| Phase 0 baseline/trace | `4dd4c4c` | `gemm_nsplit_l2_phase_4dd4c4c_phase0` | warp 0의 추가 pipe-1 wait 221--524 cycles | `early_B0` 측정 진행 |
| Phase 1A C-store hint | `4dd4c4c` | `gemm_nsplit_l2_phase_4dd4c4c_phase1a` | -0.0870% / +0.0261% | 미채택 |
| Phase 1B wave mapping | `4dd4c4c` | `gemm_nsplit_l2_phase_4dd4c4c_phase1b` | best `wave_a`도 -0.644% / -0.561% | 전부 미채택 |
| Phase 1C serpentine gate | `4dd4c4c` | Phase 1B offline/GPU gate | stronger locality가 모두 regression | GPU sweep 미진행 |
| Phase 2 structural phase | `4dd4c4c` | `gemm_nsplit_l2_phase_4dd4c4c_phase2` (partial raw) | `issue first` -0.743%/-0.449%; `early` -1.032%/-0.830% | 전부 미채택 |
| Phase 3 stage control | `08e5d62` | pending | B200/credit 대기 | pending |
| Phase 4 combined winner | pending | pending | pending | pending |
| Phase 5 8K/32K | pending | pending | pending | pending |
