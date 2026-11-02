# Week 7 컴파일 산출물 핵심 설명

**같은 행렬곱이 TTIR에서는 타일 연산으로, TTGIR에서는 스레드·메모리 배치로, PTX와 SASS에서는 GPU 명령으로 구체화됩니다.**

A100에서 컴파일한 실제 산출물을 기준으로 설명합니다. 출력 타일은 `64×32`, K 타일 폭은 `32`, CTA는 `2 warp = 64 threads`입니다. 입력은 FP16, 누산은 FP32입니다. 아래 코드는 읽기 쉽게 위치 정보와 일부 타입·인자를 생략한 발췌입니다. 전체 코드는 [원본](artifacts/a100/)과 [주석 사본](artifacts/a100_annotated/)에 있습니다.

## 1. TTIR: 어떤 타일을 계산하는가

```mlir
%pid = tt.get_program_id x : i32
```

현재 CTA의 번호입니다. 소스의 `tl.program_id(axis=0)`, CUDA의 `blockIdx.x`에 대응합니다. 이 번호를 그룹핑 수식으로 출력 타일의 행·열 좌표로 바꿉니다.

```mlir
%accumulator = arith.constant dense<0.000000e+00> : tensor<64x32xf32>
```

CTA가 계산할 `64×32` 출력 누산기를 FP32의 0으로 초기화합니다. 각 스레드가 이 텐서 전체를 갖는 것이 아니라, CTA의 스레드들이 나눠 가집니다.

```mlir
%a_ptrs = tt.expand_dims %offs_am_12 {axis = 1 : i32}
```

행 인덱스의 shape를 `(64,) → (64, 1)`로 바꿉니다. 소스의 `offs_am[:, None]`입니다. K 인덱스를 `(1, 32)`로 만들고 브로드캐스팅하면 `64×32` 타일의 주소를 계산할 수 있습니다.

```mlir
%a_55 = tt.load %a_ptrs_49, %a_54, %cst_0
%b_58 = tt.load %b_ptrs_50, %b_57, %cst
```

첫 줄은 A의 `64×32` 타일, 둘째 줄은 B의 `32×32` 타일을 읽습니다. 두 번째 인자는 마스크, 세 번째는 범위 밖 원소에 사용할 0입니다.

```mlir
%accumulator_59 = tt.dot %a_55, %b_58, %accumulator_51
```

다음 계산입니다.

```text
accumulator[64×32] += A[64×32] @ B[32×32]
```

FP16 입력을 곱하고 FP32로 누산합니다. K 루프에서 입력 포인터를 다음 타일로 옮기며 반복합니다.

```mlir
%c = arith.truncf %accumulator_34#2
    : tensor<64x32xf32> to tensor<64x32xf16>
```

누적이 끝난 결과를 FP16으로 변환합니다. `%accumulator_34#2`는 루프가 반환한 세 값 중 세 번째 값입니다.

```mlir
 tt.store %c_ptrs_42, %c, %c_mask_48
```

실제 출력 행렬 범위 안의 원소만 C에 저장합니다.

**나머지 코드**는 그룹핑 좌표, 포인터 오프셋, 경계 마스크를 계산합니다. `splat`은 스칼라 확장, `broadcast`는 shape 확장, `addptr`는 원소 단위 포인터 이동입니다. `%a_55`의 `55`는 식별 번호이며 주소가 아닙니다.

## 2. TTGIR: 스레드와 메모리에 어떻게 나누는가

```mlir
#blocked = #ttg.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [8, 4],
  warpsPerCTA = [2, 1],
  order = [1, 0]
}>
```

일반 텐서의 원소를 스레드에 나누는 방식입니다.

- 스레드 하나의 기본 원소 묶음: `1×8`
- warp의 스레드 배치: `8×4 = 32 threads`
- CTA의 warp 배치: `2×1`
- 열 방향 우선 배치: `order=[1,0]`

이 기본 단위는 `16×32`를 덮고, 더 큰 텐서는 이 배치를 반복해 덮습니다. Tensor Core 누산기는 별도의 `#mma` 레이아웃을 사용합니다.

```mlir
%a = ttg.local_alloc : () -> !ttg.memdesc<4x64x32xf16, ...>
%b = ttg.local_alloc : () -> !ttg.memdesc<4x32x32xf16, ...>
```

A·B 입력용 shared memory 버퍼입니다.

```text
A: 4 × 64 × 32 × 2 bytes = 16 KiB
B: 4 × 32 × 32 × 2 bytes =  8 KiB
합계:                       24 KiB
```

설정은 `num_stages=5`지만 **실제 shared 버퍼는 각각 4개**입니다. stage 설정을 그대로 메모리 배수로 해석하면 안 됩니다.

```mlir
%a_43 = ttg.async_copy_global_to_local ...
%a_44 = ttg.async_commit_group tokens %a_43
```

첫 줄은 global→shared 비동기 복사이고, 둘째 줄은 그 복사를 그룹으로 확정합니다. 확정은 완료 대기와 다릅니다.

```mlir
%a_146 = ttg.async_wait %a_133, %b_137 {num = 6 : i32}
```

현재 계산에 필요한 오래된 복사가 완료되도록 기다립니다. 이후 단계의 복사는 진행 중으로 둘 수 있습니다. `6`은 복사 그룹의 대기 조건이며 stage 수가 아닙니다.

```mlir
%a_148 = ttg.local_load %a_147 token %a_146 ...
%b_150 = ttg.local_load %b_149 token %a_146 ...
```

Shared memory의 A·B를 Tensor Core 입력에 맞는 레지스터 fragment로 읽습니다.

**나머지 코드**는 다음 K 타일을 미리 적재하고, 원형 버퍼의 슬롯을 바꾸고, 스레드별 데이터 배치를 변환합니다. Shared memory의 swizzling은 bank 충돌을 줄이는 배치이며, L2 재사용을 위한 CTA 그룹핑과는 별개입니다.

## 3. LLVM IR: GPU 명령으로 연결하는 중간 단계

```llvm
%12 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
```

CTA 번호 읽기가 NVIDIA용 intrinsic으로 바뀌었습니다.

```llvm
; 실제 줄에서 주요 부분만 발췌
asm sideeffect "cp.async.cg.shared.global ...", "r,l,r"(...)
```

복사 연산은 inline PTX로 내려갑니다. 주변의 `addrspace(1)`은 global, `addrspace(3)`은 shared memory를 나타냅니다.

**나머지 코드**는 주소 계산, 루프 분기, fragment 값의 조합과 추출을 처리합니다. 여기서는 전체 줄을 따라가기보다, TTGIR의 복사·동기화·행렬곱이 어떤 intrinsic이나 inline PTX로 연결되는지 보면 됩니다.

## 4. PTX: 복사하고, 기다리고, 곱한다

```ptx
cp.async.cg.shared.global [%r26 + 0], [%rd12 + 0], 0x10, %r28;
```

Global 주소 `%rd12`에서 shared 주소 `%r26`으로 비동기 복사합니다. `0x10`은 16바이트이고, `%r28`은 실제 원본에서 읽을 유효 바이트 수입니다. 부족한 부분은 0으로 채웁니다. L2에 적중하면 HBM까지 읽으러 가지 않아도 됩니다.

```ptx
cp.async.commit_group;
```

앞서 발행한 복사들을 그룹으로 확정합니다.

```ptx
cp.async.wait_group 6;
bar.sync 0;
```

첫 줄은 오래된 복사의 완료를 기다리고, 둘째 줄은 CTA의 스레드들을 동기화합니다. 스레드별 복사 완료 대기와 CTA 동기화는 서로 다른 역할입니다.

```ptx
ldmatrix.sync.aligned.m8n8.x4.shared.b16
    {%r130, %r131, %r132, %r133}, [%r174];
```

Warp가 협력해 shared memory에서 16비트 `8×8` 행렬 네 개를 레지스터 fragment로 읽습니다. 각 스레드가 행렬 네 개를 통째로 갖는다는 뜻은 아닙니다.

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%r232, %r233, %r234, %r235},
    {%r130, %r131, %r132, %r133},
    {%r134, %r135},
    {%r232, %r233, %r234, %r235};
```

Warp 단위로 `M=16, N=8, K=16` 행렬곱을 누산합니다. A·B는 FP16, 누산은 FP32입니다. 첫 번째와 마지막 레지스터 묶음이 같으므로 기존 누산값에 더합니다.

CTA의 `64×32` 출력과 K 폭 `32`는 이 명령 여러 개로 계산합니다.

**나머지 코드**는 인자 읽기, 주소 계산, 조건 검사, 루프 제어, FP16 변환과 출력 저장입니다. PTX의 `%r`는 가상 레지스터이므로 선언 개수를 실제 레지스터 사용량으로 세면 안 됩니다.

## 5. SASS: GPU에서 실행할 실제 명령

| 실제 명령 계열 | 역할 | 연결되는 PTX |
|---|---|---|
| `LDGSTS` | global→shared 복사 | `cp.async` |
| `LDGDEPBAR`, `DEPBAR` | 비동기 복사 그룹 관리·대기 | `commit_group`, `wait_group` |
| `BAR` | CTA 동기화 | `bar.sync` |
| `LDSM` | shared→레지스터 행렬 fragment 로드 | `ldmatrix` |
| `HMMA` | Tensor Core 행렬곱 누산 | `mma.sync` |
| `STG` | global 출력 저장 | `st.global` |

단계 사이에 명령 분할과 스케줄링이 있어 항상 1:1로 대응하지는 않습니다. SASS 앞부분의 ELF·debug 데이터는 묶어서 건너뛰고, `.text.matmul_kernel_single`의 명령을 보면 됩니다.

전체 흐름은 다음과 같습니다.

```text
CTA 출력 좌표 계산
    ↓
A·B를 global → shared로 미리 복사
    ↓
필요한 복사 완료 대기와 CTA 동기화
    ↓
shared → 레지스터 fragment 로드
    ↓
Tensor Core로 FP32 누산 ← K축을 따라 반복
    ↓
FP16 변환 → C에 저장
```

[주석이 포함된 전체 산출물 보기](artifacts/a100_annotated/)
