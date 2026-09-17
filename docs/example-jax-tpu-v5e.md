# 실제 TPU에서 따라가기: Y = ReLU(XW + b), JAX on TPU v5e

> Colab CLI로 TPU v5e 세션을 띄워 `jax.jit`의 각 단계 출력을 그대로 가져왔다. [example-relu-linear.md](./example-relu-linear.md)의 CPU 결과와 비교하면 **StableHLO까지는 글자 하나 다르지 않고, 그 아래 XLA TPU 백엔드에서 모든 것이 달라진다.** 타일 레이아웃, VMEM 메모리 공간, DMA 프리페치, 패딩된 버퍼 크기가 HLO 텍스트에 그대로 드러난다.

| 항목 | 값 |
|---|---|
| 하드웨어 | TPU v5e (`device_kind = "TPU v5 lite"`), 칩 1개, 코어 1개, HBM 한도 15.75 GiB |
| 소프트웨어 | JAX 0.7.2, Python 3.13 |
| 실행 방법 | `colab new -s tpu --tpu v5e1` → `colab exec -s tpu -f tpu_demo.py` |

```
jax.nn.relu(x @ w + b)              x[16,8], w[8,4], b[4]
   │ ① jax.jit 추적                  jaxpr (dot_general, broadcast_in_dim, add, custom_jvp relu)
   ▼
   │ ② lowering                      StableHLO — CPU 결과와 완전히 동일
   ▼
   │ ③ XLA TPU 백엔드                HLO: fusion 1개 (convolution + add + maximum), 타일 레이아웃,
   │                                VMEM 프리페치 copy-start/copy-done, estimated_cycles=3874
   ▼
   │ ④ LLO → TPU 바이너리            비공개. generated_code_size = 34,816 바이트
   ▼
   │ ⑤ PjRt 디스패치                 jax.Array 즉시 반환, block_until_ready에서 대기
   ▼
Y[16,4] on TpuDevice(id=0), SingleDeviceSharding
```

---

## 0. 코드

```python
import jax, jax.numpy as jnp

def f(x, w, b):
    return jax.nn.relu(x @ w + b)

x = jnp.ones((16, 8), jnp.float32)
w = jnp.ones((8, 4), jnp.float32)
b = jnp.ones((4,), jnp.float32)

print(jax.make_jaxpr(f)(x, w, b))                             # ①
lowered  = jax.jit(f).lower(x, w, b);  print(lowered.as_text())   # ②
compiled = lowered.compile();          print(compiled.as_text())  # ③
print(compiled.cost_analysis(), compiled.memory_analysis())
y = jax.jit(f)(x, w, b)                                        # ⑤
```

## 1. 추적 → jaxpr

```
{ lambda ; a:f32[16,8] b:f32[8,4] c:f32[4]. let
    d:f32[16,4] = dot_general[
      dimension_numbers=(([1], [0]), ([], []))
      preferred_element_type=float32
    ] a b
    e:f32[1,4] = broadcast_in_dim[broadcast_dimensions=(1,) shape=(1, 4) sharding=None] c
    f:f32[16,4] = add d e
    g:f32[16,4] = custom_jvp_call[
      name=relu
      call_jaxpr={ lambda ; h:f32[16,4]. let
          i:f32[16,4] = jit[name=relu jaxpr={ lambda ; h:f32[16,4]. let
                i:f32[16,4] = max h 0.0:f32[]
              in (i,) }] h
        in (i,) }
      jvp=jvp
      symbolic_zeros=False
    ] f
  in (g,) }
```

CPU와 동일하다. jaxpr은 디바이스를 모른다. 유일한 차이는 JAX 0.7.2가 `broadcast_in_dim`에 `sharding=None` 파라미터를 추가로 찍는다는 것뿐이다.

## 2. lowering → StableHLO

```mlir
module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<16x8xf32>, %arg1: tensor<8x4xf32>, %arg2: tensor<4xf32>)
      -> (tensor<16x4xf32> {jax.result_info = "result"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT]
         : (tensor<16x8xf32>, tensor<8x4xf32>) -> tensor<16x4xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<4xf32>) -> tensor<1x4xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x4xf32>) -> tensor<16x4xf32>
    %3 = stablehlo.add %0, %2 : tensor<16x4xf32>
    %4 = call @relu(%3) : (tensor<16x4xf32>) -> tensor<16x4xf32>
    return %4 : tensor<16x4xf32>
  }
  func.func private @relu(%arg0: tensor<16x4xf32>) -> tensor<16x4xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x4xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x4xf32>
    return %1 : tensor<16x4xf32>
  }
}
```

**CPU에서 뽑은 StableHLO와 바이트 단위로 같다.** Scaling Book이 StableHLO를 "platform-agnostic IR"라고 부르는 이유가 이것이다. 레이아웃도, 메모리 공간도, 퓨전도 없다. 이 텍스트가 XLA TPU 백엔드로 넘어간다.

## 3. XLA TPU 백엔드 → 최적화된 HLO

메타데이터(`metadata={op_name=... source_line=...}`)를 걷어낸 전문:

```
HloModule jit_f, is_scheduled=true,
  entry_computation_layout={(f32[16,8]{0,1:T(8,128)}, f32[8,4]{0,1:T(4,128)}, f32[4]{0:T(128)})->f32[16,4]{0,1:T(4,128)}}

%bitcast_fusion (bitcast_input: f32[16,8]) -> f32[16,8] {
  %bitcast_input = f32[16,8]{0,1:T(8,128)S(1)} parameter(0)
  ROOT %bitcast = f32[16,8]{0,1:T(8,128)} bitcast(%bitcast_input)
}

%bitcast_fusion.1 (bitcast_input.1: f32[8,4]) -> f32[8,4] {
  %bitcast_input.1 = f32[8,4]{0,1:T(4,128)} parameter(0)
  ROOT %bitcast.1 = f32[8,4]{0,1:T(4,128)} bitcast(%bitcast_input.1)
}

%fused_computation.1 (param_0.5: f32[16,8], param_1.7: f32[8,4], param_2.5: f32[4]) -> f32[16,4] {
  %param_0.5 = f32[16,8]{0,1:T(8,128)S(1)} parameter(0)
  %fusion.1  = f32[16,8]{0,1:T(8,128)} fusion(%param_0.5), kind=kLoop, calls=%bitcast_fusion
  %param_1.7 = f32[8,4]{0,1:T(4,128)} parameter(1)
  %fusion.2  = f32[8,4]{0,1:T(4,128)} fusion(%param_1.7), kind=kLoop, calls=%bitcast_fusion.1
  %convolution.3 = f32[16,4]{0,1:T(4,128)} convolution(%fusion.1, %fusion.2), dim_labels=bf_io->bf
  %param_2.5 = f32[4]{0:T(128)S(1)} parameter(2)
  %broadcast.2 = f32[16,4]{0,1:T(4,128)} broadcast(%param_2.5), dimensions={1}
  %add.2 = f32[16,4]{0,1:T(4,128)} add(%convolution.3, %broadcast.2)
  %constant.2 = f32[]{:T(128)} constant(0)
  %broadcast.3 = f32[16,4]{0,1:T(4,128)} broadcast(%constant.2), dimensions={}
  ROOT %maximum.1 = f32[16,4]{0,1:T(4,128)} maximum(%add.2, %broadcast.3)
}

ENTRY %main.16 (Arg_0.1: f32[16,8], Arg_1.2: f32[8,4], Arg_2.3: f32[4]) -> f32[16,4] {
  %Arg_0.1 = f32[16,8]{0,1:T(8,128)} parameter(0)
  %copy-start   = (f32[16,8]{0,1:T(8,128)S(1)}, f32[16,8]{0,1:T(8,128)}, u32[]{:S(2)}) copy-start(%Arg_0.1), cross_program_prefetch_index=0
  %Arg_2.3 = f32[4]{0:T(128)} parameter(2)
  %copy-start.1 = (f32[4]{0:T(128)S(1)}, f32[4]{0:T(128)}, u32[]{:S(2)}) copy-start(%Arg_2.3)
  %Arg_1.2 = f32[8,4]{0,1:T(4,128)} parameter(1)
  %copy-done    = f32[16,8]{0,1:T(8,128)S(1)} copy-done(%copy-start)
  %copy-done.1  = f32[4]{0:T(128)S(1)} copy-done(%copy-start.1)
  ROOT %fusion = f32[16,4]{0,1:T(4,128)} fusion(%copy-done, %Arg_1.2, %copy-done.1), kind=kOutput, calls=%fused_computation.1,
       backend_config={"window_config":{"kernel_window_bounds":["1","1"],"output_window_bounds":["1","1"],
                       "input_window_bounds":["1","1"],"estimated_cycles":"3874","iteration_bounds":["1","1","1"]},
                       "used_scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"4096"}],
                       "convolution_algorithm_config":{"emitter":"EmitInputFeaturePackedInputBatchInLanes"}}
}
```

CPU HLO에는 없던 것들이 한꺼번에 나타난다. 하나씩 읽는다.

### 3-1. 퓨전: fusion 하나로 전부

CPU에서는 matmul fusion과 `add+maximum` fusion **2개**였다. TPU에서는 `convolution + broadcast + add + maximum`이 **`kOutput` fusion 하나**다. matmul 결과가 MXU에서 나오는 즉시 VPU가 bias를 더하고 max를 취해 HBM에 쓴다. 중간 텐서 `x@w`와 `x@w+b`는 HBM은 물론 VMEM에도 내려가지 않는다. [③](./03-ir-to-executable.md)에서 "TPU였다면 matmul + epilogue로 합쳐질 것"이라고 예상한 것이 확인됐다.

### 3-2. `dot`이 `convolution`으로

`stablehlo.dot_general`이 `convolution(...), dim_labels=bf_io->bf`로 바뀌었다. TPU 백엔드는 행렬곱을 1×1 convolution의 특수한 경우로 다루고, MXU용 emitter(`EmitInputFeaturePackedInputBatchInLanes`)가 그 위에서 코드를 만든다. `dim_labels`의 `b`는 batch(=16행), `f`는 feature(=contracting 8), `i/o`는 input/output feature다.

### 3-3. 레이아웃: `{0,1:T(8,128)}` 읽는 법

CPU의 `{1,0}`(row-major) 대신 `{0,1:T(8,128)}`이 붙었다.

| 표기 | 의미 |
|---|---|
| `{0,1}` | minor-to-major 순서. **축 0(행)이 가장 minor**, 즉 열 우선(column-major)이다. XLA가 MXU에 맞춰 뒤집었다. |
| `T(8,128)` | 타일. 텐서를 8×128 원소 블록으로 나눠 각 블록을 연속 저장한다. Scaling Book의 "f32는 8×128 타일"이 이것이다. |
| `T(4,128)` | `[8,4]`와 `[16,4]`처럼 한 축이 8 미만이면 더 작은 타일을 쓴다. |
| `T(128)` | 1차원 `b[4]`는 128 원소 타일. |
| `S(1)` | **memory space 1 = VMEM**. 표기가 없으면 HBM. |
| `S(2)` | copy-start의 동기화 플래그가 사는 메모리 공간(SMEM). |

타일은 **패딩**을 뜻한다. `[16,4]`를 `T(4,128)`로 저장하면 major 축 4는 4로 채워지고 minor 축 16은 128로 패딩되어 4×128 = 512 원소가 된다. 실제 원소는 64개다.

### 3-4. 메모리 공간과 DMA 프리페치: `copy-start` / `copy-done`

```
%copy-start = (f32[16,8]{...S(1)}, f32[16,8]{...}, u32[]{:S(2)}) copy-start(%Arg_0.1), cross_program_prefetch_index=0
...
%copy-done  = f32[16,8]{...S(1)} copy-done(%copy-start)
```

`x`와 `b`를 HBM에서 VMEM(`S(1)`)으로 옮기는 **비동기 DMA**다. `copy-start`가 DMA를 시작하고 `copy-done`이 완료를 기다린다. 그 사이에 다른 파라미터를 준비하도록 스케줄러가 순서를 짰다(`is_scheduled=true`). `cross_program_prefetch_index=0`은 **이전 실행이 끝나기 전에 다음 실행의 입력을 미리 VMEM에 올려 두는** 프로그램 간 프리페치다. [⑤](./05-kernel-to-chip.md)에서 "TPU는 캐시가 없고 컴파일러가 모든 이동을 정적으로 지시한다"고 한 것의 실물이다.

`w`(`Arg_1.2`)는 프리페치 없이 HBM에서 직접 fusion으로 들어간다. 컴파일러가 VMEM 예산과 이득을 따져 무엇을 올릴지 골랐다.

`bitcast_fusion`은 `S(1)` 버퍼를 메모리 공간 표기 없는 타입으로 재해석하는 것으로, 데이터 이동은 없다.

### 3-5. `backend_config`: LLO가 남긴 흔적

```
"estimated_cycles":"3874"
"used_scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"4096"}]
"iteration_bounds":["1","1","1"]
```

- 이 fusion이 약 **3,874 사이클**로 추정됐다. v5e 클럭에서 수 µs다. 실제 FLOPs(1,152개)는 MXU 한 사이클 분량도 안 되므로, 시간은 전부 DMA·파이프라인 채우기·비우기다.
- fusion이 VMEM에서 **4,096바이트**의 스크래치 공간을 쓴다.
- `iteration_bounds` 1×1×1: 타일 하나에 다 들어가서 루프가 없다.

### 3-6. 비용·메모리 분석

```python
compiled.cost_analysis()
# {'flops': 1152.0, 'bytes accessed': 19968.0, 'optimal_seconds': 2.7e-08,
#  'bytes accessed0{}': 9728.0, 'bytes accessed1{}': 2048.0, 'bytes accessed2{}': 512.0, 'bytes accessedout{}': 7680.0, ...}
compiled.memory_analysis()
# CompiledMemoryStats(generated_code_size_in_bytes=34816, argument_size_in_bytes=6656,
#                     output_size_in_bytes=2048, temp_size_in_bytes=0, ...)
```

**패딩이 숫자로 보인다.** 인자의 실제 크기와 XLA가 잡은 크기를 비교하면:

| 텐서 | 실제 바이트 | 레이아웃 | 패딩 후 원소 | 패딩 후 바이트 |
|---|---|---|---|---|
| `x[16,8]` | 512 | `{0,1:T(8,128)}` → major 8, minor 16→128 | 8×128 | 4,096 |
| `w[8,4]` | 128 | `{0,1:T(4,128)}` → major 4, minor 8→128 | 4×128 | 2,048 |
| `b[4]` | 16 | `{0:T(128)}` → 4→128 | 128 | 512 |
| **인자 합계** | **656** | | | **6,656** ← `argument_size_in_bytes` |
| `y[16,4]` | 256 | `{0,1:T(4,128)}` → major 4, minor 16→128 | 4×128 | 2,048 ← `output_size_in_bytes` |

실제 데이터 656바이트가 칩 위에서는 6,656바이트를 차지한다. 10배다. [①](./01-model-to-ops.md)에서 "D, F 같은 차원을 128의 배수로 잡는" 이유가 여기 있다. 이 예제의 shape은 MXU에 맞지 않는 극단적인 경우다.

`flops: 1152`는 matmul 2·16·8·4 = 1,024에 add·max 64×2 = 128을 더한 값이다. `generated_code_size_in_bytes: 34816`은 LLO가 만든 TPU 바이너리 크기다.

### 3-7. bf16으로 바꾸면

같은 함수를 bf16 입력으로 컴파일하면 레이아웃에 `(2,1)`이 추가된다.

```
entry_computation_layout={(bf16[16,8]{0,1:T(8,128)(2,1)}, bf16[8,4]{0,1:T(4,128)(2,1)}, bf16[4]{0:T(256)(128)(2,1)})
                          ->bf16[16,4]{0,1:T(4,128)(2,1)}}
...
"estimated_cycles":"3870", "used_scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"2048"}]
```

`(2,1)`은 bf16 원소 2개를 32비트 워드 하나에 패킹한다는 뜻이다. `b[4]`의 타일이 `T(256)`으로 커진 것도 같은 이유다(256 bf16 = 128 워드). VMEM 스크래치가 4,096 → 2,048바이트로 절반이 됐다. 사이클은 거의 같다. 이 크기에서는 데이터 이동이 아니라 파이프라인 고정 비용이 지배하기 때문이다.

## 4. LLO → TPU 바이너리

이 단계는 텍스트로 볼 수 없다. 남는 흔적은 `generated_code_size_in_bytes = 34816`과 `backend_config`의 `estimated_cycles`뿐이다. Scaling Book에 따르면 LLO가 fusion마다 MXU 명령, VPU 명령(8×128 타일 단위 add·max), DMA 명령을 VLIW 번들로 묶고 소프트웨어 파이프라이닝으로 겹친 뒤, 결과 바이너리를 TPU IMEM에 로드한다. `estimated_cycles`가 그 스케줄의 길이다.

## 5. 실행과 비동기 디스패치

```
type(y) = jaxlib._jax.ArrayImpl, shape (16, 4), dtype float32
y.sharding = SingleDeviceSharding(device=TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0), memory_kind=device)
y[:2] = [[9. 9. 9. 9.]
         [9. 9. 9. 9.]]
```

값은 `relu(8·1 + 1) = 9`로 맞다. `sharding`에 디바이스 좌표 `coords=(0,0,0)`가 있다. 다중 칩 ICI 토러스에서의 위치이고, 여기서는 칩 하나뿐이다.

비동기 디스패치 측정 (4096×4096 bf16 행렬곱, 웜업 후):

| 측정 | 시간 |
|---|---|
| `z = xx @ xx` 반환까지 | 177 µs |
| `z.block_until_ready()`까지 | 1.028 ms |

2·4096³ ≈ 137 GFLOP를 1.03 ms에 처리했으니 약 134 TFLOP/s, v5e 피크(197 TFLOP/s)의 약 68%다. 디스패치 177 µs는 [④](./04-dispatch-to-kernel.md)의 JAX 문서 예제(269 µs)와 같은 크기다.

## 6. 큰 행렬로 보는 타일링: 2048×2048 bf16

작은 예제는 패딩만 보여 준다. `relu(a @ b)`를 2048×2048 bf16으로 컴파일하면 컴파일러의 결정이 달라진다.

```
entry_computation_layout={(bf16[2048,2048]{1,0:T(8,128)(2,1)}, bf16[2048,2048]{1,0:T(8,128)(2,1)})->bf16[2048,2048]{1,0:T(8,128)(2,1)}}

ENTRY %main.10 (Arg_0.1: bf16[2048,2048], Arg_1.2: bf16[2048,2048]) -> bf16[2048,2048] {
  %copy-start = (...S(1)...) copy-start(%Arg_0.1), cross_program_prefetch_index=0
  %copy-done  = bf16[2048,2048]{1,0:T(8,128)(2,1)S(1)} copy-done(%copy-start)
  ROOT %convolution_maximum_fusion = fusion(%copy-done, %Arg_1.2), kind=kOutput,
       backend_config={"window_config":{"kernel_window_bounds":["256","1"],"output_window_bounds":["256","1"],
                       "input_window_bounds":["256","16"],"estimated_cycles":"143032","iteration_bounds":["16","1","1"]},
                       "used_scoped_memory_configs":[{"memory_space":"1","offset":"0","size":"2985984"}],
                       "convolution_algorithm_config":{"emitter":"EmitAllBatchInSublanes"}}
}
```

```
cost_analysis: {'flops': 17184063488.0, 'bytes accessed': 41944064.0, 'optimal_seconds': 0.00011, ...}
```

| 결정 | 작은 예제 | 2048×2048 |
|---|---|---|
| 레이아웃 | `{0,1}` (열 우선) | `{1,0}` (행 우선) |
| MXU emitter | `EmitInputFeaturePackedInputBatchInLanes` | `EmitAllBatchInSublanes` |
| 반복 | `iteration_bounds` 1×1×1 | **16×1×1**, 윈도우 256행씩 |
| VMEM 스크래치 | 4,096 B | **2,985,984 B (≈2.85 MiB)** |
| 추정 사이클 | 3,874 | 143,032 |
| 프리페치 대상 | `x`, `b` | `a` 전체(8 MiB)를 VMEM으로 |

같은 함수인데 **레이아웃, MXU 알고리즘, 루프 구조, VMEM 사용량이 모두 바뀌었다.** 2048×2048 bf16은 8 MiB라 v5e VMEM(128 MiB)에 통째로 들어가므로 XLA가 `a`를 프리페치하고 `b`는 256행 윈도우 16번으로 스트리밍한다. arithmetic intensity는 17.2 GFLOP / 41.9 MB ≈ **410 FLOPs/byte**로, v5e의 HBM 임계값(197e12 / 8.2e11 ≈ 240)을 넘는다. [⑤](./05-kernel-to-chip.md)의 기준으로 **compute-bound**이며, `optimal_seconds` 110 µs는 FLOPs / 피크 FLOPs/s에 가깝다.

## 7. CPU 결과와의 차이 요약

| | CPU (XLA CPU) | TPU v5e (XLA TPU) |
|---|---|---|
| StableHLO | 동일 | 동일 |
| fusion 수 | 2 (`kCustom` dot, `kLoop` add+max) | **1** (`kOutput` conv+add+max) |
| matmul 표현 | `dot` → YNN 라이브러리 | `convolution` → MXU emitter |
| 레이아웃 | `{1,0}` | `{0,1:T(8,128)}` 타일, bf16은 `(2,1)` 패킹 |
| 메모리 공간 | 없음 | `S(1)` VMEM, `S(2)` 플래그 |
| 데이터 이동 | 암묵적 (캐시) | `copy-start`/`copy-done` 명시적 DMA, cross-program prefetch |
| 패딩 | 없음 | 인자 656 B → 6,656 B |
| 코드 크기 흔적 | 없음 | `generated_code_size_in_bytes=34816`, `estimated_cycles=3874` |

## 8. 재현

```bash
uv tool install google-colab-cli
colab new -s tpu --tpu v5e1
colab exec -s tpu -f tpu_demo.py --timeout 900    # jaxpr, StableHLO, compiled HLO(f32/bf16), cost/memory analysis, async timing
colab exec -s tpu -f tpu_extra.py --timeout 600   # memory_stats, 2048² bf16 HLO
colab stop -s tpu
```

`compiled.as_text()`의 `metadata={...}`는 가독성을 위해 이 문서에서 제거했다. 원문에는 각 명령이 나온 Python 소스 줄·열이 붙어 있어 프로파일러가 HLO를 코드로 되돌리는 데 쓴다.
