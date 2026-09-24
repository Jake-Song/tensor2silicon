# 실제 GPU에서 따라가기: Y = ReLU(XW + b), PyTorch 2 on NVIDIA A100

> Colab CLI로 A100 GPU 세션을 띄워 `torch.compile`의 각 단계 출력을 그대로 가져왔다. [example-relu-linear.md](./example-relu-linear.md)의 CPU 결과와 비교하면 **같은 그래프가 백엔드에 따라 어떻게 다르게 컴파일되는지**가 보인다. 이 문서는 Inductor가 만든 Triton 커널이 **Triton IR → TTGIR → PTX → SASS**로 내려가는 마지막 구간까지 실물로 담았고, 같은 코드를 T4에서 돌린 결과와의 차이를 끝에 정리했다.

| 항목 | 값 |
|---|---|
| 하드웨어 | NVIDIA A100-SXM4-40GB (Ampere, compute capability 8.0), SM 108개, HBM 40 GB, L2 40 MB, SM당 SMEM 164 KB, 최대 1410 MHz |
| 소프트웨어 | PyTorch 2.11.0+cu128, CUDA 12.8, Triton 3.6.0, Python 3.13 |
| 실행 방법 | `colab new -s a100 --gpu A100` → `colab exec -s a100 -f gpu_demo.py` |

```
torch.relu(x @ w + b)               x[16,8], w[8,4], b[4]  (w, b는 requires_grad)
   │ ① Dynamo                       FX 그래프 + guard (device=cuda:0 포함)
   ▼
   │ ② AOT Autograd                 ATen forward / backward 그래프
   ▼
   │ ③ Inductor                     forward: cuBLAS mm + Triton 커널 1개
   │                                backward: Triton 커널 1개 + cuBLAS mm
   ▼
   │ ④ Triton 컴파일러              TTIR → TTGIR → LLVM IR → PTX (sm_80) → cubin (SASS 56개 명령)
   ▼
   │ ⑤ 런타임                       CUDA 스트림에 런치, 즉시 반환, synchronize에서 대기
   ▼
Y[16,4] on cuda:0, grad_fn=CompiledFunctionBackward
```

---

## 0. 코드

```python
import torch

def f(x, w, b):
    return torch.relu(x @ w + b)

x = torch.ones(16, 8, device="cuda")
w = torch.ones(8, 4, device="cuda", requires_grad=True)
b = torch.ones(4, device="cuda", requires_grad=True)
y = torch.compile(f)(x, w, b)
y.sum().backward()
```

로그는 `TORCH_LOGS="graph_code,guards,aot_graphs,output_code"`로 켰다.

## 1. TorchDynamo: FX 그래프와 guard

```python
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[16, 8][8, 1]cuda:0", L_w_: "f32[8, 4][4, 1]cuda:0", L_b_: "f32[4][1]cuda:0"):
        l_x_ = L_x_
        l_w_ = L_w_
        l_b_ = L_b_
        matmul: "f32[16, 4][4, 1]cuda:0" = l_x_ @ l_w_;  l_x_ = l_w_ = None
        add: "f32[16, 4][4, 1]cuda:0" = matmul + l_b_;  matmul = l_b_ = None
        relu: "f32[16, 4][4, 1]cuda:0" = torch.relu(add);  add = None
        return (relu,)
```

CPU 결과와 다른 점은 타입 주석의 `cuda:0`뿐이다. Dynamo는 디바이스를 신경 쓰지 않고 Python 바이트코드만 본다.

**guard** (일부):

```
TREE_GUARD_MANAGER:
+- RootGuardManager
| +- GLOBAL_STATE: ___check_global_state() against {"allow_tf32":false, "grad_mode":true, ...}
| +- GuardManager: source=L['x']
| | +- TENSOR_MATCH: check_tensor(L['x'], Tensor, DispatchKeySet(CUDA, BackendSelect, ADInplaceOrView, AutogradCUDA),
| |                  torch.float32, device=0, requires_grad=False, size=[16, 8], stride=[8, 1])
| | +- NO_TENSOR_ALIASING: check_no_aliasing(L['b'], L['w'], L['x'])
| +- GuardManager: source=L['w']
| | +- TENSOR_MATCH: check_tensor(L['w'], ..., device=0, requires_grad=True, size=[8, 4], stride=[4, 1])
| +- GuardManager: source=L['b']
| | +- TENSOR_MATCH: check_tensor(L['b'], ..., device=0, requires_grad=True, size=[4], stride=[1])
| +- GuardManager: source=G['torch'].relu
| | +- ID_MATCH: ___check_obj_id(G['torch'].relu, 131994283891648)

Guard eval latency = 35.51 us
```

읽을 점:

- 입력 텐서마다 dtype, **device=0**, `requires_grad`, size, stride를 검사한다. 같은 함수를 CPU 텐서로 부르면 guard가 실패해 재컴파일된다.
- `allow_tf32: false` 같은 전역 상태도 guard다. A100은 TF32 Tensor Core가 있어서 이 플래그를 켜면 cuBLAS가 고르는 matmul 커널이 바뀌므로, 플래그를 바꾸면 재컴파일된다(5절에서 실제 속도 차이를 잰다).
- `torch.relu`가 **객체 ID**로 고정돼 있다. 누가 `torch.relu`를 몽키패치하면 재컴파일된다.
- guard 평가에 약 36 µs가 든다. 이것이 컴파일된 함수를 부를 때마다 내는 고정 비용이다.

## 2. AOT Autograd: ATen forward / backward 그래프

```python
# ===== Forward graph 0 =====
def forward(self, primals_1: "f32[16, 8]cuda:0", primals_2: "f32[8, 4]cuda:0", primals_3: "f32[4]cuda:0"):
    mm:   "f32[16, 4]" = torch.ops.aten.mm.default(primals_1, primals_2)
    add:  "f32[16, 4]" = torch.ops.aten.add.Tensor(mm, primals_3)
    relu: "f32[16, 4]" = torch.ops.aten.relu.default(add)
    le:      "b8[16, 4]"  = torch.ops.aten.le.Scalar(relu, 0)            # backward용 마스크
    permute: "f32[8, 16]" = torch.ops.aten.permute.default(primals_1, [1, 0])   # Xᵀ
    return (relu, le, permute)

# ===== Backward graph 0 =====
def forward(self, le: "b8[16, 4]", permute: "f32[8, 16]", tangents_1: "f32[16, 4]"):
    full_default = torch.ops.aten.full.default([], 0.0, device=cuda:0)
    where: "f32[16, 4]" = torch.ops.aten.where.self(le, full_default, tangents_1)   # ReLU 미분
    sum_1: "f32[1, 4]"  = torch.ops.aten.sum.dim_IntList(where, [0], True)          # ∂L/∂b
    view:  "f32[4]"     = torch.ops.aten.view.default(sum_1, [4])
    mm_1:  "f32[8, 4]"  = torch.ops.aten.mm.default(permute, where)                 # ∂L/∂W = Xᵀ·∂L/∂Y
    return (None, mm_1, view)
```

CPU와 **완전히 동일**하다. AOT Autograd까지는 디바이스 독립적인 단계다. 차이는 다음 단계에서 시작된다.

## 3. TorchInductor: cuBLAS 호출 + Triton 커널

### 3-1. forward 래퍼

```python
def call(args):
    primals_1, primals_2, primals_3 = args
    assert_size_stride(primals_1, (16, 8), (8, 1))
    assert_size_stride(primals_2, (8, 4), (4, 1))
    assert_size_stride(primals_3, (4, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
        # Topologically Sorted Source Nodes: [matmul], Original ATen: [aten.mm]
        extern_kernels.mm(primals_1, primals_2, out=buf0)
        buf1 = buf0; del buf0  # reuse
        buf2 = empty_strided_cuda((16, 4), (4, 1), torch.bool)
        # Topologically Sorted Source Nodes: [add, relu, le], Original ATen: [aten.add, aten.relu, aten.threshold_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_relu_threshold_backward_0.run(buf1, primals_3, buf2, 64, stream=stream0)
    return (buf1, buf2, reinterpret_tensor(primals_1, (8, 16), (1, 8), 0), )
```

**CPU와 결정이 다르다.** CPU에서는 `mm + add(bias)`를 `addmm` 하나로 묶고 relu만 따로 커널을 만들었다. GPU에서는 matmul을 **순수 `mm`(cuBLAS)** 로 보내고, **bias 덧셈을 relu·마스크와 함께 Triton 커널 하나**로 묶었다. 커널 이름 `fused_add_relu_threshold_backward`에 `add`가 들어간 것이 그 증거다. 어느 쪽이 빠른지는 백엔드 휴리스틱이 정한다. GPU에서는 작은 elementwise 연산을 라이브러리 matmul의 epilogue에 넣는 것보다 자체 Triton 커널에서 한 번에 처리하는 편을 택했다.

`stream0 = get_raw_stream(0)`과 `.run(..., stream=stream0)`이 [④](./04-dispatch-to-kernel.md)에서 말한 **CUDA 스트림에 큐잉**하는 지점이다. `extern_kernels.mm`도 같은 스트림에 들어가므로 둘의 순서가 보장된다.

### 3-2. forward Triton 커널

```python
@triton_heuristics.pointwise(
    size_hints={'x': 64},
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'out_ptr0': '*i1',
                               'xnumel': 'i32', 'XBLOCK': 'constexpr'},
                 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=108, cc=80,
                                            regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048,
                                            max_threads_per_block=1024, warp_size=32),
                 ...},
    inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_relu_threshold_backward_0',
                   'mutated_arg_names': ['in_out_ptr0'], 'num_load': 2, 'num_store': 2, 'num_reduction': 0, ...},
)
@triton.jit
def triton_poi_fused_add_relu_threshold_backward_0(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 4)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)                              # mm 결과 [16,4]
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')   # bias [4], 브로드캐스트
    tmp2 = tmp0 + tmp1                                                     # + b
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)                              # relu
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tmp4 <= tmp5                                                    # le (backward 마스크)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)                              # in-place
    tl.store(out_ptr0 + (x2), tmp6, xmask)
```

읽을 점:

- [④](./04-dispatch-to-kernel.md)의 벡터 덧셈 커널과 뼈대가 같다. `program_id`, `arange`, `xmask`, `load`, 계산, `store`.
- **브로드캐스트가 인덱스 계산으로 구현**됐다. `x0 = xindex % 4`로 bias의 인덱스를 만들고 `eviction_policy='evict_last'`로 4개짜리 bias가 캐시에서 밀리지 않게 힌트를 줬다. jaxpr의 `broadcast_in_dim` 노드가 여기서는 나머지 연산 하나다.
- `DeviceProperties(multi_processor_count=108, cc=80, ...)`가 컴파일 메타데이터에 박혀 있다. **커널이 이 GPU에 특수화**됐고, 다른 GPU에서는 다시 컴파일된다. T4에서는 같은 자리에 `multi_processor_count=40, cc=75`가 찍혔고 `backend_hash`도 달랐다.
- `num_load: 2, num_store: 2, num_reduction: 0`. 원소 64개에 대해 HBM 접근이 읽기 2 + 쓰기 2뿐이다. eager였다면 add, relu, le가 각각 커널이라 읽기 3 + 쓰기 3에 런치 3번이다.

### 3-3. backward

```python
@triton_heuristics.persistent_reduction(size_hints={'x': 4, 'r0_': 16}, ...)
@triton.jit
def triton_per_fused_sum_threshold_backward_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0 + 4*r0_1), xmask, other=0.0).to(tl.int1)   # le 마스크
    tmp1 = tl.load(in_ptr1 + (x0 + 4*r0_1), xmask, other=0.0)               # tangents (∂L/∂Y)
    tmp2 = tl.full([1, 1], 0.0, tl.float32)
    tmp3 = tl.where(tmp0, tmp2, tmp1)                                        # ReLU 미분
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None].to(tl.float32)                           # sum over rows → ∂L/∂b
    tl.store(out_ptr0 + (x0 + 4*r0_1), tmp3, xmask)                          # where 결과 (mm에 씀)
    tl.store(out_ptr1 + (x0), tmp7, xmask)                                   # ∂L/∂b
```

```python
def call(args):
    le, permute, tangents_1 = args
    ...
    buf0 = empty_strided_cuda((16, 4), (4, 1), torch.float32)
    buf1 = empty_strided_cuda((1, 4), (4, 1), torch.float32)
    stream0 = get_raw_stream(0)
    triton_per_fused_sum_threshold_backward_0.run(le, tangents_1, buf0, buf1, 4, 16, stream=stream0)
    buf2 = empty_strided_cuda((8, 4), (4, 1), torch.float32)
    extern_kernels.mm(permute, buf0, out=buf2)                               # ∂L/∂W = Xᵀ · where
    return (None, buf2, reinterpret_tensor(buf1, (4, ), (1, ), 0), )
```

읽을 점:

- 이번에는 **pointwise + reduction 퓨전**이다. `where`(pointwise)와 `sum(dim=0)`(reduction)이 커널 하나가 됐다. 이름의 `per`는 persistent reduction, 즉 reduction 축(16행) 전체를 블록 하나가 레지스터에 들고 처리한다는 뜻이다.
- 2차원 인덱싱 `[:, None]`과 `[None, :]`로 `[XBLOCK, R0_BLOCK]` 타일을 만들고 축 1로 `tl.sum`한다.
- `where` 결과를 HBM에 한 번 쓰고(`out_ptr0`) 그것을 cuBLAS `mm`이 다시 읽는다. matmul은 Triton 커널에 퓨전되지 않으므로 여기서는 HBM 왕복이 불가피하다.

## 4. Triton 컴파일러: TTIR → TTGIR → PTX → SASS

Inductor 캐시 디렉터리(`/tmp/ind_cache_fresh/triton/0/<hash>/`)에 forward 커널의 전 단계 산출물이 남아 있었다.

| 파일 | 단계 | 줄 수 |
|---|---|---|
| `.source` | Inductor가 생성한 Python 소스 (3-2) | |
| `.ttir` | Triton IR (MLIR `tt` 방언). 하드웨어 독립 | 74 |
| `.ttgir` | Triton GPU IR. 스레드 배치 레이아웃이 붙음 | 75 |
| `.llir` | LLVM IR | |
| `.ptx` | NVIDIA PTX 가상 어셈블리 (`sm_80`) | 326 |
| `.cubin` | 실제 바이너리. `cuobjdump -sass`로 SASS 확인 | 명령 56개 |

### 4-1. TTIR (하드웨어 독립)

```mlir
tt.func public @triton_poi_fused_add_relu_threshold_backward_0(
    %in_out_ptr0: !tt.ptr<f32>, %in_ptr0: !tt.ptr<f32>, %out_ptr0: !tt.ptr<i1>, %xnumel: i32) {
  %xoffset   = tt.get_program_id x : i32
  %xoffset_0 = arith.muli %xoffset, %c64_i32 : i32
  %xindex    = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
  %xindex_2  = arith.addi (splat %xoffset_0), %xindex : tensor<64xi32>
  %xmask_3   = arith.cmpi slt, %xindex_2, dense<64> : tensor<64xi32>
  %x0_4      = arith.remsi %xindex_2, dense<4> : tensor<64xi32>
  %tmp0_6 = tt.load (tt.addptr %in_out_ptr0, %xindex_2), %xmask_3 : tensor<64x!tt.ptr<f32>>
  %tmp1_8 = tt.load (tt.addptr %in_ptr0,     %x0_4),     %xmask_3 evictionPolicy = evict_last
  %tmp2 = arith.addf %tmp0_6, %tmp1_8 : tensor<64xf32>
  %mask = arith.cmpf ogt, dense<0.0>, %tmp2
  %tmp4 = arith.select %mask, dense<0.0>, %tmp2 : tensor<64xf32>      # maximum(0, x)
  %tmp6 = arith.cmpf ole, %tmp4, dense<0.0>
  tt.store %tmp0_5, %tmp4, %xmask_3
  tt.store (bitcast to i8 ptr), (extui %tmp6 to i8), %xmask_3
  tt.return
}
```

Python 커널이 거의 1:1로 MLIR 연산이 됐다. 값은 아직 `tensor<64xf32>`, 즉 **블록 전체가 하나의 값**이고 스레드 개념이 없다.

### 4-2. TTGIR (GPU 레이아웃 결정)

```mlir
#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32} {
  ...
  %xindex = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
  %tmp0_6 = tt.load %tmp0_5, %xmask : tensor<64x!tt.ptr<f32>, #blocked>
  ...
```

여기서 **컴파일러가 스레드 배치를 결정**했다. `#blocked` 레이아웃은 "원소 64개를 warp 1개(스레드 32개)가 스레드당 2개씩 맡는다"는 뜻이다. `ttg.target = "cuda:80"`가 A100에 특수화됐음을 보여준다. 레이아웃 자체(`#blocked`)는 T4와 같다. 64원소 1-warp 커널에서는 아키텍처가 달라도 스레드 배치를 바꿀 이유가 없기 때문이다. 이 결정이 아래 PTX의 `ld.global.v2`(스레드당 float 2개를 한 번에 로드)로 이어진다.

### 4-3. PTX (sm_80)

```
.version 8.7
.target sm_80
.address_size 64

.visible .entry triton_poi_fused_add_relu_threshold_backward_0(
	.param .u64 .ptr .global .align 1 ..._param_0,     // in_out_ptr0
	.param .u64 .ptr .global .align 1 ..._param_1,     // in_ptr0 (bias)
	.param .u64 .ptr .global .align 1 ..._param_2,     // out_ptr0 (mask)
	.param .u32 ..._param_3                            // xnumel
)
.reqntid 32                                            // 블록당 스레드 32개
{
	.reg .pred 	%p<6>;
	.reg .b16 	%rs<5>;
	.reg .b32 	%r<20>;
	.reg .b64 	%rd<8>;

	mov.u32 	%r7, %ctaid.x;                          // program_id(0)
	shl.b32 	%r8, %r7, 6;                            // * 64 (XBLOCK)
	mov.u32 	%r9, %tid.x;                            // 스레드 인덱스
	shl.b32 	%r10, %r9, 1;                           // * 2 (sizePerThread)
	and.b32 	%r11, %r10, 62;
	or.b32  	%r12, %r11, %r8;                        // xindex
	setp.lt.s32 	%p1, %r12, 64;                      // xmask
	... (xindex % 4 계산) ...
	mad.wide.s32 	%rd1, %r12, 4, %rd4;                // in_out_ptr0 + xindex*4
	@%p1 ld.global.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];   // mm 결과 2개 로드 (마스크 predicated)
	mad.wide.s32 	%rd2, %r17, 4, %rd6;                // in_ptr0 + (xindex%4)*4
	createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;              // L2 캐시 정책 디스크립터 생성 (Ampere+)
	@%p1 ld.global.L1::evict_last.L2::cache_hint.v2.b32 { %r3, %r4 }, [ %rd2 + 0 ], %rd3;   // bias 로드 + L1/L2 힌트
	add.f32 	%r18, %r1, %r3;                         // + b
	add.f32 	%r19, %r2, %r4;
	setp.lt.f32 	%p2, %r19, 0f00000000;
	setp.lt.f32 	%p3, %r18, 0f00000000;
	selp.f32 	%r5, 0f00000000, %r18, %p3;             // relu = select(x<0, 0, x)
	selp.f32 	%r6, 0f00000000, %r19, %p2;
	setp.le.f32 	%p4, %r6, 0f00000000;               // le
	setp.le.f32 	%p5, %r5, 0f00000000;
	@%p1 st.global.v2.b32 [ %rd1 + 0 ], { %r5, %r6 };   // relu 결과 in-place 저장
	selp.b16 	%rs2, 1, 0, %p5;
	selp.b16 	%rs3, 1, 0, %p4;
	shl.b16 	%rs4, %rs3, 8;
	or.b16  	%rs1, %rs2, %rs4;                       // bool 2개를 16비트로 패킹
	@%p1 st.global.b16 [ %rd3 + 0 ], { %rs1 };          // 마스크 저장
	ret;
}
```

읽을 점:

- `%ctaid.x`가 `tl.program_id(0)`, `%tid.x`가 TTGIR가 정한 스레드 배치다. Python 소스에는 없던 `%tid.x`가 여기서 처음 등장한다.
- `xmask`는 **predicate 레지스터** `%p1`이 되어 `@%p1 ld.global` 형태로 로드·스토어를 조건부 실행한다. 분기가 아니라 predication이다.
- `evict_last`가 **두 단계 캐시 힌트**로 내려갔다. `createpolicy.fractional.L2::evict_last`로 L2 정책 디스크립터를 만들고, `ld.global.L1::evict_last.L2::cache_hint`가 그것을 인자로 받는다. 이 `L2::cache_hint` 기능은 Ampere(sm_80)부터 있어서 T4(sm_75)에서는 `L1::evict_last`만 나왔다. 같은 Triton 소스 한 줄이 아키텍처에 따라 다른 명령이 되는 지점이다.
- `.v2.b32`: 스레드당 float 2개를 한 번에 읽고 쓴다. TTGIR의 `sizePerThread = [2]`가 이것이다.
- `relu`는 `maximum`이 아니라 `setp + selp`(비교 + 선택)로 구현됐다.

### 4-4. SASS (실제 A100 명령, 56개)

```
S2R R0, SR_TID.X                       // 스레드 ID
S2R R3, SR_CTAID.X                     // 블록 ID
IMAD R0, R3, 0x40, R0                  // xindex = ctaid*64 + tid*2 (앞 명령과 합쳐서)
ISETP.GE.AND P0, PT, R0, 0x40, PT      // xmask (반전: >=64면 건너뜀)
LEA R2, P1, R0, c[0x0][0x160], 0x2     // in_out_ptr0 주소 계산 (커널 파라미터는 constant bank c[0x0]에)
@!P0 IMAD.MOV.U32 R7, RZ, RZ, 0x14f00000   // L2 캐시 정책 디스크립터 상수
@!P0 R2UR UR6, R6                      // 디스크립터를 uniform 레지스터로
@!P0 LDG.E.64 R8, [R2.64]              // 64비트(float 2개) 글로벌 로드
@!P0 LDG.E.EL.64 R10, desc[UR6][R4.64] // bias 로드 (.EL = evict_last, desc[] = L2 힌트)
FADD R8, R10, R8                       // + b
FADD R6, R11, R9
FSEL R10, R8, RZ, P2                   // relu (select)
FSEL R11, R6, RZ, P1
FSETP.GTU.AND P3, PT, R10, RZ, PT      // le
SEL R8, RZ, 0x1, P3
@!P0 STG.E.64 [R2.64], R10             // relu 결과 저장
STG.E.U16 [R6.64], R5                  // 마스크 저장
EXIT
```

이것이 A100의 SM에서 실제로 실행되는 명령이다. Python 커널 12줄이 SASS 56개가 됐고(패딩 `NOP` 13개 제외 43개), 그중 산술은 `FADD` 2개, `FSEL` 2개, `FSETP` 4개뿐이며 나머지는 주소 계산, 캐시 정책 디스크립터 설정, 로드·스토어다. T4의 40개보다 많은 것은 `R2UR`, `desc[UR6]` 같은 L2 힌트 관련 명령이 추가됐기 때문이다. [⑤](./05-kernel-to-chip.md)에서 말한 "elementwise 커널은 memory-bound"가 명령 수준에서도 보인다.

## 5. 실행과 비동기 디스패치

```
type(y) = torch.Tensor, shape (16, 4), device cuda:0
y.grad_fn = <torch.autograd.function.CompiledFunctionBackward>
y.sum().backward() → w.grad.shape = (8, 4), b.grad = [16., 16., 16., 16.]
```

`b.grad`가 16인 것은 마스크 `le`가 전부 False(입력이 모두 양수)라서 `where`가 tangent(전부 1)를 그대로 통과시키고 16행을 더했기 때문이다. backward 그래프가 실제로 실행됐다는 확인이다.

비동기 디스패치 측정 (4096×4096 f32 행렬곱, 웜업 후):

| 측정 | 시간 |
|---|---|
| `z = x @ x` 반환까지 | 176 µs |
| `torch.cuda.synchronize()`까지 | 9.47 ms |
| `z.sum().item()`까지 (암묵적 동기화) | 9.70 ms |

Python 함수는 176 µs 만에 돌아오지만 실제 계산은 9.5 ms 걸린다. [④](./04-dispatch-to-kernel.md)의 JAX 예제와 같은 현상이 PyTorch에서도 그대로다. 웜업 없이 첫 호출을 재면 디스패치가 5.3 ms로 나오는데, 이는 cuBLAS 핸들 초기화가 포함된 값이다.

**같은 matmul을 어느 유닛이 계산하느냐** (4096×4096, 2·4096³ ≈ 137 GFLOP):

| 설정 | 완료 시간 | 처리량 | 연산 유닛 |
|---|---|---|---|
| f32, `allow_tf32=False` | 8.38 ms | 16.4 TFLOP/s | CUDA core (fp32, 피크 19.5) |
| f32, `allow_tf32=True` | 1.21 ms | 114 TFLOP/s | Tensor Core, TF32 (피크 156) |
| bf16 | 0.63 ms | 218 TFLOP/s | Tensor Core, bf16 (피크 312) |

`allow_tf32` 플래그 하나로 **7배**, dtype을 bf16으로 바꾸면 **13배** 빨라진다. Python 코드는 `x @ x` 그대로이고 cuBLAS가 다른 커널을 고른 것뿐이다. [⑤](./05-kernel-to-chip.md)에서 "Tensor Core가 FLOPs의 대부분"이라고 한 것이 이 표다. 1절의 guard에 `allow_tf32`가 들어 있는 이유도 여기서 분명해진다.

## 6. 칩 위에 놓아 보기

| 커널 실행 정보 | A100에서의 의미 |
|---|---|
| `xnumel = 64`, `XBLOCK = 64`, `Grid1D` | 그리드 = 블록 1개. **SM 108개 중 1개만** 쓴다. |
| `num-warps = 1`, `.reqntid 32` | 블록 = warp 1개 = 스레드 32개. 스레드당 원소 2개. |
| `LDG.E.64` × 2, `STG.E.64` + `STG.E.U16` | 원소당 읽기 8바이트 + 쓰기 5바이트. FLOPs는 원소당 3(add, max, cmp). 40 MB L2에 전부 들어가므로 두 번째 실행부터는 HBM에 가지도 않는다. |
| `extern_kernels.mm` | cuBLAS가 별도 커널로 실행. 입력이 f32이고 guard의 `allow_tf32=false`이므로 CUDA core(fp32 유닛)로 계산된다. TF32를 켜면 같은 그래프가 재컴파일되고 Tensor Core 커널이 선택된다. |

이 크기에서는 커널 런치 오버헤드(수 µs)가 계산 시간(수십 ns)을 압도한다. Inductor가 3개 연산을 커널 1개로 퓨전한 것이 여기서는 "메모리 절약"이 아니라 "런치 2번 절약"으로 효과가 난다.

## 7. CPU · T4 결과와의 차이 요약

같은 스크립트를 CPU([example-relu-linear.md](./example-relu-linear.md))와 Tesla T4에서도 돌렸다.

| | CPU (Inductor C++) | T4 (Turing, sm_75) | **A100 (Ampere, sm_80)** |
|---|---|---|---|
| matmul | `addmm` (bias 흡수) | `mm` (bias는 Triton 커널로) | `mm` (동일) |
| forward 퓨전 커널 | `relu + le` | `add + relu + le` | `add + relu + le` (동일) |
| Dynamo / AOT / Inductor Python 출력 | — | — | **T4와 완전히 동일** (`DeviceProperties`, `backend_hash`만 다름) |
| TTIR / TTGIR | — | `cuda:75`, `#blocked` sizePerThread=[2] | `cuda:80`, 같은 `#blocked` |
| PTX 캐시 힌트 | — | `ld.global.L1::evict_last` | `createpolicy ... L2::evict_last` + `ld.global.L1::evict_last.L2::cache_hint` |
| SASS 명령 수 | — | 40 | 56 (L2 디스크립터 `R2UR`, `desc[]` 추가) |
| SM 수 / L2 | — | 40 / 4 MB | 108 / 40 MB |
| 4096² f32 matmul | — | 32.1 ms | 9.47 ms (CUDA core), TF32 1.21 ms, bf16 0.63 ms |
| Tensor Core로 f32 가속 | — | 불가 (fp16/int8만) | TF32로 가능 (`allow_tf32=True`) |
| 병렬 단위 | OpenMP 스레드 | 블록 1 × warp 1 × 스레드 32 | 동일 |
| 런치 / 동기화 | 함수 호출 / 불필요 | `.run(..., stream=)` / `synchronize()` | 동일 |

정리하면 **프레임워크와 Inductor 수준(①~③)은 GPU 세대에 무관**하고, 차이는 전부 **Triton 컴파일러 이하(④)와 실제 실행 속도(⑤)** 에서 난다. TTIR까지도 같고, TTGIR의 `ttg.target`에서 처음 갈라지며, PTX에서 아키텍처 전용 명령이 나타난다.

## 8. 재현

```bash
uv tool install google-colab-cli
colab new -s a100 --gpu A100
colab exec -s a100 -f gpu_demo.py --timeout 900     # TORCH_LOGS를 subprocess 환경변수로 넘기는 스크립트
colab exec -s a100 -f a100_extra.py --timeout 600   # 디바이스 정보, 비동기 타이밍, f32/TF32/bf16 Tensor Core 비교
colab exec -s a100 -f gpu_chain.py --timeout 300    # /tmp/<inductor cache>/triton/0/*/ 의 .ttir .ttgir .ptx .cubin 덤프
colab stop -s a100
```

Inductor 캐시 위치는 `TORCHINDUCTOR_CACHE_DIR`로 고정하면 찾기 쉽다. SASS는 VM에 있는 `cuobjdump -sass <file>.cubin`으로 얻었다.
