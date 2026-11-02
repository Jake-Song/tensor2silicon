import os

INTERPRET = False  # CPU 해석 모드: True로 변경 시 런타임 재시작 후 처음부터 실행
os.environ["TRITON_INTERPRET"] = "1" if INTERPRET else "0"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import json
import platform
import sys
import torch
import triton
import triton.language as tl
import triton.testing

if not INTERPRET and not torch.cuda.is_available():
    raise RuntimeError("GPU 런타임을 선택하거나 재시작 후 INTERPRET=True로 실행하세요.")

device = torch.device("cpu" if INTERPRET else "cuda")
torch.manual_seed(0)

cc = None
smem_limit_bytes = None
if not INTERPRET:
    props = torch.cuda.get_device_properties(device)
    cc = f"{props.major}.{props.minor}"
    smem_limit_bytes = getattr(props, "shared_memory_per_block_optin", props.shared_memory_per_block)

env_info = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "torch": torch.__version__,
    "triton": triton.__version__,
    "cuda_build": torch.version.cuda,
    "gpu": None if INTERPRET else torch.cuda.get_device_name(device),
    "compute_capability": cc,
    "shared_memory_per_block_optin_kib": None if smem_limit_bytes is None else smem_limit_bytes / 1024,
    "mode": "CPU interpreter" if INTERPRET else "GPU",
}
print(json.dumps(env_info, ensure_ascii=False, indent=2))

@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def matmul_kernel_single(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_remaining = K - k * BLOCK_SIZE_K
        a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def get_cuda_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    ]


matmul_kernel_autotuned = triton.autotune(
    configs=get_cuda_autotune_config(),
    key=["M", "N", "K"],
)(matmul_kernel_single)


def matmul_single(
    a, b, activation="",
    block_m=64, block_n=32, block_k=32, group_m=8,
    num_stages=5, num_warps=2,
    return_kernel=False,
):
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.float16
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    compiled = matmul_kernel_single[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        ACTIVATION=activation,
        num_stages=num_stages,
        num_warps=num_warps,
    )
    if return_kernel:
        return c, compiled
    return c


def matmul_autotuned(a, b, activation=""):
    assert a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0]
    assert a.dtype == b.dtype == torch.float16
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
    matmul_kernel_autotuned[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c

ir_summary = {}
if INTERPRET:
    print("CPU 해석 모드: 컴파일 파이프라인을 우회하므로 .metadata 및 .asm 확인을 생략합니다.")
else:
    a_ir = torch.randn((512, 512), device=device, dtype=torch.float16)
    b_ir = torch.randn((512, 512), device=device, dtype=torch.float16)
    _, compiled_kernel = matmul_single(
        a_ir, b_ir,
        block_m=64, block_n=32, block_k=32, group_m=8,
        num_stages=5, num_warps=2,
        return_kernel=True,
    )

    meta = compiled_kernel.metadata
    ir_summary = {
        "shared_bytes": getattr(meta, "shared", None),
        "num_warps": getattr(meta, "num_warps", None),
        "num_stages": getattr(meta, "num_stages", None),
        "n_regs": getattr(compiled_kernel, "n_regs", None),
        "n_spills": getattr(compiled_kernel, "n_spills", None),
        "asm_keys": list(compiled_kernel.asm.keys()),
    }
    print("=== CompiledKernel Metadata ===")
    print(json.dumps(ir_summary, ensure_ascii=False, indent=2))

    def grep_lines(text, keywords, max_lines=12):
        matched = [line.strip() for line in text.splitlines() if any(k in line for k in keywords)]
        return matched[:max_lines]

    for stage, kw in [
        ("ttir", ["tt.get_program_id", "tt.load", "scf.for", "tt.dot", "tt.store"]),
        ("ttgir", ["#ttg.", "local_alloc", "async_copy", "tt.dot", "convert_layout"]),
        ("ptx", [".target", "cp.async", "ldmatrix", "mma.sync"]),
    ]:
        if stage in compiled_kernel.asm:
            print(f"\n=== .asm['{stage}'] 핵심 줄 발췌 ===")
            for ln in grep_lines(compiled_kernel.asm[stage], kw):
                print(" ", ln)
from pathlib import Path
import subprocess, shutil, hashlib
out = Path('/content/week7_compile_artifacts')
out.mkdir(exist_ok=True)
for name, value in compiled_kernel.asm.items():
    path = out / ('kernel.' + name)
    if isinstance(value, bytes): path.write_bytes(value)
    else: path.write_text(str(value))
(out / 'environment.json').write_text(json.dumps(env_info, indent=2))
(out / 'metadata.json').write_text(json.dumps(ir_summary, indent=2))
for tool, args in [('nvdisasm', []), ('cuobjdump', ['--dump-sass'])]:
    exe=shutil.which(tool)
    if exe:
        r=subprocess.run([exe, *args, str(out/'kernel.cubin')], capture_output=True,text=True)
        if r.returncode == 0:
            (out/'kernel.sass').write_text(r.stdout)
            break
(out/'manifest.json').write_text(json.dumps({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()},indent=2))
shutil.make_archive('/content/week7_compile_artifacts','zip',out)
print('ARTIFACT_ZIP=/content/week7_compile_artifacts.zip')
