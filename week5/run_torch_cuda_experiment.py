"""Paired A100 PyTorch eager vs CUDA rerun, preserving the earlier JAX run."""
import os
import ctypes
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import time

import numpy as np
import torch

target = os.environ['WEEK5_TARGET']
root = Path('/content/week5-torch-cuda')
root.mkdir(exist_ok=True)
src = Path('/content/week5-torch-cuda-src')
data_path = Path('/content/week5-inputs.npz')
data = np.load(data_path)
assert target == 'A100'
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = False
native = 'cuda'
meta = dict(
    target=target, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=platform.python_version(), numpy=np.__version__,
    device_kind=torch.cuda.get_device_name(0), devices=[str(torch.cuda.get_device_properties(0))],
    baseline='torch', baseline_implementation='torch.mm(out_dtype=float32).add_(b).relu_(), eager',
    allow_bf16_reduced_precision_reduction=False, allow_tf32=False,
    input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
    operation='ReLU(XW+b)', input_dtype='bfloat16', accumulation='float32',
    output_dtype='float32', warmup=50, repetitions_per_seed=1000,
    seeds=list(range(5)), concurrency=1, atol=1e-4, rtol=1e-4,
    timing='perf_counter_ns: host call through device completion; resident inputs; no copies',
    cache='same resident inputs repeated, no cache flush',
    selection='fixed prior CUDA selection: M=1 16 warps, M=128 8 warps; alternate order by seed',
    native=native, cuda_graph=False,
    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()},
)

meta.update(torch=torch.__version__, cuda=torch.version.cuda,
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
            nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))
cc = torch.cuda.get_device_capability()
cmd = ['nvcc', '-O3', '-std=c++17', '--shared', '-Xcompiler', '-fPIC',
       f'-arch=sm_{cc[0]}{cc[1]}', str(src/'native_relu.cu'), '-o', str(root/'native_relu.so')]
start = time.perf_counter()
proc = subprocess.run(cmd, text=True, capture_output=True)
(root/'compile.log').write_text(proc.stdout + proc.stderr)
if proc.returncode: raise RuntimeError(proc.stderr)
meta['cuda_build_s'] = time.perf_counter() - start
meta['cuda_build_command'] = cmd
lib = ctypes.CDLL(str(root/'native_relu.so'))
lib.launch.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
lib.launch.restype = ctypes.c_int


def prepare_inputs(m, seed, kind, edge=False, perturb=False):
    x, w, b = data[f'{seed}_x'][:m].copy(), data[f'{seed}_w'].copy(), data[f'{seed}_b'].copy()
    if edge:
        x[:] = 0
        b = np.resize(np.array([-.1, 0, .1, -1e-7, 1e-7], np.float32), 1024)
    if perturb:
        b = b + np.float32(0.125)
    args = tuple(torch.from_numpy(a).to('cuda', dtype=dt) for a, dt in
                 zip((x,w,b), (torch.bfloat16,torch.bfloat16,torch.float32)))
    torch.cuda.synchronize()
    ref = np.maximum(b, 0)[None,:].repeat(m,axis=0) if edge else data[f'{seed}_ref'][:m]
    if perturb:
        # Recompute before ReLU, since the saved reference has already clipped negatives.
        ref = np.maximum(x.astype(np.float64) @ w.astype(np.float64) + b, 0)
    return args, ref

def prepare_fn(kind, m, config, args):
    start = time.perf_counter()
    if kind == 'cuda':
        out = torch.empty((m,1024), device='cuda', dtype=torch.float32)
        ptrs = [ctypes.c_void_p(a.data_ptr()) for a in (*args,out)]
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        def fn():
            status = lib.launch(*ptrs, m, config, stream)
            if status: raise RuntimeError(f'CUDA launch status {status}')
            return out
        sync = lambda y: torch.cuda.synchronize()
        host = lambda y: y.cpu().numpy()
    else:
        assert kind == 'torch'
        def fn():
            x, w, b = args
            return torch.mm(x, w, out_dtype=torch.float32).add_(b).relu_()
        sync = lambda y: torch.cuda.synchronize()
        host = lambda y: y.cpu().numpy()
    y = fn()
    sync(y)
    return fn, sync, host, time.perf_counter()-start

def validate(y, ref):
    assert y.dtype == np.float32, y.dtype
    err = np.abs(y.astype(np.float64)-ref)
    failed = (~np.isfinite(y)) | (err > 1e-4 + 1e-4*np.abs(ref))
    return dict(passed=not bool(failed.any()), failed_elements=int(failed.sum()), max_abs_error=float(err.max()))

def measure(fn, sync, count, warmup):
    for _ in range(warmup): sync(fn())
    times = []
    start_total = time.perf_counter_ns()
    for _ in range(count):
        start = time.perf_counter_ns()
        sync(fn())
        times.append((time.perf_counter_ns()-start)/1e6)
    return times, (time.perf_counter_ns()-start_total)/1e9

print('DEVICE', json.dumps(meta), flush=True)
selected = {1:16, 128:8}  # Same CUDA configurations as the earlier run.

rows, outputs, edges = [], {}, []
for m in [1,128]:
    for seed in range(5):
        order = ['torch',native] if seed % 2 == 0 else [native,'torch']
        for kind in order:
            args, ref = prepare_inputs(m, seed, kind)
            fn, sync, host, prep = prepare_fn(kind,m,selected[m],args)
            sync(fn())
            check = validate(host(fn()), ref)
            assert check['passed'], (kind,m,seed,check)
            times, elapsed = measure(fn,sync,1000,50)
            actual = host(fn())
            check = validate(actual,ref)
            assert check['passed'], (kind,m,seed,check)
            outputs[f'{kind}_{seed}_m{m}'] = actual
            row = dict(implementation=kind,m=m,seed=seed,selected_config=selected[m] if kind != 'torch' else None,
                       prepare_and_first_call_s=prep,times_ms=times,elapsed_s=elapsed,
                       p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                       mean_ms=float(np.mean(times)),samples_per_s=1000*m/elapsed,**check)
            rows.append(row)
            print('RESULT', json.dumps({k:v for k,v in row.items() if k!='times_ms'}),flush=True)
    for kind in ['torch',native]:
        for name in ['zero_bias_boundary','changed_bias']:
            args, ref = prepare_inputs(m,4,kind,edge=name=='zero_bias_boundary',perturb=name=='changed_bias')
            fn,sync,host,_ = prepare_fn(kind,m,selected[m],args)
            sync(fn())
            check = validate(host(fn()),ref)
            edges.append(dict(implementation=kind,m=m,test=name,**check))
            assert check['passed'], edges[-1]

result = dict(meta=meta,selected=selected,rows=rows,edge_tests=edges)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(root/'outputs.npz',**outputs)
with tarfile.open('/content/week5-torch-cuda-results.tar.gz','w:gz') as tar:
    for p in root.iterdir():
        if p.suffix != '.so': tar.add(p,arcname='results/'+p.name)
print('COMPLETE',target,flush=True)
