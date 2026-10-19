"""Run paired JAX vs native CUDA/Pallas benchmarks inside one Colab runtime."""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import ctypes
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import time

import numpy as np
import jax
import jax.numpy as jnp

target = os.environ['WEEK5_TARGET']
root = Path('/content/week5-native')
root.mkdir(exist_ok=True)
src = Path('/content/week5-native-src')
sys.path.insert(0, str(src))
data_path = Path('/content/week5-inputs.npz')
data = np.load(data_path)
is_tpu = target.startswith('v')
assert jax.default_backend() == ('tpu' if is_tpu else 'gpu'), jax.devices()
assert len(jax.devices()) == 1
assert jax.__version__ == '0.11.1', jax.__version__
native = 'pallas' if is_tpu else 'cuda'
meta = dict(
    target=target, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    python=platform.python_version(), numpy=np.__version__,
    device_kind=jax.devices()[0].device_kind, devices=[str(d) for d in jax.devices()],
    jax=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
    input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
    operation='ReLU(XW+b)', input_dtype='bfloat16', accumulation='float32',
    output_dtype='float32', warmup=50, repetitions_per_seed=1000,
    seeds=list(range(5)), concurrency=1, atol=1e-4, rtol=1e-4,
    timing='perf_counter_ns: host call through device completion; resident inputs; no copies',
    cache='same resident inputs repeated, no cache flush',
    tuning='seed 0, 100 calls after 20 warmups; final measurements separately; alternate order by seed',
    native=native, cuda_graph=False,
    sources={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in src.iterdir() if p.is_file()},
)

@jax.jit
def baseline(x, w, b):
    acc = jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())),
                             precision=jax.lax.Precision.HIGHEST,
                             preferred_element_type=jnp.float32)
    return jnp.maximum(acc + b, jnp.float32(0))

if is_tpu:
    from pallas_relu import build_pallas
    meta['libtpu'] = importlib.metadata.version('libtpu')
else:
    import torch
    assert torch.cuda.device_count() == 1
    expected = 'A100' if target == 'A100' else 'RTX PRO 6000'
    assert expected in torch.cuda.get_device_name(0)
    torch.set_grad_enabled(False)
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
    if kind == 'cuda':
        args = tuple(torch.from_numpy(a).to('cuda', dtype=dt) for a, dt in
                     zip((x,w,b), (torch.bfloat16,torch.bfloat16,torch.float32)))
        torch.cuda.synchronize()
    else:
        args = tuple(jax.device_put(a.astype(dt)).block_until_ready() for a, dt in
                     zip((x,w,b), (jnp.bfloat16,jnp.bfloat16,jnp.float32)))
    ref = np.maximum(b, 0)[None,:].repeat(m,axis=0) if edge else data[f'{seed}_ref'][:m]
    if perturb:
        # Recompute before ReLU, since the saved reference has already clipped negatives.
        ref = np.maximum(x.astype(np.float64) @ w.astype(np.float64) + b, 0)
    return args, ref

def prepare_fn(kind, m, config, args, save=False):
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
        graph = baseline if kind == 'jax' else build_pallas(m, config)
        lowered = graph.lower(*args)
        compiled = lowered.compile()
        if save:
            (root/f'{kind}-m{m}-stablehlo.txt').write_text(lowered.as_text())
            (root/f'{kind}-m{m}-optimized.txt').write_text(compiled.as_text())
        fn = lambda: compiled(*args)
        sync = lambda y: y.block_until_ready()
        host = lambda y: np.asarray(y)
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
tuning, selected = [], {}
for m in [1,128]:
    args, ref = prepare_inputs(m, 0, native)
    configs = [128,256,512,1024] if is_tpu else ([4,8,16] if m == 1 else [4,8])
    for config in configs:
        item = dict(m=m, config=config)
        try:
            fn, sync, host, prep = prepare_fn(native, m, config, args)
            item.update(validate(host(fn()), ref))
            if item['passed']:
                times, elapsed = measure(fn, sync, 100, 20)
                item.update(p50_ms=float(np.median(times)), prepare_s=prep)
        except Exception as e:
            item.update(passed=False, error=str(e))
        tuning.append(item)
        print('TUNE', json.dumps(item), flush=True)
    valid = [r for r in tuning if r['m'] == m and r['passed']]
    if not valid:
        (root/'tuning-failure.json').write_text(json.dumps(tuning,indent=2))
        raise RuntimeError(f'No valid {native} kernel for M={m}')
    selected[m] = min(valid, key=lambda r:r['p50_ms'])['config']

rows, outputs, edges = [], {}, []
for m in [1,128]:
    for seed in range(5):
        order = ['jax',native] if seed % 2 == 0 else [native,'jax']
        for kind in order:
            args, ref = prepare_inputs(m, seed, kind)
            fn, sync, host, prep = prepare_fn(kind,m,selected[m],args,save=seed==0)
            sync(fn())
            check = validate(host(fn()), ref)
            assert check['passed'], (kind,m,seed,check)
            times, elapsed = measure(fn,sync,1000,50)
            actual = host(fn())
            check = validate(actual,ref)
            assert check['passed'], (kind,m,seed,check)
            outputs[f'{kind}_{seed}_m{m}'] = actual
            row = dict(implementation=kind,m=m,seed=seed,selected_config=selected[m] if kind != 'jax' else None,
                       prepare_and_first_call_s=prep,times_ms=times,elapsed_s=elapsed,
                       p50_ms=float(np.median(times)),p95_ms=float(np.percentile(times,95)),
                       mean_ms=float(np.mean(times)),samples_per_s=1000*m/elapsed,**check)
            rows.append(row)
            print('RESULT', json.dumps({k:v for k,v in row.items() if k!='times_ms'}),flush=True)
    for kind in ['jax',native]:
        for name in ['zero_bias_boundary','changed_bias']:
            args, ref = prepare_inputs(m,4,kind,edge=name=='zero_bias_boundary',perturb=name=='changed_bias')
            fn,sync,host,_ = prepare_fn(kind,m,selected[m],args)
            sync(fn())
            check = validate(host(fn()),ref)
            edges.append(dict(implementation=kind,m=m,test=name,**check))
            assert check['passed'], edges[-1]

result = dict(meta=meta,selected=selected,tuning=tuning,rows=rows,edge_tests=edges)
(root/'result.json').write_text(json.dumps(result,indent=2)+'\n')
np.savez_compressed(root/'outputs.npz',**outputs)
with tarfile.open('/content/week5-native-results.tar.gz','w:gz') as tar:
    for p in root.iterdir():
        if p.suffix != '.so': tar.add(p,arcname='results/'+p.name)
print('COMPLETE',target,flush=True)
