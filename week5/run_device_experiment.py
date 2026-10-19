"""Colab payload: TPU uses JAX; GPU uses eager PyTorch/CUDA. No installation."""
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np

backend = os.environ['WEEK5_BACKEND']
target = os.environ['WEEK5_TARGET']
root = Path('/content/week5-06')
root.mkdir(exist_ok=True)
data_path = Path('/content/week5-inputs.npz')
data = np.load(data_path)
meta = dict(target=target, backend=backend, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            python=platform.python_version(), numpy=np.__version__, input_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
            operation='ReLU(XW+b)', contract='A', input_dtype='bfloat16', accumulation='float32', output_dtype='float32',
            warmup=50, repetitions_per_seed=1000, seeds=list(range(5)), concurrency=1,
            timing='perf_counter_ns: host invocation through completion, resident X/W/b, no output copy',
            cache='repeated same resident inputs; no cache flush', atol=1e-4, rtol=1e-4)
if backend == 'jax':
    import jax
    import jax.numpy as jnp
    assert jax.default_backend() == 'tpu', jax.devices()
    devices = jax.devices()
    assert len(devices) == 1, devices
    dev = devices[0]
    meta.update(framework_version=jax.__version__, jaxlib=importlib.metadata.version('jaxlib'),
                libtpu=importlib.metadata.version('libtpu'), devices=[str(d) for d in devices], device_kind=dev.device_kind)
    def put(x, dtype):
        return jax.device_put(x.astype(dtype), dev).block_until_ready()
    bf = jnp.bfloat16
    fp = jnp.float32
    @jax.jit
    def graph(x, w, b):
        y = jax.lax.dot_general(x, w, (((1,), (0,)), ((), ())),
                               precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)
        return jnp.maximum(y + b, jnp.float32(0))
    def sync(y):
        y.block_until_ready()
    def host(y):
        return np.asarray(y)
else:
    import torch
    assert torch.cuda.is_available()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_tf32 = False
    meta.update(framework_version=torch.__version__, cuda=torch.version.cuda,
                device_kind=torch.cuda.get_device_name(0), devices=[str(torch.cuda.get_device_properties(0))],
                allow_bf16_reduced_precision_reduction=False, implementation='torch.mm(out_dtype=float32), FP32 add_, relu_ (eager)')
    meta['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
    bf = torch.bfloat16
    fp = torch.float32
    def put(x, dtype):
        y = torch.from_numpy(np.array(x, copy=True)).to(device='cuda', dtype=dtype)
        torch.cuda.synchronize()
        return y
    def graph(x, w, b):
        return torch.mm(x, w, out_dtype=torch.float32).add_(b).relu_()
    def sync(y):
        torch.cuda.synchronize()
    def host(y):
        return y.cpu().numpy()

print('DEVICE', json.dumps(meta), flush=True)
rows = []
outputs = {}
for m in [1, 128]:
    for seed in range(5):
        x = put(data[f'{seed}_x'][:m], bf)
        w = put(data[f'{seed}_w'], bf)
        b = put(data[f'{seed}_b'], fp)
        start = time.perf_counter()
        if backend == 'jax':
            lowered = graph.lower(x, w, b)
            fn = lowered.compile()
            (root / f'm{m}-stablehlo.txt').write_text(lowered.as_text())
            (root / f'm{m}-optimized.txt').write_text(fn.as_text())
        else:
            fn = graph
        y = fn(x, w, b)
        sync(y)
        prepare = time.perf_counter() - start
        actual = host(y)
        assert actual.dtype == np.float32
        ref = data[f'{seed}_ref'][:m]
        error = np.abs(actual.astype(np.float64) - ref)
        failed = (~np.isfinite(actual)) | (error > 1e-4 + 1e-4 * np.abs(ref))
        outputs[f'{seed}_m{m}'] = actual
        for _ in range(50):
            sync(fn(x, w, b))
        times = []
        start_total = time.perf_counter_ns()
        for _ in range(1000):
            start = time.perf_counter_ns()
            y = fn(x, w, b)
            sync(y)
            times.append((time.perf_counter_ns() - start) / 1e6)
        elapsed = (time.perf_counter_ns() - start_total) / 1e9
        row = dict(m=m, seed=seed, prepare_and_first_call_s=prepare, times_ms=times,
                   p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)),
                   mean_ms=float(np.mean(times)), elapsed_s=elapsed, samples_per_s=1000*m/elapsed,
                   max_abs_error=float(error.max()), failed_elements=int(failed.sum()), passed=not bool(failed.any()))
        rows.append(row)
        print('RESULT', json.dumps({k:v for k,v in row.items() if k != 'times_ms'}), flush=True)

# Dedicated boundary test: zero X gives exactly broadcast ReLU(b), both shapes.
edges = []
for m in [1, 128]:
    edge_b = np.resize(np.array([-.1, 0, .1, -1e-7, 1e-7], np.float32), 1024)
    y = graph(put(np.zeros((m,1024), np.float32), bf), put(data['0_w'], bf), put(edge_b, fp))
    sync(y)
    expected = np.broadcast_to(np.maximum(edge_b, 0), (m,1024))
    edges.append(dict(m=m, passed=bool(np.array_equal(host(y), expected))))
result = dict(meta=meta, rows=rows, edge_tests=edges)
(root / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
np.savez_compressed(root / 'outputs.npz', **outputs)
import tarfile
with tarfile.open('/content/week5-results.tar.gz', 'w:gz') as tar:
    tar.add(root, arcname='results')
print('COMPLETE', target, all(r['passed'] for r in rows), edges, flush=True)
