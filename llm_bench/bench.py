"""Reproducible same-device BF16 inference comparison.

Example: python -m llm_bench --backend all --preset colab --out results/run1.json
Compilation, packing, reference checks and profiling are outside steady timing.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time
import traceback

from .metrics import accept, summarize, RMS_FLOOR


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


def map_tree(tree, fn):
    if isinstance(tree, dict):
        return {k: map_tree(v, fn) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(map_tree(v, fn) for v in tree)
    return fn(tree)


def environment():
    import torch
    import triton
    p = torch.cuda.get_device_properties(0)
    sources = {}
    root = Path(__file__).resolve().parents[1]
    for directory in ('llm_bench', 'llm_roofline'):
        for path in sorted((root / directory).glob('*.py')):
            sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        telemetry = subprocess.check_output([
            'nvidia-smi', '--query-gpu=driver_version,pstate,temperature.gpu,power.limit,clocks.sm,clocks.mem,memory.used,utilization.gpu',
            '--format=csv'], text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError) as error:
        telemetry = f'unavailable: {type(error).__name__}'
    return dict(gpu=p.name, memory_gib=p.total_memory / 2**30,
                capability=list(torch.cuda.get_device_capability()), sm_count=p.multi_processor_count,
                torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                python=platform.python_version(), source_sha256=sources,
                nvidia_smi_at_start=telemetry,
                matmul_tf32=False, storage_dtype='bfloat16', accumulation_dtype='float32',
                compilation_cache_policy='runtime persistent caches reused; first call is not guaranteed cold startup')


def validate_tensor(actual, expected, label):
    import torch
    if actual.shape != expected.shape or actual.dtype != torch.bfloat16:
        raise AssertionError(f'{label}: shape/dtype {actual.shape}/{actual.dtype}, expected {expected.shape}/BF16')
    a, e = actual.float(), expected.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(e).all()):
        raise AssertionError(f'{label}: nonfinite values')
    rms = max(e.square().mean().sqrt().item(), RMS_FLOOR)
    d = a - e
    rmse, maximum = d.square().mean().sqrt().item(), d.abs().max().item()
    result = dict(label=label, reference_rms=rms, rmse=rmse, max_abs_error=maximum,
                  nrmse=rmse / rms, normalized_max=maximum / rms)
    result['passed'] = accept(result['nrmse'], result['normalized_max'])
    if not result['passed']:
        raise AssertionError(json.dumps(result))
    return result


def validate_output(actual, expected, valid_length=None):
    if len(actual) != 2 or len(actual[1]) != len(expected[1]):
        raise AssertionError('logits and every KV cache are required')
    checks = [validate_tensor(actual[0], expected[0], 'logits')]
    for i, (pair, ref) in enumerate(zip(actual[1], expected[1])):
        if len(pair) != 2 or len(ref) != 2:
            raise AssertionError(f'layer{i}: both K and V cache tensors are required')
        for name, a, e in zip(('k', 'v'), pair, ref):
            if valid_length is not None:
                a, e = a[:, :, :valid_length], e[:, :, :valid_length]
            checks.append(validate_tensor(a, e, f'layer{i}.{name}'))
    return checks


def reference(cfg, params, tokens, cache=None, position=None):
    """Independent original model in FP32 using the same quantized weights."""
    import torch
    from llm_roofline import spec, torch_llm
    b, t = tokens.shape
    length = t if cache is None else int(position) + 1
    phase = spec.PhaseConfig('prefill' if cache is None else 'decode', b, t, length)
    consts = torch_llm.make_consts(cfg, phase, torch.float32, tokens.device)
    state = None if cache is None else [(k[:, :, :length].float().contiguous().clone(),
                                        v[:, :, :length].float().contiguous().clone()) for k, v in cache]
    return torch_llm.forward(params, tokens, consts, state)


class Captured:
    def __init__(self, fn):
        import torch
        self.fn = fn
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = fn()
        torch.cuda.synchronize()

    def __call__(self):
        self.graph.replay()
        return self.output


def time_fixed(fn, warmup=20, calls=100, rounds=5):
    import torch
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    device_rounds, wall_samples = [], []
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(calls):
            output = fn()
        b.record()
        b.synchronize()
        device_rounds.append(a.elapsed_time(b) / calls)
        # Per-call synchronization measures request latency, independently of
        # batched GPU event measurements (whose samples are round averages).
        for _ in range(calls):
            start = time.perf_counter()
            output = fn()
            torch.cuda.synchronize()
            wall_samples.append((time.perf_counter() - start) * 1000)
    return dict(device_round_average=summarize(device_rounds),
                synchronized_wall=summarize(wall_samples),
                gpu_sample_unit='average of consecutive full forwards',
                warmup=warmup, calls_per_round=calls, rounds=rounds)


def profile_call(fn, path, native=False):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                           torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=True) as p:
        fn()
        torch.cuda.synchronize()
    p.export_chrome_trace(str(path))
    # p.events() also contains CUDA-side compiled-graph annotations whose
    # duration covers child kernels. Use leaf Chrome-trace categories so the
    # summary never counts an annotation and its kernels twice.
    trace = json.loads(Path(path).read_text())
    events = [e for e in trace['traceEvents']
              if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset') and e.get('ph') == 'X']
    kernels = {}
    for e in events:
        row = kernels.setdefault(e['name'], {'calls': 0, 'total_us': 0.0})
        row['calls'] += 1
        row['total_us'] += float(e['dur'])
    owned = ('_embedding', '_layernorm', '_add', '_add_layernorm', '_swiglu', '_rope_qkv',
             '_rope_qkv_compact_position', '_gemm', '_small_gemm', '_sum_parts',
             '_tma_gemm',
             '_prefill', '_decode_parts', '_decode_reduce', '_argmax', '_advance',
             '_copy_prefix', '_copy_token', '_record_token', '_kv_write')
    def allowed(name):
        if 'memcpy' in name.lower() or 'memset' in name.lower():
            return True
        return any(re.search(r'(?<![A-Za-z0-9_])' + re.escape(kernel) + r'(?![A-Za-z0-9_])', name)
                   for kernel in owned)
    forbidden = [name for name in kernels if not allowed(name)] if native else []
    if native and not kernels:
        raise AssertionError('native profiler returned no CUDA events; cannot verify compute provenance')
    if native and forbidden:
        raise AssertionError(f'native compute fallback detected: {forbidden}')
    return dict(trace=Path(path).name, cuda_kernels=kernels, forbidden_native_kernels=forbidden)


def make_runner(name, cfg, params, capacity, compile_model=True, attention_backend='sdpa', matmul_backend='torch',
                torch_attention='sdpa', decode_full_context=False, flex_kernel_options=None):
    if name == 'triton':
        from .native import NativeRunner
        return NativeRunner(cfg, params, capacity)
    from .torch_backend import TorchRunner
    return TorchRunner(cfg, params, capacity, hybrid=name == 'hybrid', compile_model=compile_model,
                       attention_backend=attention_backend if name == 'hybrid' else torch_attention,
                       matmul_backend=matmul_backend if name == 'hybrid' else 'torch',
                       decode_full_context=decode_full_context, flex_kernel_options=flex_kernel_options)


def make_cache(cfg, batch, capacity, generator, random=True):
    import torch
    shape = (batch, cfg.n_kv_heads, capacity, cfg.head_dim)
    factory = (lambda: torch.randn(shape, generator=generator, dtype=torch.bfloat16, device='cuda')) if random else (
        lambda: torch.zeros(shape, dtype=torch.bfloat16, device='cuda'))
    return [(factory(), factory()) for _ in range(cfg.n_layers)]


def tuning_results():
    result = {}
    for module in ('triton_matmul', 'triton_attention'):
        try:
            import importlib
            m = importlib.import_module(f'llm_bench.{module}')
            result[module] = m.tuning_results()
        except (ImportError, AttributeError):
            pass
    return result


def fixed_workload(name, cfg, params, fp32_params, phase, args):
    import torch
    is_decode = phase.name == 'decode'
    g = torch.Generator(device='cuda').manual_seed(args.seed + 100)
    tokens = torch.randint(cfg.vocab, (phase.batch, phase.q_len), generator=g, device='cuda')
    cache = make_cache(cfg, phase.batch, phase.kv_len, g) if is_decode else None
    position = torch.tensor(phase.kv_len - 1, dtype=torch.int64, device='cuda')
    original_cache = map_tree(cache, lambda x: x.clone()) if is_decode else None
    start = time.perf_counter()
    runner = make_runner(name, cfg, params, phase.kv_len, not args.no_compile,
                         args.hybrid_attention, args.hybrid_matmul, args.torch_attention,
                         decode_full_context=is_decode, flex_kernel_options=args.flex_kernel_options)
    torch.cuda.synchronize()
    prepare_s = time.perf_counter() - start
    fn = (lambda: runner.decode(tokens, cache, position)) if is_decode else (lambda: runner.prefill(tokens))
    start = time.perf_counter()
    actual = fn()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - start
    expected = reference(cfg, fp32_params, tokens, original_cache,
                         phase.kv_len - 1 if is_decode else None)
    checks = validate_output(actual, expected)
    if is_decode:
        for pairs, old in zip(cache, original_cache):
            for a, b in zip(pairs, old):
                if not torch.equal(a[:, :, :-1], b[:, :, :-1]):
                    raise AssertionError('decode modified existing cache prefix')
    del actual, expected, original_cache
    captured = Captured(fn)
    # A changed token buffer must change the graph output, not return constants.
    tokens.copy_(torch.randint(cfg.vocab, tokens.shape, generator=g, device='cuda'))
    changed_expected = reference(cfg, fp32_params, tokens, cache,
                                 phase.kv_len - 1 if is_decode else None)
    changed_checks = validate_output(captured(), changed_expected)
    del changed_expected
    count, rounds, warmup = (5, 2, 3) if args.quick else (100, 5, 20)
    torch.cuda.reset_peak_memory_stats()
    timing = time_fixed(captured, warmup, count, rounds)
    memory = dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  scope='whole process, including BF16 source/packed weights and FP32 validation weights')
    no_graph = time_fixed(fn, 2, 5 if args.quick else 20, 2)
    prof = profile_call(fn, Path(args.out).with_name(f'{Path(args.out).stem}-{name}-{phase.name}-trace.json'),
                        native=name == 'triton') if args.profile else None
    gpu_ms = timing['device_round_average']['median_ms']
    return dict(status='ok', backend=name, workload=phase.name, shape=dataclasses.asdict(phase),
                correct=True, correctness=checks, changed_input_correctness=changed_checks,
                prepare_s=prepare_s, compile_autotune_first_call_s=first_s, timing=timing,
                no_graph_timing=no_graph, tokens_per_second=phase.batch*phase.q_len/(gpu_ms/1000),
                memory=memory, profile=prof, tuning=tuning_results())


def generation_workload(name, cfg, params, fp32_params, prompt_length, args):
    import torch
    count = 128 if not args.quick else 8
    capacity = prompt_length + count
    g = torch.Generator(device='cuda').manual_seed(args.seed + 200)
    prompt = torch.randint(cfg.vocab, (1, prompt_length), generator=g, device='cuda')
    cache = make_cache(cfg, 1, capacity, g, random=False)
    token = torch.zeros((1, 1), dtype=torch.int64, device='cuda')
    position = torch.tensor(prompt_length, dtype=torch.int64, device='cuda')
    start = time.perf_counter()
    runner = make_runner(name, cfg, params, capacity, not args.no_compile,
                         args.hybrid_attention, args.hybrid_matmul, args.torch_attention,
                         flex_kernel_options=args.flex_kernel_options)
    torch.cuda.synchronize()
    prepare_s = time.perf_counter() - start

    def prefill():
        output = runner.prefill(prompt)
        runner.copy_prefill_cache(output[1], cache)
        selected = runner.select_token(output[0])
        if name == 'triton':
            from .triton_ops import copy_token
            copy_token(selected, token)
        else:
            token.copy_(selected)
        return output

    def decode():
        output = runner.decode(token, cache, position)
        selected = runner.select_token(output[0])
        if name == 'triton':
            from .triton_ops import copy_token
            copy_token(selected, token)
        else:
            token.copy_(selected)
        runner.advance(position)
        return output

    start = time.perf_counter()
    initial = prefill()
    torch.cuda.synchronize()
    expected = reference(cfg, fp32_params, prompt)
    checks = validate_output(initial, expected)
    # Teacher forcing uses FP32-reference tokens in BOTH paths. This checks
    # accumulated cache errors without confusing numerical drift with sampling.
    ref_cache = [(torch.zeros_like(k, dtype=torch.float32), torch.zeros_like(v, dtype=torch.float32))
                 for k, v in cache]
    for pair, values in zip(ref_cache, expected[1]):
        for dst, src in zip(pair, values):
            dst[:, :, :prompt_length].copy_(src)
    teacher_token = expected[0][:, -1].argmax(-1, keepdim=True)
    generation_checks = []
    # Validate every step in quick/tiny mode; full model covers the entire 127
    # step cache trajectory, recording all output tensor metrics at checkpoints.
    checkpoints = {0, 1, 7, 31, 63, count - 2}
    for step in range(count - 1):
        pos = prompt_length + step
        position.fill_(pos)
        actual = runner.decode(teacher_token, cache, position)
        ref = reference(cfg, fp32_params, teacher_token, ref_cache, pos)
        for pair, values in zip(ref_cache, ref[1]):
            for dst, src in zip(pair, values):
                dst[:, :, pos:pos+1].copy_(src[:, :, pos:pos+1])
        if step in checkpoints or args.preset == 'tiny':
            generation_checks.append(dict(step=step, checks=validate_output(actual, ref, pos+1)))
        teacher_token = ref[0][:, -1].argmax(-1, keepdim=True)
    del initial, expected, ref_cache, teacher_token, actual, ref
    # Compile select/advance and warm full decode path, then capture once.
    position.fill_(prompt_length)
    prefill()
    decode()
    torch.cuda.synchronize()
    first_s = time.perf_counter() - start
    pre_graph = Captured(prefill)
    position.fill_(prompt_length)
    dec_graph = Captured(decode)
    # Validate the captured graphs themselves at two consecutive GPU-resident
    # positions. This catches stale position specialization or replay aliasing
    # independently of the pre-capture teacher-forcing trajectory.
    position.fill_(prompt_length)
    pre_graph()
    graph_checks = []
    for step in range(2):
        pos = prompt_length + step
        expected = reference(cfg, fp32_params, token, cache, pos)
        actual = dec_graph()
        graph_checks.append(dict(step=step, checks=validate_output(actual, expected, pos + 1)))
        if int(position.item()) != pos + 1:
            raise AssertionError('captured decode failed to advance the GPU cache position')
    del actual, expected

    def run_once(record=False):
        position.fill_(prompt_length)
        torch.cuda.synchronize()
        start, first, end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        wall_start = time.perf_counter()
        start.record()
        pre_graph()
        first.record()
        for _ in range(count - 1):
            dec_graph()
        end.record()
        end.synchronize()
        wall = (time.perf_counter() - wall_start) * 1000
        return dict(device_ms=start.elapsed_time(end), ttft_device_ms=start.elapsed_time(first),
                    decode_device_ms=first.elapsed_time(end)/(count-1), wall_ms=wall)

    for _ in range(2):
        run_once()
    torch.cuda.reset_peak_memory_stats()
    rows = [run_once() for _ in range(3 if args.quick else 20)]
    memory = dict(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  scope='whole process, including BF16 source/packed weights and FP32 validation weights')
    # Diagnostic generation collects actual sampled tokens outside timing.
    position.fill_(prompt_length)
    pre_graph()
    sampled = [int(token.item())]
    for _ in range(count-1):
        dec_graph()
        sampled.append(int(token.item()))
    final_cache_position = int(position.item())
    if final_cache_position != prompt_length + count - 1:
        raise AssertionError('generation graph did not consume the expected number of cache positions')
    prof = None
    if args.profile:
        position.fill_(prompt_length)
        pre_graph()
        prof = profile_call(decode, Path(args.out).with_name(f'{Path(args.out).stem}-{name}-generate-trace.json'),
                            native=name == 'triton')
    return dict(status='ok', backend=name, workload='generate', correct=True,
                batch=1, prompt_length=prompt_length, generated_tokens=count,
                prepare_s=prepare_s, prepare_compile_and_validation_s=first_s,
                correctness=checks, teacher_forced_checks=generation_checks, graph_replay_checks=graph_checks,
                timing={key: summarize([row[key] for row in rows]) for key in rows[0]},
                memory=memory, sampled_tokens=sampled, final_cache_position=final_cache_position,
                profile=prof, tuning=tuning_results())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('torch', 'hybrid', 'triton', 'all'), default='all')
    parser.add_argument('--preset', choices=('tiny', 'small', 'colab'), default='colab')
    parser.add_argument('--workload', choices=('prefill', 'decode', 'generate', 'all'), default='all')
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--quick', action='store_true', help='smoke timing; not a final benchmark')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--no-compile', action='store_true', help='diagnostics only; never a compiled result')
    parser.add_argument('--torch-attention', choices=('sdpa', 'flex-decode'), default='sdpa')
    parser.add_argument('--hybrid-attention', choices=('sdpa', 'triton', 'decode-triton'), default='sdpa')
    parser.add_argument('--hybrid-matmul', choices=('torch', 'decode-triton', 'm1-triton'), default='torch')
    parser.add_argument('--tuning-config', help='JSON containing measured split-K, split-KV and optional FlexAttention choices')
    args = parser.parse_args()
    import torch
    from llm_roofline import spec, torch_llm
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA GPU with native BF16 support required')
    args.flex_kernel_options = None
    if args.tuning_config:
        from .triton_matmul import set_split_k_overrides, set_tma_overrides
        from .triton_attention import set_attention_split_overrides
        tuning_config = json.loads(Path(args.tuning_config).read_text())
        set_split_k_overrides(tuning_config.get('split_k_overrides', {}))
        set_tma_overrides(tuning_config.get('tma_overrides', {}))
        set_attention_split_overrides(tuning_config.get('attention_split_overrides', {}))
        args.flex_kernel_options = tuning_config.get('flex_kernel_options')
    preset = spec.PRESETS[args.preset]
    names = ['torch', 'hybrid', 'triton'] if args.backend == 'all' else [args.backend]
    if args.reverse:
        names.reverse()
    phases = ['prefill', 'decode', 'generate'] if args.workload == 'all' else [args.workload]
    result = dict(environment=environment(), arguments=vars(args), model=dataclasses.asdict(preset.model),
                  started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), results=[])
    if args.tuning_config:
        result['tuning_config_sha256'] = hashlib.sha256(Path(args.tuning_config).read_bytes()).hexdigest()
    write_json(args.out, result)
    with torch.inference_mode():
        params = torch_llm.random_params(preset.model, torch.bfloat16, torch.device('cuda'), seed=args.seed)
        parameter_sizes = []
        map_tree(params, lambda tensor: parameter_sizes.append(tensor.numel()))
        result['parameter_count'] = sum(parameter_sizes)
        fp32_params = map_tree(params, lambda x: x.float())
        for name in names:
            for phase in phases:
                print(f'BEGIN {name} {phase}', flush=True)
                try:
                    if phase == 'generate':
                        row = generation_workload(name, preset.model, params, fp32_params, preset.prefill.q_len, args)
                    else:
                        row = fixed_workload(name, preset.model, params, fp32_params, getattr(preset, phase), args)
                    print(f'OK {name} {phase}', flush=True)
                except Exception as error:
                    row = dict(status='error', backend=name, workload=phase, correct=False,
                               error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
                    print(row['traceback'], flush=True)
                result['results'].append(row)
                result['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                write_json(args.out, result)
                gc.collect()
                torch.cuda.empty_cache()
    result['complete'] = True
    result['all_passed'] = all(r['status'] == 'ok' for r in result['results'])
    write_json(args.out, result)
    print('LLM_BENCH_RUN_COMPLETE', flush=True)
    return 0 if result['all_passed'] else 1
