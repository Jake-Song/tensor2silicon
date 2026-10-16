"""Run correctness-gated kernel 7 experiments on the GPU hosting this script."""
import argparse
import json
import os
from pathlib import Path
import random
import re
import statistics
import subprocess


def measure(binary, kernel, size, env, log):
    p = subprocess.run([str(binary), str(kernel), str(size), str(size)],
                       env=env, text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=180)
    log.write_text(p.stdout)
    if p.returncode or '-> PASS' not in p.stdout or 'FAIL' in p.stdout:
        raise RuntimeError(f'Execution or correctness failed: {log}')
    matches = re.findall(r'performance: \(\s*([\d.]+)\) GFLOPS', p.stdout)
    if len(matches) != 1:
        raise RuntimeError(f'Missing or ambiguous measurement: {log}')
    return float(matches[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', nargs='+', type=int, default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--configs', nargs='+', type=int, choices=range(7), default=list(range(7)))
    parser.add_argument('--copies', nargs='+', type=int, choices=range(5), default=[0])
    parser.add_argument('--orders', nargs='+', type=int, choices=range(7), default=list(range(7)))
    parser.add_argument('--output', type=Path, default=Path('results/tuning'))
    args = parser.parse_args()
    if args.rounds < 1 or any(s < 128 or s & (s - 1) for s in args.sizes):
        parser.error('rounds must be positive and sizes must be powers of two >= 128')
    args.output.mkdir(parents=True, exist_ok=True)
    binary = Path(__file__).resolve().parent / 'build/tc_gemm'
    base = {k: v for k, v in os.environ.items() if not k.startswith('TC7_')}
    if base.get('NVIDIA_TF32_OVERRIDE') == '0':
        parser.error('NVIDIA_TF32_OVERRIDE=0 disables the TF32 baseline')
    rows = []
    rng = random.Random(7)
    for size in args.sizes:
        samples = {}
        for trial in range(args.rounds):
            variants = [('cublas', 100, {}), ('default', 7, {})]
            variants += [(f'cfg{cfg}-order{order}-copy{copy}', 7,
                          {'TC7_CFG': str(cfg), 'TC7_ORDER': str(order), 'TC7_COPY': str(copy)})
                         for cfg in args.configs for order in args.orders for copy in args.copies
                         if not (size == 128 and cfg in (0, 1))
                         and (copy != 4 or order == 6)]
            rng.shuffle(variants)
            for name, kernel, settings in variants:
                gflops = measure(binary, kernel, size, base | settings,
                                 args.output / f'{size}-{trial}-{name}.log')
                samples.setdefault(name, []).append(gflops)
                print(f'{size} round={trial} {name}: {gflops / 1000:.2f} TFLOPS', flush=True)
        reference = statistics.median(samples['cublas'])
        for name, values in samples.items():
            median = statistics.median(values)
            rows.append(dict(size=size, variant=name, gflops=median,
                             percent_cublas=100 * median / reference, samples=values))
        (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
        best = max((r for r in rows if r['size'] == size and r['variant'] != 'cublas'),
                   key=lambda r: r['gflops'])
        print(f'BEST {size}: {best}', flush=True)


if __name__ == '__main__':
    main()
