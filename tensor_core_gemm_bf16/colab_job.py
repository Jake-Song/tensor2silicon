"""Execute reproducible subprocess jobs inside an already allocated Colab G4.

Usage: python colab_job.py --name screening --mode tune --size 4096 --rounds 3
No allocation, authentication, or runtime cleanup is performed by this file.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--mode', choices=['check', 'bench', 'tune', 'list'], default='tune')
    parser.add_argument('--size', type=int, default=4096)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--group', type=int, default=8)
    parser.add_argument('--variants', help='Comma-separated explicit variant names; cuBLAS is always included')
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if Path(args.name).name != args.name or args.name in {'.', '..'}:
        parser.error('name must be one filename component')
    root = Path(__file__).resolve().parent
    args.output.mkdir(parents=True, exist_ok=True)
    command = [str(root / 'build/bf16_gemm'), args.mode]
    if args.mode in {'bench', 'tune'}:
        command += [str(args.size), str(args.rounds), str(args.seed)]
    env = dict(os.environ, BF16_GROUP=str(args.group))
    if args.variants:
        env['BF16_VARIANTS'] = args.variants
    manifest = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*')
        if p.is_file() and (p.suffix in {'.cu', '.cuh', '.py', '.json'} or p.name == 'Makefile')
        and not {'build', 'results', 'artifacts'}.intersection(p.relative_to(root).parts)
    }
    start = time.time()
    telemetry_path = args.output / f'{args.name}.telemetry.csv'
    with telemetry_path.open('w') as telemetry_file:
        telemetry = subprocess.Popen(
            ['nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu',
             '--format=csv', '--loop-ms=200'], stdout=telemetry_file, stderr=subprocess.DEVNULL)
        try:
            with (args.output / f'{args.name}.jsonl').open('w') as stdout, (args.output / f'{args.name}.stderr').open('w') as stderr:
                completed = subprocess.run(command, cwd=root, env=env, stdout=stdout, stderr=stderr, timeout=900)
        finally:
            telemetry.terminate()
            telemetry.wait(timeout=10)
    metadata = dict(command=command, group=args.group, variants=args.variants,
                    started_unix=start, elapsed_s=time.time() - start,
                    returncode=completed.returncode, source_sha256=manifest,
                    operation='D=A@B', input_dtype='bfloat16', output_dtype='float32', accumulation='float32')
    (args.output / f'{args.name}.meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
