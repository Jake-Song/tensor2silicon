"""Benchmark the saved size-specific native policy against cuBLASLt on one GPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument('--policy', type=Path, default=root / 'selected.json')
    parser.add_argument('--output', type=Path, default=root / 'results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--include-stages', action='store_true', help='Also compare SIMT and the five teaching stages')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    for item in policy['conditions']:
        name = f"selected-{args.seed}-{item['size']}"
        variants = item['variant']
        if args.include_stages:
            variants += ',s0_simt,s1_wmma_global,s2_wmma_smem,s3_mma_ldmatrix,s4_async,s5_frag_double'
        subprocess.run([
            sys.executable, str(root / 'colab_job.py'), '--name', name, '--mode', 'tune',
            '--size', str(item['size']), '--rounds', str(args.rounds), '--seed', str(args.seed),
            '--group', str(item['group']), '--variants', variants, '--output', str(args.output),
        ], check=True)


if __name__ == '__main__':
    main()
