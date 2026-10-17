"""Summarize saved JSONL samples without allocating or accessing a GPU."""
import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics


def summarize(paths):
    samples = defaultdict(list)
    checks_by_file = {}
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        invalid = [r for r in rows if r.get('kind') == 'check' and not r.get('pass')]
        summaries = [r for r in rows if r.get('kind') == 'check_summary']
        if invalid or any(r['failed'] for r in summaries):
            raise ValueError(f'Correctness failure in {path}')
        checks_by_file[str(path)] = [r for r in rows if r.get('kind') == 'check']
        for row in rows:
            if row.get('kind') == 'timing':
                if row.get('operation') != 'D=A@B':
                    raise ValueError(f'Non-matmul result: {path}')
                key = (str(path), row['size'], row['variant'], row['group'])
                samples[key].append(row)
    result = []
    for (path, size, name, group), rows in sorted(samples.items()):
        matching_checks = [r for r in checks_by_file[path] if r['variant'] == name and r['m'] == size
                           and r['n'] == size and r['k'] == size and r['seed'] == rows[0]['seed']
                           and r['group'] == group]
        if len({r['seed'] for r in rows}) != 1 or len({r.get('timing', 'event') for r in rows}) != 1:
            raise ValueError(f'Mixed seed or timing mode in {path}: {name}/{size}')
        if any(not math.isfinite(r['ms']) or r['ms'] <= 0 for r in rows):
            raise ValueError(f'Invalid timing in {path}: {name}/{size}')
        if not matching_checks:
            raise ValueError(f'No full-output correctness record for {path}: {name}/{size}')
        phases = {r.get('phase') for r in matching_checks}
        if any(phases) and not {'before', 'after'} <= phases:
            raise ValueError(f'Missing pre/post timing check in {path}: {name}/{size}')
        median = statistics.median(r['ms'] for r in rows)
        baseline = samples.get((path, size, 'cublas', group))
        baseline_ms = statistics.median(r['ms'] for r in baseline) if baseline else None
        result.append(dict(source=path, size=size, variant=name, stage=rows[0]['stage'], group=group,
                           seed=rows[0]['seed'],
                           timing=rows[0].get('timing', 'event'),
                           samples=len(rows), median_ms=median, min_ms=min(r['ms'] for r in rows),
                           max_ms=max(r['ms'] for r in rows), tflops=2 * size**3 / median / 1e9,
                           cublas_ms=baseline_ms, cublas_ratio=baseline_ms / median if baseline_ms else None,
                           full_output_pass=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(rows, indent=2) + '\n')
    with args.output.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['source'])
        writer.writeheader()
        writer.writerows(rows)
    for size in sorted({r['size'] for r in rows}):
        best = sorted((r for r in rows if r['size'] == size), key=lambda r: r['median_ms'])[:8]
        print(size, [(r['variant'], round(r['tflops'], 2), round(r['cublas_ratio'], 3) if r['cublas_ratio'] else None) for r in best])


if __name__ == '__main__':
    main()
