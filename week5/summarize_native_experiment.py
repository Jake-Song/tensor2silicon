"""Revalidate recovered CUDA/Pallas outputs and aggregate paired timing records."""
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

ROOT = Path(__file__).parent
RESULTS = ROOT / 'results/2026-10-19-native'


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    input_path = ROOT / 'results/2026-10-19-colab/inputs.npz'
    input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
    summary = []
    with np.load(input_path, allow_pickle=False) as inputs:
        for target in ['A100', 'v5e1']:
            archive = RESULTS / f'{target}.tar.gz'
            if not archive.exists():
                continue
            with tarfile.open(archive) as tar:
                tar.extractall(RESULTS / target, filter='data')
            folder = RESULTS / target / 'results'
            result = json.loads((folder/'result.json').read_text())
            meta = result['meta']
            assert meta['target'] == target
            assert meta['input_sha256'] == input_hash
            assert meta['jax'] == meta['jaxlib'] == '0.11.1'
            assert meta['output_dtype'] == meta['accumulation'] == 'float32'
            assert len(meta['devices']) == 1
            assert ('A100' if target == 'A100' else 'TPU v5 lite') in meta['device_kind']
            kernel = 'native_relu.cu' if target == 'A100' else 'pallas_relu.py'
            for source in ['run_native_experiment.py', kernel]:
                assert meta['sources'][source] == hashlib.sha256((ROOT/source).read_bytes()).hexdigest()
            if target == 'v5e1':
                for m in [1,128]:
                    hlo = (folder/f'pallas-m{m}-optimized.txt').read_text()
                    assert 'custom_call_target="tpu_custom_call"' in hlo
                    assert ' dot(' not in hlo
            assert len(result['rows']) == 20
            assert len(result['edge_tests']) == 8
            assert all(e['passed'] for e in result['edge_tests'])
            seen = set()
            with np.load(folder/'outputs.npz', allow_pickle=False) as outputs:
                for row in result['rows']:
                    key = (row['implementation'],row['m'],row['seed'])
                    assert key not in seen
                    seen.add(key)
                    assert row['passed'] and row['failed_elements'] == 0
                    assert len(row['times_ms']) == 1000
                    actual = outputs[f"{row['implementation']}_{row['seed']}_m{row['m']}"]
                    ref = inputs[f"{row['seed']}_ref"][:row['m']]
                    assert actual.dtype == np.float32
                    assert np.isfinite(actual).all()
                    np.testing.assert_allclose(actual,ref,atol=1e-4,rtol=1e-4)
            for m in [1,128]:
                for implementation in ['jax',meta['native']]:
                    rows = [r for r in result['rows'] if r['m']==m and r['implementation']==implementation]
                    assert sorted(r['seed'] for r in rows) == list(range(5))
                    times = np.concatenate([r['times_ms'] for r in rows])
                    elapsed = sum(r['elapsed_s'] for r in rows)
                    summary.append(dict(
                        target=target,device=meta['device_kind'],M=m,implementation=implementation,
                        n=len(times),p50_ms=float(np.median(times)),
                        p95_ms=float(np.percentile(times,95)),mean_ms=float(times.mean()),
                        elapsed_s=elapsed,samples_per_s=len(times)*m/elapsed,
                        max_abs_error=max(r['max_abs_error'] for r in rows),
                        passed=True,selected_config=rows[0]['selected_config'],
                    ))
    assert {r['target'] for r in summary} == {'A100','v5e1'}, 'Both devices are required.'
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    sources = {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
               ['native_relu.cu','pallas_relu.py','run_native_experiment.py','summarize_native_experiment.py']}
    (RESULTS/'source-manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    for row in summary:
        print(f"{row['target']:5} M={row['M']:3} {row['implementation']:6} "
              f"p50={row['p50_ms']:.6f} ms, {row['samples_per_s']:,.0f} samples/s")


if __name__ == '__main__':
    main()
