"""Verify downloaded artifacts and summarize pooled raw invocation times."""
import hashlib
import json
from pathlib import Path
import tarfile
import numpy as np

root = Path(__file__).parent / 'results/2026-10-19-colab-jax'
input_root = Path(__file__).parent / 'results/2026-10-19-colab'
root.mkdir(parents=True, exist_ok=True)
manifest = json.loads((input_root / 'inputs-manifest.json').read_text())
assert hashlib.sha256((input_root / 'inputs.npz').read_bytes()).hexdigest() == manifest['archive_sha256']
summary = []
with np.load(input_root / 'inputs.npz') as inputs:
    for hw in ['v5e1', 'v6e1', 'A100', 'G4']:
        archive = root / f'{hw}.tar.gz'
        if not archive.exists():
            continue
        with tarfile.open(archive) as tar:
            tar.extractall(root / hw, filter='data')
        folder = root / hw / 'results'
        result = json.loads((folder / 'result.json').read_text())
        assert result['meta']['backend'] == 'jax'
        assert result['meta']['framework_version'] == '0.11.1'
        assert result['meta']['jaxlib'] == '0.11.1'
        assert result['meta']['platform'] == ('tpu' if hw.startswith('v') else 'gpu')
        assert result['meta']['input_sha256'] == manifest['archive_sha256']
        assert len(result['rows']) == 10
        assert all(e['passed'] for e in result['edge_tests'])
        with np.load(folder / 'outputs.npz') as outputs:
            for row in result['rows']:
                actual = outputs[f"{row['seed']}_m{row['m']}"]
                ref = inputs[f"{row['seed']}_ref"][:row['m']]
                assert np.isfinite(actual).all()
                np.testing.assert_allclose(actual, ref, atol=1e-4, rtol=1e-4)
        for m in [1, 128]:
            rows = [r for r in result['rows'] if r['m'] == m]
            times = np.concatenate([r['times_ms'] for r in rows])
            assert len(times) == 5000
            elapsed = sum(r['elapsed_s'] for r in rows)
            summary.append(dict(target=hw, device=result['meta']['device_kind'], m=m,
                                n=len(times), p50_ms=float(np.percentile(times, 50)),
                                p95_ms=float(np.percentile(times, 95)), mean_ms=float(times.mean()),
                                samples_per_s=5000*m/elapsed,
                                max_abs_error=max(r['max_abs_error'] for r in rows),
                                failed_elements=sum(r['failed_elements'] for r in rows)))
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*jax_experiment.py')}
(root / 'source-manifest.json').write_text(json.dumps(sources, indent=2) + '\n')
for r in summary:
    print(f"| {r['target']} | {r['m']} | {r['p50_ms']:.6f} / {r['p95_ms']:.6f} | {r['samples_per_s']:,.0f} | {r['max_abs_error']:.3g} / 통과 |")
