"""Generate the shared, rounded BF16 values and CPU FP64 references once."""
import hashlib
import json
from pathlib import Path
import numpy as np


def bf16(x):
    u = x.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def main():
    root = Path(__file__).parent / 'results/2026-10-19-colab'
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-1, 1, (128, 1024)).astype(np.float32)
        w = (rng.uniform(-1, 1, (1024, 1024)) / np.sqrt(1024)).astype(np.float32)
        b = rng.uniform(-.1, .1, 1024).astype(np.float32)
        for name, value in [('x_original', x), ('w_original', w), ('x', bf16(x)), ('w', bf16(w)), ('b', b)]:
            arrays[f'{seed}_{name}'] = value
        arrays[f'{seed}_ref'] = np.maximum(bf16(x).astype(np.float64) @ bf16(w).astype(np.float64) + b.astype(np.float64), 0).astype(np.float32)
    path = root / 'inputs.npz'
    np.savez_compressed(path, **arrays)
    manifest = {'archive_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}}
    (root / 'inputs-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(path, path.stat().st_size, manifest['archive_sha256'])


if __name__ == '__main__':
    main()
