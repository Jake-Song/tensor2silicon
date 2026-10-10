"""Result-schema compatibility and self-contained CUDA source packaging."""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

from llm_bench.report import BACKENDS, LEGACY_BACKENDS, load_runs, build_notebook
from llm_roofline.spec import PRESETS


class ReportTests(unittest.TestCase):
    def fixture(self, backends, explicit=True):
        value = dict(complete=True, arguments=dict(preset='colab'),
                     model=dataclasses.asdict(PRESETS['colab'].model),
                     results=[dict(backend=b, workload=w, generated_tokens=128)
                              for b in backends for w in ('prefill','decode','generate')])
        if explicit:
            value['backends'] = list(backends)
        return value

    def load(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1,2):
                (Path(tmp)/f'final-run{n}.json').write_text(json.dumps(value))
            return load_runs(tmp)

    def test_old_three_backend_results_remain_readable(self):
        self.assertEqual(len(self.load(self.fixture(LEGACY_BACKENDS,False))),2)

    def test_five_backend_results_require_every_workload(self):
        value=self.fixture(BACKENDS)
        self.assertEqual(len(self.load(value)),2)
        value['results'].pop()
        with self.assertRaises(ValueError): self.load(value)

    def test_notebook_embeds_cuda_and_cpp_sources(self):
        import base64,gzip,re
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'report.ipynb'
            build_notebook([],root,path,results_dir=Path(tmp))
            notebook=json.loads(path.read_text())
            source=''.join(notebook['cells'][1]['source'])
            payload=re.search(r"PAYLOAD = '([^']+)'",source)[1]
            data=json.loads(gzip.decompress(base64.b64decode(payload)))
            for name in ('llm_bench/csrc/bindings.cpp','llm_bench/csrc/kernels.cu'):
                self.assertEqual(data['sources'][name],(root/name).read_text())


if __name__ == '__main__':
    unittest.main()
