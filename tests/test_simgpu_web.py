import importlib.util
import json
import unittest
from unittest.mock import patch

from toy.web import Handler, WEB_ROOT, examples, run_lesson, run_worker, validate_config


class WebContractTests(unittest.TestCase):
    def test_assets_and_defaults(self):
        for filename, _ in Handler.ASSETS.values():
            self.assertTrue((WEB_ROOT / filename).is_file(), filename)
        self.assertEqual(examples()["defaults"]["backend"], "c")

    def test_input_limits(self):
        for bad in ([], {"backend": "unknown"}, {"backend": "simgpu", "m": 17},
                    {"backend": "simgpu", "tiled": True}, {"m": True}, {"k": 0}, {"source": "x"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_config(bad)
        self.assertEqual(validate_config({"backend": "simgpu", "m": 16})["m"], 16)

    def test_missing_simulator_is_visible(self):
        with patch.dict("sys.modules", {"simgpu": None}):
            response = run_lesson({"backend": "simgpu"})
        self.assertEqual(response["checks"]["simgpu"]["status"], "unavailable")
        self.assertIsNone(response["simulation"])

    @unittest.skipUnless(importlib.util.find_spec("simgpu"), "optional simulator not installed")
    def test_all_presets_and_fusion_modes(self):
        for example in ("matmul", "relu_linear", "shared"):
            for fusion in ("none", "epilogue", "full"):
                with self.subTest(example=example, fusion=fusion):
                    response = run_lesson(dict(backend="simgpu", example=example, fusion=fusion, m=3, k=5, n=3))
                    self.assertTrue(all(c["status"] == "pass" for c in response["checks"].values()))
                    json.dumps(response, allow_nan=False)
                    sim = response["simulation"]
                    ids = {n["id"] for n in response["optimized"]["nodes"]}
                    for kernel in sim["kernels"]:
                        self.assertIn(kernel["node_id"], ids)
                        self.assertTrue(kernel["report"]["samples"])
                        self.assertTrue(kernel["source_map"])
                    self.assertEqual(sim["stats"]["cycles"], sum(k["report"]["cycles"] for k in sim["kernels"]))

    @unittest.skipUnless(importlib.util.find_spec("simgpu"), "optional simulator not installed")
    def test_isolated_worker(self):
        result = run_worker(dict(backend="simgpu", m=3, k=4, n=5))
        self.assertEqual(result["checks"]["simgpu"]["status"], "pass")


if __name__ == "__main__":
    unittest.main()
