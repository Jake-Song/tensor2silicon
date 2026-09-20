import json
import shutil
import subprocess
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from toy import web


class LessonTests(unittest.TestCase):
    def test_presets_and_defaults(self):
        data = web.examples()
        self.assertEqual(len(data["examples"]), 3)
        self.assertEqual(web.validate_config({}), web.DEFAULTS)
        self.assertIn("return relu", data["examples"][1]["source"])

    @unittest.skipUnless(shutil.which("gcc"), "C compiler not available")
    def test_all_examples_fusion_modes_and_tiles(self):
        for example in web.PRESETS:
            baseline = None
            for fusion in ("none", "epilogue", "full"):
                for tiled in (False, True):
                    with self.subTest(example=example, fusion=fusion, tiled=tiled):
                        result = web.run_lesson({"example": example, "fusion": fusion,
                                                 "tiled": tiled, "m": 3, "k": 5, "n": 2})
                        self.assertTrue(all(c["status"] == "pass" for c in result["checks"].values()))
                        self.assertEqual(result["output"]["shape"], [3, 2])
                        self.assertEqual(len(result["output"]["preview"]), 3)
                        if baseline is None:
                            baseline = result["raw"]
                        self.assertEqual(result["raw"], baseline)
                        self.assertNotIn("tile", result["raw"]["ir"])
                        self.assertEqual("tile=" in result["optimized"]["ir"], tiled)

    def test_serialization_and_shared_intermediate(self):
        with patch.object(web.shutil, "which", return_value=None):
            result = web.run_lesson({"example": "shared"})
        graph = result["optimized"]
        self.assertIn("matmul", [n["op"] for n in graph["nodes"]])
        fusion = next(n for n in graph["nodes"] if n["op"] == "fusion")
        inner_ids = {n["id"] for n in fusion["body"]}
        self.assertIn(fusion["root"], inner_ids)
        for node in fusion["body"]:
            self.assertTrue(set(node["inputs"]) <= inner_ids)
        ids = {n["id"] for n in graph["nodes"]}
        self.assertTrue(set(graph["outputs"]) <= ids)
        self.assertEqual(result, json.loads(json.dumps(result)))
        self.assertIn("cannot branch", result["branch"]["error"])

    def test_missing_compiler_preserves_python_results(self):
        with patch.object(web.shutil, "which", return_value=None):
            result = web.run_lesson({})
        self.assertEqual(result["checks"]["c"]["status"], "unavailable")
        self.assertEqual(result["checks"]["python"]["status"], "pass")
        self.assertIn("void toy_run", result["c_code"])
        self.assertTrue(result["raw"]["nodes"])

    def test_compiler_failure_preserves_results(self):
        with patch.object(web.shutil, "which", return_value="gcc"), patch.object(
            web, "compile_graph", side_effect=subprocess.CalledProcessError(1, "gcc")
        ):
            result = web.run_lesson({})
        self.assertEqual(result["checks"]["c"]["status"], "unavailable")
        self.assertEqual(result["checks"]["interpreter"]["status"], "pass")

    def test_invalid_settings(self):
        values = [[], None, {"source": "print('no')"}, {"example": []},
                  {"example": "unknown"}, {"fusion": {}}, {"fusion": "auto"},
                  {"tiled": 1}, {"m": 0}, {"k": 129}, {"n": 1.5}, {"m": True},
                  {"m": "16"}, {"n": None}]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                web.validate_config(value)

    def test_dimension_boundaries(self):
        with patch.object(web.shutil, "which", return_value=None):
            for size in (1, 128):
                result = web.run_lesson({"m": size, "k": size, "n": size})
                self.assertEqual(result["output"]["shape"], [size, size])
                self.assertEqual(len(result["output"]["preview"]), min(size, 4))

    def test_worker_integration(self):
        result = web.run_worker(web.validate_config({"m": 2, "k": 3, "n": 2}))
        self.assertEqual(result["checks"]["python"]["status"], "pass")
        self.assertEqual(result["output"]["shape"], [2, 2])

    def test_worker_error_and_timeout(self):
        with patch.object(web.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                web.run_worker(web.DEFAULTS)
        with patch.object(web.subprocess, "run", side_effect=subprocess.TimeoutExpired("worker", 30)) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                web.run_worker(web.DEFAULTS)
            self.assertEqual(run.call_args.kwargs["timeout"], 30)


class QuietHandler(web.Handler):
    def log_message(self, *args):
        pass


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = web.ThreadingHTTPServer(("127.0.0.1", 0), QuietHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def request(self, path, data=None, content_type="application/json"):
        request = Request(self.base + path, data=data, headers={"Content-Type": content_type})
        try:
            response = urlopen(request, timeout=35)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, response.headers, response.read()

    def test_static_assets_and_catalog(self):
        for path in ("/", "/style.css", "/app.js", "/architecture.js", "/architecture.css", "/favicon.svg"):
            status, headers, body = self.request(path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertEqual(headers["Cache-Control"], "no-store")
        status, _, body = self.request("/api/examples")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["defaults"], web.DEFAULTS)

    def test_files_outside_web_assets_are_not_served(self):
        for path in ("/README.md", "/../pyproject.toml", "/%2e%2e/pyproject.toml", "/toy/web.py"):
            self.assertEqual(self.request(path)[0], 404)

    def test_run_api_and_bad_requests(self):
        with patch.object(web, "run_worker", return_value={"ok": True}) as worker:
            status, _, body = self.request("/api/run", b'{"m": 3}')
            self.assertEqual(status, 200)
            self.assertTrue(json.loads(body)["ok"])
            self.assertEqual(worker.call_args.args[0]["m"], 3)
        for body in (b"{", b"[]", b'{"n": 0}', b"x" * 4097):
            self.assertEqual(self.request("/api/run", body)[0], 400)
        self.assertEqual(self.request("/api/run", b"{}", "text/plain")[0], 415)
        self.assertEqual(self.request("/api/unknown", b"{}")[0], 404)

    def test_timeout_and_worker_failure_recover(self):
        for error, status in ((subprocess.TimeoutExpired("worker", 30), 504), (RuntimeError("failed"), 500)):
            with patch.object(web, "run_worker", side_effect=error):
                actual, _, body = self.request("/api/run", b"{}")
                self.assertEqual(actual, status)
                self.assertIn("error", json.loads(body))
        with patch.object(web, "run_worker", return_value={"recovered": True}):
            self.assertEqual(self.request("/api/run", b"{}")[0], 200)

    def test_concurrent_requests_serialize_compiler_work(self):
        active = 0
        peak = 0
        guard = threading.Lock()

        def worker(config):
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(active, peak)
            time.sleep(.02)
            with guard:
                active -= 1
            return {"ok": True}

        with patch.object(web, "run_worker", side_effect=worker), ThreadPoolExecutor(4) as pool:
            responses = list(pool.map(lambda _: self.request("/api/run", b"{}"), range(4)))
        self.assertTrue(all(status == 200 for status, _, _ in responses))
        self.assertEqual(peak, 1)


if __name__ == "__main__":
    unittest.main()
