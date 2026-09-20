"""Local compiler lessons. Run ``python -m toy.web`` and open localhost:8000."""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from . import compile_graph, compile_simgpu, emit_c, fuse, relu, run, symbolic_trace, tile_matmuls
from .trace import TraceError

ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = ROOT / "web"
DEFAULTS = {"example": "relu_linear", "m": 16, "k": 8, "n": 4, "fusion": "full", "tiled": False, "backend": "c"}
RTOL, ATOL = 1e-4, 1e-4
WORKER_TIMEOUT = 30


def matmul(x, w):
    return x @ w


def relu_linear(x, w, b):
    return relu(x @ w + b)


def shared(x, w, b):
    h = x @ w
    return relu(h + b) + h


def data_branch(x):
    if x:
        return relu(x)
    return x


PRESETS = {
    "matmul": (matmul, "행렬 곱", "Y = XW", "두 행렬의 축이 어떻게 출력 shape을 결정하는지 살펴봅니다."),
    "relu_linear": (relu_linear, "Linear + ReLU", "Y = ReLU(XW + b)", "브로드캐스팅과 fusion을 한 식에서 따라갑니다."),
    "shared": (shared, "공유되는 중간값", "H = XW · Y = ReLU(H + b) + H", "두 곳에서 쓰는 H가 왜 독립된 연산으로 남는지 확인합니다."),
}


def examples() -> dict:
    return {
        "defaults": DEFAULTS,
        "examples": [
            {"id": key, "title": title, "formula": formula, "description": desc,
             "source": inspect.getsource(fn)}
            for key, (fn, title, formula, desc) in PRESETS.items()
        ],
    }


def validate_config(value) -> dict:
    if not isinstance(value, dict):
        raise ValueError("설정은 JSON object여야 합니다.")
    if set(value) - DEFAULTS.keys():
        raise ValueError("알 수 없는 설정입니다. 예제, 차원, fusion, tiling, backend만 선택할 수 있습니다.")
    config = DEFAULTS | value
    if not isinstance(config["example"], str) or config["example"] not in PRESETS:
        raise ValueError("지원하지 않는 예제입니다.")
    for key in ("m", "k", "n"):
        if type(config[key]) is not int or not 1 <= config[key] <= 128:
            raise ValueError(f"{key.upper()}은 1–128 사이 정수여야 합니다.")
    if not isinstance(config["fusion"], str) or config["fusion"] not in ("none", "epilogue", "full"):
        raise ValueError("fusion은 none, epilogue, full 중 하나여야 합니다.")
    if type(config["tiled"]) is not bool:
        raise ValueError("tiled는 boolean이어야 합니다.")
    if config["backend"] not in ("c", "simgpu"):
        raise ValueError("backend는 c 또는 simgpu여야 합니다.")
    if config["backend"] == "simgpu":
        if any(config[key] > 16 for key in ("m", "k", "n")):
            raise ValueError("시뮬레이터에서는 M, K, N을 1–16으로 설정해 주세요.")
        if config["tiled"]:
            raise ValueError("CPU tiling은 sim-gpu에서 지원하지 않습니다.")
    return config


def serialize_graph(graph) -> dict:
    nodes = [*graph.inputs, *graph.nodes]
    ids = {node: f"n{i}" for i, node in enumerate(nodes)}

    def record(node, mapping):
        item = {"id": mapping[node], "name": node.name, "op": node.op,
                "inputs": [mapping[n] for n in node.inputs], "shape": list(node.shape),
                "dtype": node.dtype,
                "attrs": {k: v for k, v in node.attrs.items() if k not in ("body", "root")}}
        if node.op == "fusion":
            inner_ids = {n: f"{mapping[node]}/{i}" for i, n in enumerate(node.attrs["body"])}
            item["body"] = [record(n, inner_ids) for n in node.attrs["body"]]
            item["root"] = inner_ids[node.attrs["root"]]
        return item

    return {"nodes": [record(n, ids) for n in nodes],
            "inputs": [ids[n] for n in graph.inputs],
            "outputs": [ids[n] for n in graph.outputs], "ir": str(graph),
            "operation_count": len(graph.nodes)}


def comparison(actual, expected) -> dict:
    return {"status": "pass" if np.allclose(actual, expected, rtol=RTOL, atol=ATOL) else "fail",
            "max_abs_error": float(np.max(np.abs(actual - expected)))}


def run_lesson(value) -> dict:
    config = validate_config(value)
    fn = PRESETS[config["example"]][0]
    rng = np.random.default_rng(0)
    m, k, n = (config[key] for key in ("m", "k", "n"))
    args = [rng.standard_normal((m, k), dtype=np.float32),
            rng.standard_normal((k, n), dtype=np.float32)]
    if config["example"] != "matmul":
        args.append(rng.standard_normal((n,), dtype=np.float32))
    gm = symbolic_trace(fn, *args)
    # tile_matmuls mutates its input, so even the no-fusion path needs a copy.
    optimized = copy.deepcopy(gm.graph)
    if config["fusion"] != "none":
        optimized = fuse(optimized, fuse_matmul=config["fusion"] == "full")
    tile_matmuls(optimized, (16, 16, 16) if config["tiled"] else None)
    expected = fn(*args)
    checks = {
        "python": comparison(gm(*args), expected),
        "interpreter": comparison(run(gm.graph, *args)[0], expected),
        "optimized_interpreter": comparison(run(optimized, *args)[0], expected),
    }
    simulation = None
    if config["backend"] == "simgpu":
        try:
            from .codegen_simgpu import _simulator
            sim = _simulator()
            compiled = compile_simgpu(optimized, config=sim.GPUConfig(max_cycles=50_000))
            result = compiled.run(*args, record=True, max_samples=100_000)
            checks["simgpu"] = comparison(result.outputs[0], expected)
            simulation = {
                "stats": vars(result.stats), "layout": compiled.layout,
                "output": result.outputs[0][:4, :4].tolist(),
                "kernels": [{"name": item.kernel.name, "node_id": item.kernel.node_id,
                             "shape": list(item.kernel.shape), "source": item.kernel.source,
                             "source_map": item.kernel.source_map,
                             "configuration": vars(item.kernel.config),
                             "report": item.report} for item in result.kernels],
            }
        except ImportError as exc:
            checks["simgpu"] = {"status": "unavailable", "message": str(exc)}
    elif not shutil.which("gcc"):
        checks["c"] = {"status": "unavailable", "message": "GCC를 찾지 못했습니다. 그래프와 Python 실행은 사용할 수 있습니다."}
    else:
        try:
            compiled = compile_graph(optimized)
            checks["c"] = comparison(compiled(*args)[0], expected)
        except (OSError, subprocess.CalledProcessError) as exc:
            checks["c"] = {"status": "unavailable", "message": f"C 컴파일 또는 로드 실패: {exc}"}

    try:
        symbolic_trace(data_branch, (m, k))
    except TraceError as exc:
        branch_error = str(exc)

    return {
        "config": config, "source": inspect.getsource(fn), "simulation": simulation,
        "raw": serialize_graph(gm.graph), "optimized": serialize_graph(optimized),
        "python_code": gm.code, "c_code": emit_c(optimized), "checks": checks,
        "output": {"shape": list(expected.shape), "dtype": "f32",
                   "preview": (result.outputs[0] if simulation is not None else expected)[:4, :4].tolist()},
        "tolerance": {"rtol": RTOL, "atol": ATOL}, "seed": 0,
        "branch": {"source": inspect.getsource(data_branch), "error": branch_error},
    }


def run_worker(config) -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "toy.web", "--worker"],
        input=json.dumps(config), capture_output=True, text=True, cwd=ROOT,
        timeout=WORKER_TIMEOUT,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-2000:] or "실행 프로세스가 종료되었습니다.")
    return json.loads(result.stdout)


class Handler(BaseHTTPRequestHandler):
    # Explicit asset routes keep repository files outside the web server.
    ASSETS = {"/": ("index.html", "text/html; charset=utf-8"),
              "/style.css": ("style.css", "text/css; charset=utf-8"),
              "/app.js": ("app.js", "text/javascript; charset=utf-8"),
              "/favicon.svg": ("favicon.svg", "image/svg+xml")}

    def send_body(self, status, data: bytes, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_json(self, status, value):
        self.send_body(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                       "application/json; charset=utf-8")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/examples":
            self.send_json(200, examples())
        elif path in self.ASSETS:
            name, mime = self.ASSETS[path]
            self.send_body(200, (WEB_ROOT / name).read_bytes(), mime)
        else:
            self.send_json(404, {"error": "페이지를 찾을 수 없습니다."})

    def do_POST(self):
        if urlsplit(self.path).path != "/api/run":
            self.send_json(404, {"error": "API를 찾을 수 없습니다."})
            return
        if self.headers.get_content_type() != "application/json":
            self.send_json(415, {"error": "application/json 요청이 필요합니다."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("요청 크기는 1–4096 bytes여야 합니다.")
            config = validate_config(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        try:
            self.send_json(200, run_worker(config))
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "실행 제한 시간(30초)을 초과했습니다. 다시 실행해 주세요."})
        except (RuntimeError, OSError, ValueError) as exc:
            self.send_json(500, {"error": f"실행 실패: {exc}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        json.dump(run_lesson(json.load(sys.stdin)), sys.stdout, ensure_ascii=False, allow_nan=False)
        return
    # A single request handler serializes compilation; each run gets fresh
    # process-local C scratch buffers and a bounded lifetime.
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        server = HTTPServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        parser.exit(1, f"Could not start localhost:{args.port}: {exc}. Try another --port.\n")
    print(f"Tensor to Silicon · http://127.0.0.1:{server.server_port} · Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
