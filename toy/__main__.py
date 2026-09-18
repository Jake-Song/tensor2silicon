"""Walk Y = ReLU(XW + b) through the toy compiler.  Run with: uv run -m toy"""

import numpy as np

from . import Graph, compile_graph, run


def build_relu_linear(M=16, K=8, N=4) -> Graph:
    g = Graph()
    x = g.input("x", (M, K))
    w = g.input("w", (K, N))
    b = g.input("b", (N,))
    xw = g.matmul(x, w)
    bb = g.broadcast_in_dim(b, (M, N), dims=(1,))
    y = g.relu(g.add(xw, bb))
    g.output(y)
    return g


def main() -> None:
    g = build_relu_linear()

    print("== 1. IR ==")
    print(g)

    compiled = compile_graph(g)
    print("\n== 2. generated C ==")
    print(compiled.source)

    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 8), dtype=np.float32)
    w = rng.standard_normal((8, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)

    (ref,) = run(g, x, w, b)
    (out,) = compiled(x, w, b)
    ok = np.allclose(ref, out, rtol=1e-5, atol=1e-6)
    print("== 3. interp vs C ==")
    print(f"max abs diff: {np.abs(ref - out).max():.3e}")
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
