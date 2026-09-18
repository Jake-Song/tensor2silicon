"""Walk Y = ReLU(XW + b) through the toy compiler.  Run with: uv run -m toy"""

import numpy as np

from . import Graph, compile_graph, fuse, run


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
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 8), dtype=np.float32)
    w = rng.standard_normal((8, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)
    (ref,) = run(g, x, w, b)

    variants = [
        ("unfused", g),
        ("fused epilogue (matmul kept separate, Inductor-style)", fuse(g, fuse_matmul=False)),
        ("fully fused (XLA-style)", fuse(g)),
    ]
    all_ok = True
    for title, graph in variants:
        print(f"==== {title} ====")
        print("-- IR --")
        print(graph)
        compiled = compile_graph(graph)
        print("-- generated C --")
        print(compiled.source)
        (interp_out,) = run(graph, x, w, b)
        (c_out,) = compiled(x, w, b)
        ok = np.allclose(ref, interp_out) and np.allclose(ref, c_out, rtol=1e-5, atol=1e-6)
        all_ok &= ok
        print(f"-- check vs unfused interpreter: max abs diff {np.abs(ref - c_out).max():.3e} -> "
              f"{'PASS' if ok else 'FAIL'}\n")
    print("ALL PASS" if all_ok else "SOME FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
