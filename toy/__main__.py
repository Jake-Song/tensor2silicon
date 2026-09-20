"""Walk Y = ReLU(XW + b) through the toy compiler.  Run with: uv run -m toy"""

import numpy as np

import toy
from toy import TraceError, jit, run


@jit
def f(x, w, b):
    return toy.relu(x @ w + b)


def main() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 8), dtype=np.float32)
    w = rng.standard_normal((8, 4), dtype=np.float32)
    b = rng.standard_normal((4,), dtype=np.float32)

    # Eager: the same Python function runs directly on NumPy arrays.
    ref = f.fn(x, w, b)

    print("==== 1. trace -> IR (like jax.make_jaxpr / FX graph) ====")
    g = f.trace(x, w, b)
    print(g)

    print("\n==== 2. fusion pass (like XLA HLO fusion) ====")
    print(f.lower(x, w, b))

    print("\n==== 3. codegen -> C (like XLA CPU / Inductor C++) ====")
    compiled = f.compile(x, w, b)
    print(compiled.source)

    print("==== 4. run and check ====")
    y = f(x, w, b)  # cache hit: no retrace
    (interp,) = run(g, x, w, b)
    ok = np.allclose(ref, interp) and np.allclose(ref, y, rtol=1e-5, atol=1e-6)
    print(f"eager vs interp vs C: max abs diff {np.abs(ref - y).max():.3e} -> {'PASS' if ok else 'FAIL'}")

    print("\n==== 5. shape specialization: a new shape retraces and recompiles ====")
    x2 = rng.standard_normal((32, 8), dtype=np.float32)
    f(x2, w, b)
    for k in f.cache:
        print("  cache key:", [s for s, _ in k])

    print("\n==== 6. graph break: Python control flow on a traced value ====")

    @jit
    def g_bad(x):
        if x:  # would need the value at trace time
            return toy.relu(x)
        return x

    try:
        g_bad(x)
    except TraceError as e:
        print("  TraceError:", e)

    print("\n==== 7. FX-style extraction: graph, table, and executable Python ====")
    gm = toy.symbolic_trace(f.fn, x, w, b)
    gm.graph.print_tabular()
    print(gm.code)
    graph_ok = np.allclose(gm(x, w, b), ref, rtol=1e-5, atol=1e-6)
    print(f"GraphModule vs eager: {'PASS' if graph_ok else 'FAIL'}")
    ok = ok and graph_ok

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
