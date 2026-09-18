"""NumPy reference interpreter: the oracle every code generator is checked against."""

from __future__ import annotations

import numpy as np

from .ir import Graph, Node

_NP_DTYPE = {"f32": np.float32}


def run(graph: Graph, *args: np.ndarray) -> list[np.ndarray]:
    """Evaluate ``graph`` on ``args`` (in ``graph.inputs`` order)."""
    if len(args) != len(graph.inputs):
        raise ValueError(f"expected {len(graph.inputs)} inputs, got {len(args)}")
    env: dict[Node, np.ndarray] = {}
    for node, arr in zip(graph.inputs, args):
        arr = np.asarray(arr, dtype=_NP_DTYPE[node.dtype])
        if arr.shape != node.shape:
            raise ValueError(f"input {node!r} got array of shape {arr.shape}")
        env[node] = arr
    for node in graph.nodes:
        env[node] = _eval(node, [env[i] for i in node.inputs])
    return [env[o] for o in graph.outputs]


def _eval(node: Node, xs: list[np.ndarray]) -> np.ndarray:
    match node.op:
        case "matmul":
            return xs[0] @ xs[1]
        case "add":
            return xs[0] + xs[1]
        case "relu":
            return np.maximum(xs[0], 0)
        case "fusion":
            body, root = node.attrs["body"], node.attrs["root"]
            env: dict[Node, np.ndarray] = {}
            for inner in body:
                if inner.op == "param":
                    env[inner] = xs[inner.attrs["index"]]
                else:
                    env[inner] = _eval(inner, [env[i] for i in inner.inputs])
            return env[root]
        case "broadcast_in_dim":
            shape, dims = node.attrs["shape"], node.attrs["dims"]
            # Insert size-1 axes for every target dim not mapped from the source,
            # then let NumPy repeat along them.
            x = xs[0]
            for d in range(len(shape)):
                if d not in dims:
                    x = np.expand_dims(x, d)
            return np.broadcast_to(x, shape)
        case _:
            raise NotImplementedError(node.op)
