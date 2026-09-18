"""Tracing front end: run a Python function on abstract ``Tensor``s to record a ``Graph``.

This is the toy version of ``jax.jit`` tracing / TorchDynamo.  A ``Tensor``
carries only a shape and a dtype; every operator on it appends a node to the
graph being traced instead of computing anything.  ``jit`` wraps a function so
that the first call for a given input signature traces, fuses and compiles it,
and later calls with the same shapes hit the cache.
"""

from __future__ import annotations

import inspect
import time
from typing import Callable

import numpy as np

from .codegen_c import Compiled, compile_graph
from .ir import F32, Graph, Node
from .passes import TILE_CANDIDATES, Tile, fuse as fuse_pass, tile_matmuls


class TraceError(RuntimeError):
    pass


class Tensor:
    """Abstract value seen by the traced function.  Records ops, never computes."""

    __array_priority__ = 1000  # so ndarray + Tensor dispatches to Tensor.__radd__

    def __init__(self, graph: Graph, node: Node):
        self._graph = graph
        self._node = node

    shape = property(lambda self: self._node.shape)
    dtype = property(lambda self: self._node.dtype)
    ndim = property(lambda self: len(self._node.shape))

    def __repr__(self) -> str:
        return f"Tensor(traced {self._node!r})"

    def __matmul__(self, other: Tensor) -> Tensor:
        other = self._check(other, "@")
        return Tensor(self._graph, self._graph.matmul(self._node, other._node))

    def __add__(self, other: Tensor) -> Tensor:
        other = self._check(other, "+")
        a, b = _broadcast_pair(self._graph, self._node, other._node)
        return Tensor(self._graph, self._graph.add(a, b))

    def __radd__(self, other: Tensor) -> Tensor:
        return self._check(other, "+").__add__(self)

    def __bool__(self):
        raise TraceError(
            f"cannot branch on a traced value {self!r}: its contents are unknown at "
            "trace time (this is what a Dynamo graph break / JAX ConcretizationError is)"
        )

    def __iter__(self):
        raise TraceError(f"cannot iterate over a traced value {self!r}")

    def _check(self, other, op: str) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError(
                f"unsupported operand for {op}: Tensor and {type(other).__name__} "
                "(constants are not supported by the toy IR yet)"
            )
        if other._graph is not self._graph:
            raise TraceError("operands come from different traces")
        return other


def relu(x):
    """Works both eagerly on ndarrays and under tracing on Tensors."""
    if isinstance(x, Tensor):
        return Tensor(x._graph, x._graph.relu(x._node))
    return np.maximum(x, 0)


def _broadcast_pair(graph: Graph, a: Node, b: Node) -> tuple[Node, Node]:
    """NumPy-style rank promotion: align the lower-rank operand to trailing dims."""
    if a.shape == b.shape:
        return a, b
    if len(a.shape) < len(b.shape):
        return _broadcast_to(graph, a, b.shape), b
    if len(b.shape) < len(a.shape):
        return a, _broadcast_to(graph, b, a.shape)
    raise TypeError(f"cannot broadcast {a!r} with {b!r} (size-1 expansion not supported)")


def _broadcast_to(graph: Graph, n: Node, shape: tuple[int, ...]) -> Node:
    offset = len(shape) - len(n.shape)
    dims = tuple(range(offset, len(shape)))
    return graph.broadcast_in_dim(n, shape, dims)  # raises ShapeError on size mismatch


# -- trace ------------------------------------------------------------------


def trace(fn: Callable, *specs) -> Graph:
    """Record ``fn`` into a Graph.  Each spec is a shape tuple or an array-like."""
    names = list(inspect.signature(fn).parameters)
    if len(names) != len(specs):
        raise TypeError(f"{fn.__name__} takes {len(names)} args, got {len(specs)} specs")
    graph = Graph()
    args = []
    for name, spec in zip(names, specs):
        shape = tuple(spec) if isinstance(spec, tuple) else tuple(np.shape(spec))
        args.append(Tensor(graph, graph.input(name, shape, F32)))
    result = fn(*args)
    outs = result if isinstance(result, tuple) else (result,)
    for o in outs:
        if not isinstance(o, Tensor):
            raise TraceError(f"{fn.__name__} returned a non-Tensor {type(o).__name__}")
        graph.output(o._node)
    return graph


# -- jit --------------------------------------------------------------------


class Jitted:
    """``tile``: ``None`` for naive loops, a ``(BM, BN, BK)`` block, or ``"auto"``
    to time each of ``TILE_CANDIDATES`` on the real inputs and keep the fastest
    (a toy ``max-autotune``).  ``self.tuning`` records what was measured."""

    def __init__(
        self,
        fn: Callable,
        *,
        fuse: bool = True,
        fuse_matmul: bool = True,
        tile: Tile | str | None = None,
    ):
        self.fn = fn
        self.fuse = fuse
        self.fuse_matmul = fuse_matmul
        self.tile = tile
        self.cache: dict[tuple, Compiled] = {}
        self.tuning: dict[tuple, dict[Tile | None, float]] = {}
        self.__name__ = getattr(fn, "__name__", "jitted")
        self.__doc__ = fn.__doc__

    @staticmethod
    def _key(args) -> tuple:
        return tuple((np.shape(a), str(np.asarray(a).dtype)) for a in args)

    def trace(self, *args) -> Graph:
        """The raw traced graph, before any pass (like ``jax.make_jaxpr``)."""
        return trace(self.fn, *args)

    def lower(self, *args, tile: Tile | str | None = "default") -> Graph:
        """The graph after passes (like ``jit(f).lower(...)``).  With
        ``tile="auto"`` this returns the unscheduled graph; scheduling happens
        in ``compile``."""
        g = self.trace(*args)
        if self.fuse:
            g = fuse_pass(g, fuse_matmul=self.fuse_matmul)
        tile = self.tile if tile == "default" else tile
        return tile_matmuls(g, None if tile == "auto" else tile)

    def compile(self, *args) -> Compiled:
        key = self._key(args)
        if key not in self.cache:
            if self.tile == "auto":
                self.cache[key] = self._autotune(key, args)
            else:
                self.cache[key] = compile_graph(self.lower(*args))
        return self.cache[key]

    def _autotune(self, key: tuple, args, repeats: int = 3) -> Compiled:
        timings: dict[Tile | None, float] = {}
        best = None
        for tile in [None, *TILE_CANDIDATES]:
            c = compile_graph(self.lower(*args, tile=tile))
            c(*args)  # warm-up
            t = min(_time_call(c, args) for _ in range(repeats))
            timings[tile] = t
            if best is None or t < timings[best[0]]:
                best = (tile, c)
        self.tuning[key] = timings
        return best[1]

    def __call__(self, *args):
        outs = self.compile(*args)(*args)
        return outs[0] if len(outs) == 1 else tuple(outs)


def jit(fn: Callable | None = None, /, **opts) -> Jitted | Callable[[Callable], Jitted]:
    """Decorator: ``@jit`` or ``@jit(fuse_matmul=False)``."""
    if fn is None:
        return lambda f: Jitted(f, **opts)
    return Jitted(fn, **opts)


def _time_call(fn, args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0
