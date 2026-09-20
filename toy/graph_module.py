"""FX-style inspection and execution of a shape-specialized toy graph.

Capture uses the existing abstract Tensor tracer.  Generated NumPy source is
the executable forward function, not a rendering of the original Python.
Graph editing/recompilation and code generation for fused graphs are out of scope.
"""

from __future__ import annotations

from typing import Callable

from .ir import F32, Graph
from .trace import _capture


class GraphModule:
    """A captured graph and its generated NumPy implementation.

    Calls accept positional array-like inputs, convert them to float32, and
    check their captured shapes.  ``graph`` is exposed for inspection and for
    compiler passes; editing it does not update the generated implementation.
    ``return_tuple`` defaults to whether the graph has other than one output.
    """

    def __init__(self, graph: Graph, *, return_tuple: bool | None = None):
        if return_tuple is None:
            return_tuple = len(graph.outputs) != 1
        if not return_tuple and len(graph.outputs) != 1:
            raise ValueError("a single-value return requires exactly one output")
        self.graph = graph
        self._code = _emit_python(graph, return_tuple)
        namespace = {}
        exec(compile(self._code, "<toy.GraphModule>", "exec"), namespace)
        self._forward = namespace["forward"]

    @property
    def code(self) -> str:
        """Standalone Python source defining the forward function."""
        return self._code

    def __call__(self, *args):
        return self._forward(*args)


def symbolic_trace(fn: Callable, *specs) -> GraphModule:
    """Capture ``fn`` with shape tuples or example arrays, without computing.

    Supports the same tensor operations as ``toy.trace`` and preserves tensor
    versus flat-tuple returns, including singleton and empty tuples.  Python
    side effects occur once while tracing.  Calls to the result execute only
    the captured graph, never the original function.
    """
    graph, return_tuple = _capture(fn, *specs)
    return GraphModule(graph, return_tuple=return_tuple)


def _emit_python(graph: Graph, return_tuple: bool) -> str:
    # Generated identifiers are independent of user names (including np,
    # forward, arg0, and names that are not valid Python identifiers).
    names = {node: f"arg{i}" for i, node in enumerate(graph.inputs)}
    lines = ["import numpy as np", "", f"def forward({', '.join(names.values())}):"]
    for node in [*graph.inputs, *graph.nodes]:
        if node.dtype != F32:
            raise TypeError(f"Python code generation only supports f32, got {node.dtype}")
    for node in graph.inputs:
        name = names[node]
        lines.append(f"    {name} = np.asarray({name}, dtype=np.float32)")
        lines.append(f"    if {name}.shape != {node.shape!r}:")
        message = f"input {node.name!r} expected shape {node.shape}, got "
        lines.append(f"        raise ValueError({message!r} + str({name}.shape))")

    for i, node in enumerate(graph.nodes):
        args = [names[n] for n in node.inputs]
        match node.op:
            case "matmul":
                expr = f"{args[0]} @ {args[1]}"
            case "add":
                expr = f"{args[0]} + {args[1]}"
            case "relu":
                expr = f"np.maximum({args[0]}, 0)"
            case "broadcast_in_dim":
                # Mirror the interpreter's axis insertion before broadcasting.
                axes = tuple(d for d in range(len(node.shape)) if d not in node.attrs["dims"])
                expr = f"np.broadcast_to(np.expand_dims({args[0]}, axis={axes!r}), {node.shape!r})"
            case _:
                raise NotImplementedError(f"Python code generation does not support {node.op!r}")
        names[node] = f"v{i}"
        lines.append(f"    v{i} = {expr}")

    outputs = [names[node] for node in graph.outputs]
    result = "(" + "".join(f"{name}, " for name in outputs) + ")" if return_tuple else outputs[0]
    lines.append(f"    return {result}")
    return "\n".join(lines) + "\n"
