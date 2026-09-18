"""Graph IR with shape inference.

A ``Graph`` is a list of ``Node``s in topological (append) order, the toy
equivalent of a jaxpr or an FX graph.  Every builder method infers the output
shape and raises ``ShapeError`` on mismatch, so a graph is always well-formed
by construction.

Supported ops: ``input``, ``matmul`` (2-D), ``add`` (identical shapes),
``relu``, ``broadcast_in_dim``.  Like StableHLO, broadcasting is explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

F32 = "f32"


class ShapeError(ValueError):
    pass


@dataclass(eq=False)
class Node:
    op: str
    inputs: list[Node]
    shape: tuple[int, ...]
    dtype: str = F32
    attrs: dict = field(default_factory=dict)
    name: str = ""

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def strides(self) -> tuple[int, ...]:
        """Row-major element strides."""
        out, acc = [], 1
        for d in reversed(self.shape):
            out.append(acc)
            acc *= d
        return tuple(reversed(out))

    def type_str(self) -> str:
        return f"{self.dtype}[{','.join(map(str, self.shape))}]"

    def __repr__(self) -> str:
        return f"{self.name}:{self.type_str()}"


class Graph:
    def __init__(self) -> None:
        self.inputs: list[Node] = []
        self.nodes: list[Node] = []  # non-input nodes, topological order
        self.outputs: list[Node] = []

    # -- builders -----------------------------------------------------------

    def input(self, name: str, shape: tuple[int, ...], dtype: str = F32) -> Node:
        n = Node("input", [], tuple(shape), dtype, name=name)
        self.inputs.append(n)
        return n

    def matmul(self, a: Node, b: Node) -> Node:
        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ShapeError(f"matmul needs 2-D operands, got {a!r} and {b!r}")
        if a.shape[1] != b.shape[0]:
            raise ShapeError(
                f"matmul inner dims differ: {a!r} @ {b!r} ({a.shape[1]} != {b.shape[0]})"
            )
        return self._add_node("matmul", [a, b], (a.shape[0], b.shape[1]))

    def add(self, a: Node, b: Node) -> Node:
        if a.shape != b.shape:
            raise ShapeError(
                f"add needs identical shapes (use broadcast_in_dim first): {a!r} + {b!r}"
            )
        return self._add_node("add", [a, b], a.shape)

    def relu(self, a: Node) -> Node:
        return self._add_node("relu", [a], a.shape)

    def broadcast_in_dim(
        self, a: Node, shape: tuple[int, ...], dims: tuple[int, ...]
    ) -> Node:
        """Map source dim ``i`` to target dim ``dims[i]``; other target dims repeat."""
        shape, dims = tuple(shape), tuple(dims)
        if len(dims) != len(a.shape):
            raise ShapeError(f"broadcast_in_dim: {a!r} has {len(a.shape)} dims, dims={dims}")
        for i, d in enumerate(dims):
            if not 0 <= d < len(shape):
                raise ShapeError(f"broadcast_in_dim: dim {d} out of range for shape {shape}")
            if a.shape[i] != shape[d]:
                raise ShapeError(
                    f"broadcast_in_dim: source dim {i} (size {a.shape[i]}) "
                    f"!= target dim {d} (size {shape[d]})"
                )
        return self._add_node("broadcast_in_dim", [a], shape, {"shape": shape, "dims": dims})

    def output(self, node: Node) -> Node:
        self.outputs.append(node)
        return node

    def _add_node(self, op, inputs, shape, attrs=None) -> Node:
        n = Node(op, list(inputs), tuple(shape), inputs[0].dtype, attrs or {})
        n.name = f"%{len(self.nodes)}"
        self.nodes.append(n)
        return n

    # -- printing -----------------------------------------------------------

    def __str__(self) -> str:
        head = " ".join(repr(n) for n in self.inputs)
        lines = [f"{{ lambda ; {head}. let"]
        for n in self.nodes:
            attrs = ""
            if n.attrs:
                attrs = "[" + ", ".join(f"{k}={v}" for k, v in n.attrs.items()) + "]"
            args = " ".join(i.name for i in n.inputs)
            lines.append(f"    {n!r} = {n.op}{attrs} {args}")
        outs = ", ".join(n.name for n in self.outputs)
        lines.append(f"  in ({outs}) }}")
        return "\n".join(lines)
