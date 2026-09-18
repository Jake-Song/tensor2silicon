"""toy: a minimal ML compiler.  Graph IR -> NumPy interpreter -> C codegen."""

from .codegen_c import Compiled, compile_graph, emit_c
from .interp import run
from .ir import F32, Graph, Node, ShapeError

__all__ = ["F32", "Graph", "Node", "ShapeError", "run", "emit_c", "compile_graph", "Compiled"]
