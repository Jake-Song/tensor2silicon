"""toy: a minimal ML compiler.  Graph IR -> NumPy interpreter -> C codegen."""

from .codegen_c import Compiled, compile_graph, emit_c
from .interp import run
from .ir import F32, Graph, Node, ShapeError
from .passes import fuse

__all__ = ["F32", "Graph", "Node", "ShapeError", "run", "fuse", "emit_c", "compile_graph", "Compiled"]
