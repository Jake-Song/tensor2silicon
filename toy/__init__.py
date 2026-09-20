"""toy: a minimal ML compiler.  Trace -> Graph IR -> fusion -> C codegen."""

from .codegen_c import Compiled, compile_graph, emit_c
from .interp import run
from .ir import F32, Graph, Node, ShapeError
from .passes import TILE_CANDIDATES, fuse, tile_matmuls
from .trace import Jitted, Tensor, TraceError, jit, relu, trace
from .graph_module import GraphModule, symbolic_trace

__all__ = ["F32", "Graph", "Node", "ShapeError", "run", "fuse", "tile_matmuls", "TILE_CANDIDATES", "jit", "Jitted", "trace", "Tensor", "TraceError", "relu", "emit_c", "compile_graph", "Compiled", "GraphModule", "symbolic_trace"]
