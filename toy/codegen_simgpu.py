"""Lower the toy graph to inspectable SIMT assembly and execute it on simgpu.

Each output element belongs to one thread. Matmul reductions are sequential;
fusion keeps the epilogue in registers. Memory addresses are word addresses.
The optional simulator is imported only when this backend is requested.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from .ir import Graph, Node

if TYPE_CHECKING:
    from simgpu import GPUConfig, Stats
    from simgpu.isa import Instruction


class SimGPUCompileError(ValueError):
    """The graph cannot be represented by this simple simulator backend."""


def _simulator():
    try:
        import simgpu
        from simgpu.isa import OPCODES
        if not {"fadd", "fmul", "fmax"} <= OPCODES.keys() or not hasattr(simgpu, "report_data"):
            raise ImportError("float32 instructions or report_data are missing")
        return simgpu
    except ImportError as exc:
        raise ImportError(
            "The simgpu backend needs the float32-enabled sim-gpu checkout. "
            "Run with: uv run --with-editable ../sim-gpu python -m toy.web --port 8010"
        ) from exc


@dataclass
class Kernel:
    name: str
    node_id: str
    shape: tuple[int, ...]
    source: str
    program: list[Instruction]
    source_map: list[str]  # instruction PC -> graph node ID (including fusion body)
    config: GPUConfig


@dataclass
class KernelRun:
    kernel: Kernel
    stats: Stats
    report: dict | None


@dataclass
class SimulationRun:
    outputs: list[np.ndarray]
    kernels: list[KernelRun]
    stats: Stats  # sums of the sequential launches, without modeled launch overhead


class _Emitter:
    def __init__(self, node, buffers, ids):
        self.node = node
        self.buffers = buffers
        self.ids = ids
        self.lines = [f"; {node.name}: {node.op} {node.shape}; one thread per output element"]
        self.line_nodes = {}
        self.free = list(reversed(range(1, 16)))  # r0 always holds the global index
        self.label_count = 0
        self.internal = {node}
        self.params = []
        if node.op == "fusion":
            self.internal = set(node.attrs["body"])
            self.params = node.inputs

    def reg(self):
        if not self.free:
            raise SimGPUCompileError(
                f"{self.node.name}: register exhaustion (16 registers); disable fusion or simplify the graph"
            )
        return f"r{self.free.pop()}"

    def release(self, *regs):
        for reg in regs:
            assert reg != "r0"
            self.free.append(int(reg[1:]))

    def emit(self, text, node=None):
        self.lines.append("    " + text)
        self.line_nodes[len(self.lines)] = self.ids[node or self.node]

    def label(self, stem):
        self.label_count += 1
        return f"{stem}_{self.label_count}"

    def mark(self, label):
        self.lines.append(label + ":")

    def load(self, node, idx, owner):
        address = self.reg()
        self.emit(f"mov {address}, {self.buffers[node]}", owner)
        for coordinate, stride in zip(idx, node.strides):
            if stride == 1:
                self.emit(f"add {address}, {address}, {coordinate}", owner)
            else:
                term = self.reg()
                self.emit(f"mul {term}, {coordinate}, {stride}", owner)
                self.emit(f"add {address}, {address}, {term}", owner)
                self.release(term)
        # Reuse the address register for the loaded word.
        self.emit(f"ld {address}, [{address}]", owner)
        return address

    def expr(self, node, idx):
        if node not in self.internal:
            return self.load(node, idx, self.node)
        if node.op == "param":
            return self.load(self.params[node.attrs["index"]], idx, node)
        if node.op == "broadcast_in_dim":
            return self.expr(node.inputs[0], tuple(idx[d] for d in node.attrs["dims"]))
        if node.op == "relu":
            value = self.expr(node.inputs[0], idx)
            self.emit(f"fmax {value}, {value}, 0", node)
            return value
        if node.op == "add":
            a = self.expr(node.inputs[0], idx)
            b = self.expr(node.inputs[1], idx)
            self.emit(f"fadd {a}, {a}, {b}", node)
            self.release(b)
            return a
        if node.op == "matmul":
            acc, k = self.reg(), self.reg()
            loop, done = self.label("reduce"), self.label("reduced")
            self.emit(f"mov {acc}, 0", node)
            self.emit(f"mov {k}, 0", node)
            self.mark(loop)
            pred = self.reg()
            self.emit(f"slt {pred}, {k}, {node.inputs[0].shape[1]}", node)
            self.emit(f"bz {pred}, {done}", node)
            self.release(pred)
            a = self.expr(node.inputs[0], (idx[0], k))
            b = self.expr(node.inputs[1], (k, idx[1]))
            self.emit(f"fmul {a}, {a}, {b}", node)
            self.emit(f"fadd {acc}, {acc}, {a}", node)
            self.release(a, b)
            self.emit(f"add {k}, {k}, 1", node)
            self.emit(f"jmp {loop}", node)
            self.mark(done)
            self.release(k)
            return acc
        raise SimGPUCompileError(f"unsupported operation: {node.op}")

    def generate(self):
        self.emit("mul r0, %bid, %bdim")
        self.emit("add r0, r0, %tid")
        pred = self.reg()
        self.emit(f"slt {pred}, r0, {self.node.size}")
        self.emit(f"bz {pred}, exit")
        self.release(pred)
        idx = []
        owned = []
        for dim, stride in zip(self.node.shape, self.node.strides):
            if len(self.node.shape) == 1:
                idx.append("r0")
                continue
            coordinate = self.reg()
            self.emit(f"div {coordinate}, r0, {stride}")
            self.emit(f"rem {coordinate}, {coordinate}, {dim}")
            idx.append(coordinate)
            owned.append(coordinate)
        root = self.node.attrs["root"] if self.node.op == "fusion" else self.node
        value = self.expr(root, tuple(idx))
        address = self.reg()
        self.emit(f"add {address}, r0, {self.buffers[self.node]}")
        self.emit(f"st [{address}], {value}")
        self.release(address, value, *owned)
        self.mark("exit")
        self.emit("ret")
        assert len(self.free) == 15
        return "\n".join(self.lines) + "\n"


def _validate(graph: Graph) -> None:
    supported = {"input", "param", "matmul", "add", "relu", "broadcast_in_dim", "fusion"}
    known = set(graph.inputs)
    for outer in [*graph.inputs, *graph.nodes]:
        if outer in graph.nodes and any(n not in known for n in outer.inputs):
            raise SimGPUCompileError("graph must be in topological order")
        known.add(outer)
        for node in [outer, *outer.attrs.get("body", [])]:
            if node.dtype != "f32":
                raise SimGPUCompileError(f"only f32 tensors are supported, got {node.dtype}")
            if node.op not in supported or (node is not outer and node.op == "fusion"):
                raise SimGPUCompileError(f"unsupported operation: {node.op}")
            if any(type(d) is not int or d < 0 for d in node.shape):
                raise SimGPUCompileError(f"invalid shape: {node.shape}")
            if "tile" in node.attrs:
                raise SimGPUCompileError("CPU tile schedules are not supported by simgpu; use tile=None")
    if any(n not in known for n in graph.outputs):
        raise SimGPUCompileError("output does not belong to the graph")


class CompiledSimGPU:
    def __init__(self, graph: Graph, config: GPUConfig | None = None):
        sim = _simulator()
        self.graph = copy.deepcopy(graph)
        _validate(self.graph)
        self.config = replace(config) if config is not None else sim.GPUConfig()
        self.buffers = {}
        self.layout = []
        nodes = [*self.graph.inputs, *self.graph.nodes]
        ids = {node: f"n{i}" for i, node in enumerate(nodes)}
        cursor = 0
        alignment = max(1, self.config.coalesce_width)
        for node in nodes:
            for i, inner in enumerate(node.attrs.get("body", [])):
                ids[inner] = f"{ids[node]}/{i}"
            cursor = (cursor + alignment - 1) // alignment * alignment
            self.buffers[node] = cursor
            self.layout.append(dict(node_id=ids[node], name=node.name, address=cursor,
                                    words=node.size, shape=list(node.shape), dtype="f32"))
            cursor += node.size
        if cursor >= 2**31:
            raise SimGPUCompileError("buffer addresses exceed signed 32-bit indexing")
        self.config = replace(self.config, mem_size=max(self.config.mem_size, cursor, 1))
        self.kernels = []
        for node in self.graph.nodes:
            if not node.size:
                continue
            emitter = _Emitter(node, self.buffers, ids)
            source = emitter.generate()
            program = sim.assemble(source)
            cfg = replace(self.config, num_blocks=(node.size + self.config.block_size - 1) // self.config.block_size)
            self.kernels.append(Kernel(node.name, ids[node], node.shape, source, program,
                                       [emitter.line_nodes[i.line] for i in program], cfg))
        self.source = "\n".join(k.source for k in self.kernels)

    def __call__(self, *args) -> list[np.ndarray]:
        return self.run(*args).outputs

    def run(self, *args, record=False, max_samples=100_000) -> SimulationRun:
        sim = _simulator()
        if len(args) != len(self.graph.inputs):
            raise ValueError(f"expected {len(self.graph.inputs)} inputs, got {len(args)}")
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError("max_samples must be a positive integer")
        gpu = sim.GPU(replace(self.config))
        for node, value in zip(self.graph.inputs, args):
            arr = np.asarray(value, dtype=np.float32, order="C")
            if arr.shape != node.shape:
                raise ValueError(f"input {node!r} got array of shape {arr.shape}")
            gpu.mem.write(self.buffers[node], arr.view(np.int32).reshape(-1).tolist())
        total, runs, sample_count = sim.Stats(), [], 0
        for kernel in self.kernels:
            recorder = sim.Recorder()

            def sample(value):
                nonlocal sample_count
                if sample_count >= max_samples:
                    raise sim.SimError(f"Recording exceeded {max_samples} warp samples; use smaller shapes")
                sample_count += 1
                recorder(value)

            gpu.config = replace(kernel.config)
            gpu.sample = sample if record else None
            stats = gpu.run(kernel.program)
            report = sim.report_data(recorder.samples, kernel.program, stats, kernel.config, kernel.name) if record else None
            runs.append(KernelRun(kernel, stats, report))
            for name, value in vars(stats).items():
                setattr(total, name, getattr(total, name) + value)
        outputs = []
        for node in self.graph.outputs:
            words = gpu.mem.read(self.buffers[node], node.size) if node.size else []
            outputs.append(np.array(words, dtype=np.int32).view(np.float32).reshape(node.shape))
        return SimulationRun(outputs, runs, total)


def compile_simgpu(graph: Graph, *, config: GPUConfig | None = None) -> CompiledSimGPU:
    """Compile f32 graph IR. Grid size and minimum memory capacity are inferred.

    ``config`` is an optional simgpu.GPUConfig; its other hardware/cycle settings
    are honored. The caller's graph and configuration are never modified.
    """
    return CompiledSimGPU(graph, config)
