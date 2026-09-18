"""Graph-to-graph optimization passes.

``fuse`` is the classic producer-into-consumer fusion: an elementwise node
absorbs each of its producers that has no other consumer, so a chain like
``relu(add(matmul(x, w), broadcast(b)))`` becomes one ``fusion`` node and,
downstream, one loop nest with no intermediate buffers.
"""

from __future__ import annotations

from .ir import Graph, Node, param

# Ops that may absorb a producer into their loop nest.
_CONSUMERS = {"add", "relu"}
# Ops that are cheap to recompute per output element and so may be absorbed.
_ELEMENTWISE = {"add", "relu", "broadcast_in_dim"}


def fuse(graph: Graph, *, fuse_matmul: bool = True) -> Graph:
    """Return a new graph with fusible chains collapsed into ``fusion`` nodes.

    With ``fuse_matmul=False`` the matmul stays a separate kernel and only the
    elementwise epilogue is fused (Inductor's ``addmm`` + pointwise kernel).
    With ``fuse_matmul=True`` the matmul is pulled in as well, giving a single
    loop nest with the epilogue applied to the accumulator (XLA-style).
    """
    producers_ok = _ELEMENTWISE | ({"matmul"} if fuse_matmul else set())

    consumers: dict[Node, list[Node]] = {n: [] for n in graph.inputs + graph.nodes}
    for n in graph.nodes:
        for i in n.inputs:
            consumers[i].append(n)
    outputs = set(graph.outputs)

    # group[n] is the list of original nodes fused together; the last is the root.
    group: dict[Node, list[Node]] = {}
    for n in graph.nodes:
        members: list[Node] = []
        if n.op in _CONSUMERS:
            for p in dict.fromkeys(n.inputs):  # each producer once
                if (
                    p.op in producers_ok
                    and p not in outputs
                    and set(consumers[p]) == {n}  # sole consumer (possibly via several operands)
                    and p in group
                ):
                    members.extend(group.pop(p))
        members.append(n)
        group[n] = members

    return _rebuild(graph, group)


def _rebuild(graph: Graph, group: dict[Node, list[Node]]) -> Graph:
    new = Graph()
    old2new: dict[Node, Node] = {}
    for n in graph.inputs:
        old2new[n] = new.input(n.name, n.shape, n.dtype)

    for root in graph.nodes:
        if root not in group:  # absorbed into a later root
            continue
        members = group[root]
        if len(members) == 1:
            old2new[root] = _clone(new, root, old2new)
            continue

        inside = set(members)
        ext_inputs: list[Node] = []  # outer nodes the body reads, in first-use order
        inner: dict[Node, Node] = {}
        body: list[Node] = []
        for m in members:
            args = []
            for i in m.inputs:
                if i in inside:
                    args.append(inner[i])
                else:
                    if i not in ext_inputs:
                        ext_inputs.append(i)
                    k = ext_inputs.index(i)
                    leaf = next((b for b in body if b.op == "param" and b.attrs["index"] == k), None)
                    if leaf is None:
                        leaf = param(k, i)
                        body.append(leaf)
                    args.append(leaf)
            node = Node(m.op, args, m.shape, m.dtype, dict(m.attrs), name=f"v{len(body)}")
            inner[m] = node
            body.append(node)
        old2new[root] = new.fusion([old2new[i] for i in ext_inputs], body, inner[root])

    for o in graph.outputs:
        new.output(old2new[o])
    return new


def _clone(new: Graph, n: Node, old2new: dict[Node, Node]) -> Node:
    args = [old2new[i] for i in n.inputs]
    match n.op:
        case "matmul":
            return new.matmul(*args)
        case "add":
            return new.add(*args)
        case "relu":
            return new.relu(*args)
        case "broadcast_in_dim":
            return new.broadcast_in_dim(args[0], n.attrs["shape"], n.attrs["dims"])
        case "fusion":
            return new.fusion(args, n.attrs["body"], n.attrs["root"])
        case _:
            raise NotImplementedError(n.op)
