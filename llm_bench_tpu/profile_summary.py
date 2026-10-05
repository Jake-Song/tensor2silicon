"""Summarize TPU XLA Ops intervals without adding nested/overlapped durations."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics


def category(event):
    name = event.get("name", "")
    for text, label in (("native_gemm", "GEMM"), ("native_kv", "KV write"),
                        ("pallas_attention", "Attention"), ("native_layernorm", "LayerNorm"),
                        ("native_rope", "RoPE/layout"), ("native_split", "RoPE/layout"),
                        ("native_merge", "RoPE/layout"), ("native_residual", "Residual"),
                        ("native_swiglu", "SwiGLU"), ("native_embed", "Embedding")):
        if text in name:
            return label
    return "XLA/" + event.get("args", {}).get("hlo_category", "other")


def exclusive_times(events):
    """A custom kernel owns its nested intervals; other intervals use the leaf.

    Only one XLA Ops track is passed here. Async/DMA tracks are never added.
    The sweep attributes each occupied microsecond to exactly one category.
    """
    boundaries = []
    for i, event in enumerate(events):
        if event.get("dur", 0) > 0:
            boundaries.extend(((event["ts"], 1, i), (event["ts"] + event["dur"], -1, i)))
    active, totals = set(), defaultdict(float)
    last = 0.0
    for position, action, idx in sorted(boundaries):
        if active and position > last:
            custom = [i for i in active if not category(events[i]).startswith("XLA/")]
            owner = min(custom or active, key=lambda i: events[i]["dur"])
            totals[category(events[owner])] += position - last
        if action == 1:
            active.add(idx)
        else:
            active.remove(idx)
        last = position
    return dict(totals)


def summarize_trace(filename):
    data = json.loads(gzip.decompress(Path(filename).read_bytes()))
    events = data["traceEvents"]
    processes, threads = {}, {}
    for event in events:
        if event.get("ph") != "M":
            continue
        if event["name"] == "process_name":
            processes[event["pid"]] = event["args"]["name"]
        if event["name"] == "thread_name":
            threads[event["pid"], event["tid"]] = event["args"]["name"]
    pids = [pid for pid, name in processes.items() if "/device:TPU:" in name]
    if len(pids) != 1:
        raise ValueError(f"expected one TPU device track, found {pids}")
    pid = pids[0]
    ops, modules = [], []
    for event in events:
        if event.get("ph") != "X" or event.get("pid") != pid:
            continue
        track = threads.get((pid, event.get("tid")))
        if track == "XLA Ops":
            ops.append(event)
        elif track == "XLA Modules":
            modules.append(event)
    if not ops or not modules:
        raise ValueError("TPU XLA Ops/Modules events are missing")
    steps = len(modules)
    totals = exclusive_times(ops)
    counts = Counter(category(e) for e in ops)
    per_op = defaultdict(lambda: {"count": 0, "inclusive_us": 0, "example_hlo": ""})
    for event in ops:
        name = re.sub(r"\.\d+$", "", event["name"])
        if category(event) == "GEMM":
            name = event["name"].split(".")[0]
            dims = re.findall(r"bf16\[([0-9,]+)\]", event.get("args", {}).get("long_name", ""))
            if len(dims) >= 3:
                name += " " + " @ ".join(dims[1:3])
        item = per_op[name]
        item["count"] += 1
        item["inclusive_us"] += event["dur"]
        item["example_hlo"] = event.get("args", {}).get("long_name", "")
    return {"file": str(filename), "device_events_verified": True, "device": processes[pid],
            "steps": steps, "device_module_mean_ms": sum(e["dur"] for e in modules) / steps / 1000,
            "device_module_median_ms": statistics.median(e["dur"] for e in modules) / 1000,
            "categories": {k: {"exclusive_ms_per_step": v / steps / 1000,
                                "events_per_step": counts[k] / steps} for k, v in
                           sorted(totals.items(), key=lambda p: -p[1])},
            "xla_ops_union_ms_per_step": sum(totals.values()) / steps / 1000,
            "method": "Union of TPU XLA Ops intervals; Pallas parent owns nested time; otherwise leaf. "
                      "Do not add inclusive op totals, async ops, DMA, module or host tracks.",
            "ops_inclusive": dict(sorted(per_op.items(), key=lambda p: -p[1]["inclusive_us"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = {}
    for path in sorted(Path(args.folder).rglob("perfetto_trace.json.gz")):
        key = str(path.relative_to(args.folder)).split("/plugins/")[0]
        row = summarize_trace(path)
        result[key] = row
        print(key, json.dumps({k: v for k, v in row.items() if k != "ops_inclusive"}), flush=True)
    if not result:
        raise ValueError("no Perfetto traces found")
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
