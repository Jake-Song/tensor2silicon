"""Audit existing G4 PyTorch traces; never imports torch or allocates a GPU.

Run from any directory: python llm_bench/profiling/analyze.py
Add --plot with an interpreter that has matplotlib to regenerate the figure.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re


REPO = Path(__file__).resolve().parents[2]
DATASETS = ("2026-10-04-g4", "2026-10-10-g4-cuda")
LEAF_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_name(name):
    # These are name-based groups, not measured per-operation contributions.
    has_mm = bool(re.search(r"(?:^|_)mm(?:_|$)", name))
    if "flex_attention" in name:
        return "FlexAttention + GEMM (mixed)" if has_mm else "FlexAttention (named)"
    if "pytorch_flash::" in name:
        return "FlashAttention"
    if has_mm or "cutlass::" in name:
        return "GEMM-containing (named)"
    return "Other kernels / copies"


def intervals_union(intervals):
    intervals = sorted(intervals)
    start, end = intervals[0]
    occupied = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            occupied += end - start
            start, end = next_start, next_end
    return occupied + end - start


def profile(trace_path, recorded):
    trace = json.loads(trace_path.read_text())
    events = [e for e in trace["traceEvents"]
              if e.get("ph") == "X" and e.get("cat") in LEAF_CATEGORIES]
    if not events:
        raise ValueError(f"No GPU leaf events: {trace_path}")
    devices = {e.get("args", {}).get("device", e.get("pid")) for e in events}
    if len(devices) != 1:
        raise ValueError(f"Multiple devices need separate timelines: {trace_path}")
    parents = collections.defaultdict(list)
    for event in trace["traceEvents"]:
        if event.get("cat") == "cpu_op" and event.get("ph") == "X":
            parents[event.get("args", {}).get("External id")].append(event)
    grouped = collections.defaultdict(list)
    for event in events:
        if not math.isfinite(event["dur"]) or event["dur"] <= 0:
            raise ValueError(f"Invalid duration: {trace_path}")
        grouped[event["name"]].append(event)
    total = math.fsum(e["dur"] for e in events)
    rows = []
    for name, items in grouped.items():
        duration = math.fsum(e["dur"] for e in items)
        saved = recorded[name]
        if len(items) != saved["calls"] or not math.isclose(duration, saved["total_us"], abs_tol=1e-6):
            raise ValueError(f"Trace/JSON mismatch: {trace_path}: {name}")
        sample = items[0]
        cpu = parents.get(sample.get("args", {}).get("External id"), [])
        resources = {key: sample.get("args", {}).get(key) for key in
                     ("grid", "block", "registers per thread", "shared memory",
                      "est. achieved occupancy %")}
        shape_events = collections.defaultdict(list)
        for event in items:
            matched = parents.get(event.get("args", {}).get("External id"), [])
            # Retain unambiguous CPU correlations only. Kernel names can be
            # reused by several GEMM shapes, so the first event is insufficient.
            if len(matched) == 1:
                parent = matched[0]
                key = json.dumps(dict(name=parent["name"],
                                      input_dims=parent.get("args", {}).get("Input Dims"),
                                      input_strides=parent.get("args", {}).get("Input Strides")), sort_keys=True)
                shape_events[key].append(event["dur"])
        cpu_shapes = [dict(**json.loads(key), calls=len(durations),
                           total_us=math.fsum(durations))
                      for key, durations in shape_events.items()]
        rows.append(dict(name=name, category=sample["cat"], group=group_name(name),
                         calls=len(items), total_us=duration, mean_us=duration / len(items),
                         share_percent=100 * duration / total, sample_resources=resources,
                         correlated_cpu_shapes=cpu_shapes,
                         cpu_ops=[dict(name=e["name"], input_dims=e.get("args", {}).get("Input Dims"),
                                       input_strides=e.get("args", {}).get("Input Strides")) for e in cpu]))
    if set(grouped) != set(recorded):
        raise ValueError(f"Kernel set mismatch: {trace_path}")
    rows.sort(key=lambda row: row["total_us"], reverse=True)
    origin = min(e["ts"] for e in events)
    intervals = [(e["ts"] - origin, e["ts"] - origin + e["dur"]) for e in events]
    span = max(end for _, end in intervals)
    occupied = intervals_union(intervals)
    groups = collections.defaultdict(float)
    for row in rows:
        groups[row["group"]] += row["total_us"]
    return dict(total_leaf_us=total, leaf_events=len(events),
                kernel_events=sum(e["cat"] == "kernel" for e in events),
                category_counts=dict(collections.Counter(e["cat"] for e in events)),
                stream_ids=sorted({e.get("args", {}).get("stream") for e in events}),
                trace_span_us=span, occupied_union_us=occupied,
                uncovered_span_us=max(0.0, span - occupied),
                overlap_us=max(0.0, total - occupied),
                groups_us=dict(groups), kernels=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "g4")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit = dict(method="GPU leaf events only; shares divide by summed leaf duration; no new GPU run",
                 inputs_sha256={}, runs=[], profiles=[])
    for dataset in DATASETS:
        root = REPO / "llm_bench/results" / dataset
        for run in ("final-run1", "final-run2"):
            path = root / f"{run}.json"
            data = json.loads(path.read_text())
            if not data["complete"] or not data["all_passed"] or data["arguments"]["quick"]:
                raise ValueError(f"Incomplete, failed or smoke-only run: {path}")
            audit["inputs_sha256"][str(path.relative_to(REPO))] = digest(path)
            hashes = data["environment"]["source_sha256"]
            source_checks = {name: digest(REPO / name) == hashes.get(name)
                             for name in ("llm_bench/bench.py", "llm_bench/torch_backend.py", "llm_roofline/spec.py")}
            audit["runs"].append(dict(dataset=dataset, run=run, started_utc=data["started_utc"],
                                      environment=data["environment"], arguments=data["arguments"],
                                      model=data["model"], source_matches_current=source_checks))
            for row in data["results"]:
                if row["backend"] != "torch":
                    continue
                if row["status"] != "ok" or not row["correct"]:
                    raise ValueError(f"Failed torch row: {path}")
                trace_path = root / row["profile"]["trace"]
                audit["inputs_sha256"][str(trace_path.relative_to(REPO))] = digest(trace_path)
                analysis = profile(trace_path, row["profile"]["cuda_kernels"])
                analysis.update(dataset=dataset, run=run, workload=row["workload"],
                                trace=str(trace_path.relative_to(REPO)), timing=row["timing"],
                                no_graph_timing=row.get("no_graph_timing"),
                                shape=row.get("shape", dict(batch=row.get("batch"), q_len=1,
                                                           kv_capacity=row.get("prompt_length", 0) + row.get("generated_tokens", 0))),
                                trace_scope="one uncaptured B1 decode with sampling at first generation position"
                                if row["workload"] == "generate" else "one uncaptured compiled forward")
                audit["profiles"].append(analysis)
    (args.out_dir / "analysis.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    fields = ["dataset", "run", "workload", "category", "group", "name", "calls", "total_us", "mean_us", "share_percent"]
    with (args.out_dir / "kernels.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in audit["profiles"]:
            for kernel in item["kernels"]:
                writer.writerow({field: item.get(field, kernel.get(field)) for field in fields})
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        latest = [item for item in audit["profiles"] if item["dataset"] == DATASETS[-1]]
        names = ["FlashAttention", "FlexAttention (named)", "FlexAttention + GEMM (mixed)",
                 "GEMM-containing (named)", "Other kernels / copies"]
        colors = ["#4477aa", "#66ccee", "#aa3377", "#ee7733", "#bbbbbb"]
        fig, ax = plt.subplots(figsize=(10, 4.8))
        for i, item in enumerate(latest):
            left = 0
            for name, color in zip(names, colors):
                share = 100 * item["groups_us"].get(name, 0) / item["total_leaf_us"]
                ax.barh(i, share, left=left, color=color, label=name if i == 0 else None)
                if share >= 9:
                    ax.text(left + share / 2, i, f"{share:.1f}%", ha="center", va="center", fontsize=9)
                left += share
        ax.set_yticks(range(len(latest)), [f"{item['workload']} / {item['run']}" for item in latest])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of summed GPU leaf-event duration (%)")
        ax.set_title("Existing G4 torch.compile traces: 2026-10-10\nUncaptured forward; generation trace is one B1 decode step")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir / "kernel-shares.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(json.dumps(dict(profiles=len(audit["profiles"]), checked_inputs=len(audit["inputs_sha256"]),
                          kernel_rows=sum(len(item["kernels"]) for item in audit["profiles"]),
                          out_dir=str(args.out_dir)), ensure_ascii=False))


if __name__ == "__main__":
    main()
