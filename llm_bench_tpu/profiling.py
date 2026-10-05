"""Device trace capture kept outside the scored measurement loop."""
from pathlib import Path
import time


def capture(fn, inputs, folder, steps=20, label="forward"):
    import jax
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    jax.block_until_ready(fn(*inputs))
    started = time.perf_counter()
    with jax.profiler.trace(str(folder), create_perfetto_link=False, create_perfetto_trace=True):
        for i in range(steps):
            with jax.profiler.StepTraceAnnotation(label, step_num=i):
                jax.block_until_ready(fn(*inputs))
    files = [{"path": str(p.relative_to(folder)), "bytes": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file()]
    from .profile_summary import summarize_trace
    traces = list(folder.rglob("perfetto_trace.json.gz"))
    if len(traces) != 1:
        raise ValueError("use a fresh profile directory for each capture")
    summary = summarize_trace(traces[0])
    if summary["steps"] != steps:
        raise ValueError(f"expected {steps} device executions, captured {summary['steps']}")
    return {"folder": str(folder), "steps": steps, "files": files,
            "capture_and_export_s": time.perf_counter() - started,
            "score_timing": False, "device_events_verified": True,
            "device_module_median_ms": summary["device_module_median_ms"],
            "categories": summary["categories"]}
