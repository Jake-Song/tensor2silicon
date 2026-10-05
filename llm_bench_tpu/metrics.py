"""Numerical gates fixed before measuring; host-only latency summaries."""
import math
import statistics

import numpy as np

NRMSE_LIMIT = 0.02
NORMALIZED_MAX_LIMIT = 0.20
RMS_FLOOR = 1e-6


def errors(actual, expected):
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    a, b = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("non-finite output/reference")
    diff = a - b
    rms = max(float(np.sqrt(np.mean(np.square(b), dtype=np.float64))), RMS_FLOOR)
    nrmse = float(np.sqrt(np.mean(np.square(diff), dtype=np.float64))) / rms
    max_abs = float(np.max(np.abs(diff)))
    normalized_max = max_abs / rms
    return {"nrmse": nrmse, "normalized_max": normalized_max, "max_abs": max_abs,
            "reference_rms": rms,
            "passed": nrmse <= NRMSE_LIMIT and normalized_max <= NORMALIZED_MAX_LIMIT}


def summarize(samples):
    vals = sorted(float(v) for v in samples)
    if not vals or not all(math.isfinite(v) and v > 0 for v in vals):
        raise ValueError("latencies must be positive finite observations")
    at = (len(vals) - 1) * .95
    lo, hi = math.floor(at), math.ceil(at)
    return {"median_ms": statistics.median(vals), "p95_ms": vals[lo] + (vals[hi] - vals[lo]) * (at - lo),
            "min_ms": vals[0], "max_ms": vals[-1], "samples_ms": list(samples)}
