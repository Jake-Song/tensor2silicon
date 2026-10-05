"""Device-independent acceptance criteria and latency summaries."""
from __future__ import annotations

import math
import statistics

NRMSE_LIMIT = 0.02
NORMALIZED_MAX_LIMIT = 0.20
RMS_FLOOR = 1e-6


def summarize(samples):
    values = sorted(float(v) for v in samples)
    if not values or not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("latencies must be positive finite observations")
    index = (len(values) - 1) * .95
    lo, hi = math.floor(index), math.ceil(index)
    return {"median_ms": statistics.median(values),
            "p95_ms": values[lo] + (values[hi] - values[lo]) * (index - lo),
            "min_ms": values[0], "max_ms": values[-1],
            "samples_ms": list(samples)}


def accept(nrmse, normalized_max):
    return (math.isfinite(nrmse) and math.isfinite(normalized_max)
            and nrmse <= NRMSE_LIMIT and normalized_max <= NORMALIZED_MAX_LIMIT)
