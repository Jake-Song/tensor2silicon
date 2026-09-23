"""Roofline and time-breakdown plots from the JSON written by torch_llm / jax_llm.

    python -m llm_roofline.plot results_gpu.json results_tpu.json --out-dir plots/
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette, first three slots (validated all-pairs for scatter).
# Kind is also carried by marker shape and the legend, never by color alone.
KINDS = {  # label, color, marker
    "matmul": ("matmul", "#2a78d6", "o"),
    "reduction": ("reduction (LayerNorm, softmax)", "#eb6834", "D"),
    "elementwise": ("elementwise / data movement", "#1baf7a", "s"),
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def kind_of(row):
    return {"norm": "reduction", "softmax": "reduction"}.get(row["kind"], row["kind"])


def load(path_or_data):
    if isinstance(path_or_data, dict):
        return path_or_data
    with open(path_or_data) as f:
        return json.load(f)


def title_of(d):
    m = d["meta"]
    return f"{m['framework']} on {m['device']} ({m['dtype']}, preset {m['preset']})"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _groups(rows):
    """Merge ops of the same shape (q/k/v/o_proj, gate/up_proj) into one labelled marker at their mean speed."""
    out = {}
    for r in rows:
        if r["flops"] <= 0:
            continue   # AI = 0 (pure data movement) has no place on a log axis
        g = out.setdefault((kind_of(r), r["desc"]), {"ai": r["ai"], "tflops": [], "names": []})
        g["tflops"].append(r["tflops"])
        g["names"].append(r["name"])
    return [{"ai": g["ai"], "tflops": float(np.mean(g["tflops"])), "names": g["names"], "kind": k[0]}
            for k, g in out.items()]


def _short(names):
    if len(names) > 1 and all(n.endswith("_proj") for n in names):
        return ",".join(n.removesuffix("_proj") for n in names) + "_proj"
    return ",".join(names)


def plot_roofline(data, axes=None):
    """One panel per phase: the roofline min(peak, AI x BW) and every op at (AI, achieved FLOP/s)."""
    d = load(data)
    peak_flops, peak_bw = d["meta"]["peaks_used"]
    phases = list(d["phases"])
    if axes is None:
        fig, axes = plt.subplots(1, len(phases), figsize=(6.2 * len(phases), 4.8), sharey=True, facecolor=SURFACE)
        axes = np.atleast_1d(axes)
    else:
        fig = axes[0].figure
    ridge = peak_flops / peak_bw
    ai = np.logspace(-1.5, 4, 400)
    pending = []
    for ax, ph in zip(axes, phases):
        _style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        roof = np.minimum(peak_flops, ai * peak_bw) / 1e12
        ax.plot(ai, roof, color=INK2, linewidth=2, solid_capstyle="round", zorder=2)
        ax.axvline(ridge, color=AXIS, linewidth=1, zorder=1)
        ax.set_xlim(ai[0], ai[-1])
        lo = min(r["tflops"] for p_ in d["phases"].values() for r in p_["rows"] if r["flops"] > 0)
        ax.set_ylim(10 ** np.floor(np.log10(lo)), peak_flops / 1e12 * 3)
        labels = []
        for g in _groups(d["phases"][ph]["rows"]):
            kind = g["kind"]
            _, color, marker = KINDS[kind]
            ax.scatter(g["ai"], g["tflops"], s=64, color=color, marker=marker, edgecolors=SURFACE, linewidths=2, zorder=4)
            if kind != "elementwise":   # label selectively; elementwise ops are in the legend and the table
                labels.append((g["ai"], g["tflops"], _short(g["names"])))
        pending.append((ax, labels))
        p = d["phases"][ph]["phase"]
        ax.set_title(f"{ph}: B={p['batch']} T={p['q_len']} S={p['kv_len']}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("arithmetic intensity (FLOP / byte)", color=INK2, fontsize=9)
        ax.set_xlim(ai[0], ai[-1])
    axes[0].set_ylabel("achieved TFLOP/s", color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], linestyle="", marker=m, markersize=8, color=c, label=lab) for lab, c, m in KINDS.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"Roofline: {title_of(d)}\nroof = min({peak_flops / 1e12:.0f} TFLOP/s, AI x {peak_bw / 1e9:.0f} GB/s)  "
                 f"| ridge (vertical line) at AI = {ridge:.0f} FLOP/B  | measured ceilings",
                 color=INK, fontsize=11, x=0.01, ha="left", y=1.06)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    for ax, labels in pending:   # after layout, so pixel positions are final
        _place_labels(ax, labels)
    return fig


def _place_labels(ax, labels, char_px=5.6, line_px=12, dx=8):
    """Put each label right of its point, then push labels down until none overlap;
    a label that had to move gets a hairline leader back to its point."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    items = sorted(((*to_px((x, y)), text) for x, y, text in labels), key=lambda t: -t[1])
    placed = []   # (x0, x1, y) in pixels
    for px, py, text in items:
        x0, x1, ly = px + dx, px + dx + char_px * len(text), py
        for _ in range(50):
            clash = [p for p in placed if p[0] < x1 and x0 < p[1] and abs(p[2] - ly) < line_px]
            if not clash:
                break
            ly = min(p[2] for p in clash) - line_px
        placed.append((x0, x1, ly))
        moved = abs(ly - py) > 3
        pt = 72 / fig.dpi   # pixel offsets -> points, so the layout survives savefig(dpi=...)
        ax.annotate(text, ax.transData.inverted().transform((px, py)), xytext=((x0 - px) * pt, (ly - py) * pt),
                    textcoords="offset points",
                    fontsize=8, color=INK, va="center", zorder=5,
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8, shrinkA=0, shrinkB=4) if moved else None)


def plot_time_breakdown(data, axes=None):
    """One panel per phase: time per forward pass spent in each op (per-op time x calls), largest first."""
    d = load(data)
    phases = list(d["phases"])
    if axes is None:
        fig, axes = plt.subplots(1, len(phases), figsize=(6.2 * len(phases), 5.2), facecolor=SURFACE)
        axes = np.atleast_1d(axes)
    else:
        fig = axes[0].figure
    for ax, ph in zip(axes, phases):
        _style(ax)
        ax.grid(False)
        ax.grid(True, axis="x", color=GRID, linewidth=1)
        rows = sorted(d["phases"][ph]["rows"], key=lambda r: r["t_s"] * r["count"])
        total = sum(r["t_s"] * r["count"] for r in rows) * 1e3
        y = np.arange(len(rows))
        ms = [r["t_s"] * r["count"] * 1e3 for r in rows]
        ax.barh(y, ms, height=0.62, color=[KINDS[kind_of(r)][1] for r in rows], zorder=3)
        ax.set_yticks(y, [f"{r['name']} x{r['count']}" for r in rows], fontsize=8, color=INK2)
        ax.tick_params(axis="y", length=0)
        for yi, v, r in zip(y, ms, rows):
            if v / total >= 0.04:   # label only the ops that matter
                ax.text(v, yi, f"  {v / total * 100:.0f}%", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, max(ms) * 1.18)
        ax.set_xlabel("ms per forward pass (per-op time x calls)", color=INK2, fontsize=9)
        ax.set_title(f"{ph}: sum of ops {total:.2f} ms", color=INK, fontsize=11, loc="left")
    handles = [plt.Line2D([], [], linestyle="", marker="s", markersize=8, color=c, label=lab) for lab, c, _ in KINDS.values()]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9, labelcolor=INK2,
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle(f"Where the time goes: {title_of(d)}", color=INK, fontsize=12, x=0.01, ha="left", y=1.03)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def summary_table(data):
    """Rows for a compact per-op table (pandas-free), e.g. for display in the notebook."""
    d = load(data)
    out = []
    for ph, pd_ in d["phases"].items():
        for r in pd_["rows"]:
            out.append({"phase": ph, "#": r["mm_index"] or "", "op": r["name"], "calls": r["count"], "shape": r["desc"],
                        "AI": round(r["ai"], 1), "theory": r["bound"], "us": round(r["t_s"] * 1e6, 1),
                        "TFLOP/s": round(r["tflops"], 1), "GB/s": round(r["gbps"]),
                        "%roofline": round(r["pct_roof"] * 100), "measured": r["measured"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+")
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.json:
        stem = os.path.splitext(os.path.basename(path))[0]
        for name, fn in (("roofline", plot_roofline), ("time", plot_time_breakdown)):
            out = os.path.join(args.out_dir, f"{stem}_{name}.png")
            fn(path).savefig(out, dpi=150, bbox_inches="tight", facecolor=SURFACE)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
