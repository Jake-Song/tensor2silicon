"""Generate the week 4 roofline figure; no GPU or measurements involved.

Run with a Python environment containing Matplotlib:
    MPLCONFIGDIR=/tmp/week4-matplotlib python3 week4/make_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def main():
    n = 4096
    flops = 2 * n**3
    peak = 20e12
    bandwidth = 1e12
    ridge = peak / bandwidth
    intensities = [10 ** (-1.1 + 3.2 * index / 400) for index in range(401)]
    fig, ax = plt.subplots(figsize=(11, 6.4), layout="constrained")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(
        intensities, [bandwidth * value / 1e12 for value in intensities],
        color="#0284c7", linestyle="--", linewidth=1.8,
        label="HBM ceiling: BW × I (BW = 1 TB/s)",
    )
    ax.axhline(peak / 1e12, color="#d97706", linestyle="--", linewidth=1.8,
               label="Compute ceiling: P = 20 TFLOPs/s")
    ax.plot(
        intensities, [min(peak, bandwidth * value) / 1e12 for value in intensities],
        color="#172554", linewidth=3, label="Roofline: min(P, BW × I)",
    )
    ax.axvline(ridge, color="#64748b", linestyle=":", linewidth=1.4)
    ax.annotate("Ridge: P / BW = 20 FLOPs/Byte", xy=(ridge, 20),
                xytext=(1.1, 55), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": "#64748b"})
    configurations = [
        (1, "No reuse", "#dc2626", (0.30, 0.16)),
        (32, "T = 32", "#059669", (1.8, 11)),
        (64, "T = 64", "#7c3aed", (7, 2.5)),
        (128, "T = 128", "#db2777", (28, 8)),
    ]
    for tile, label, color, label_position in configurations:
        traffic = 8 * n**3 // tile + 4 * n**2
        intensity = flops / traffic
        rate = min(peak, bandwidth * intensity) / 1e12
        ax.scatter([intensity], [rate], s=65, color=color, edgecolor="white", zorder=5)
        ax.annotate(f"{label}\nI = {intensity:.3f}; R = {rate:.3f}",
                    xy=(intensity, rate), xytext=label_position, textcoords="data",
                    color=color, fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": color})
    ax.text(0.11, 3.7, "HBM bandwidth\nsets the lower ceiling", color="#0369a1", fontsize=11)
    ax.text(38, 3.7, "Compute sets\nthe lower ceiling", color="#92400e", fontsize=11)
    ax.set(xlim=(0.08, 100), ylim=(0.08, 90),
           xlabel="HBM arithmetic intensity I [FLOPs/Byte] — log scale",
           ylabel="Performance upper bound R [TFLOPs/s] — log scale",
           title="4096 × 4096 FP32 MatMul · hypothetical hardware")
    ax.set_xticks([0.1, 0.25, 1, 4, 8, 20, 32, 100])
    ax.set_yticks([0.1, 0.25, 1, 4, 10, 20, 80])
    formatter = FuncFormatter(lambda value, _: f"{value:g}")
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid(which="major", alpha=0.22)
    ax.legend(loc="upper left", fontsize=9)
    fig.supxlabel("Dots are calculated ceilings, not measurements. Final C writes are included.",
                  fontsize=10)
    output = Path(__file__).resolve().parent / "assets"
    output.mkdir(exist_ok=True)
    fig.savefig(output / "roofline.png", dpi=180)
    fig.savefig(output / "roofline.svg")
    plt.close(fig)


if __name__ == "__main__":
    main()
