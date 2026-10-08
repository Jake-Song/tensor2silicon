"""Print the week 5 arithmetic model, not hardware benchmark results.

BF16 X/W, FP32 bias/output, one read of each input and one output write.
Intermediates stay on chip. Bias/ReLU counts are reported separately and
are not charged to matrix peak throughput. No external packages required.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    batch: int
    k: int = 1024
    n: int = 1024

    @property
    def matmul_flops(self) -> int:
        return 2 * self.batch * self.k * self.n

    @property
    def output_elements(self) -> int:
        return self.batch * self.n

    @property
    def traffic(self) -> dict[str, int]:
        return {
            "X read": 2 * self.batch * self.k,
            "W read": 2 * self.k * self.n,
            "bias read": 4 * self.n,
            "Y write": 4 * self.output_elements,
        }

    @property
    def total_bytes(self) -> int:
        return sum(self.traffic.values())

    @property
    def intensity(self) -> float:
        return self.matmul_flops / self.total_bytes


@dataclass(frozen=True)
class VirtualDevice:
    name: str
    peak_flops_per_second: float
    bandwidth_bytes_per_second: float


WORKLOADS = (Workload(1), Workload(128))
DEVICES = (
    VirtualDevice("A", 200e12, 1e12),
    VirtualDevice("B", 80e12, 2e12),
)


def main() -> None:
    print("Arithmetic model only; these are NOT measured device latencies.\n")
    print("| Item | M=1 | M=128 |")
    print("|---|---:|---:|")
    rows = {
        "MatMul FLOPs (2MKN)": [w.matmul_flops for w in WORKLOADS],
        "Bias adds": [w.output_elements for w in WORKLOADS],
        "ReLU max/comparisons": [w.output_elements for w in WORKLOADS],
    }
    for name in WORKLOADS[0].traffic:
        rows[f"{name} (Byte)"] = [w.traffic[name] for w in WORKLOADS]
    rows["Total Q (Byte)"] = [w.total_bytes for w in WORKLOADS]
    for name, values in rows.items():
        print(f"| {name} | {values[0]:,} | {values[1]:,} |")
    intensities = [w.intensity for w in WORKLOADS]
    print(f"| I (FLOPs/Byte) | {intensities[0]:.6f} | {intensities[1]:.6f} |")

    print("\n| Device | P (TFLOP/s) | BW (TB/s) | Ridge (FLOPs/Byte) |")
    print("|---|---:|---:|---:|")
    for device in DEVICES:
        peak = device.peak_flops_per_second
        bandwidth = device.bandwidth_bytes_per_second
        print(f"| {device.name} | {peak / 1e12:g} | {bandwidth / 1e12:g} | {peak / bandwidth:g} |")

    print("\n| M | Device | Compute (us) | Memory (us) | Ideal (us) | Bound |")
    print("|---:|---|---:|---:|---:|---|")
    for workload in WORKLOADS:
        for device in DEVICES:
            compute_us = workload.matmul_flops / device.peak_flops_per_second * 1e6
            memory_us = workload.total_bytes / device.bandwidth_bytes_per_second * 1e6
            ideal_us = max(compute_us, memory_us)
            bound = "memory" if memory_us > compute_us else "compute"
            print(
                f"| {workload.batch} | {device.name} | {compute_us:.6f} | "
                f"{memory_us:.6f} | {ideal_us:.6f} | {bound} |"
            )


if __name__ == "__main__":
    main()
