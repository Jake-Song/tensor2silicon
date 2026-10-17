"""Read `ncu --import report.ncu-rep --csv --page raw` exports offline."""
import argparse
import csv
import json
from pathlib import Path

METRICS = {
    'tensor_elapsed_pct': 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed',
    'registers_per_thread': 'launch__registers_per_thread',
    'occupancy_active_pct': 'sm__warps_active.avg.pct_of_peak_sustained_active',
    'barrier_per_issue': 'smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio',
    'wait_per_issue': 'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',
    'short_scoreboard_per_issue': 'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
    'long_scoreboard_per_issue': 'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'instructions': 'smsp__inst_executed.sum',
    'shared_ldsm_bank_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm.sum',
    'shared_ld_bank_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'shared_st_bank_conflicts': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'spilling_instructions': 'sass__inst_executed_register_spilling',
}


def number(value):
    if value is None or value in {'', 'no data', 'n/a'}:
        return None
    return float(value.replace(',', ''))


def read(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f'No profiler metrics in {path}')
    units = rows[0] if not rows[0]['ID'] else {}
    results = []
    for row in rows:
        if not row['ID']:
            continue
        result = dict(source=str(path), kernel=row['Kernel Name'])
        result.update({label: number(row.get(metric)) for label, metric in METRICS.items()})
        metric = 'gpu__time_duration.sum'
        value = number(row.get(metric))
        unit = units.get(metric)
        scale = {'nsecond': 1e-6, 'usecond': 1e-3, 'msecond': 1, 'second': 1000,
                 'ns': 1e-6, 'us': 1e-3, 'ms': 1, 's': 1000}.get(unit)
        result['profile_ms'] = value * scale if value is not None and scale is not None else None
        result['profile_time_unit'] = unit
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [row for path in args.inputs for row in read(path)]
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
