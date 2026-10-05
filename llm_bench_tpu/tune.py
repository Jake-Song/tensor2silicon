"""Bounded whole-model search with fixed accuracy gates and two-run acceptance."""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def merge(target, patch):
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--initial-config")
    parser.add_argument("--deadline", type=float, required=True, help="Unix time after which no candidate starts")
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()
    folder = Path(args.out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    incumbent_file = folder / "selected.json"
    ledger = folder / "candidates.jsonl"
    if incumbent_file.exists():
        incumbent = json.loads(incumbent_file.read_text())
    elif args.initial_config:
        incumbent = json.loads(Path(args.initial_config).read_text())
    else:
        incumbent = {"version": 1, "native": {}, "hybrid": {}}
    incumbent_file.write_text(json.dumps(incumbent, indent=2) + "\n")
    previous = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    done = {r["id"] for r in previous}
    for candidate in json.loads(Path(args.candidates).read_text()):
        if candidate["id"] in done:
            continue
        if len(done) >= min(args.max_candidates, 30) or time.time() >= args.deadline:
            print("SEARCH_BUDGET_REACHED", flush=True)
            break
        started = time.time()
        row = dict(candidate, started_utc=datetime.now(timezone.utc).isoformat(), runs=[])
        backend, phase = candidate["backend"], candidate["phase"]
        proposed = copy.deepcopy(incumbent)
        merge(proposed.setdefault(backend, {}).setdefault(phase, {}), candidate["changes"])
        trial = folder / candidate["id"]
        trial.mkdir(exist_ok=True)
        config_file, before_file = trial / "config.json", trial / "incumbent.json"
        config_file.write_text(json.dumps(proposed, indent=2) + "\n")
        before_file.write_text(json.dumps(incumbent, indent=2) + "\n")
        row["config"] = proposed
        print("CANDIDATE", candidate["id"], candidate["hypothesis"], flush=True)
        row["decision"] = "rejected"
        for seed in (123, 321):
            out = trial / f"seed-{seed}.json"
            command = [sys.executable, "-u", "-m", "llm_bench_tpu", "--phase", phase,
                       "--seed", str(seed), "--samples", "60", "--warmup", "5",
                       "--backends", f"incumbent_{backend},{backend}",
                       "--tuning-config", str(config_file), "--incumbent-config", str(before_file),
                       "--out", str(out)]
            if seed == 321:
                command.append("--reverse")
            try:
                with (trial / f"seed-{seed}.log").open("w") as log:
                    run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=max(1, 300 - (time.time() - started)))
                if run.returncode:
                    row["decision"] = "failed"
                    row["reason"] = f"seed {seed} exited {run.returncode}; see saved log"
                    if out.exists():
                        row["error"] = json.loads(out.read_text()).get("error")
                    break
                result = json.loads(out.read_text())
                assert result["complete"] and result["status"] == "ok"
                rows = result["phases"][phase]["backends"]
                old = rows[f"incumbent_{backend}"]["latency"]["median_ms"]
                new = rows[backend]["latency"]["median_ms"]
                row["runs"].append({"seed": seed, "incumbent_ms": old, "candidate_ms": new,
                                    "improvement_fraction": 1 - new / old, "result": str(out)})
                print("TRIAL", candidate["id"], seed, old, new, flush=True)
                if new >= old * .99:
                    row["reason"] = "whole-model improvement did not exceed 1%"
                    break
            except subprocess.TimeoutExpired:
                row["decision"] = "timeout"
                row["reason"] = "300 second total candidate budget"
                break
            except Exception as exc:
                row["decision"] = "failed"
                row["reason"] = f"{type(exc).__name__}: {exc}"
                break
        else:
            row["decision"] = "accepted"
            incumbent = proposed
            incumbent_file.write_text(json.dumps(incumbent, indent=2) + "\n")
        row["elapsed_s"] = time.time() - started
        with ledger.open("a") as log:
            log.write(json.dumps(row) + "\n")
        done.add(candidate["id"])
        print("DECISION", candidate["id"], row["decision"], row.get("reason", "two independent runs passed"), flush=True)
    print("TPU_TUNING_BATCH_COMPLETE", json.dumps(incumbent), flush=True)


if __name__ == "__main__":
    main()
