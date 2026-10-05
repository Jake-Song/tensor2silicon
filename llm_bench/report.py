"""Build the Korean benchmark report and a self-contained results notebook.

Only final-run1.json and final-run2.json contribute benchmark numbers. Other
JSON files are listed as tuning evidence, never silently pooled with final runs.
This module imports no accelerator framework when producing the report.
"""

from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import shlex


BACKENDS = ("torch", "hybrid", "triton")
LABELS = {"torch": "PyTorch compile", "hybrid": "PyTorch + Triton compile", "triton": "Native Triton"}
FINAL_FILES = ("final-run1.json", "final-run2.json")
CANDIDATE_FILES = ("hybrid-attention-candidate.json", "hybrid-matmul-candidate.json", "hybrid-both-candidate.json")
TORCH_CANDIDATE_FILES = ("torch-flex-candidate.json",)
NATIVE_CANDIDATE_FILES = ("optimized-candidate.json", "native-tma-candidate.json")
METRIC_LABELS = {
    "graph_device": "CUDA Graph GPU 시간", "graph_wall": "CUDA Graph 동기화 wall 시간",
    "no_graph_device": "Graph 없는 GPU 시간", "no_graph_wall": "Graph 없는 동기화 wall 시간",
    "generation_device": "128-token 생성 GPU 시간", "generation_wall": "128-token 생성 wall 시간",
    "ttft_device": "GPU TTFT", "decode_token_device": "GPU decode ms/token",
}


def load_runs(results_dir):
    """Require the two named final runs; never promote a quick/tuning file."""
    from llm_roofline.spec import PRESETS

    root = Path(results_dir)
    expected_model = dataclasses.asdict(PRESETS["colab"].model)
    runs = []
    for name in FINAL_FILES:
        path = root / name
        data = json.loads(path.read_text())
        if not data.get("complete"):
            raise ValueError(f"{name} is incomplete")
        if data.get("arguments", {}).get("quick") or data.get("arguments", {}).get("no_compile"):
            raise ValueError(f"{name} is a quick or uncompiled diagnostic, not a final run")
        if data.get("arguments", {}).get("preset") != "colab":
            raise ValueError(f"{name} must use the full colab model preset")
        if data.get("model") != expected_model:
            raise ValueError(f"{name} model dimensions do not match the colab preset")
        expected = {(backend, phase) for backend in BACKENDS for phase in ("prefill", "decode", "generate")}
        rows = data.get("results", [])
        keys = [(row.get("backend"), row.get("workload")) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != expected:
            raise ValueError(f"{name} requires all nine unique backend/workload rows; got {keys}")
        for row in rows:
            if row.get("status") == "ok" and row["workload"] == "generate":
                if row.get("generated_tokens") != 128:
                    raise ValueError(f"{name}/{row['backend']} must measure generation of 128 tokens")
        runs.append((name.removesuffix(".json"), data))
    return runs


def load_tuning_configs(runs, results_dir):
    configs = {}
    for _, data in runs:
        configured = data.get("arguments", {}).get("tuning_config")
        if configured:
            name = Path(configured).name
            path = Path(results_dir) / name
            if not path.exists():
                raise FileNotFoundError(f"standalone reproduction needs the final tuning config: {path}")
            raw = path.read_bytes()
            expected_hash = data.get("tuning_config_sha256")
            if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
                raise ValueError(f"{name} bytes do not match the tuning config used by the measured run")
            configs[name] = json.loads(raw)
    return configs


def load_candidates(results_dir):
    candidates = []
    for name in CANDIDATE_FILES:
        path = Path(results_dir) / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        args = data.get("arguments", {})
        if not data.get("complete") or args.get("quick") or args.get("no_compile") or args.get("preset") != "colab":
            raise ValueError(f"{name} must be a complete, compiled, non-quick colab candidate")
        for row in data.get("results", []):
            if row.get("status") == "ok" and row.get("workload") == "generate" and row.get("generated_tokens") != 128:
                raise ValueError(f"{name} candidate generation must contain 128 tokens")
        candidates.append((name, data))
    return candidates


def load_torch_candidates(results_dir):
    """Keep the official PyTorch Flex baseline diagnostics in their own table."""
    candidates = []
    for name in TORCH_CANDIDATE_FILES:
        path = Path(results_dir) / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        args = data.get("arguments", {})
        if (not data.get("complete") or args.get("quick") or args.get("no_compile")
                or args.get("preset") != "colab" or args.get("backend") != "torch"):
            raise ValueError(f"{name} must be a complete compiled colab PyTorch candidate")
        for row in data.get("results", []):
            if row.get("status") == "ok" and row.get("workload") == "generate" and row.get("generated_tokens") != 128:
                raise ValueError(f"{name} candidate generation must contain 128 tokens")
        candidates.append((name, data))
    return candidates


def load_native_candidates(results_dir):
    """Load explicitly named native whole-model diagnostics separately."""
    candidates = []
    for name in NATIVE_CANDIDATE_FILES:
        path = Path(results_dir) / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        args = data.get("arguments", {})
        if (not data.get("complete") or args.get("quick") or args.get("no_compile")
                or args.get("preset") != "colab"):
            raise ValueError(f"{name} must be a complete non-quick colab native candidate")
        for row in data.get("results", []):
            if (row.get("backend") == "triton" and row.get("status") == "ok"
                    and row.get("workload") == "generate" and row.get("generated_tokens") != 128):
                raise ValueError(f"{name} native generation must contain 128 tokens")
        candidates.append((name, data))
    return candidates


def hybrid_policy(args):
    matmul = {
        "torch": "compiled PyTorch GEMM",
        "decode-triton": "prefill PyTorch GEMM + 모든 decode projection/LM head Triton GEMM",
        "m1-triton": "M=1 decode projection/LM head만 Triton GEMM; B=8 decode와 prefill은 PyTorch GEMM",
    }.get(args.get("hybrid_matmul", "torch"), f"GEMM policy={args.get('hybrid_matmul')}")
    attention = {
        "sdpa": "PyTorch SDPA",
        "triton": "prefill/decode Triton attention",
        "decode-triton": "prefill SDPA + decode Triton attention",
    }.get(args.get("hybrid_attention", "sdpa"), f"attention policy={args.get('hybrid_attention')}")
    return matmul, attention


def reproduction_command(run, data, results_dir="."):
    """Preserve final settings while relocating result/config paths to this folder."""
    args = data.get("arguments", {})
    command = ["python", "-m", "llm_bench", "--backend", args.get("backend", "all"),
               "--preset", args.get("preset", "colab"), "--workload", args.get("workload", "all"),
               "--seed", str(args.get("seed", 123))]
    if args.get("profile"):
        command.append("--profile")
    if args.get("reverse"):
        command.append("--reverse")
    if args.get("hybrid_attention"):
        command.extend(("--hybrid-attention", args["hybrid_attention"]))
    if args.get("hybrid_matmul"):
        command.extend(("--hybrid-matmul", args["hybrid_matmul"]))
    if args.get("torch_attention"):
        command.extend(("--torch-attention", args["torch_attention"]))
    if args.get("tuning_config"):
        command.extend(("--tuning-config", str(Path(results_dir) / Path(args["tuning_config"]).name)))
    command.extend(("--out", str(Path(results_dir) / (run + ".json"))))
    return shlex.join(command)


def correctness_metrics(row):
    checks = []

    def visit(value):
        if isinstance(value, dict):
            if "nrmse" in value and "normalized_max" in value:
                checks.append(value)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for key in ("correctness", "changed_input_correctness", "teacher_forced_checks", "graph_replay_checks"):
        visit(row.get(key, []))
    return {
        "checked_tensors": len(checks),
        "worst_nrmse": max((c["nrmse"] for c in checks), default=None),
        "worst_normalized_max": max((c["normalized_max"] for c in checks), default=None),
        "worst_max_abs_error": max((c.get("max_abs_error", 0) for c in checks), default=None),
    }


def records_from_runs(runs):
    """Flatten distributions while retaining run, workload and timing semantics."""
    records = []
    for run, data in runs:
        for row in data.get("results", []):
            if row.get("status") != "ok":
                continue
            timing = row["timing"]
            if row["workload"] == "generate":
                measures = {"generation_device": timing["device_ms"], "generation_wall": timing["wall_ms"],
                            "ttft_device": timing["ttft_device_ms"], "decode_token_device": timing["decode_device_ms"]}
            else:
                measures = {"graph_device": timing["device_round_average"], "graph_wall": timing["synchronized_wall"]}
                if row.get("no_graph_timing"):
                    measures.update(no_graph_device=row["no_graph_timing"]["device_round_average"],
                                    no_graph_wall=row["no_graph_timing"]["synchronized_wall"])
            accuracy = correctness_metrics(row)
            for metric, value in measures.items():
                records.append(dict(run=run, backend=row["backend"], workload=row["workload"], metric=metric,
                                    median_ms=value["median_ms"], p95_ms=value["p95_ms"],
                                    min_ms=value["min_ms"], max_ms=value["max_ms"],
                                    sample_count=len(value.get("samples_ms", [])),
                                    speedup_vs_torch=None, **accuracy))
    baselines = {(r["run"], r["workload"], r["metric"]): r["median_ms"]
                 for r in records if r["backend"] == "torch"}
    for row in records:
        baseline = baselines.get((row["run"], row["workload"], row["metric"]))
        row["speedup_vs_torch"] = baseline / row["median_ms"] if baseline is not None else None
    return records


def _number(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def _table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
                     + ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |"
                        for row in rows])


def _index(records):
    return {(r["run"], r["backend"], r["workload"], r["metric"]): r for r in records}


def render_report(runs, records, inventory=(), tuning_configs=None, candidates=(),
                  reproduction_dir="llm_bench/results/2026-10-04-g4", torch_candidates=(), native_candidates=()):
    """Render measured observations separately from optimization interpretation."""
    lookup = _index(records)
    tuning_configs = tuning_configs or {}
    first = runs[0][1]
    model = first["model"]
    hybrid_choices = []
    torch_choices = []
    for run, data in runs:
        args = data.get("arguments", {})
        matmul, attention = hybrid_policy(args)
        hybrid_choices.append(f"{run}: {matmul}, {attention}")
        if args.get("torch_attention") == "flex-decode":
            torch_choices.append(f"{run}: prefill 및 전체 KV가 유효한 고정 decode는 SDPA; 연속 생성 decode는 공식 PyTorch FlexAttention")
        else:
            torch_choices.append(f"{run}: PyTorch SDPA")
    sections = ["# PyTorch·Triton 실제 추론 비교", ""]
    lines = []
    for workload, metric, label in (("prefill", "graph_device", "prefill"),
                                    ("decode", "graph_device", "decode 한 step"),
                                    ("generate", "generation_device", "128-token 생성")):
        winners = []
        for run, _ in runs:
            measured_rows = [r for r in records if r["run"] == run and r["workload"] == workload and r["metric"] == metric]
            if measured_rows:
                winner = min(measured_rows, key=lambda r: r["median_ms"])
                winners.append(f"{run}: {LABELS[winner['backend']]} {_number(winner['median_ms'])} ms")
        if winners:
            lines.append(f"- **{label} GPU 중앙값 최저** — " + "; ".join(winners) + ".")
    sections.extend(lines + ["", "각 실행의 중앙값을 따로 비교했다. 두 실행의 원시 표본을 합치지 않았으며, 아래 배속은 모두 같은 실행의 PyTorch compile 대비 값이다.", ""])
    near_ties = []
    for run, _ in runs:
        for workload, label in (("prefill", "prefill"), ("decode", "B=8 decode")):
            measured = sorted((r for r in records if r["run"] == run and r["workload"] == workload
                               and r["metric"] == "graph_device"), key=lambda r: r["median_ms"])
            if len(measured) >= 2:
                first_two = measured[:2]
                gap = first_two[1]["median_ms"] / first_two[0]["median_ms"] - 1
                if gap < .01:
                    near_ties.append(f"{run} {label}: {LABELS[first_two[0]['backend']]}와 "
                                     f"{LABELS[first_two[1]['backend']]} 차이 {gap * 100:.2f}%")
        hybrid = lookup.get((run, "hybrid", "generate", "generation_device"))
        native = lookup.get((run, "triton", "generate", "generation_device"))
        if hybrid and native:
            a, b = hybrid["median_ms"], native["median_ms"]
            gap = max(a, b) / min(a, b) - 1
            if gap < .01:
                near_ties.append(f"{run} 128-token 생성: PyTorch + Triton {_number(a)} ms와 "
                                 f"Native Triton {_number(b)} ms, 차이 {gap * 100:.2f}%")
    if near_ties:
        sections += ["**중앙값 차이가 1% 미만인 조건은 거의 동급으로 해석한다.** "
                     "작은 순위 차이만으로 의미 있는 우위를 주장하지 않으며 통계적 동등성 검정을 수행한 것은 아니다.", ""]
        sections += [f"- {line}." for line in near_ties] + [""]
    failures = [(run, r) for run, d in runs for r in d.get("results", []) if r.get("status") != "ok"]
    if failures:
        sections += ["**실패한 조건이 있어 전체 비교는 불완전하다.**", "",
                     _table(["실행", "구현", "조건", "실패"], [(run, r.get("backend"), r.get("workload"), r.get("error")) for run, r in failures]), ""]
    sections += ["## 모델과 측정 조건", "",
                 f"기존 `llm_roofline` decoder를 사용했다: {model['n_layers']} layers, D={model['d_model']}, "
                 f"F={model['d_ff']}, query/KV heads={model['n_heads']}/{model['n_kv_heads']}, "
                 f"head_dim={model['head_dim']}, vocab={model['vocab']}. Affine LayerNorm(ε=1e-5), "
                 "split-half RoPE, causal attention, SwiGLU, residual, LM head를 모두 실행한다. "
                 "학습된 언어 모델 checkpoint가 아닌 프로젝트의 고정 seed 합성 가중치이므로 생성 토큰의 언어 품질을 평가하지 않는다.", ""]
    parameter_rows = [(run, f"{data['parameter_count']:,}", f"{data['parameter_count'] / 1e6:.3f} M")
                      for run, data in runs if data.get("parameter_count") is not None]
    if parameter_rows:
        sections += [_table(["실행", "실제 원본 parameter 수", "백만 parameter"], parameter_rows), "",
                     "Parameter 수는 실제 생성한 원본 가중치 tensor에서 합산했다. 실행용 packing으로 추가한 복사본은 별도 모델 parameter로 세지 않았다.", ""]
    environment_rows = []
    for run, data in runs:
        env = data["environment"]
        environment_rows.append((run, env.get("gpu"), "/".join(map(str, env.get("capability", []))),
                                 _number(env.get("memory_gib"), 1), env.get("torch"), env.get("triton"), env.get("cuda")))
    sections += [_table(["실행", "GPU", "Compute capability", "VRAM GiB", "PyTorch", "Triton", "CUDA"], environment_rows), "",
                 "BF16 저장·출력, FP32 GEMM 누산 및 normalization/RoPE/SwiGLU 중간 계산을 사용했다. "
                 "동일한 BF16 반올림 가중치를 FP32로 변환해 기존 독립 모델과 검증했고 TF32와 BF16 reduced-precision reduction을 껐다. "
                 "PyTorch 두 경로는 `fullgraph=True`, `dynamic=False`, `max-autotune-no-cudagraphs`로 컴파일한 뒤 "
                 "세 경로 모두 같은 수동 CUDA Graph 실행 조건을 적용한다. CUDA Graph 없는 측정도 별도로 보존했다.", ""]
    if any(data.get("arguments", {}).get("torch_attention") == "flex-decode" for _, data in runs):
        sections += ["순수 PyTorch 기준선도 추론용 attention 경로를 사용한다. `--torch-attention flex-decode`는 연속 생성에서 "
                     "공식 `torch.nn.attention.flex_attention`을 컴파일하고, GPU 위치 tensor를 참조하는 `score_mod`로 미래 cache 슬롯을 제외한다. "
                     "이는 사용자 작성 Triton kernel이 없는 PyTorch 경로다. 짧은 query에서 decoding 전용 kernel로 내리는 동작과 "
                     "tensor 위치 closure는 [PyTorch 공식 문서](https://pytorch.org/blog/flexattention-for-inference/)에 설명되어 있다. "
                     "고정 B=8 decode의 SDPA는 전체 KV가 유효하므로 `attn_mask=None`, `is_causal=False`를 사용한다.", ""]
    shapes = []
    for row in first.get("results", []):
        if row.get("backend") != "torch":
            continue
        if row["workload"] == "generate":
            shapes.append(("generate", row.get("batch"), row.get("prompt_length"), "prompt부터 증가", row.get("generated_tokens")))
        else:
            p = row.get("shape", {})
            shapes.append((row["workload"], p.get("batch"), p.get("q_len"), p.get("kv_len"), "—"))
    sections += [_table(["조건", "Batch", "입력/query 길이", "KV 길이", "생성 토큰"], shapes), "",
                 "Prefill은 **모든 prompt 위치의 logits**를 계산한다. 마지막 prompt 위치의 logits만 계산하도록 "
                 "LM head를 줄이는 production 최적화는 이번 비교 범위 밖이다. 생성의 첫 prefill도 같은 계산을 수행한다.", "",
                 "고정 decode는 같은 KV 길이에서 한 토큰을 처리하는 지연이다. 생성 측정은 GPU에서 greedy argmax로 "
                 "다음 토큰을 선택하고 KV cache와 위치를 갱신하는 실제 연속 실행이다. 토큰화, 네트워크, queueing, "
                 "모델 로딩, 컴파일, autotune, Graph capture는 steady-state 시간에 포함하지 않는다. "
                 "생성 시간에는 prefill·cache 복사·첫 토큰 선택과 이후 decode가 포함된다.", "",
                 "## 실제 시간", ""]
    timing_rows = []
    for run, data in runs:
        for r in data.get("results", []):
            if r.get("status") != "ok":
                continue
            generate = r["workload"] == "generate"
            gpu = lookup[(run, r["backend"], r["workload"], "generation_device" if generate else "graph_device")]
            wall = lookup[(run, r["backend"], r["workload"], "generation_wall" if generate else "graph_wall")]
            timing_rows.append((run, LABELS[r["backend"]], r["workload"], _number(gpu["median_ms"]),
                                _number(gpu["p95_ms"]), _number(wall["median_ms"]),
                                _number(wall["p95_ms"]), _number(gpu["speedup_vs_torch"]) + "×"))
    sections += [_table(["실행", "구현", "조건", "GPU median ms", "GPU p95 ms", "Wall median ms", "Wall p95 ms", "GPU 배속"], timing_rows), "",
                 "고정 prefill/decode의 GPU 표본은 연속 호출을 CUDA event로 잰 **round 평균**이며 p95도 round 평균들의 p95다. "
                 "Wall 표본은 호출마다 CUDA 완료를 동기화한 지연이다. 생성의 GPU/wall 표본은 각각 한 번의 전체 생성 실행이다. "
                 "GPU 값과 wall 값을 같은 분포로 해석하지 않는다.", ""]
    generation_rows = []
    for run, _ in runs:
        for backend in BACKENDS:
            ttft = lookup.get((run, backend, "generate", "ttft_device"))
            tpot = lookup.get((run, backend, "generate", "decode_token_device"))
            if ttft and tpot:
                generation_rows.append((run, LABELS[backend], _number(ttft["median_ms"]), _number(tpot["median_ms"]),
                                        _number(tpot["speedup_vs_torch"]) + "×"))
    sections += [_table(["실행", "구현", "GPU TTFT ms", "GPU decode ms/token", "Decode 배속"], generation_rows), "",
                 "TTFT는 prefill·cache 복사·첫 argmax를 포함한다. Decode ms/token은 첫 토큰 이후 전체 decode GPU 시간의 토큰당 평균이며, "
                 "매 토큰 지연 분포의 p95를 의미하지 않는다.", ""]
    contribution_rows = []
    decode_dominated = []
    for run, data in runs:
        baseline = {metric: lookup.get((run, "torch", "generate", metric))
                    for metric in ("generation_device", "ttft_device", "decode_token_device")}
        if not all(baseline.values()):
            continue
        for backend in ("hybrid", "triton"):
            measured = {metric: lookup.get((run, backend, "generate", metric)) for metric in baseline}
            row = next((r for r in data.get("results", [])
                        if r.get("backend") == backend and r.get("workload") == "generate" and r.get("status") == "ok"), {})
            steps = row.get("generated_tokens", 0) - 1
            if not all(measured.values()) or steps < 1:
                continue
            differences = {metric: baseline[metric]["median_ms"] - measured[metric]["median_ms"] for metric in baseline}
            total = differences["generation_device"]
            decode = differences["decode_token_device"] * steps
            contribution_rows.append((run, LABELS[backend], f"{total:+.3f}",
                                      f"{differences['ttft_device']:+.3f}",
                                      f"{differences['decode_token_device']:+.5f} × {steps} = {decode:+.3f}"))
            if total > 0 and decode >= .75 * total:
                decode_dominated.append(f"{run} {LABELS[backend]}")
    if contribution_rows:
        sections += [_table(["실행", "PyTorch 대비 비교 경로", "전체 생성 시간 차이 ms", "TTFT 차이 ms", "TPOT 차이 ms/token × decode 횟수"], contribution_rows), "",
                     "차이는 모두 `PyTorch 시간 − 비교 경로 시간`으로 계산해 양수이면 비교 경로가 빠르다. "
                     "전체·TTFT·TPOT의 중앙값을 각각 사용하므로 열 사이의 합은 전체 차이와 정확히 일치하지 않을 수 있다.", ""]
        if decode_dominated:
            sections += ["전체 생성 시간 차이는 다음 조건에서 주로 **B=1 연속 decode 구간**에 나타났다: "
                         + "; ".join(decode_dominated) + ". 위 `TPOT 차이 × 127`이 그 크기를 보여준다. "
                         "이는 측정 구간의 분해이며 특정 attention kernel만의 인과적 효과는 아니다. "
                         "GEMM 선택과 compiler fusion 경계도 함께 달라질 수 있다.", ""]
    sections += ["## 정확성 및 준비 비용", ""]
    validation_rows, prepare_rows, memory_rows = [], [], []
    for run, data in runs:
        for row in data.get("results", []):
            if row.get("status") != "ok":
                continue
            accuracy = correctness_metrics(row)
            validation_rows.append((run, LABELS[row["backend"]], row["workload"], accuracy["checked_tensors"],
                                    _number(accuracy["worst_nrmse"], 5), _number(accuracy["worst_normalized_max"], 5)))
            gen = row["workload"] == "generate"
            first_call = row.get("prepare_compile_and_validation_s") if gen else row.get("compile_autotune_first_call_s")
            prepare_rows.append((run, LABELS[row["backend"]], row["workload"], _number(row.get("prepare_s")),
                                 _number(first_call), "compile+autotune+reference 검증" if gen else "첫 호출 compile/autotune 포함"))
            memory = row.get("memory", {})
            memory_rows.append((run, LABELS[row["backend"]], row["workload"],
                                _number(memory.get("peak_allocated_gib")), _number(memory.get("peak_reserved_gib"))))
    sections += [_table(["실행", "구현", "조건", "검증 tensor 수", "최대 NRMSE", "최대 normalized max"], validation_rows), "",
                 "합격 기준은 NRMSE ≤ 0.02, 최대 절대 오차 / reference RMS ≤ 0.20이다. "
                 "생성 검증은 같은 FP32 reference 토큰을 입력하는 teacher forcing으로 127회 decode를 진행하고, "
                 "0부터 센 step 0·1·7·31·63·126의 여섯 checkpoint에서 logits와 유효 KV의 누적 오차를 확인한다. "
                 "측정 전 캡처된 decode Graph를 두 번 연속 replay해, 갱신된 token·KV·GPU 위치를 사용하는 출력도 FP32 reference와 검증한다. "
                 "위 최대 오차와 검증 tensor 수에는 이 Graph replay 검사도 포함된다. "
                 "실제 greedy 생성은 각 경로의 출력을 사용하므로 BF16 오차와 가까운 logit 순위 때문에 토큰열이 달라질 수 있다. "
                 "이 결과를 bitwise 동등성이나 언어 품질 동등성으로 해석하지 않는다.", ""]
    token_rows = []
    for run, data in runs:
        sampled = {row["backend"]: row.get("sampled_tokens") for row in data.get("results", [])
                   if row.get("status") == "ok" and row.get("workload") == "generate"}
        for left, right in itertools.combinations(BACKENDS, 2):
            a, b = sampled.get(left), sampled.get(right)
            if not a or not b:
                continue
            count = min(len(a), len(b))
            same = sum(x == y for x, y in zip(a, b))
            prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), count)
            first_difference = "없음" if prefix == count and len(a) == len(b) else str(prefix + 1)
            token_rows.append((run, f"{LABELS[left]} / {LABELS[right]}", f"{same}/{count}",
                               f"{same / count * 100:.1f}%", prefix, first_difference))
    if token_rows:
        sections += [_table(["실행", "Greedy token열 비교", "동일 위치 일치", "일치율", "공통 prefix 길이", "첫 차이 위치(1부터)"], token_rows), "",
                     "이 표는 각 구현이 독립적으로 생성한 token열의 진단값이다. 첫 차이 이후에는 서로 다른 토큰을 입력하므로 "
                     "이후 불일치는 같은 입력에 대한 수치 오차를 뜻하지 않는다. 일치율은 정확성 합격 기준이나 언어 품질 지표가 아니다.", ""]
    sections += [_table(["실행", "구현", "조건", "모델 준비 s", "첫 실행/검증 구간 s", "구간 의미"], prepare_rows), "",
                 "첫 호출 비용은 프로세스·디스크 compiler cache 상태에 영향을 받는다. 별도 cache 제거를 수행한 cold-start 배포 비용으로 주장하지 않는다.", "",
                 _table(["실행", "구현", "조건", "Peak allocated GiB", "Peak reserved GiB"], memory_rows), "",
                 "메모리는 해당 steady 측정 구간의 PyTorch CUDA allocator peak이며 실행 중인 프로세스 전체를 포함한다. "
                 "BF16 원본·packed 가중치, **정확성 검증용 FP32 reference 가중치**, KV cache, CUDA Graph pool과 workspace가 포함되므로 "
                 "제품 배포 시 모델만의 최소 VRAM이나 GPU 전체 사용량으로 해석하지 않는다. Reserved는 allocator가 보유한 용량이다.", "",
                 "## CUDA Graph 효과와 실행 커널", ""]
    ablation_rows = []
    for run, _ in runs:
        for backend in BACKENDS:
            for phase in ("prefill", "decode"):
                graph = lookup.get((run, backend, phase, "graph_device"))
                plain = lookup.get((run, backend, phase, "no_graph_device"))
                if graph and plain:
                    ablation_rows.append((run, LABELS[backend], phase, _number(graph["median_ms"]),
                                          _number(plain["median_ms"]), _number(plain["median_ms"] / graph["median_ms"]) + "×"))
    sections += [_table(["실행", "구현", "조건", "Graph GPU ms", "Graph 없는 GPU ms", "Graph 배속"], ablation_rows), ""]
    profile_rows = []
    for run, data in runs:
        for row in data.get("results", []):
            profile = row.get("profile")
            if not profile:
                continue
            kernels = profile.get("cuda_kernels", {})
            top = sorted(kernels, key=lambda key: kernels[key].get("total_us", 0), reverse=True)[:3]
            audit = "—"
            if row.get("backend") == "triton":
                audit = "검증 통과" if kernels and not profile.get("forbidden_native_kernels") else "미통과"
            profile_rows.append((run, LABELS[row["backend"]], row["workload"],
                                 sum(k.get("calls", 0) for k in kernels.values()), audit,
                                 "; ".join("`" + name[:95] + "`" for name in top)))
    sections += [_table(["실행", "구현", "조건", "CUDA event 수", "Native kernel 출처", "시간 상위 kernel"], profile_rows), "",
                 "Profiler는 steady timing과 분리한 일반 forward/decode를 관찰했다. Chrome trace의 leaf `kernel`, `gpu_memcpy`, `gpu_memset` "
                 "event만 집계해 compiled-region annotation과 자식 kernel 시간을 중복 계산하지 않는다. Native 경로는 프로젝트 Triton kernel 이름의 allowlist와 "
                 "memcpy/memset만 허용하며 알 수 없는 CUDA kernel이 있으면 실패한다. CPU 쪽 ATen allocation/metadata 이벤트만으로 "
                 "library compute를 판단하지 않는다. 전체 kernel 이름과 시간은 final JSON, Chrome trace는 옆의 trace 파일에 보존했다.", "",
                 "## 최적화 범위와 결과 해석", "",
                 _table(["경로", "실제 구현/탐색"], [
                     ("공통", "고정 shape, BF16, 사전 QKV·gate/up weight packing, precomputed FP32 RoPE, static KV cache, CUDA Graph, 워밍업 후 측정"),
                     ("PyTorch compile", "; ".join(torch_choices) + ". Inductor max-autotune와 compiler fusion; 내부 생성 Triton 및 vendor GEMM 허용"),
                     ("PyTorch + Triton", "; ".join(hybrid_choices) + ". 공통 수동 Triton embedding, affine LN, residual+LN, RoPE+KV write, SwiGLU, argmax"),
                     ("Native Triton", "모든 model compute를 Triton으로 실행. grouped tiled/persistent GEMM 및 선택 shape의 TMA 후보, small-M split-K, 타일·warps·stages autotune, causal online-softmax prefill, split-KV decode 및 작은 연산 fusion"),
                 ]), ""]
    for run, data in runs:
        sections.append(f"- `{run}` 인자: `{json.dumps(data.get('arguments', {}), ensure_ascii=False, sort_keys=True)}`")
    split_rows = []
    for run, data in runs:
        configured = data.get("arguments", {}).get("tuning_config")
        if not configured:
            continue
        name = Path(configured).name
        config = tuning_configs.get(name, {})
        overrides = config.get("split_k_overrides", {})
        if not overrides:
            for row in data.get("results", []):
                overrides = row.get("tuning", {}).get("triton_matmul", {}).get("split_k_overrides", {})
                if overrides:
                    break
        for shape, factor in sorted(overrides.items(), key=lambda pair: tuple(int(x) for x in pair[0].split(","))):
            measured = config.get("shapes", {}).get(shape, {}).get("selected_median_ms")
            split_rows.append((run, name, shape, factor, _number(measured * 1000 if measured is not None else None),
                               config.get("method", {}).get("cache_mode", "기록 없음")))
    if split_rows:
        sections += ["", _table(["실행", "선택 config", "GEMM M,K,N", "측정 후 선택 split-K", "개별 GEMM μs", "Cache 조건"], split_rows), "",
                     "Split-K는 각 shape의 후보를 측정해 선택한 factor다. 개별 GEMM의 후보 선택 시간은 전체 모델 추론 시간과 "
                     "다른 실험이며 합산하지 않았다. Cold cache 모드이면 cache flush는 후보 시간 바깥에서 실행됐다."]
    attention_rows, flex_rows = [], []
    for run, data in runs:
        configured = data.get("arguments", {}).get("tuning_config")
        if not configured:
            continue
        name = Path(configured).name
        config = tuning_configs.get(name, {})
        attention_tuning = config.get("attention_tuning", {})
        for shape, factor in sorted(config.get("attention_split_overrides", {}).items(),
                                    key=lambda pair: tuple(int(x) for x in pair[0].split(","))):
            selected = attention_tuning.get("shape_selection", {}).get(shape, {})
            measured = selected.get("mean_position_median_ms_by_split", {}).get(str(factor))
            attention_rows.append((run, name, shape, ", ".join(map(str, selected.get("positions", []))), factor,
                                   _number(measured * 1000 if measured is not None else None),
                                   attention_tuning.get("method", {}).get("cache_mode", "기록 없음")))
        if "flex_kernel_options" in config:
            options = config["flex_kernel_options"]
            tuning = config.get("flex_tuning", {})
            selected = next((row for row in tuning.get("results", [])
                             if row.get("status") == "ok" and row.get("options") == options), {})
            warm = selected.get("warm", {}).get("median_ms")
            evicted = selected.get("after_eviction", {}).get("median_ms")
            flex_rows.append((run, name, json.dumps(options, sort_keys=True), tuning.get("timed_position", "기록 없음"),
                              _number(warm * 1000 if warm is not None else None),
                              _number(evicted * 1000 if evicted is not None else None),
                              sum(row.get("status") == "error" for row in tuning.get("results", []))))
    if attention_rows:
        sections += ["", _table(["실행", "선택 config", "Attention B,NQ,NK,S,H", "측정 위치", "선택 split-KV", "위치별 중앙값의 평균 μs", "Cache 조건"], attention_rows), "",
                     "Native split-KV는 각 shape의 측정 위치들에서 얻은 중앙값의 산술 평균이 가장 작은 factor를 선택한다. "
                     "Attention 부분 결과를 합치는 reduction까지 포함한 구성요소 시간이며 전체 모델 시간에 합산하지 않았다. "
                     "후보 간 차이가 1% 안팎이면 측정 변동 수준일 수 있으며, 선택된 순위만으로 유의한 성능 향상을 주장하지 않는다."]
    if flex_rows:
        sections += ["", _table(["실행", "선택 config", "Flex kernel_options", "측정 위치", "Warm μs", "Eviction 후 μs", "실패 후보 수"], flex_rows), "",
                     "FlexAttention 옵션은 동일 BF16 Q/K/V에서 측정한 eviction 후 중앙값으로 선택했다. "
                     "Warm은 여러 captured attention 호출의 평균을 표본으로 사용하고, eviction 후 값은 각 호출 전에 큰 buffer를 "
                     "접근한 뒤 측정했다. Eviction 작업은 시간에서 제외되며 모든 cache line 제거를 보장하지 않는다. "
                     "`SPLIT_KV` 등 compiler 옵션은 버전에 따라 달라질 수 있어 설치된 source의 지원 근거와 실패 후보를 JSON에 기록했다. "
                     "이 표는 구성요소 실험이며 최종 전체 모델 시간으로 확인해야 한다."]
    if attention_rows or flex_rows:
        sections += ["", "최종 `--tuning-config`는 GEMM `split_k_overrides`, native attention `attention_split_overrides`, "
                     "PyTorch `flex_kernel_options`, 선택 shape의 `tma_overrides`를 함께 로드한다. 합쳐진 JSON은 원래 GEMM 결과와 "
                     "`attention_tuning`, `flex_tuning`, `tma_tuning`, `source_files`를 보존하고, 노트북에도 동일한 전체 config가 내장된다."]
    tma_rows, tma_evidence = [], []
    for run, data in runs:
        configured = data.get("arguments", {}).get("tuning_config")
        if not configured:
            continue
        config = tuning_configs.get(Path(configured).name, {})
        tuning = config.get("tma_tuning", {})
        overrides = config.get("tma_overrides", {})
        successes = total = 0
        resources = []
        for shape, row in tuning.get("shapes", {}).items():
            best = row.get("best_tma_config")
            baseline, measured = row.get("baseline_median_ms"), row.get("best_tma_median_ms")
            tma_rows.append((run, shape, best or "지원되지 않음", _number(baseline * 1000 if baseline is not None else None),
                             _number(measured * 1000 if measured is not None else None),
                             _number(row.get("best_tma_speedup")) + "×", overrides.get(shape, "미채택")))
            for name, candidate in row.get("candidates", {}).items():
                if name != "linear":
                    total += 1
                    successes += candidate.get("status") == "ok"
            if shape in overrides:
                resource = row.get("candidates", {}).get(overrides[shape], {}).get("resources")
                if resource:
                    resources.append(resource)
        if total:
            loads = bool(resources) and all(any(".shared::cluster.global" in line
                                                for line in resource.get("tma_ptx_instructions", [])) for resource in resources)
            stores = bool(resources) and all(any(".global.shared::cta" in line
                                                 for line in resource.get("tma_ptx_instructions", [])) for resource in resources)
            tma_evidence.append((run, f"{successes}/{total}", "확인" if loads else "미확인", "확인" if stores else "미확인",
                                 ", ".join(str(x) for x in sorted({resource.get("mma_sync_instructions", 0) for resource in resources})),
                                 sum(resource.get("wgmma_instructions", 0) for resource in resources),
                                 sum(resource.get("tcgen05_instructions", 0) for resource in resources)))
    if tma_rows:
        sections += ["", "### SM120 TMA 구성요소 후보", "",
                     _table(["실행", "GEMM M,K,N", "최고 TMA config", "기존 linear μs", "TMA μs", "구성요소 배속", "최종 shape별 선택"], tma_rows), "",
                     _table(["실행", "성공한 TMA 후보", "Global→shared TMA load", "Shared→global TMA store", "선택 kernel PTX mma.sync 수", "WGMMA 수", "TCGen05 수"], tma_evidence), "",
                     "Row-major 가중치를 transpose/packing하지 않고 동일 BF16 입력으로 비교했다. TMA 사용은 실제 컴파일 PTX의 "
                     "`cp.async.bulk.tensor` load/store로 확인했으며 `mma.sync` 경로를 사용한다. PTX 명령 수는 정적 코드 내 개수로 "
                     "실행한 명령 수나 Tensor Core 이용률이 아니다. 공유 메모리·register·spill 및 실패 후보는 raw JSON에 있다. "
                     "구성요소에서 1% 넘게 빠른 shape만 전체 모델 후보로 전달했으며, 이 배속을 전체 모델 배속으로 해석하지 않는다."]
    swiglu_rows, swiglu_summaries, swiglu_decisions = [], [], []
    for run, data in runs:
        configured = data.get("arguments", {}).get("tuning_config")
        config = tuning_configs.get(Path(configured).name, {}) if configured else {}
        tuning = config.get("swiglu_tuning", {})
        if not tuning:
            continue
        fused_candidates = [(name, row) for name, row in tuning.get("candidates", {}).items()
                            if name != "linear_plus_swiglu"]
        passed = sum(row.get("status") == "ok" and row.get("fp32_accuracy", {}).get("passed", False)
                     and row.get("existing_bf16_closeness", {}).get("bitwise_equal_fraction") == 1.0
                     for _, row in fused_candidates)
        baseline, best = tuning.get("baseline_median_ms"), tuning.get("best_fused_median_ms")
        if not config.get("swiglu_adopted", False) and baseline is not None and best is not None:
            reason = ("사전에 정한 시간 1% 초과 감소 기준에 못 미쳐" if best >= baseline * .99
                      else "구성요소 시간 외의 최종 채택 판단에 따라")
            swiglu_decisions.append(f"{run}: 최고 후보는 {baseline / best:.5f}×이며 {reason} 최종 runner에는 적용하지 않았다.")
        swiglu_summaries.append((run, tuning.get("best_fused_config"),
                                _number(baseline * 1000 if baseline is not None else None),
                                _number(best * 1000 if best is not None else None),
                                _number(tuning.get("best_fused_speedup"), 5) + "×",
                                f"{passed}/{len(fused_candidates)}",
                                "채택" if config.get("swiglu_adopted", False) else "미채택",
                                _number(tuning.get("saved_intermediate_io_bytes_per_call", 0) / 1e6, 3)))
        for name, row in fused_candidates:
            resource = row.get("resources") or {}
            median = row.get("median_ms")
            swiglu_rows.append((run, name, row.get("status"),
                                _number(median * 1000 if median is not None else None),
                                _number(row.get("fp32_accuracy", {}).get("nrmse"), 5),
                                _number(row.get("existing_bf16_closeness", {}).get("bitwise_equal_fraction"), 3),
                                resource.get("registers_per_thread", "—"), resource.get("local_spills", "—")))
    if swiglu_summaries:
        sections += ["", "### Gate/up GEMM + SwiGLU fusion 후보", "",
                     _table(["실행", "최고 fused config", "기존 두 연산 μs", "Fused μs", "구성요소 배속", "FP32 합격 및 기존 BF16 bitwise 일치", "최종 채택", "논리적 중간 I/O 절감 MB"], swiglu_summaries), "",
                     _table(["실행", "Fused config", "상태", "전체 fused 연산 μs", "FP32 NRMSE", "기존 BF16 원소 일치율", "Registers/thread", "Local spills"], swiglu_rows), "",
                     "GEMM의 FP32 누산을 BF16으로 반올림한 뒤 FP32 SiLU/곱을 수행하는 기존 순서를 보존했다. "
                     "기준 시간은 linear와 별도 SwiGLU를 모두 포함하며, 후보 순서를 회전하고 cache eviction은 시간 밖에서 수행했다. "
                     "미채택 후보는 소스와 측정 기록만 보존하며 전체 모델 시간에 이 후보의 개선을 더하지 않는다. "
                     "표의 I/O 절감은 중간 BF16 tensor의 쓰기·읽기를 제거하는 논리적 byte 수이며 측정한 DRAM traffic이 아니다. "
                     "느린 큰 타일 후보의 register/spill 증가는 fusion의 자원 비용을 시사하지만, "
                     "register pressure가 시간 차이의 원인이라는 확정에는 추가 counter 실험이 필요하다."]
        sections += [""] + [f"- {line}" for line in swiglu_decisions]
    if native_candidates:
        native_rows, native_index = [], {}
        for name, data in native_candidates:
            rows = {row["workload"]: row for row in data.get("results", [])
                    if row.get("backend") == "triton" and row.get("status") == "ok"}
            native_index[name] = rows
            native_rows.append((name,
                                _number(rows.get("prefill", {}).get("timing", {}).get("device_round_average", {}).get("median_ms")),
                                _number(rows.get("decode", {}).get("timing", {}).get("device_round_average", {}).get("median_ms")),
                                _number(rows.get("generate", {}).get("timing", {}).get("device_ms", {}).get("median_ms"))))
        sections += ["", "### Native 전체 모델 후보 관찰", "",
                     _table(["후보 파일", "Prefill GPU ms", "B=8 decode GPU ms", "128-token 생성 GPU ms"], native_rows), "",
                     "이 후보 기록은 최종 반복 실행의 표본과 합치지 않는다. TMA는 `tma_overrides`에 지정한 큰 M의 projection에만 "
                     "적용하고 다른 shape는 기존 native GEMM을 사용한다. Prefill 개선 여부와 decode/생성의 작은 변동을 구분한다."]
        before = native_index.get("optimized-candidate.json", {}).get("prefill")
        after = native_index.get("native-tma-candidate.json", {}).get("prefill")
        if before and after:
            a, b = before["timing"]["device_round_average"]["median_ms"], after["timing"]["device_round_average"]["median_ms"]
            sections += ["", f"기존 최적화 후보와 TMA 후보의 prefill은 {_number(a, 4)} → {_number(b, 4)} ms로 "
                         f"관찰되었다 ({(1 - b / a) * 100:.2f}% 감소). 순차 후보 측정이므로 효과 확정은 최종 반복 실행과 함께 판단한다. "
                         "Decode/생성 시간의 작은 차이는 별도 개선으로 주장하지 않는다."]
    if torch_candidates:
        flex_candidate_rows = []
        for name, data in torch_candidates:
            rows = {row["workload"]: row for row in data.get("results", [])
                    if row.get("backend") == "torch" and row.get("status") == "ok"}
            generation = rows.get("generate", {}).get("timing", {})
            flex_candidate_rows.append((name, data.get("arguments", {}).get("torch_attention", "sdpa"),
                                        _number(rows.get("prefill", {}).get("timing", {}).get("device_round_average", {}).get("median_ms")),
                                        _number(rows.get("decode", {}).get("timing", {}).get("device_round_average", {}).get("median_ms")),
                                        _number(generation.get("device_ms", {}).get("median_ms")),
                                        _number(generation.get("decode_device_ms", {}).get("median_ms"))))
        sections += ["", "### 공식 PyTorch FlexAttention 기준선 진단", "",
                     _table(["후보 파일", "PyTorch attention", "Prefill GPU ms", "B=8 decode GPU ms", "128-token 생성 GPU ms", "생성 decode ms/token"], flex_candidate_rows), "",
                     "이 기록은 Flex `kernel_options`를 별도로 선택하기 전의 공식 PyTorch 기준선 후보다. "
                     "최종 실행의 표본에 합치지 않았으며, 아래의 이전 hybrid masked-SDPA 후보 표와 구분한다."]
    if candidates:
        candidate_rows = []
        candidate_index = {}
        for name, data in candidates:
            rows = {row["workload"]: row for row in data.get("results", [])
                    if row.get("backend") == "hybrid" and row.get("status") == "ok"}
            candidate_index[name] = (data, rows)
            matmul, attention = hybrid_policy(data.get("arguments", {}))
            prefill = rows.get("prefill", {}).get("timing", {}).get("device_round_average", {}).get("median_ms")
            decode = rows.get("decode", {}).get("timing", {}).get("device_round_average", {}).get("median_ms")
            generation = rows.get("generate", {}).get("timing", {})
            candidate_rows.append((name, attention, matmul, _number(prefill), _number(decode),
                                   _number(generation.get("device_ms", {}).get("median_ms")),
                                   _number(generation.get("decode_device_ms", {}).get("median_ms"))))
        sections += ["", "### 최종 선택 전 후보 비교", "",
                     _table(["후보 파일", "Attention", "GEMM", "Prefill GPU ms", "B=8 decode GPU ms", "128-token 생성 GPU ms", "생성 decode ms/token"], candidate_rows), "",
                     "이 표는 이전 **명시적 boolean mask를 사용한 SDPA** 구현을 포함한 후보 진단 기록이다. "
                     "최종 PyTorch 기준선의 FlexAttention 또는 전체 유효 KV용 mask 없는 SDPA 결과를 대신하지 않는다. "
                     "같은 GPU에서 순차 실행한 후보 측정이며 최종 두 실행의 표본이나 평균에 합치지 않았다. "
                     "기록된 정책·seed·환경·source hash를 기준으로 비교하며, 동시 무작위 대조 실험이나 통계적 인과 증명으로 해석하지 않는다."]
        a = candidate_index.get("hybrid-matmul-candidate.json")
        b = candidate_index.get("hybrid-both-candidate.json")
        if a and b and "generate" in a[1] and "generate" in b[1]:
            da, ra = a
            db, rb = b
            aa, ab = da.get("arguments", {}), db.get("arguments", {})
            fields = ("gpu", "capability", "torch", "triton", "cuda", "source_sha256")
            controlled = (aa.get("hybrid_matmul") == ab.get("hybrid_matmul") == "decode-triton"
                          and aa.get("hybrid_attention") == "sdpa" and ab.get("hybrid_attention") == "decode-triton"
                          and aa.get("seed") == ab.get("seed") and da.get("model") == db.get("model")
                          and all(da.get("environment", {}).get(field) == db.get("environment", {}).get(field) for field in fields))
            if controlled:
                ga, gb = ra["generate"]["timing"], rb["generate"]["timing"]
                total_a, total_b = ga["device_ms"]["median_ms"], gb["device_ms"]["median_ms"]
                decode_a, decode_b = ga["decode_device_ms"]["median_ms"], gb["decode_device_ms"]["median_ms"]
                sections += ["", f"GEMM을 같은 `decode-triton`으로 고정하고 decode attention만 SDPA에서 Triton으로 바꾼 후보에서 "
                             f"생성 GPU 시간은 {_number(total_a)} → {_number(total_b)} ms ({total_a / total_b:.3f}×), "
                             f"decode 시간은 {_number(decode_a)} → {_number(decode_b)} ms/token ({decode_a / decode_b:.3f}×)이었다. "
                             "두 후보의 GPU·소프트웨어·seed·model·source hash가 같으며 prefill은 모두 SDPA였다. "
                             "이 비교에서 관찰된 개선은 이전 masked-SDPA decode attention 교체와 연결된다. "
                             "최종 FlexAttention 기준선 대비 개선이나 모든 batch/context의 개선으로 해석하지 않는다."]
    sections += ["", "Persistent GEMM은 autotune 후보이며 모든 shape에서 선택된다는 뜻이 아니다. 선택된 configuration은 final JSON의 `tuning`에 있다. "
                 "‘PyTorch native’는 사용자 작성 Triton이 없다는 경계이며, Inductor가 Triton을 생성하지 않는다는 뜻이 아니다.", "",
                 "약 2시간의 실행 예산 안에서 적용 가능한 후보를 구현·측정했다. 알려진 모든 최적화나 세계 최고 성능을 보증하지 않는다. "
                 "Warp specialization, CLC, 별도 CUDA megakernel, 대규모 compiler/version 탐색, multi-GPU/continuous batching은 이번 최종 비교에서 측정하지 않았다. "
                 "SM120에는 SM100/B200의 TCGen05/TMEM 경로를 그대로 적용할 수 없다. FP8/FP4 양자화, sparsity, speculative decoding 및 모델 구조 변경은 "
                 "동일 BF16 계산 비교의 범위 밖이다. [Triton Blackwell API](https://triton-lang.org/main/gluon/api/nvidia.blackwell.html)", "",
                 "관찰값은 표의 실행 시간·오차·kernel trace다. 원인 해석은 이를 바탕으로 한 제한된 추론이다: "
                 "prefill은 큰 GEMM/attention 효율, 작은 batch decode는 weight/KV 읽기와 launch 비용의 영향을 받는다. "
                 "Native Triton이 느린 조건은 그 구현의 shape별 tiling·occupancy·추가 reduction 비용을 반영할 수 있다. "
                 "정확한 병목을 확정하려면 Nsight counter와 개별 최적화 ablation이 추가로 필요하며 이번 결과만으로 확정하지 않는다.", "",
                 "## 재현과 산출물", "",
                 "아래 명령은 **저장소 루트**에서 실행한다. 각 최종 실행의 실제 옵션을 보존하고, "
                 "결과/config 경로를 로컬 결과 디렉터리로 지정했으므로 디렉터리를 옮기거나 PYTHONPATH를 설정할 필요가 없다.", "",
                 "```bash\n" + "\n".join(reproduction_command(run, data, reproduction_dir) for run, data in runs) + "\n```", "",
                 "`summary.csv`는 두 실행의 "
                 "GPU/wall·Graph 유무·생성 TTFT/TPOT를 구분한 전체 수치다. "
                 "[실행별 GPU·wall 시간 차트](charts.png)에서 세 구현을 비교할 수 있다. "
                 "노트북은 source와 최종 JSON을 내장하고 기본값 `RUN_BENCHMARK=False`로 저장 결과를 재분석한다. "
                 "GPU 재측정 시 런타임 GPU·PyTorch·Triton 버전을 다시 확인한다.", ""]
    if inventory:
        sections += ["다른 JSON 파일은 후보 탐색/진단 기록이며 최종 시간 표에 합산하지 않았다:", ""]
        sections += [f"- `{name}`" for name in inventory]
        sections.append("")
    return "\n".join(sections)


def plot_records(records, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    run_names = list(dict.fromkeys(r["run"] for r in records))
    lookup = _index(records)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    panels = (("prefill", "graph_device", "Prefill: GPU ms"), ("decode", "graph_device", "Decode: GPU ms/step"),
              ("generate", "generation_device", "128-token generation: GPU ms"),
              ("prefill", "graph_wall", "Prefill: synchronized wall ms"),
              ("decode", "graph_wall", "Decode: synchronized wall ms/step"),
              ("generate", "generation_wall", "128-token generation: wall ms"))
    colors = ("#4477AA", "#EE6677")
    for ax, (phase, metric, title) in zip(axes.flat, panels):
        x = np.arange(len(BACKENDS))
        width = 0.36 if len(run_names) == 2 else 0.7 / max(1, len(run_names))
        for index, run in enumerate(run_names):
            values = [lookup.get((run, backend, phase, metric), {}).get("median_ms", float("nan")) for backend in BACKENDS]
            bars = ax.bar(x + (index - (len(run_names) - 1) / 2) * width, values, width,
                          label=run, color=colors[index % len(colors)])
            ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        ax.set_xticks(x, ["PyTorch\ncompile", "PyTorch +\nTriton", "Native\nTriton"])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("milliseconds; lower is faster")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Same-model BF16 inference — independent run medians", fontsize=15)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def build_notebook(runs, root, destination, inventory=(), results_dir=None, candidates=(), torch_candidates=(), native_candidates=()):
    """Use plain JSON so building does not require nbformat/nbclient installed."""
    source_files = {}
    for package in ("llm_bench", "llm_roofline"):
        for path in sorted((root / package).glob("*.py")):
            source_files[str(path.relative_to(root))] = path.read_text()
    sidecars = load_tuning_configs(runs, results_dir or Path.cwd())
    sidecars_raw = {name: base64.b64encode((Path(results_dir or Path.cwd()) / name).read_bytes()).decode()
                    for name in sidecars}
    payload = {"sources": source_files, "runs": dict(runs), "inventory": list(inventory),
               "sidecars": sidecars, "sidecars_raw": sidecars_raw,
               "candidates": dict(candidates), "torch_candidates": dict(torch_candidates),
               "native_candidates": dict(native_candidates)}
    compressed = gzip.compress(json.dumps(payload, ensure_ascii=False).encode(), mtime=0)
    encoded = base64.b64encode(compressed).decode()
    digest = hashlib.sha256(compressed).hexdigest()

    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}

    def code(text):
        return {"cell_type": "code", "metadata": {}, "source": text.splitlines(keepends=True),
                "execution_count": None, "outputs": []}

    bootstrap = f'''# 기본 실행은 저장된 실측 결과만 재분석합니다. GPU를 사용하려면 True로 바꾸세요.
RUN_BENCHMARK = False
import base64, gzip, hashlib, json, sys
from pathlib import Path
PAYLOAD = {encoded!r}
compressed = base64.b64decode(PAYLOAD)
assert hashlib.sha256(compressed).hexdigest() == {digest!r}
archive = json.loads(gzip.decompress(compressed))
base = Path('/content') if Path('/content').exists() else Path.cwd()
REPRO_ROOT = base / 'tensor2silicon_benchmark_reproduction'
for relative, source in archive['sources'].items():
    target = REPRO_ROOT / relative
    assert target.resolve().is_relative_to(REPRO_ROOT.resolve())
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source)
RESULTS_DIR = REPRO_ROOT / 'llm_bench' / 'results' / 'embedded-final'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
for name, data in archive['runs'].items():
    (RESULTS_DIR / (name + '.json')).write_text(json.dumps(data, ensure_ascii=False, indent=2))
for name, data in archive['sidecars'].items():
    assert Path(name).name == name
    (RESULTS_DIR / name).write_bytes(base64.b64decode(archive['sidecars_raw'][name]))
for name, data in archive['candidates'].items():
    assert Path(name).name == name
    (RESULTS_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
for name, data in archive['torch_candidates'].items():
    assert Path(name).name == name
    (RESULTS_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
for name, data in archive['native_candidates'].items():
    assert Path(name).name == name
    (RESULTS_DIR / name).write_text(json.dumps(data, ensure_ascii=False, indent=2))
sys.path.insert(0, str(REPRO_ROOT))
runs = list(archive['runs'].items())
print('복원 위치:', REPRO_ROOT)
source_mismatches = {{}}
for name, data in runs:
    env = data['environment']
    print(name, env['gpu'], 'PyTorch', env['torch'], 'Triton', env['triton'], 'CUDA', env['cuda'])
    configured = data.get('arguments', {{}}).get('tuning_config')
    if configured and data.get('tuning_config_sha256'):
        actual_hash = hashlib.sha256((RESULTS_DIR / Path(configured).name).read_bytes()).hexdigest()
        assert actual_hash == data['tuning_config_sha256'], 'Tuning config SHA-256 mismatch'
    source_mismatches[name] = [path for path, digest in env.get('source_sha256', {{}}).items()
                               if path != 'llm_bench/report.py' and
                               (path not in archive['sources'] or
                                hashlib.sha256(archive['sources'][path].encode()).hexdigest() != digest)]
    print('측정 때의 compute source와 다른 파일:', source_mismatches[name])
'''
    rerun = '''# GPU 재측정은 최대 수십 분 이상 걸릴 수 있습니다. 모델/패키지 다운로드는 하지 않습니다.
if RUN_BENCHMARK:
    import os, subprocess, torch, triton
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    assert not any(source_mismatches.values()), '기록된 source와 다른 파일을 먼저 확인하세요.'
    print('재측정 GPU:', torch.cuda.get_device_name(), 'torch:', torch.__version__, 'triton:', triton.__version__)
    refreshed = []
    for name, saved in runs:
        args = saved['arguments']
        output = RESULTS_DIR / (name + '-rerun.json')
        command = [sys.executable, '-m', 'llm_bench', '--backend', 'all', '--preset', args.get('preset', 'colab'),
                   '--workload', 'all', '--seed', str(args.get('seed', 123)), '--profile', '--out', str(output)]
        if args.get('reverse'):
            command.append('--reverse')
        if args.get('hybrid_attention'):
            command += ['--hybrid-attention', args['hybrid_attention']]
        if args.get('hybrid_matmul'):
            command += ['--hybrid-matmul', args['hybrid_matmul']]
        if args.get('torch_attention'):
            command += ['--torch-attention', args['torch_attention']]
        if args.get('tuning_config'):
            tuning_config = RESULTS_DIR / Path(args['tuning_config']).name
            assert tuning_config.exists()
            command += ['--tuning-config', str(tuning_config)]
        subprocess.run(command, cwd=REPRO_ROOT, check=True)
        refreshed.append((name + '-rerun', json.loads(output.read_text())))
    runs = refreshed
else:
    print('저장된 실제 측정 결과를 사용합니다. RUN_BENCHMARK=False')
'''
    analysis = '''from llm_bench.report import records_from_runs, render_report, plot_records
from IPython.display import Markdown, display, Image
records = records_from_runs(runs)
display(Markdown(render_report(runs, records, archive['inventory'], archive['sidecars'],
                              list(archive['candidates'].items()), str(RESULTS_DIR),
                              list(archive['torch_candidates'].items()), list(archive['native_candidates'].items()))))
'''
    chart = '''chart_path = RESULTS_DIR / 'charts.png'
plot_records(records, chart_path)
display(Image(filename=str(chart_path)))
'''
    raw = '''# 표본과 실행별 정밀 수치를 보존합니다. 결과를 합쳐 표본 수를 늘리지 않습니다.
import csv
csv_path = RESULTS_DIR / 'summary.csv'
with csv_path.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print('CSV:', csv_path)
print('결과 JSON:', RESULTS_DIR)
print('수치 행:', len(records))
'''
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python", "version": "3"},
                             "colab": {"name": destination.name, "provenance": []}},
                "cells": [md("# 프로젝트 모델의 PyTorch·Triton 실제 추론 비교\n\n"
                             "실측 JSON과 재현 가능한 모델·벤치마크 source를 내장한 노트북입니다. 기본 실행은 GPU 없이 결과를 재분석합니다. "
                             "NumPy와 Matplotlib이 필요하며 Colab 기본 환경에서 사용할 수 있습니다.\n\n"
                             "PyTorch는 컴파일 버전, native Triton은 GPU 계산 전체를 수동 Triton kernel로 수행합니다. "
                             "하드웨어별 최적화의 제한과 미수행 후보를 결과와 함께 기록했습니다."),
                          code(bootstrap), md("## 선택: GPU 재측정\n\n기본값은 저장 결과 분석입니다. 재측정하려면 위 cell의 "
                                              "`RUN_BENCHMARK=True`로 바꿉니다. 기록된 GPU/소프트웨어 환경과 현재 런타임을 비교하세요. "
                                              "컴파일·autotune·정확성 검증은 steady-state 측정 전에 수행됩니다."),
                          code(rerun), code(analysis), code(chart), code(raw)]}
    for i, cell in enumerate(notebook["cells"]):
        cell["id"] = f"llm-bench-{i:02d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--notebook", type=Path, required=True)
    args = parser.parse_args(argv)
    runs = load_runs(args.results)
    records = records_from_runs(runs)
    if not records:
        raise ValueError("the final runs contain no successful measured workloads")
    inventory = sorted(p.name for p in args.results.glob("*.json")
                       if p.name not in FINAL_FILES and not p.name.endswith("-trace.json"))
    tuning_configs = load_tuning_configs(runs, args.results)
    candidates = load_candidates(args.results)
    torch_candidates = load_torch_candidates(args.results)
    native_candidates = load_native_candidates(args.results)
    source_root = Path(__file__).resolve().parents[1]
    try:
        reproduction_dir = str(args.results.resolve().relative_to(source_root))
    except ValueError:
        reproduction_dir = str(args.results.resolve())
    (args.results / "report.md").write_text(render_report(runs, records, inventory, tuning_configs, candidates,
                                                       reproduction_dir, torch_candidates, native_candidates))
    with (args.results / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot_records(records, args.results / "charts.png")
    build_notebook(runs, source_root, args.notebook, inventory, args.results, candidates, torch_candidates, native_candidates)
    print(f"Wrote {args.results / 'report.md'}, summary.csv, charts.png, {args.notebook}")


if __name__ == "__main__":
    main()
