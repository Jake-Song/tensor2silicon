"""Build a standalone Colab notebook from the measured implementation sources."""
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zlib


def build():
    root = Path(__file__).resolve().parents[1]
    paths = sorted(p for p in (root / "llm_bench_tpu").glob("*.py") if p.name != "build_notebook.py")
    paths += sorted((root / "llm_bench_tpu" / "frozen").glob("*.py"))
    config_path = root / "llm_bench_tpu" / "configs" / "optimized-v6e.json"
    if not config_path.exists():
        raise FileNotFoundError("select a measured config before building the optimized notebook")
    paths.append(config_path)
    paths += [root / "llm_roofline" / name for name in ("__init__.py", "spec.py", "jax_llm.py")]
    paths += [root / "tests" / "test_llm_bench_tpu.py"]
    payload = json.dumps({str(p.relative_to(root)): p.read_text() for p in paths}).encode()
    encoded = base64.b64encode(zlib.compress(payload, 9)).decode()
    digest = hashlib.sha256(payload).hexdigest()
    cells = []
    def md(source):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"})
    def code(source):
        cells.append({"cell_type": "code", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n",
                      "execution_count": None, "outputs": []})

    md('''
    # TPU v6e 프로파일링과 최적화: JAX · JAX+Pallas · native Pallas

    Colab **TPU v6e**에서 같은 BF16 모델의 전체 prefill/decode를 비교합니다.
    소스를 내장했으므로 GitHub의 최신 상태나 별도 모델 다운로드에 의존하지 않습니다.
    런타임 유형을 TPU v6e로 선택한 뒤 위에서부터 실행하세요.
    두 독립 실행에서 전체 모델 성능 개선을 확인한 단계별 설정을 사용합니다.
    `original_native`는 수정 전 커널과 attention을 함께 동결한 비교 대상입니다.

    | 구현 | 실행 범위 |
    |---|---|
    | JAX | 원본 JAX decoder 전체를 JIT/XLA 컴파일 |
    | JAX+Pallas | attention의 QK → causal softmax → PV를 직접 작성한 Pallas로 교체 |
    | native Pallas | embedding, LayerNorm, 모든 GEMM, RoPE, head 배치, attention, residual, SwiGLU, KV 갱신 모두 Pallas |
    | original native Pallas | 최적화 전 native 코드와 attention의 동결본; 같은 입력으로 재측정 |

    native Pallas도 JAX의 배열·컴파일·호출 기능을 사용합니다. 모델 계산이 Pallas 밖으로
    빠지지 않았는지는 StableHLO로 검사합니다. 모든 경로가 동일한 BF16 가중치·입력·cache를 공유합니다.
    ''')
    md('''
    ## 모델과 공정한 비교 조건

    8층 decoder, D=2048, FFN=5632, query/KV heads=16, head_dim=128, vocabulary=32000.
    LayerNorm·RoPE·causal attention·SwiGLU를 사용하는 무작위 가중치 모델입니다.

    - **Prefill:** B=1, T=S=2048. 모든 prompt 위치의 logits를 계산합니다.
    - **Decode:** B=8, T=1, S=2048. 마지막 cache 슬롯을 갱신하는 한 번의 forward입니다.
    - 가중치·activation·KV·logits는 BF16, GEMM 누산과 정규화·softmax 내부 계산은 FP32입니다.
    - 컴파일·입력 생성·host 전송·검증을 제외하고, warmup 후 호출마다 모든 출력의 완료를 기다립니다.
    - 60개 표본의 측정 순서를 모든 경로 사이에서 순환·역전합니다. seed를 바꾼 별도 프로세스로 재확인합니다.
    - cache donation은 사용하지 않습니다. 기존 cache 복사 비용도 시간에 포함됩니다.
    - 여러 토큰을 연속 생성하는 serving/학습 성능을 측정하는 실험은 아닙니다.

    정확도 기준은 실행 전에 고정합니다: FP32로 승격한 같은 BF16 가중치·입력의 원본 모델 대비
    모든 logits/KV의 NRMSE ≤ 0.02, max_abs/RMS ≤ 0.20, RMS 분모 하한 1e-6.
    Decode의 새 K/V를 별도로 검사하고 기존 cache prefix를 정확히 보존해야 합니다.
    측정 후에는 원본 BF16 JAX와 다시 비교합니다. 반올림 차이를 허용하며 비트 단위 동일성은 요구하지 않습니다.
    ''')
    code('''
    import importlib.metadata as metadata
    import subprocess
    import sys
    expected_versions = {"jax": "0.11.2", "jaxlib": "0.11.2", "libtpu": "0.0.48"}
    try:
        installed = {name: metadata.version(name) for name in expected_versions}
    except metadata.PackageNotFoundError:
        installed = {}
    if installed != expected_versions:
        setup = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "jax[tpu]==0.11.2", "libtpu==0.0.48"],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(setup.stdout)
        setup.check_returncode()
    print({name: metadata.version(name) for name in ("jax", "jaxlib", "libtpu")})
    # TPU는 아래 자식 프로세스에서만 초기화합니다. 이 notebook kernel에서 jax를 import하지 않습니다.
    ''')
    md("## 실행 소스 준비\n\n아래 압축 묶음은 저장소 구현의 정확한 snapshot이며 SHA-256을 확인한 뒤 풉니다.")
    code(f'''
    import base64, hashlib, json, os, zlib
    from pathlib import Path
    from datetime import datetime, timezone
    BUNDLE_B64 = {encoded!r}
    payload = zlib.decompress(base64.b64decode(BUNDLE_B64))
    assert hashlib.sha256(payload).hexdigest() == {digest!r}
    ROOT = Path("/content/pallas-opt") if Path("/content").exists() else Path.cwd() / "tpu-colab-run"
    ROOT.mkdir(parents=True, exist_ok=True)
    SOURCE_FILES = json.loads(payload)
    for name, source in SOURCE_FILES.items():
        destination = ROOT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
    RESULTS = ROOT / "results"
    RESULTS.mkdir(exist_ok=True)
    PROFILE_ROOT = RESULTS / datetime.now(timezone.utc).strftime("profiles-after-%Y%m%dT%H%M%S")
    print("Source SHA-256:", hashlib.sha256(payload).hexdigest())
    print("Files:", *SOURCE_FILES, sep="\\n")
    ''')
    md('''
    ## Pallas의 교체 범위 확인

    아래 소스의 attention은 query tile마다 KV context 전체를 읽고,
    QK와 softmax probability를 BF16로 변환해 VMEM에 유지합니다.
    온라인 softmax/FlashAttention 구현은 아닙니다.
    Native의 모든 모델 연산은 `native.py`의 Pallas kernel들로 구성됩니다.
    ''')
    code('''
    from IPython.display import Code, Markdown, display
    display(Code(SOURCE_FILES["llm_bench_tpu/kernels.py"], language="python"))
    display(Code(SOURCE_FILES["llm_bench_tpu/native.py"], language="python"))
    ''')
    code('''
    def run_job(arguments, log_name, extra_env=None):
        environment = dict(os.environ, JAX_PLATFORMS="tpu", PYTHONUNBUFFERED="1")
        environment.update(extra_env or {})
        with (RESULTS / log_name).open("w") as log:
            job = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in job.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            exit_code = job.wait()
        assert exit_code == 0, f"Process failed: see {RESULTS / log_name}"

    def benchmark(name, seed, *flags):
        run_job(["-m", "llm_bench_tpu", "--seed", str(seed), "--out", str(RESULTS / f"{name}.json"),
                 "--tuning-config", str(ROOT / "llm_bench_tpu/configs/optimized-v6e.json"),
                 "--backends", "jax,hybrid,native,original_native",
                 "--hlo-dir", str(RESULTS / f"hlo-{name}"), *flags], f"{name}.log")
        result = json.loads((RESULTS / f"{name}.json").read_text())
        assert result["complete"] and result["status"] == "ok"
        return result
    ''')
    md("## 작은 모델 및 TPU 경계조건 검사\n\n작은 모델의 시간은 최종 성능표에 포함하지 않습니다.")
    code('''
    smoke = benchmark("smoke", 123, "--smoke", "--samples", "3")
    run_job(["-m", "unittest", "discover", "-s", "tests", "-p", "test_llm_bench_tpu.py", "-v"],
            "device-tests.log", {"RUN_TPU_PALLAS_TESTS": "1"})
    ''')
    md("## 전체 모델 측정 1\n\n동일한 장치·같은 가중치 배열로 세 구현을 실행합니다. 컴파일 시간은 별도 기록합니다.")
    code('primary = benchmark("run1", 123, "--profile-dir", str(PROFILE_ROOT), "--profile-steps", "20")')
    md("## 전체 모델 측정 2\n\n새 가중치·입력(seed 321), 반대 컴파일 순서, 별도 프로세스로 확인합니다.")
    code('confirmation = benchmark("run2", 321, "--reverse")')
    md("## 실측 결과\n\n배속은 **같은 실행의 JAX 중앙값 ÷ 각 구현 중앙값**입니다. 1보다 크면 빠릅니다.")
    code('''
    labels = {"jax": "JAX", "hybrid": "JAX+Pallas", "native": "native Pallas (optimized)", "original_native": "native Pallas (original)"}
    lines = ["| Seed | 단계 | 구현 | p50 ms | p95 ms | JAX 대비 배속 | 컴파일 s |",
             "|---|---|---|---:|---:|---:|---:|"]
    for result in (primary, confirmation):
        for phase, data in result["phases"].items():
            for backend, label in labels.items():
                row = data["backends"][backend]
                latency = row["latency"]
                assert row["accuracy_fp32"]["passed"] and row["post_timing_vs_jax"]["passed"]
                lines.append(f"| {result['arguments']['seed']} | {phase} | {label} | {latency['median_ms']:.4f} | "
                             f"{latency['p95_ms']:.4f} | {row['speedup_vs_jax']:.3f}× | {row['compile_s']:.2f} |")
    display(Markdown("\\n".join(lines)))
    print("실제 런타임:", primary["runtime"])
    print("매개변수 수:", f"{primary['parameter_count']:,}")
    ''')
    code('''
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, phase in zip(axes, ("prefill", "decode")):
        x = np.arange(len(labels))
        for offset, result, color in ((-.18, primary, "#3070b3"), (.18, confirmation, "#39a58c")):
            values = [result["phases"][phase]["backends"][b]["latency"]["median_ms"] for b in labels]
            bars = ax.bar(x + offset, values, .35, color=color, label=f"seed {result['arguments']['seed']}")
            ax.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        ax.set_xticks(x, ["JAX", "JAX+Pallas", "native\\noptimized", "native\\noriginal"])
        ax.set_ylabel("Whole-model latency (ms, lower is better)")
        ax.set_title(phase.capitalize() + " · TPU v6e · BF16")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison.png", dpi=160)
    fig.savefig(RESULTS / "comparison.svg")
    plt.show()
    ''')
    code('''
    lines = ["| 단계 | 구현 | Pallas 호출 수 | 최대 FP32 NRMSE (두 실행) |",
             "|---|---|---:|---:|"]
    for phase in ("prefill", "decode"):
        for backend, label in labels.items():
            rows = [r["phases"][phase]["backends"][backend] for r in (primary, confirmation)]
            lines.append(f"| {phase} | {label} | {rows[0]['hlo_audit']['pallas_calls']} | "
                         f"{max(r['accuracy_fp32']['max_nrmse'] for r in rows):.6f} |")
    display(Markdown("\\n".join(lines)))
    for phase in ("prefill", "decode"):
        for backend in ("hybrid", "native"):
            speedups = [r["phases"][phase]["backends"][backend]["speedup_vs_jax"] for r in (primary, confirmation)]
            print(f"{phase} {labels[backend]}: {min(speedups):.3f}–{max(speedups):.3f}× vs same-run JAX")
    ''')
    md('''
    ## 장치 프로파일: 원인과 개선의 확인

    점수 측정이 끝난 뒤, 컴파일·warmup이 완료된 모델을 경로별로 20회 실행해
    XPlane과 Perfetto를 저장했습니다. 아래는 실제 TPU `XLA Ops` 트랙의 시간입니다.
    중첩된 커널과 DMA/비동기 트랙의 시간을 더하지 않고, 겹치는 구간을 한 번만 집계합니다.
    장치 시간에는 호스트 호출·동기화 비용이 포함되지 않으므로 위 전체 지연과 구분합니다.
    일반 XLA 연산의 분류는 프로파일러가 제공한 분류를 그대로 사용합니다.
    `convolution fusion`을 CNN 연산으로 해석하지 않습니다.

    최적화 전 최초 프로파일에서 native decode는 장치 시간 9.84ms 중 KV 갱신이
    7.42ms(약 75%)였고, GEMM은 0.87ms였습니다. Prefill은 GEMM 5.33ms가 가장 컸습니다.
    이는 2026-10-05의 초기 측정이며, 아래 표는 **이 노트북 실행에서 재측정한 결과**입니다.
    설정 파일을 생략하면 원본 설정으로 동작합니다. 모든 커널이 모든 shape에서 빠르다는 뜻은 아닙니다.
    [TPU 블록 제약](https://docs.jax.dev/en/latest/pallas/tpu/details.html),
    [TPU GEMM](https://docs.jax.dev/en/latest/pallas/tpu/matmul.html),
    [프로파일 수집 API](https://docs.jax.dev/en/latest/_autosummary/jax.profiler.trace.html).
    ''')
    code('''
    run_job(["-m", "llm_bench_tpu.profile_summary", str(PROFILE_ROOT),
             "--out", str(RESULTS / "profile-after.json")], "profile-summary.log")
    profiles = json.loads((RESULTS / "profile-after.json").read_text())
    lines = ["| 단계 | 구현 | 장치 평균 ms | KV 갱신 ms | GEMM ms | Attention ms |",
             "|---|---|---:|---:|---:|---:|"]
    for phase in ("prefill", "decode"):
        for backend in ("original_native", "native"):
            row = profiles[f"{phase}-{backend}"]
            assert row["device_events_verified"] and row["steps"] == 20
            category_ms = lambda c: row["categories"].get(c, {}).get("exclusive_ms_per_step", 0)
            lines.append(f"| {phase} | {labels[backend]} | {row['device_module_mean_ms']:.3f} | "
                         f"{category_ms('KV write'):.3f} | {category_ms('GEMM'):.3f} | {category_ms('Attention'):.3f} |")
    display(Markdown("\\n".join(lines)))
    print("실측으로 선택한 설정:", json.dumps(primary["tuning_config"], indent=2))
    ''')
    code('''
    import tarfile
    archive = ROOT / "tpu-comparison-results.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(RESULTS, arcname="results")
    print("결과 묶음:", archive)
    print("COLAB_TPU_THREE_WAY_NOTEBOOK_COMPLETE")
    # 브라우저에서 내려받기: from google.colab import files; files.download(str(archive))
    ''')
    for i, cell in enumerate(cells):
        cell["id"] = f"tpu-threeway-{i:02d}"
    notebook = {"cells": cells, "metadata": {"accelerator": "TPU",
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"}}, "nbformat": 4, "nbformat_minor": 5}
    out = root / "notebooks" / "llm_backend_comparison_tpu_v6e.ipynb"
    out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(out)


if __name__ == "__main__":
    build()
