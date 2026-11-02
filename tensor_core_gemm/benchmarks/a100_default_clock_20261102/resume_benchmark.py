"""Run inside a fresh Colab A100; poll status.json from short CLI calls."""
import os, subprocess, json, pathlib, time, hashlib, threading, csv
root=pathlib.Path('/content/tc-default/tensor_core_gemm')
out=root/'benchmarks/a100_default_clock_20261102'
status=out/'status.json'
def update(stage,**kw):
    status.write_text(json.dumps(dict(stage=stage,**kw),indent=2))
def command(args,filename):
    with (out/filename).open('w') as f:
        p=subprocess.run(args,cwd=root,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError(f'{filename}: exit {p.returncode}')
def main():
    if (out/'measurements.jsonl').exists():
        (out/'measurements.jsonl').rename(out/'measurements-initial-failed.jsonl')
    if status.exists(): (out/'status-initial.json').write_text(status.read_text())
    command(['nvcc','-O3','-std=c++17','-arch=sm_80','-lineinfo','-I.',str(out/'benchmark.cu'),'build/runner.o','-lcublas','-lcublasLt','-o','build/default_clock_bench'],'build-benchmark.log')
    paths=[out/'benchmark.cu',out/'resume_benchmark.py',root/'runner.cu',root/'runner.cuh',*sorted((root/'kernels').rglob('*.cuh'))]
    (out/'benchmark-source-sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    update('benchmark')
    stop=threading.Event()
    fields='clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,pstate,clocks_throttle_reasons.active'
    def sample():
        with (out/'telemetry.jsonl').open('w') as f:
            while not stop.is_set():
                t=time.time_ns()//1000
                p=subprocess.run(['nvidia-smi','--query-gpu='+fields,'--format=csv,noheader,nounits'],capture_output=True,text=True)
                f.write(json.dumps(dict(timestamp_us=t,returncode=p.returncode,values=p.stdout.strip(),error=p.stderr.strip()))+'\n'); f.flush()
                stop.wait(0.1)
    th=threading.Thread(target=sample); th.start()
    try: command(['./build/default_clock_bench'],'measurements.jsonl')
    finally: stop.set(); th.join()
    command(['nvidia-smi','-q'],'gpu-after.log')
    update('complete',exit_code=0)
try: main()
except Exception as e:
    update('failed',error=str(e)); raise
