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
    update('environment')
    import torch
    props=torch.cuda.get_device_properties(0)
    if 'A100' not in props.name: raise RuntimeError(f'Unexpected GPU: {props.name}')
    env=dict(gpu=props.name,memory_gib=props.total_memory/2**30,capability=[props.major,props.minor],torch=torch.__version__,clock_policy='unlocked default',source_commit='dcdefd41649df3bf0e0d0f1098a1a27c92cbde70')
    (out/'environment.json').write_text(json.dumps(env,indent=2))
    # Return driver overrides to defaults. Fail instead of silently measuring a locked GPU.
    for option in ['-rgc','-rmc','-rac']:
        command(['nvidia-smi','-i','0',option],f'reset{option}.log')
    command(['nvidia-smi','-q'], 'gpu-before.log')
    for k in list(os.environ):
        if k.startswith('TC7_') or k=='NVIDIA_TF32_OVERRIDE': os.environ.pop(k)
    paths=[root/'runner.cu',root/'runner.cuh',root/'check.cu',root/'Makefile',*sorted((root/'kernels').rglob('*.cuh')),out/'benchmark.cu',out/'run_remote.py']
    (out/'source-sha256.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2))
    update('build')
    command(['make','-j2','build','build/tc_check'],'build.log')
    command(['nvcc','-O3','-std=c++17','-arch=sm_80','-lineinfo','-I.',str(out/'benchmark.cu'),'build/runner.o','-lcublas','-lcublasLt','-o','build/default_clock_bench'],'build-benchmark.log')
    update('accuracy')
    command(['./build/tc_check'],'correctness.log')
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
