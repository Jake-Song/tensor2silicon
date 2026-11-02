"""Summarize saved paired default-clock measurements and sampled telemetry."""
from pathlib import Path
import csv,json,statistics
root=Path(__file__).resolve().parent
rows=[json.loads(x) for x in (root/'measurements.jsonl').read_text().splitlines() if x.startswith('{')]
assert len(rows)==50,len(rows)
assert all(r['pass'] for r in rows)
telemetry=[json.loads(x) for x in (root/'telemetry.jsonl').read_text().splitlines()]
samples=[]
for r in telemetry:
    if r['returncode']: continue
    v=next(csv.reader([r['values']]))
    samples.append(dict(timestamp_us=r['timestamp_us'],sm_mhz=float(v[0]),memory_mhz=float(v[1]),power_w=float(v[2]),power_limit_w=float(v[3]),temperature_c=float(v[4]),pstate=v[5].strip(),throttle_reasons=v[6].strip()))
summary=[]
for size in [4096,8192]:
    base=statistics.median(r['tflops'] for r in rows if r['size']==size and r['kernel']==100)
    for kernel in [7,8,9,10,100]:
        trials=[r for r in rows if r['size']==size and r['kernel']==kernel]
        assert len(trials)==5
        s=[t for t in samples if any(r['start_us']<=t['timestamp_us']<=r['stop_us'] for r in trials)]
        result=dict(size=size,kernel=kernel,tflops_median=statistics.median(r['tflops'] for r in trials),ms_median=statistics.median(r['ms'] for r in trials),tflops_min=min(r['tflops'] for r in trials),tflops_max=max(r['tflops'] for r in trials),max_fp32_error=max(r['fp32_error'] for r in trials),max_quantized_tf32_error=max(r['quantized_tf32_error'] for r in trials),max_cublas_tf32_error=max(r['cublas_tf32_error'] for r in trials),quantized_fp32_diagnostic_all_within_1e5=all(r['quantized_fp32_within_1e5'] for r in trials),clock_samples=len(s))
        result['percent_cublas']=result['tflops_median']/base*100
        if s:
            for field in ['sm_mhz','memory_mhz','power_w','temperature_c']:
                result[field]=dict(min=min(t[field] for t in s),max=max(t[field] for t in s),median=statistics.median(t[field] for t in s))
            result['throttle_reasons']=sorted(set(t['throttle_reasons'] for t in s))
        summary.append(result)
(root/'summary.json').write_text(json.dumps(dict(method=dict(sizes=[4096,8192],kernels=[7,8,9,10,100],rounds=5,repeats=50,warmup=5,alpha=0.5,beta=3,clock_policy='driver defaults, unlocked',seed_inputs=20261016,seed_order=7,timing='CUDA event; excludes allocation, reference, validation and host/device transfer; includes possible host dispatch gaps',c_reset='before each accuracy check and timed batch, not between 50 repeated updates',correctness_gates='original FP32 reference <1e-3 and same-input cuBLAS TF32 <1e-5; tc_check independently quantized short-K <1e-5',long_k_quantized_fp32='diagnostic only; initial attempt failed; identical error reproduced with original implementation and cuBLAS TF32; see diagnosis.log'),environment=json.loads((root/'environment.json').read_text()),results=summary),indent=2)+'\n')
with (root/'summary.csv').open('w',newline='') as f:
    fields=['size','kernel','tflops_median','ms_median','percent_cublas','clock_samples']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n'); w.writeheader(); w.writerows(summary)
for r in summary:
    print(r['size'],r['kernel'],f"{r['tflops_median']:.2f} TFLOPS",f"{r['percent_cublas']:.1f}%",r.get('sm_mhz',{}),r['clock_samples'])
