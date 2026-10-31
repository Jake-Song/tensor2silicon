import json, pathlib, subprocess, torch
root = pathlib.Path('/content/tc-library-run')
root.mkdir(exist_ok=True)
meta = dict(name=torch.cuda.get_device_name(0), memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30, capability=torch.cuda.get_device_capability(0), torch=torch.__version__)
assert 'A100' in meta['name'], meta
meta['nvcc'] = subprocess.check_output(['nvcc','--version'],text=True)
(root/'hardware.json').write_text(json.dumps(meta,indent=2))
print(json.dumps(meta,indent=2))
subprocess.run(['tar','xzf','/content/tc-libraries-source.tgz','-C',str(root)],check=True)
script = r'''#!/bin/bash
set -euo pipefail
cd /content/tc-library-run
git clone --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git cutlass
git -C cutlass rev-parse HEAD > cutlass-sha.txt
cd tensor_core_gemm
make libraries CUTLASS_DIR=/content/tc-library-run/cutlass
make BUILD_DIR=build-libraries NVCCFLAGS='-O3 -std=c++17 -arch=sm_80 -lineinfo -DTC_ENABLE_CUTLASS -I/content/tc-library-run/cutlass/include' build-libraries/tc_check_libraries
./build-libraries/tc_check_libraries | tee ../correctness.log
for id in 100 101 102; do
  ./build-libraries/tc_gemm "$id" 8192 128 | tee "../benchmark_${id}.log"
done
'''
(root/'run.sh').write_text(script)
wrapper = 'bash /content/tc-library-run/run.sh > /content/tc-library-run/run.log 2>&1; echo $? > /content/tc-library-run/exit-code.txt'
p = subprocess.Popen(['bash','-c',wrapper],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
(root/'pid.txt').write_text(str(p.pid))
print('Started job',p.pid)
