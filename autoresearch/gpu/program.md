# autoresearch: Triton kernel for Y = ReLU(XW + b) on A100

You are an autonomous kernel engineer. Your goal is to make `kernel.py` as fast as possible on this GPU while keeping it correct. Work in a loop: form a hypothesis, edit, benchmark, keep or revert, log, and repeat.

## Setup

Do these steps once at the start, together with the human:

1. **Agree on a run tag**, such as today's date (`sep23`). The branch `autoresearch/<tag>-gpu` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>-gpu`.
3. **Read the in-scope files** for full context:
   - `bench.py` is the fixed harness. It handles correctness, the anti-cheat guard, timing, and the summary. **Do not modify it.**
   - `kernel.py` holds the kernel. It is the only file you edit.
4. **Check the device**: `nvidia-smi` should show an A100.
5. **Run the baseline**: `timeout 300 python bench.py > run.log 2>&1`. Read the summary.
6. **Create `results.tsv`** with the header row, then the baseline row. Do not commit this file; it is gitignored. Also create `report.md` from the skeleton in *Report* below, and log the baseline as experiment `#0`.
7. Confirm with the human, then start the loop. After that, do not ask for permission again.

## The task

`kernel.py` must export `relu_linear(x, w, b) -> y`:
- `x` is `[M, K]` bf16, `w` is `[K, N]` bf16, `b` is `[N]` bf16, all contiguous CUDA tensors.
- `y = relu(x @ w + b)` is `[M, N]` bf16. Accumulate in fp32.
- Timed shape: M = N = K = 4096. It must also be correct on (M, K, N) = (1024, 2048, 512). Every dimension is a multiple of 512.

## Rules

**You CAN:**
- Change anything in `kernel.py`: tile sizes, `num_warps`, `num_stages`, grouped or swizzled program ordering, `triton.autotune` configs, persistent kernels, split-K, pointer and mask arithmetic, the loop structure, and the epilogue.

**You CANNOT:**
- Modify `bench.py`, install packages, or add dependencies. Only what is already installed (torch, triton) is available.
- Compute the matmul with vendor libraries: `torch.matmul`/`@`/`mm`/`addmm`/`einsum`, cuBLAS, cuDNN, CUTLASS, or `torch.compile`. The guard fails any run that launches a vendor GEMM kernel. The matmul must be your Triton kernel.
- Cache outputs across calls, special-case the benchmark's input values, or skip work based on the data. The harness re-checks with fresh data in the same buffers after timing.
- Loosen precision below bf16 inputs with fp32 accumulation. No fp8, int8, or tf32 downcasts of the inputs.

**Time budget:** each `python bench.py` has to finish within `timeout 300`, including JIT compile and autotuning. Keep autotune config lists small, around 8 configs or fewer.

## Metric

The primary metric is `time_ms`: lower is better. The harness times the kernel on 5 separately allocated input sets, with the L2 cache flushed between reps, and reports the median of the 5. Each allocation's time is printed on the `[timing]` lines of `run.log`. One slow allocation, usually the first, is normal and doesn't affect the median. `speedup` = cuBLAS time / your time; above 1.0 means you beat cuBLAS plus a separate ReLU. `pct_peak` is measured against 312 TFLOP/s (A100 dense bf16). A run only counts if it prints `status: ok`.

**Noise:** across separate `bench.py` runs, the baseline's `time_ms` varied by less than 0.1%. A single allocation can be up to 20% slow, which is why the harness takes a median. Treat a change as an improvement only if it is **more than 1% faster** than the current best. When a result is borderline, run it again before keeping it.

**Simplicity criterion:** at equal speed, simpler code wins. A 0.5% gain that adds 40 lines of hacks is not worth keeping. Deleting code at equal or better speed is a win.

## Output format

The benchmark ends with:

```
---
status:      ok
correct:     True
max_abs_err: 0.01565
time_ms:     1.0998
tflops:      125.0
pct_peak:    40.1
baseline_ms: 0.5673
speedup:     0.516
```

Extract the key lines with `grep '^status:\|^time_ms:\|^tflops:\|^speedup:' run.log`. If the output is empty or the status is `error`, run `tail -n 50 run.log` to read the traceback.

## Logging results

`results.tsv` is **tab-separated**, because commas break descriptions. It has 6 columns:

```
commit	time_ms	tflops	speedup	status	description
```

- `commit`: the short git hash (7 chars).
- `time_ms`, `tflops`, `speedup`: from the summary. Use `0` if the run crashed.
- `status`: `keep`, `discard`, `incorrect`, `forbidden`, or `crash`.
- `description`: a short note on what you tried.

Example:

```
commit	time_ms	tflops	speedup	status	description
a1b2c3d	1.0998	125.0	0.516	keep	baseline 64x64x32 w4 s2
b2c3d4e	0.8233	167.0	0.689	keep	128x128x32 w4 s3
c3d4e5f	0.9708	141.6	0.584	discard	128x128x64 w4 s4
d4e5f6a	0.0000	0.0	0.000	crash	BLOCK_K=128 s5 (out of shared memory)
```

## Report (`report.md`)

`results.tsv` holds the numbers. `report.md` is the human-readable story of the run, and it records **every** trial: the failures too, not only the wins. A discarded commit is gone from the branch after `git reset`, so its entry in `report.md` is the only lasting record of what was tried and why it didn't work. Write failed trials with the same care as successes: a well-explained dead end saves the next run from repeating it.

Create it during setup with this skeleton, and keep it **untracked** while the loop runs (never `git add` it mid-run: `git reset --hard` leaves untracked files alone, but it would roll back a tracked one):

```markdown
# Autoresearch report: <tag> (A100)

## Summary
(filled in at the end)

## Kept: improvements
| # | commit | time_ms | vs previous best | change |
|---|---|---|---|---|

## Failed trials: slower, incorrect, forbidden, crash
| # | commit | time_ms | vs best at the time | status | change | why it failed |
|---|---|---|---|---|---|---|

## Experiment log
```

After **every** experiment, do two things:

1. Add one row to the matching table: *Kept* for `keep`, *Failed trials* for everything else. `#` is the experiment number (the baseline is `0`). Compute "vs best" as a percentage, e.g. `+17.9% slower` or `-25.5% faster`. For crashes, write `-` for `time_ms`.
2. Append a short entry to *Experiment log*:

```markdown
### #{n}: {status}, {change}
- **Hypothesis:** why this should be faster.
- **Result:** time_ms, and the delta vs the best at the time. For crashes and incorrect runs, the key error line from `run.log`.
- **Takeaway:** what this tells you about the kernel or the hardware, and what it rules in or out for later experiments.
```

Example of a failed-trial entry:

```markdown
### #2: discard, 128x128x64 tiles, num_stages 4
- **Hypothesis:** a bigger BLOCK_K halves the loop trips, and more stages hide global-load latency.
- **Result:** 0.9708 ms vs best 0.8233 ms (+17.9% slower).
- **Takeaway:** 4 stages × (128+128)×64×2 B = 128 KB of shared memory leaves room for only one CTA per SM, so occupancy dropped. Raise num_stages only while smem per CTA stays under ~80 KB. Next: try 128x256 tiles with 8 warps and 3 stages.
```

**At the end of the run**, whether you hit the experiment limit or the human stops you:

- Fill in *Summary*:
  - baseline → best `time_ms` and the overall gain;
  - best `speedup` vs cuBLAS;
  - the experiment count by status (kept / discard / incorrect / forbidden / crash);
  - the best commit hash;
  - the 3–5 lessons that mattered most, including what did **not** help;
  - the most promising untried ideas.
- Commit it to the branch: `git add report.md && git commit -m "report: <tag>"`.

## The experiment loop

LOOP until the human stops you or you reach **30 experiments** (the human may give a different number):

1. Look at the git state, `results.tsv`, and the takeaways in `report.md`. Pick the next idea and say it in one sentence.
2. Edit `kernel.py`.
3. `git commit -am "<short description>"`.
4. `timeout 300 python bench.py > run.log 2>&1`. Always redirect: do not let the output flood your context.
5. `grep '^status:\|^time_ms:\|^tflops:\|^speedup:' run.log`. If there is a problem, `tail -n 50 run.log`.
6. Note the short hash (`git rev-parse --short HEAD`) now, because a reset makes it unreachable. If `status: ok` and the run is more than 1% faster than the best kept result, **keep** the commit and the branch advances. Otherwise, `git reset --hard HEAD~1` back to the best commit.
7. **Then** log the experiment, after the reset so it can never undo the log: append one row to `results.tsv`, and add a table row plus a log entry to `report.md`. This applies to failures too.
8. After a crash from a simple bug (a typo, a shape error), fix it and rerun once. If the idea itself is broken, log `crash` and move on.

When the loop ends, finish `report.md` as described in *Report* and commit it.

Be autonomous. Do not stop to ask whether you should continue. If you run out of ideas, re-read `kernel.py` and your results, combine the near-misses, and try something more radical.

## Hardware notes (A100 SXM/PCIe, sm_80)

- 108 SMs. Each SM has 164 KB of shared memory (a single block can use about 163 KB) and a 256 KB register file.
- bf16 Tensor Core peak is 312 TFLOP/s dense. HBM is about 1.5–2.0 TB/s. At 4096³ the problem is strongly compute-bound (arithmetic intensity ~1365 FLOP/B), so the job is keeping the tensor cores fed.
- `cp.async` pipelining is driven by `num_stages`. Each stage costs `(BLOCK_M + BLOCK_N) * BLOCK_K * 2` bytes of shared memory.
- The grid for 4096² with 128×128 tiles is 1024 CTAs, or about 9.5 waves over 108 SMs. Tail effects and L2 reuse depend on program ordering; see the grouped ordering in the Triton matmul tutorial.
- The bias + ReLU epilogue should stay fused. It is nearly free compared with the K loop.
