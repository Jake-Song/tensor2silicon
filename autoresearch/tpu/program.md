# autoresearch: Pallas TPU kernel for Y = ReLU(XW + b) on v6e-1

You are an autonomous kernel engineer. Your goal is to make `kernel.py` as fast as possible on this TPU while keeping it correct. Work in a loop: form a hypothesis, edit, benchmark, keep or revert, log, and repeat.

## Setup

Do these steps once at the start, together with the human:

1. **Agree on a run tag**, such as today's date (`sep23`). The branch `autoresearch/<tag>-tpu` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>-tpu`.
3. **Read the in-scope files** for full context:
   - `bench.py` is the fixed harness. It handles correctness, the anti-cheat guard, timing, and the summary. **Do not modify it.**
   - `kernel.py` holds the kernel. It is the only file you edit.
4. **Run the baseline**: `timeout 300 python bench.py > run.log 2>&1`. The first line of `run.log` should say `TPU v6 lite`. If the *baseline* fails to compile (a Mosaic or serialization error), stop and tell the human, because the jax/libtpu install is broken and that is not a kernel problem.
5. **Create `results.tsv`** with the header row, then the baseline row. Do not commit this file; it is gitignored. Also create `report.md` from the skeleton in *Report* below, and log the baseline as experiment `#0`.
6. Confirm with the human, then start the loop. After that, do not ask for permission again.

## The task

`kernel.py` must export `relu_linear(x, w, b) -> y`:
- `x` is `[M, K]` bf16, `w` is `[K, N]` bf16, `b` is `[N]` bf16, all `jax.Array`s on the TPU.
- `y = relu(x @ w + b)` is `[M, N]` bf16. Accumulate in fp32.
- Timed shape: M = N = K = 4096. It must also be correct on (M, K, N) = (1024, 2048, 512). Every dimension is a multiple of 512.

## Rules

**You CAN:**
- Change anything in `kernel.py`: block sizes `bm`/`bk`/`bn`, grid order and `dimension_semantics`, `BlockSpec` index maps, VMEM scratch layout, `vmem_limit_bytes` in `pltpu.CompilerParams`, pipelining (`pipeline_mode=pl.Buffered(n)` on a `BlockSpec`), manual DMA with `pltpu.emit_pipeline` or `pltpu.make_async_copy`, K-loop structure, and the epilogue. `jnp.dot`/`jax.lax.dot_general` **inside** the kernel body is the MXU instruction, so it is expected there. Plain JAX reshapes or pads outside the kernel are fine too.

**You CANNOT:**
- Modify `bench.py`, install packages, upgrade jax/libtpu, or add dependencies.
- Compute the matmul with XLA outside a Pallas kernel (`x @ w`, `jnp.dot`, `jnp.einsum`, or `lax.dot_general` in the wrapper), or run Pallas with `interpret=True`. The guard fails any run whose lowered StableHLO contains a `dot_general` or `convolution`, or has no `tpu_custom_call`.
- Cache outputs across calls, special-case the benchmark's input values, or skip work based on the data. The harness re-checks with fresh data after timing.
- Loosen precision below bf16 inputs with fp32 accumulation. No fp8 or int8 downcasts of the inputs.

**Time budget:** each `python bench.py` has to finish within `timeout 300`, including Mosaic compile. There is no autotuner: to compare block sizes, try them one experiment at a time.

## Metric

The primary metric is `time_ms`: lower is better. It is the median of 7 rounds of 50 back-to-back calls. `speedup` = XLA time / your time, where XLA's fused `relu(x @ w + b)` is the baseline; above 1.0 means you beat XLA. `pct_peak` is measured against 918 TFLOP/s (v6e dense bf16). A run only counts if it prints `status: ok`.

**Noise:** timings vary by about 1%. Treat a change as an improvement only if it is **more than 1% faster** than the current best. When a result is borderline, run it again before keeping it.

**Simplicity criterion:** at equal speed, simpler code wins. A 0.5% gain that adds 40 lines of hacks is not worth keeping. Deleting code at equal or better speed is a win.

## Output format

The benchmark ends with:

```
---
status:      ok
correct:     True
max_abs_err: 0.01562
time_ms:     14.0450
tflops:      9.8
pct_peak:    1.1
baseline_ms: 0.1733
speedup:     0.012
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
a1b2c3d	14.0450	9.8	0.012	keep	baseline 128x128x128
b2c3d4e	2.4223	56.7	0.072	keep	256x256x256 (8x fewer grid steps)
c3d4e5f	2.4500	56.1	0.071	discard	256x256x256 + bias block hoisted
d4e5f6a	0.0000	0.0	0.000	crash	huge blocks (VMEM OOM)
```

## Report (`report.md`)

`results.tsv` holds the numbers. `report.md` is the human-readable story of the run, and it records **every** trial: the failures too, not only the wins. A discarded commit is gone from the branch after `git reset`, so its entry in `report.md` is the only lasting record of what was tried and why it didn't work. Write failed trials with the same care as successes: a well-explained dead end saves the next run from repeating it.

Create it during setup with this skeleton, and keep it **untracked** while the loop runs (never `git add` it mid-run: `git reset --hard` leaves untracked files alone, but it would roll back a tracked one):

```markdown
# Autoresearch report: <tag> (TPU v6e-1)

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
### #4: discard, bk=4096 (single K step), bm=bn=512
- **Hypothesis:** one K step removes the accumulator read-modify-write and most of the grid steps.
- **Result:** 0.2932 ms vs best 0.2740 ms (+7.0% slower).
- **Takeaway:** 512×4096 input blocks are large DMAs with nothing to overlap them against when there is only one K step, so pipelining is lost. Grid-step overhead is no longer the bottleneck at this size. Next: keep K split into 2–4 steps and grow bm/bn instead.
```

**At the end of the run**, whether you hit the experiment limit or the human stops you:

- Fill in *Summary*:
  - baseline → best `time_ms` and the overall gain;
  - best `speedup` vs XLA;
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

## Hardware notes (TPU v6e, "Trillium", 1 chip)

- There is one TensorCore per chip. Its MXUs are 256×256 systolic arrays, so blocks that are multiples of 256 fill them. The bf16 peak is 918 TFLOP/s dense and HBM is about 1.6 TB/s. At 4096³ the problem is strongly compute-bound (~1365 FLOP/B).
- A Pallas grid runs **sequentially** on one core, like a nested loop. Each grid step costs roughly 0.35 µs of overhead, so tiny blocks mean too many steps. The baseline's 128³ blocks give 32×32×32 = 32768 steps.
- `BlockSpec` inputs are double-buffered HBM→VMEM DMAs that the pipeline issues for you. VMEM is small (tens of MiB; the default scoped limit is smaller). The footprint is roughly `2*(bm*bk + bk*bn)*2 B + 2*bm*bn*2 B (out) + bm*bn*4 B (acc)`. Raise `vmem_limit_bytes` if big blocks fail to compile.
- The native tile for 32-bit data is (8, 128), and bf16 packs into (16, 128). The last two block dimensions must be divisible by 8/16 and 128, or equal the full array dimension.
- `dimension_semantics=("parallel", "parallel", "arbitrary")`: K must stay `"arbitrary"`, because the accumulator carries across it.
- The bias + ReLU epilogue should stay fused. It is nearly free compared with the K loop.
