# Full-model autoresearch: JAX + Pallas on TPU v6e or v5e

## Task and edit boundary

Optimize the complete decoder-only Transformer forward pass from `llm_roofline`.
Preserve the model, weights, inputs, outputs, and bf16 precision. This is inference
implementation research with random weights, not training or model-quality research.

Edit **only this directory's `model.py`**. `bench.py`, `../llm_common.py`, all
`llm_roofline/` files, dependency versions, permissions, and this protocol are fixed.
The existing ReLU kernel experiments are separate and must not be changed.

Public contract:

```python
forward(params, tokens, consts, cache=None) -> (logits, kv_cache)
```

Use the reference parameter tree and constants. Return bf16 logits `[B,T,32000]`
and all eight layers' `(K,V)` arrays. A prefill call returns new caches; decode
updates only the last slot and preserves every earlier slot. Do not mutate weights,
tokens or constants. Keep LayerNorm (not RMSNorm), RoPE, causal masking, SwiGLU,
residuals, and the output projection. Q is already scaled by `1/sqrt(H)` in
`rope_split_heads`; fused attention must not apply that scale a second time.

JAX/XLA operations and custom Pallas kernels are allowed. The evaluator applies jax.jit to forward, donating only the decode cache. Do not donate parameters, tokens, or constants. Never use a donated input again; return the updated cache. Do not use interpret=True for scored runs.

No quantization, altered dimensions/heads/layers, omitted logits, skipped layers,
output caching, data-dependent benchmark shortcuts, reference monkeypatching,
evaluator introspection, or host computation replacing device work. Shape-based
specialization and compile caches are allowed; caching outputs or parameter-derived
values across calls is not. Do not install or upgrade packages during a search.

## Setup and workloads

Use the user's tag and phase; otherwise use a UTC timestamp tag and `decode`.
Create an isolated worktree/branch named `autoresearch/<tag>-llm-tpu-<phase>`.
Never switch/reset a dirty shared checkout. A launch request with a tag and phase
already authorizes setup and the bounded loop; do not ask for another confirmation.
Stop if a branch name already exists instead of overwriting it.

The fixed `colab` model has D=2048, F=5632, N=K=16, H=128, L=8, vocabulary=32000.

| Phase | B | T | S | Objective |
|---|---:|---:|---:|---|
| prefill | 1 | 2048 | 2048 | Complete prompt forward, all logits and KV outputs |
| decode | 8 | 1 | 2048 | One complete token forward, updated KV outputs |

Decode repeatedly overwrites the last slot of a fixed context. It does **not**
measure growing-context generation or end-to-end tokenization/sampling latency.
Prefill and decode are independent searches with separate best implementations.
Both phases must pass tiny-model correctness gates, including GQA (N=4, K=2).

Run from this directory, substituting the chosen phase:

```bash
timeout 900 python bench.py --phase decode --json baseline.json > baseline.log 2>&1
```

Require `status: ok`, `correct: True`, `scored: True` before optimizing. `--smoke`
uses the tiny preset and is never eligible for scoring. A scored run requires
TPU v6e or v5e. Record the actual device and do not compare incumbents across devices. If the initial baseline cannot run, report the environment failure;
do not modify the evaluator to make it pass. Default budget: 30 candidate experiments,
900 seconds per invocation, including compilation and correctness checks.

## Fixed evaluator

Checks all logits and every K/V array with `atol=rtol=0.02`, plus structure,
shapes, bf16 dtype and finite values. Old cache prefixes must be unchanged.
It tests two fresh tiny-model seeds for both phases and the selected full-size phase.
After timing it changes weights, tokens and cache values and checks again; CUDA
reuses the original buffers, while JAX receives fresh immutable arrays.

Both baseline and candidate return **logits and all caches** across the executable
boundary. Every call is synchronized: CUDA completion on GPU, the entire output
pytree with `jax.block_until_ready` on TPU. No timed wrapper may discard logits.
The returned decode cache becomes the next call's cache.

Timing excludes input creation/transfers, compilation and five warmup calls. It is
the median of seven rounds of ten synchronized full-forward calls. First-call time
(including compilation and one execution) is reported separately. The evaluator
compares with the unmodified eager PyTorch reference on GPU or jitted JAX reference
on TPU in the same process, using identical values and independent allocations.
`speedup = baseline_ms / time_ms`; the selection metric is candidate `time_ms`.
These synchronized timings are not interchangeable with the old roofline timings.

The final `result_json:` line and optional `--json` file include `status`, `correct`,
`scored`, `phase`, `preset`, device/software versions, latency samples, `time_ms`,
`baseline_ms`, `speedup`, `max_abs_err`, and compile/first-call times. Failures produce
nonzero exit codes and `incorrect` or `error`. Timeout exit 124 is a failed trial,
never a result; do not read an older JSON file as its output.

## Experiment loop

1. Read current results and report. Choose one hypothesis using measured bottlenecks
   and earlier evidence; explain it before editing. No random method selection or
   mandatory literature queue. Start with attention fusion for prefill; for decode,
   inspect launch overhead, small-op fusion and cache traffic, then matmul/layout tuning.
2. Edit `model.py` and commit **that file only**. Save its parent and candidate hash.
3. Run the fixed evaluator with the selected phase, a unique JSON filename and log
   for this trial. Redirect full output and read the summary; keep traceback details
   for a failed experiment.
4. A successful candidate must beat the incumbent by **strictly more than 1%**.
   Run it again in a new process. Keep only if both scored, correct runs pass that
   threshold; store the slower of the two latencies as the new incumbent. If the
   confirmation fails, discard. Start the incumbent from the baseline latency.
5. For discarded trials, restore **only `model.py`** from the saved parent and commit
   that restoration. Do not use `git reset --hard`, `git clean`, or a blanket add.
   Candidate commits and trial logs remain available for reviewing failed ideas.
6. Append the TSV row and update the report for every attempt, including failures.
   Stop after the experiment limit, user stop, or a broken runtime. Do not silently
   turn a failed accelerator run into CPU or tiny-model performance evidence.

At equal speed prefer simpler code, but do not call it a speed improvement.

## Persistent results

Create `results.tsv` as tab-separated text, with this exact header:

```text
experiment	commit	phase	time_ms	confirmation_ms	baseline_ms	speedup	status	description
```

Baseline is experiment 0, status `keep`, confirmation blank. Other statuses are
`keep`, `discard`, `incorrect`, `forbidden`, `crash`, `timeout`. Invalid/unmeasured
latencies are blank, not zero. Use `forbidden` for a protocol violation found in
review; the evaluator is a correctness/timing gate, not a security sandbox.
For kept candidates, `time_ms` holds the slower of the first and confirmation runs.
Keep reports and logs outside the candidate restore operation. Use `runs/` for
unique trial JSON/log files and ignore it in Git; no evidence should be overwritten.

Create `report.md` with Summary, Kept improvements, Failed trials, and Experiment
log sections. Every trial records hypothesis, change, timing/error evidence, delta
from incumbent, and takeaway. At the end include baseline→best latency, best
commit, status counts, lessons, and promising untried ideas. Commit the final report
explicitly. Do not push or provision more runtimes unless the user requests it.
