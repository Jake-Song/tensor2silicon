"""Y = ReLU(XW + b) as one Pallas TPU kernel. The ONLY file the agent edits.

Baseline: the blocked matmul from the Pallas TPU docs with an fp32 VMEM accumulator
and bias + ReLU fused into the last K step. Deliberately untuned (128^3 blocks).
"""
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def relu_linear_kernel(x_ref, w_ref, b_ref, y_ref, acc_ref, *, nsteps):
    @pl.when(pl.program_id(2) == 0)
    def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

    # bf16 x bf16 -> fp32 on the MXU
    acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

    # epilogue: bias + ReLU on the last K step, while the tile is still in VMEM
    @pl.when(pl.program_id(2) == nsteps - 1)
    def _():
        y = acc_ref[...] + b_ref[...].astype(jnp.float32)
        y_ref[...] = jnp.maximum(y, 0.0).astype(y_ref.dtype)


@functools.partial(jax.jit, static_argnames=["bm", "bk", "bn"])
def relu_linear(x, w, b, *, bm=128, bk=128, bn=128):
    M, K = x.shape
    _, N = w.shape
    return pl.pallas_call(
        functools.partial(relu_linear_kernel, nsteps=K // bk),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(M // bm, N // bn, K // bk),
            in_specs=[
                pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
                pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
                pl.BlockSpec((1, bn), lambda i, j, k: (0, j)),
            ],
            out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
            scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
        ),
        out_shape=jax.ShapeDtypeStruct((M, N), x.dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")),
    )(x, w, b.reshape(1, N))
