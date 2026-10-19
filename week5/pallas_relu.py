"""Fused TPU matmul/bias/ReLU, native Pallas (interpret=False).

Reference: https://docs.jax.dev/en/latest/pallas/tpu/matmul.html
"""
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def build_pallas(m, bn):
    def kernel(x_ref, w_ref, b_ref, y_ref):
        acc = jax.lax.dot_general(
            x_ref[...], w_ref[...], (((1,), (0,)), ((), ())),
            # Native BF16 MXU inputs, with explicit FP32 accumulation below.
            precision=jax.lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )
        y_ref[...] = jnp.maximum(acc + b_ref[...][None, :], jnp.float32(0))

    return jax.jit(pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, 1024), jnp.float32),
        grid=(1024 // bn,),
        in_specs=[pl.BlockSpec((m, 1024), lambda j: (0, 0)),
                  pl.BlockSpec((1024, bn), lambda j: (0, j)),
                  pl.BlockSpec((bn,), lambda j: (j,))],
        out_specs=pl.BlockSpec((m, bn), lambda j: (0, j)),
        compiler_params=pltpu.CompilerParams(dimension_semantics=('parallel',)),
        interpret=False,
    ))
