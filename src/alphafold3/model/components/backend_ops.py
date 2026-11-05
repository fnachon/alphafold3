# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with the
# License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at

"""Backend-dispatched ops: tokamax kernels on CUDA/CPU, plain JAX on Apple MPS.

tokamax has no Metal kernels and its XLA fallbacks do not know the `mps`
platform (they raise `KeyError: 'mps'` for float32 inputs and log a traceback
for every op that has no implementation for the device). On MPS we therefore
call plain `jax.numpy` implementations with the same semantics, which jax-mps
compiles to Metal, and use jax-mps's fused MLX `sdpa` Metal kernel for
attention (set `AF3_MPS_FUSED_ATTENTION=0` to use plain JAX instead). Every
other platform keeps using tokamax unchanged.
"""

from collections.abc import Callable
import functools
import os

import jax
import jax.numpy as jnp
import tokamax


def default_platform() -> str:
  """Returns the platform ops are traced for (the default device's platform)."""
  device = getattr(jax.config, 'jax_default_device', None)
  platform = getattr(device, 'platform', None)
  return platform if isinstance(platform, str) else jax.default_backend()


def use_mps_ops() -> bool:
  return default_platform() == 'mps'


def _xla_dot_product_attention(q, k, v, *, mask, bias):
  """Softmax attention over `[*B, T, H, D]` q and `[*B, S, H, D]` k, v."""
  scale = q.shape[-1] ** -0.5
  logits = jnp.einsum(
      '...thd,...shd->...hts', q, k, preferred_element_type=jnp.float32
  )
  logits *= scale
  if bias is not None:
    logits += bias.astype(jnp.float32)
  if mask is not None:
    logits = jnp.where(mask, logits, -0.7 * jnp.finfo(jnp.float32).max)
  weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
  return jnp.einsum('...hts,...shd->...thd', weights, v)


@functools.cache
def _mps_fused_ops():
  """Returns the jax-mps fused (MLX Metal) ops module, or None if unusable."""
  if os.environ.get('AF3_MPS_FUSED_ATTENTION', '1') == '0':
    return None
  try:
    from jax_plugins.mps import ops  # pylint: disable=g-import-not-at-top
  except ImportError:
    return None
  return ops if hasattr(ops, 'sdpa') else None


def _fused_sdpa(ops, q, k, v, additive):
  """`[b, T, H, D]` q/k/v with a `[b|1, H|1, T|1, S]` additive bias or None."""

  @jax.custom_batching.custom_vmap
  def attend(q, k, v, additive):
    out = ops.sdpa(
        jnp.swapaxes(q, -2, -3),
        jnp.swapaxes(k, -2, -3),
        jnp.swapaxes(v, -2, -3),
        mask=additive,
    )
    return jnp.swapaxes(out, -2, -3)

  @attend.def_vmap
  def attend_vmap(axis_size, in_batched, q, k, v, additive):
    # mps.sdpa has no batching rule: fold the mapped axis into the batch axis.
    def fold(x, batched, batch_size):
      if not batched:
        x = jnp.broadcast_to(x, (axis_size,) + x.shape)
      if x.shape[1] != batch_size:  # Broadcast `1` batch of mask/bias.
        x = jnp.broadcast_to(x, (axis_size, batch_size) + x.shape[2:])
      return x.reshape((axis_size * batch_size,) + x.shape[2:])

    batch_size = q.shape[1]
    q, k, v = (fold(x, b, batch_size) for x, b in zip((q, k, v), in_batched))
    if additive is not None:
      additive = fold(additive, in_batched[3], batch_size)
    out = attend(q, k, v, additive)
    return out.reshape((axis_size, batch_size) + out.shape[1:]), True

  return attend(q, k, v, additive)


def _mps_fused_dot_product_attention(ops, q, k, v, *, mask, bias):
  """Attention with the fused MLX `sdpa` Metal kernel.

  The kernel takes a single mask operand, so the boolean mask and the float
  bias are merged into one additive bias. Layout is `[B, H, T, D]`.
  """
  additive = None
  if mask is not None:
    additive = jnp.where(mask, 0.0, -1e9).astype(jnp.float32)
  if bias is not None:
    bias = bias.astype(jnp.float32)
    additive = bias if additive is None else additive + bias
  if additive is not None:
    additive = additive.astype(q.dtype)
  return _fused_sdpa(ops, q, k, v, additive)


def dot_product_attention(q, k, v, *, mask=None, bias=None, implementation):
  """Same contract as `tokamax.dot_product_attention` (no scale/causal args)."""
  if use_mps_ops():
    ops = _mps_fused_ops()
    if ops is not None and q.ndim == 4 and k.ndim == 4 and v.ndim == 4:
      return _mps_fused_dot_product_attention(
          ops, q, k, v, mask=mask, bias=bias
      )
    return _xla_dot_product_attention(q, k, v, mask=mask, bias=bias)
  return tokamax.dot_product_attention(
      q, k, v, mask=mask, bias=bias, implementation=implementation
  )


def _xla_gated_linear_unit(x, weights, activation):
  """`activation(x @ weights[:, 0]) * (x @ weights[:, 1])`, `[K, 2, N]` weights."""
  y = jnp.einsum('...k,kjn->...jn', x, weights)
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  gate = y[..., 0, :].astype(dtype)
  projection = y[..., 1, :].astype(dtype)
  return (activation(gate) * projection).astype(x.dtype)


def gated_linear_unit(
    x, weights, activation: Callable[[jax.Array], jax.Array]
):
  """Same contract as `tokamax.gated_linear_unit`."""
  if use_mps_ops():
    return _xla_gated_linear_unit(x, weights, activation)
  return tokamax.gated_linear_unit(x=x, weights=weights, activation=activation)
