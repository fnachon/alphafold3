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

"""Tests for backend-dispatched ops and the MSA row shuffle."""

from absl.testing import absltest
from alphafold3.model.components import backend_ops
from alphafold3.model.network import featurization
import jax
import jax.numpy as jnp
import numpy as np
import tokamax


# tokamax is the reference for these tests, but it has no `mps` platform.
_NO_TOKAMAX = 'tokamax cannot run float32 ops on the mps platform'


@absltest.skipIf(jax.default_backend() == 'mps', _NO_TOKAMAX)
class BackendOpsTest(absltest.TestCase):

  def test_xla_attention_matches_tokamax(self):
    rng = np.random.RandomState(0)
    q = jnp.asarray(rng.randn(3, 10, 4, 16), jnp.float32)
    k = jnp.asarray(rng.randn(3, 12, 4, 16), jnp.float32)
    v = jnp.asarray(rng.randn(3, 12, 4, 16), jnp.float32)
    bias = jnp.asarray(rng.randn(1, 4, 10, 12), jnp.float32)
    mask = jnp.asarray(rng.rand(3, 1, 1, 12) > 0.2)
    expected = tokamax.dot_product_attention(
        q, k, v, mask=mask, bias=bias, implementation='xla'
    )
    actual = backend_ops._xla_dot_product_attention(
        q, k, v, mask=mask, bias=bias
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

  def test_xla_attention_without_bias_or_mask(self):
    rng = np.random.RandomState(1)
    q = jnp.asarray(rng.randn(2, 5, 2, 8), jnp.float32)
    expected = tokamax.dot_product_attention(q, q, q, implementation='xla')
    actual = backend_ops._xla_dot_product_attention(
        q, q, q, mask=None, bias=None
    )
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

  def test_xla_gated_linear_unit_matches_tokamax(self):
    rng = np.random.RandomState(2)
    x = jnp.asarray(rng.randn(6, 7, 16), jnp.float32)
    weights = jnp.asarray(rng.randn(16, 2, 24), jnp.float32)
    expected = tokamax.gated_linear_unit(
        x=x, weights=weights, activation=jax.nn.swish, implementation='xla'
    )
    actual = backend_ops._xla_gated_linear_unit(x, weights, jax.nn.swish)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

  def test_xla_gated_linear_unit_keeps_dtype(self):
    x = jnp.ones((2, 8), jnp.bfloat16)
    weights = jnp.ones((8, 2, 4), jnp.bfloat16)
    out = backend_ops._xla_gated_linear_unit(x, weights, jax.nn.sigmoid)
    self.assertEqual(out.dtype, jnp.bfloat16)


@absltest.skipIf(jax.default_backend() != 'mps', 'needs the jax-mps backend')
class MpsFusedAttentionTest(absltest.TestCase):

  def test_fused_attention_matches_plain_jax(self):
    ops = backend_ops._mps_fused_ops()
    if ops is None:
      self.skipTest('jax-mps fused ops are unavailable')
    rng = np.random.RandomState(0)
    q = jnp.asarray(rng.randn(8, 24, 4, 32), jnp.bfloat16)
    k = jnp.asarray(rng.randn(8, 24, 4, 32), jnp.bfloat16)
    v = jnp.asarray(rng.randn(8, 24, 4, 32), jnp.bfloat16)
    bias = jnp.asarray(rng.randn(1, 4, 24, 24), jnp.bfloat16)
    mask = jnp.asarray(rng.rand(8, 1, 1, 24) > 0.2)
    expected = backend_ops._xla_dot_product_attention(
        q, k, v, mask=mask, bias=bias
    )
    actual = jax.jit(
        lambda *a: backend_ops._mps_fused_dot_product_attention(
            ops, *a[:3], mask=a[3], bias=a[4]
        )
    )(q, k, v, mask, bias)
    np.testing.assert_allclose(
        np.asarray(actual, np.float32),
        np.asarray(expected, np.float32),
        atol=5e-2,
        rtol=5e-2,
    )

  def test_fused_attention_under_vmap(self):
    ops = backend_ops._mps_fused_ops()
    if ops is None:
      self.skipTest('jax-mps fused ops are unavailable')
    rng = np.random.RandomState(1)
    q = jnp.asarray(rng.randn(3, 4, 12, 2, 32), jnp.float32)
    k = jnp.asarray(rng.randn(3, 4, 12, 2, 32), jnp.float32)
    v = jnp.asarray(rng.randn(3, 4, 12, 2, 32), jnp.float32)
    bias = jnp.asarray(rng.randn(1, 2, 12, 12), jnp.float32)  # Unbatched.
    mask = jnp.asarray(rng.rand(3, 4, 1, 1, 12) > 0.2)

    def fused(q, k, v, mask):
      return backend_ops._mps_fused_dot_product_attention(
          ops, q, k, v, mask=mask, bias=bias
      )

    def plain(q, k, v, mask):
      return backend_ops._xla_dot_product_attention(
          q, k, v, mask=mask, bias=bias
      )

    actual = jax.jit(jax.vmap(fused))(q, k, v, mask)
    expected = jax.vmap(plain)(q, k, v, mask)
    np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=2e-3)


class ShuffleTest(absltest.TestCase):

  def test_gumbel_argsort_orders_valid_rows_first(self):
    valid = np.zeros(2048, np.float32)
    valid[:300] = 1.0
    logits = jnp.asarray((valid - 1.0) * 1e6)
    key = jax.random.PRNGKey(0)
    order = np.asarray(jax.jit(featurization.gumbel_argsort_sample_idx)(key, logits))
    self.assertTrue(np.all(valid[order[:300]] == 1.0))
    self.assertEqual(sorted(order.tolist()), list(range(2048)))

  def test_reversed_gather_matches_numpy(self):
    # jax-mps used to miscompile `rows[idx[::-1]]`; the shuffle must not.
    rng = np.random.RandomState(0)
    rows = jnp.asarray(rng.randint(0, 30, size=(4096, 64)), jnp.int32)
    key = jax.random.PRNGKey(5)
    logits = jnp.zeros((4096,), jnp.float32)
    order = jax.jit(featurization.gumbel_argsort_sample_idx)(key, logits)
    gathered = jax.jit(lambda r, i: r[i, :])(rows, order)
    np.testing.assert_array_equal(
        np.asarray(gathered), np.asarray(rows)[np.asarray(order)]
    )


if __name__ == '__main__':
  absltest.main()
