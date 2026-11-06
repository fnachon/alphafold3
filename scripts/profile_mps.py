#!/usr/bin/env python3
"""Profile AlphaFold 3 inference on one JAX backend (built for Apple MPS).

Two reports:

  trunk      Time per Pairformer layer and per trunk pass, and how a layer splits
             into triangle multiplication / triangle attention / transition /
             single track (isolated modules, real weights, bf16 activations).
  end2end    Whole-model run times from slopes: trunk pass (recycles), Pairformer
             layer, diffusion step and diffusion sample count.

Example:
  python scripts/profile_mps.py --input_json=fold_input.json --bucket=512 \
      --model_dir=weights --backend=mps --report=trunk,end2end

The input must already carry its MSA/templates (or be MSA-free). Timings are
medians/means of warm runs; expect ~10-15% run-to-run noise.
"""

import argparse
import contextlib
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import haiku as hk  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import run_alphafold as ra  # noqa: E402
from alphafold3.common import folding_input  # noqa: E402
from alphafold3.constants import chemical_components  # noqa: E402
from alphafold3.data import featurisation  # noqa: E402
from alphafold3.model import model  # noqa: E402
from alphafold3.model.components import utils  # noqa: E402
from alphafold3.model.network import modules  # noqa: E402

_PAIRFORMER = 'diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer/'


def _time(fn, *args, repeat=3):
  """Seconds per call, after one warm-up (which includes compilation)."""
  jax.block_until_ready(fn(*args))
  times = []
  for _ in range(repeat):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    times.append(time.perf_counter() - t0)
  return float(np.median(times))


def _load_batch(args, device):
  fold_input = next(iter(
      folding_input.load_fold_inputs_from_path(pathlib.Path(args.input_json))
  ))
  example = featurisation.featurise_input(
      fold_input=fold_input,
      buckets=[args.bucket],
      ccd=chemical_components.Ccd(user_ccd=fold_input.user_ccd),
      verbose=False,
  )[0]
  batch = jax.tree_util.tree_map(
      jnp.asarray, ra.utils.remove_invalidly_typed_feats(example)
  )
  return jax.device_put(batch, device)


def _config(**overrides):
  return ra.make_model_config(flash_attention_implementation='xla', **overrides)


def report_trunk(args, device):
  """Isolated Pairformer layer breakdown (layer 0 weights)."""
  n = args.bucket
  cfg = _config()
  params = ra.ModelRunner(cfg, device, args.model_dir).model_params
  rs = np.random.RandomState(0)
  pair = jnp.asarray(rs.randn(n, n, 128), jnp.bfloat16)
  pair_mask = jnp.ones((n, n), jnp.bfloat16)
  single = jnp.asarray(rs.randn(n, 384), jnp.bfloat16)
  single_mask = jnp.ones((n,), jnp.bfloat16)
  pf_cfg, gc = cfg.evoformer.pairformer, cfg.global_config

  def layer0(prefix=''):
    return {
        k[len(_PAIRFORMER) - len('trunk_pairformer/'):]: {
            name: v[0] for name, v in d.items()
        }
        for k, d in params.items()
        if k.startswith(_PAIRFORMER + prefix)
    }

  def timed(make, inputs, prefix=''):
    def fwd(*a):
      with utils.bfloat16_context():
        return make()(*a)

    f = hk.transform(fwd)
    jf = jax.jit(lambda p, *a: f.apply(p, None, *a))
    return _time(jf, layer0(prefix), *inputs)

  layer = lambda ws: (lambda: modules.PairFormerIteration(
      pf_cfg, gc, with_single=ws, name='trunk_pairformer'))
  t_full = timed(layer(True), (pair, pair_mask, single, single_mask))
  t_pair = timed(layer(False), (pair, pair_mask))

  # Ablate sub-blocks by zeroing them inside the jit.
  def ablated(attr):
    orig = getattr(modules, attr)
    class Zero(hk.Module):
      def __init__(self, *a, **k):
        super().__init__(name=k.get('name'))
      def __call__(self, act, *a, **k):
        return jnp.zeros_like(act)
    setattr(modules, attr, Zero)
    try:
      return timed(layer(False), (pair, pair_mask))
    finally:
      setattr(modules, attr, orig)

  t_tri_mul = t_pair - ablated('TriangleMultiplication')
  t_tri_att = t_pair - ablated('GridSelfAttention')
  t_trans = t_pair - ablated('TransitionBlock')

  print(f'\n== Pairformer layer, N={n}, bf16 activations ==')
  rows = [
      ('full layer (pair + single)', t_full),
      ('pair track', t_pair),
      ('  triangle multiplication (x2)', t_tri_mul),
      ('  triangle attention (x2)', t_tri_att),
      ('  pair transition', t_trans),
      ('single track', t_full - t_pair),
  ]
  for label, t in rows:
    print(f'{label:34s} {t * 1e3:8.1f} ms   {100 * t / t_full:5.1f}%')
  print(f'48 layers: {48 * t_full:.1f} s per trunk pass (isolated estimate)')


def report_end2end(args, device):
  """Whole-model timings; costs are the slopes between runs."""
  batch = _load_batch(args, device)
  base = _config()
  full = ra.ModelRunner(base, device, args.model_dir).model_params

  def sliced(m, p):
    out = {}
    for mod, d in full.items():
      out[mod] = {}
      for name, v in d.items():
        if 'layer_stack' in mod and 'msa_stack' in mod:
          v = v[:m]
        elif 'layer_stack' in mod and 'trunk_pairformer' in mod:
          v = v[:p]
        out[mod][name] = v
    return out

  def run(m=4, p=48, steps=1, recycles=0, samples=1):
    cfg = _config(num_diffusion_samples=samples, num_recycles=recycles)
    cfg.evoformer.msa_stack.num_layer = m
    cfg.evoformer.pairformer.num_layer = p
    cfg.heads.diffusion.eval.steps = steps
    f = hk.transform(lambda b: model.Model(cfg)(b))
    jf = jax.jit(f.apply)
    prm = sliced(m, p)
    key = jax.random.PRNGKey(0)
    t = _time(lambda: jf(prm, key, batch), repeat=2)
    print(f'  m={m:2d} p={p:2d} steps={steps:3d} recycles={recycles} '
          f'samples={samples}: {t:7.2f} s', flush=True)
    return t

  print(f'\n== End to end, N={args.bucket}, backend={device.platform} ==')
  t_r0 = run(recycles=0)
  t_r1 = run(recycles=1)
  trunk_pass = t_r1 - t_r0
  t_p12 = run(p=12)
  per_layer = (t_r0 - t_p12) / 36  # widest baseline, least noise
  t_s1 = run(steps=1)
  t_s11 = run(steps=11)
  per_step = (t_s11 - t_s1) / 10
  t_x5 = run(samples=5, steps=11)

  print('\nDerived:')
  print(f'  trunk pass (48 layers)     {trunk_pass:7.2f} s  '
        f'(x{args.recycles + 1} = {trunk_pass * (args.recycles + 1):.1f} s)')
  print(f'  Pairformer layer (slope)   {per_layer * 1e3:7.1f} ms')
  print(f'  diffusion step, 1 sample   {per_step * 1e3:7.1f} ms '
        f'(x200 = {per_step * 200:.1f} s)')
  print(f'  diffusion step, 5 samples  {(t_x5 - t_s1) / 10 * 1e3:7.1f} ms')


def main():
  ap = argparse.ArgumentParser(description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument('--input_json', required=True)
  ap.add_argument('--model_dir', default='weights')
  ap.add_argument('--bucket', type=int, default=512)
  ap.add_argument('--backend', default='mps')
  ap.add_argument('--recycles', type=int, default=10,
                  help='recycle count used to extrapolate total trunk time')
  ap.add_argument('--report', default='trunk,end2end')
  args = ap.parse_args()

  device = jax.local_devices(backend=args.backend)[0]
  jax.config.update('jax_default_device', device)
  reports = set(args.report.split(','))
  if 'trunk' in reports:
    report_trunk(args, device)
  if 'end2end' in reports:
    report_end2end(args, device)


if __name__ == '__main__':
  main()
