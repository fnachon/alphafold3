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

import os
import pathlib
import stat
import sys
import textwrap

from absl.testing import absltest
from alphafold3.data import msa
from alphafold3.data import msa_config
from alphafold3.data.tools import mmseqs

# A stand-in `mmseqs` that fakes createdb/search/result2msa/unpackdb. Every
# query gets two hits; mode 3 (headers) carries a UniRef-style description.
_FAKE_MMSEQS = textwrap.dedent(
    f"""\
    #!{sys.executable}
    import pathlib, sys
    args = sys.argv[1:]
    cmd = args[0]
    log = pathlib.Path(__file__).with_suffix('.log')
    with log.open('a') as f:
        f.write(' '.join(args) + '\\n')
    if cmd == 'createdb':
        pathlib.Path(args[2]).write_text(pathlib.Path(args[1]).read_text())
    elif cmd == 'search':
        pathlib.Path(args[4]).with_suffix('.search').write_text('ok')
    elif cmd == 'result2msa':
        mode = args[args.index('--msa-format-mode') + 1]
        pathlib.Path(args[4]).write_text(mode)
    elif cmd == 'unpackdb':
        mode = pathlib.Path(args[1]).read_text()
        qdb = [l for l in pathlib.Path(args[1]).parent.joinpath('qdb').read_text().splitlines()]
        out = pathlib.Path(args[2]); out.mkdir()
        for i in range(0, len(qdb), 2):
            seq = qdb[i + 1]
            hdr = ('UniRef90_P11111 x Tax=Foo TaxID=1 RepID=P11111_HUMAN'
                   if mode == '3' else 'UniRef90_P11111')
            (out / f'{{i // 2}}.a3m').write_text(
                f'>{{qdb[i][1:]}}\\n{{seq}}\\n>{{hdr}}\\n{{seq[:-1].lower()}}{{seq[-1]}}\\n')
    """
)


class MMseqsTest(absltest.TestCase):

  def _make_tool(self, **kwargs) -> tuple[mmseqs.MMseqs, pathlib.Path]:
    tmp = pathlib.Path(self.create_tempdir().full_path)
    binary = tmp / 'mmseqs'
    binary.write_text(_FAKE_MMSEQS)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    db = tmp / 'db'
    (tmp / 'db.dbtype').write_text('x')
    tool = mmseqs.MMseqs(
        binary_path=str(binary), database_path=str(db), **kwargs
    )
    return tool, binary.with_suffix('.log')

  def test_query_returns_a3m_starting_with_query(self):
    tool, _ = self._make_tool(gpu=False)
    result = tool.query('MKV')
    self.assertEqual(result.target_sequence, 'MKV')
    self.assertEqual(result.a3m, '>query\nMKV\n>UniRef90_P11111\nmkV\n')

  def test_gpu_flag_and_batching(self):
    tool, log = self._make_tool(gpu=True)
    results = tool.query_many(['MKV', 'AAA', 'MKV'])
    self.assertEqual(set(results), {'MKV', 'AAA'})
    lines = log.read_text().splitlines()
    self.assertEqual(sum(l.startswith('search') for l in lines), 1)
    self.assertIn('--gpu 1', [l for l in lines if l.startswith('search')][0])

  def test_cache_avoids_second_search(self):
    cache = self.create_tempdir().full_path
    tool, log = self._make_tool(cache_dir=cache)
    tool.query_many(['MKV'])
    tool.query_many(['MKV', 'AAA'])
    searches = [l for l in log.read_text().splitlines() if l.startswith('search')]
    self.assertLen(searches, 2)  # The second search only holds the new one.

  def test_max_sequences_truncates(self):
    tool, _ = self._make_tool(max_sequences=1)
    self.assertEqual(tool.query('MKV').a3m, '>query\nMKV\n')

  def test_uniprot_style_headers_enable_species_parsing(self):
    tool, _ = self._make_tool(uniprot_style_headers=True)
    a3m = tool.query('MKV').a3m
    self.assertIn('>tr|P11111|P11111_HUMAN\n', a3m)
    parsed = msa.Msa.from_a3m(
        query_sequence='MKV', chain_poly_type='polypeptide(L)', a3m=a3m
    )
    self.assertEqual(list(parsed.descriptions), ['query', 'tr|P11111|P11111_HUMAN'])

  def test_failure_raises_with_output(self):
    tool, binary_log = self._make_tool()
    broken = pathlib.Path(str(binary_log.with_suffix('')))
    broken.write_text('#!/bin/sh\necho boom >&2\nexit 3\n')
    with self.assertRaisesRegex(RuntimeError, r'(?s)createdb failed.*boom'):
      tool.query('MKV')

  def test_missing_database_raises(self):
    with self.assertRaisesRegex(RuntimeError, 'database not found'):
      mmseqs.MMseqs(binary_path=sys.executable, database_path='/no/such/db')

  def test_get_msa_dispatches_to_mmseqs(self):
    tool, _ = self._make_tool(gpu=False)
    config = msa_config.RunConfig(
        config=msa_config.MMseqsConfig(
            binary_path=tool._binary_path,
            database_config=msa_config.DatabaseConfig(
                name='db', path=tool._database_path
            ),
            gpu=False,
        ),
        chain_poly_type='polypeptide(L)',
        crop_size=None,
    )
    result = msa.get_msa('MKV', config, 'polypeptide(L)')
    self.assertEqual(result.depth, 2)


if __name__ == '__main__':
  absltest.main()
