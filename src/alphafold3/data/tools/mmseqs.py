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

"""MMseqs2 MSA tool: a drop-in replacement for Jackhmmer on machines where HMMER
is slow or unavailable (e.g. Apple Silicon with the MMseqs2 Metal GPU backend).

The search follows the usual GPU recipe: `search --gpu 1 -s 7.5 -a 1 -e 1e-4`
followed by `result2msa` in A3M mode. Loading a large database dominates the
cost of a search, so `query_many` searches several sequences in one MMseqs2 run.
"""

from collections.abc import Sequence
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from absl import logging
from alphafold3.data.tools import msa_tool
from alphafold3.data.tools import subprocess_utils

# Matches the `RepID=ACC_SPECIES` field of UniRef/UniProt headers. The species
# mnemonic is at most 5 alphanumeric characters, which is what the AF3 species
# parser (msa_identifiers) accepts.
_REPID_RE = re.compile(r'RepID=(?P<acc>[A-Za-z0-9]+)_(?P<species>[A-Za-z0-9]{1,5})\b')


def _read_fasta(text: str) -> list[tuple[str, str]]:
  """Returns (header, sequence) pairs from FASTA/A3M text."""
  records = []
  header, seq = None, []
  for line in text.splitlines():
    if line.startswith('>'):
      if header is not None:
        records.append((header, ''.join(seq)))
      header, seq = line[1:], []
    elif header is not None:
      seq.append(line.strip())
  if header is not None:
    records.append((header, ''.join(seq)))
  return records


def _uniprot_style_headers(
    a3m_records: list[tuple[str, str]], full_headers: dict[str, str]
) -> list[tuple[str, str]]:
  """Rewrites `UniRef90_ACC` hit headers to `tr|ACC|ACC_SPECIES`."""
  out = [a3m_records[0]]
  for header, seq in a3m_records[1:]:
    hit_id = header.split()[0]
    match = _REPID_RE.search(full_headers.get(hit_id, ''))
    if match:
      acc = hit_id.rsplit('_', 1)[-1]
      header = f'tr|{acc}|{acc}_{match.group("species")}'
    out.append((header, seq))
  return out


class MMseqs(msa_tool.MsaTool):
  """Python wrapper of the MMseqs2 search + result2msa pipeline."""

  def __init__(
      self,
      *,
      binary_path: str,
      database_path: str,
      n_cpu: int = 0,
      e_value: float = 1e-4,
      max_sequences: int = 10_000,
      sensitivity: float = 7.5,
      gpu: bool = True,
      uniprot_style_headers: bool = False,
      cache_dir: str | os.PathLike[str] | None = None,
      timeout_s: float | None = None,
  ):
    """Initializes the wrapper.

    Args:
      binary_path: The path to the mmseqs executable.
      database_path: MMseqs2 target database. For `gpu=True` this must be a
        padded database (`mmseqs makepaddedseqdb`).
      n_cpu: Number of threads; 0 lets MMseqs2 use all cores.
      e_value: The E-value threshold.
      max_sequences: Maximum number of sequences to return (query included).
      sensitivity: MMseqs2 `-s` value.
      gpu: Run the gapless prefilter on the GPU.
      uniprot_style_headers: See `msa_config.MMseqsConfig`.
      cache_dir: If set, A3Ms are cached on disk keyed by sequence and settings.
      timeout_s: Optional timeout for each mmseqs invocation.
    """
    subprocess_utils.check_binary_exists(path=binary_path, name='MMseqs2')
    if not pathlib.Path(f'{database_path}.dbtype').exists():
      raise RuntimeError(f'MMseqs2 database not found at {database_path}')
    self._binary_path = binary_path
    self._database_path = str(database_path)
    self._n_cpu = n_cpu
    self._e_value = e_value
    self._max_sequences = max_sequences
    self._sensitivity = sensitivity
    self._gpu = gpu
    self._uniprot_style_headers = uniprot_style_headers
    self._cache_dir = pathlib.Path(cache_dir) if cache_dir else None
    self._timeout_s = timeout_s

  def _cache_path(self, sequence: str) -> pathlib.Path | None:
    if self._cache_dir is None:
      return None
    settings = '|'.join(
        map(
            str,
            (
                pathlib.Path(self._database_path).name,
                self._sensitivity,
                self._e_value,
                self._max_sequences,
                self._uniprot_style_headers,
            ),
        )
    )
    tag = hashlib.sha1(settings.encode()).hexdigest()[:8]
    digest = hashlib.sha1(sequence.encode()).hexdigest()[:20]
    return self._cache_dir / tag / f'{digest}.a3m'

  def _run(self, *args: str) -> None:
    proc = subprocess.run(
        [self._binary_path, *args],
        capture_output=True,
        text=True,
        timeout=self._timeout_s,
        check=False,
    )
    if proc.returncode != 0:
      raise RuntimeError(
          f'mmseqs {args[0]} failed (exit {proc.returncode}):\n'
          f'{(proc.stdout + proc.stderr)[-3000:]}'
      )

  def _unpack(self, work: pathlib.Path, name: str, mode: int) -> pathlib.Path:
    threads = ['--threads', str(self._n_cpu)] if self._n_cpu else []
    self._run(
        'result2msa', str(work / 'qdb'), self._database_path,
        str(work / 'res'), str(work / name),
        '--msa-format-mode', str(mode), '--filter-msa', '0', '-v', '1',
        *threads,
    )  # fmt: skip
    out = work / f'{name}_files'
    self._run(
        'unpackdb', str(work / name), str(out),
        '--unpack-suffix', '.a3m', '--unpack-name-mode', '1', '-v', '1',
    )  # fmt: skip
    return out

  def _search(self, sequences: Sequence[str]) -> list[str]:
    """Runs one MMseqs2 search for all sequences and returns their A3Ms."""
    work = pathlib.Path(tempfile.mkdtemp(prefix='af3_mmseqs_'))
    try:
      (work / 'queries.fa').write_text(
          ''.join(f'>{i}\n{s}\n' for i, s in enumerate(sequences))
      )
      threads = ['--threads', str(self._n_cpu)] if self._n_cpu else []
      self._run('createdb', str(work / 'queries.fa'), str(work / 'qdb'), '-v', '1')
      self._run(
          'search', str(work / 'qdb'), self._database_path,
          str(work / 'res'), str(work / 'tmp'),
          '--gpu', '1' if self._gpu else '0',
          '-s', str(self._sensitivity), '-a', '1', '-e', str(self._e_value),
          '--max-seqs', str(max(self._max_sequences, 1000)), '-v', '1',
          *threads,
      )  # fmt: skip
      a3m_dir = self._unpack(work, 'msa', 5)
      header_dir = (
          self._unpack(work, 'msa_hdr', 3)
          if self._uniprot_style_headers
          else None
      )
      a3ms = []
      for i, seq in enumerate(sequences):
        path = a3m_dir / f'{i}.a3m'
        records = (
            _read_fasta(path.read_text())
            if path.exists() and path.stat().st_size
            else []
        )
        if not records:
          records = [('query', seq)]
        if header_dir is not None and (header_dir / f'{i}.a3m').exists():
          full = {
              h.split()[0]: h
              for h, _ in _read_fasta((header_dir / f'{i}.a3m').read_text())
          }
          records = _uniprot_style_headers(records, full)
        records = records[: self._max_sequences]
        # The first record is the query, make sure it carries the full header.
        records[0] = ('query', records[0][1])
        a3ms.append(''.join(f'>{h}\n{s}\n' for h, s in records))
      return a3ms
    finally:
      shutil.rmtree(work, ignore_errors=True)

  def query_many(self, sequences: Sequence[str]) -> dict[str, msa_tool.MsaToolResult]:
    """Queries all unique `sequences`, searching only those that are not cached."""
    unique = list(dict.fromkeys(sequences))
    a3ms: dict[str, str] = {}
    missing = []
    for seq in unique:
      path = self._cache_path(seq)
      if path is not None and path.exists():
        a3ms[seq] = path.read_text()
      else:
        missing.append(seq)
    if missing:
      logging.info(
          'MMseqs2: searching %d new sequence(s), %d cached.',
          len(missing), len(unique) - len(missing),
      )
      for seq, a3m in zip(missing, self._search(missing)):
        a3ms[seq] = a3m
        path = self._cache_path(seq)
        if path is not None:
          path.parent.mkdir(parents=True, exist_ok=True)
          tmp = path.with_suffix('.a3m.tmp')
          tmp.write_text(a3m)
          tmp.replace(path)  # Atomic, so concurrent runs never read a partial file.
    return {
        seq: msa_tool.MsaToolResult(
            target_sequence=seq, e_value=self._e_value, a3m=a3ms[seq]
        )
        for seq in unique
    }

  def query(self, target_sequence: str) -> msa_tool.MsaToolResult:
    """Queries the database with a single sequence."""
    return self.query_many([target_sequence])[target_sequence]
