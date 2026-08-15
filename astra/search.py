
import gc
import os
import sys
import time
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
import pyhmmer
from concurrent.futures import ThreadPoolExecutor
from astra import initialize
from astra import rp16 as rp16_module


PRESSED_SUFFIXES = ('h3m', 'h3i', 'h3f', 'h3p')
HMM_CHUNK_SIZE = 2000
GPU_CELL_CAP = 100_000_000
GPU_TIMING_ENV = 'ASTRA_GPU_OVERLAP_TIMING'
GPU_SERIAL_ENV = 'ASTRA_GPU_PROFILE_SERIAL'


class GPUConfigurationError(ValueError):
    """Raised when an explicit installed-database GPU request is invalid."""


class GPUOverlapMetrics:
    """Opt-in counters for the bounded profile-generation pipeline."""

    def __init__(self):
        self.requested_thread_count = 0
        self.profile_worker_count = 0
        self.profile_build_worker_count = 0
        self.profile_selection_worker_count = 0
        self.producer_slot_count = 0
        self.continuation_worker_count = 0
        self.profile_overlap_enabled = False
        self.profile_host_bytes = 0
        self.chunk_count = 0
        self.generated_chunk_count = 0
        self.consumed_chunk_count = 0
        self.ready_without_wait_count = 0
        self.profile_load_seconds = 0.0
        self.session_build_seconds = 0.0
        self.target_batch_seconds = 0.0
        self.preflight_seconds = 0.0
        self.generation_seconds = 0.0
        self.selection_seconds = 0.0
        self.generation_wait_seconds = 0.0
        self.initial_generation_wait_seconds = 0.0
        self.pipeline_stall_seconds = 0.0
        self.continuation_seconds = 0.0
        self.overlap_seconds = 0.0
        self.pipeline_wall_seconds = 0.0

    def snapshot(self):
        return {
            'requested_thread_count': self.requested_thread_count,
            'profile_worker_count': self.profile_worker_count,
            'profile_build_worker_count': self.profile_build_worker_count,
            'profile_selection_worker_count': (
                self.profile_selection_worker_count
            ),
            'producer_slot_count': self.producer_slot_count,
            'continuation_worker_count': self.continuation_worker_count,
            'profile_overlap_enabled': self.profile_overlap_enabled,
            'profile_host_bytes': self.profile_host_bytes,
            'chunk_count': self.chunk_count,
            'generated_chunk_count': self.generated_chunk_count,
            'consumed_chunk_count': self.consumed_chunk_count,
            'ready_without_wait_count': self.ready_without_wait_count,
            'profile_load_seconds': self.profile_load_seconds,
            'session_build_seconds': self.session_build_seconds,
            'target_batch_seconds': self.target_batch_seconds,
            'preflight_seconds': self.preflight_seconds,
            'generation_seconds': self.generation_seconds,
            'selection_seconds': self.selection_seconds,
            'generation_wait_seconds': self.generation_wait_seconds,
            'initial_generation_wait_seconds': self.initial_generation_wait_seconds,
            'pipeline_stall_seconds': self.pipeline_stall_seconds,
            'continuation_seconds': self.continuation_seconds,
            'overlap_seconds': self.overlap_seconds,
            'pipeline_wall_seconds': self.pipeline_wall_seconds,
        }


def gpu_hmm_chunk_size(sequence_count):
    """Bound one GPU candidate matrix while retaining Astra's 2,000-HMM cap."""
    if sequence_count > GPU_CELL_CAP:
        raise GPUConfigurationError(
            f"explicit GPU search supports at most {GPU_CELL_CAP:,} targets; "
            f"received {sequence_count:,}"
        )
    return min(
        HMM_CHUNK_SIZE,
        max(1, GPU_CELL_CAP // max(1, sequence_count)),
    )


def gpu_profile_worker_allocation(threads, overlap_requested):
    """Return overlap and compute-worker slots under Astra's CLI convention.

    ``--threads`` counts compute/search workers, as in Astra's CPU path; the
    common main writer is excluded. Serial control reserves the same producer
    slot as overlap so the continuation width remains identical.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise GPUConfigurationError(
            "GPU profile sessions require a positive integer compute-worker budget"
        )
    if type(overlap_requested) is not bool:
        raise TypeError("overlap_requested must be bool")
    producer_slots = int(threads >= 2)
    overlap_enabled = overlap_requested and bool(producer_slots)
    return overlap_enabled, producer_slots, threads - producer_slots


def parse_gpu_manifest_mappings(entries):
    """Parse repeatable ``DB=PATH`` values without accepting ambiguity."""
    mappings = {}
    for entry in entries or ():
        if not isinstance(entry, str):
            raise GPUConfigurationError("--gpu-manifest values must be DB=PATH strings")
        db_name, separator, manifest_path = entry.partition('=')
        db_name = db_name.strip()
        manifest_path = manifest_path.strip()
        if not separator or not db_name or not manifest_path:
            raise GPUConfigurationError(
                f"malformed --gpu-manifest value {entry!r}; expected DB=PATH"
            )
        if db_name in mappings:
            raise GPUConfigurationError(
                f"duplicate --gpu-manifest mapping for database {db_name!r}"
            )
        mappings[db_name] = os.path.expandvars(os.path.expanduser(manifest_path))
    return mappings


def discover_pressed_base(installation_dir):
    """Return the unambiguous complete pressed base in an installation directory.

    Astra's own press operation normally names the base after the directory.
    Older installations such as HydDB instead keep a source-derived base name,
    so they are accepted when exactly one complete pressed set is present.
    """
    try:
        raw_directory = os.fspath(installation_dir)
    except TypeError as error:
        raise GPUConfigurationError("installed database has no installation directory") from error
    if not raw_directory:
        raise GPUConfigurationError("installed database has no installation directory")

    db_dir = Path(os.path.expandvars(os.path.expanduser(raw_directory)))
    if not db_dir.is_dir():
        raise GPUConfigurationError(
            f"installed database directory does not exist: {db_dir}"
        )
    db_dir = db_dir.resolve()

    def complete(base):
        return all(Path(f"{base}.{suffix}").is_file() for suffix in PRESSED_SUFFIXES)

    candidates = set()
    for member in db_dir.iterdir():
        if member.suffix in {f'.{suffix}' for suffix in PRESSED_SUFFIXES}:
            base = member.with_suffix('')
            if complete(base):
                candidates.add(base)

    if not candidates:
        raise GPUConfigurationError(
            f"no complete pressed HMM set found in {db_dir}; expected .h3m/.h3i/.h3f/.h3p"
        )
    if len(candidates) != 1:
        names = ', '.join(sorted(base.name for base in candidates))
        raise GPUConfigurationError(
            f"ambiguous pressed HMM sets in {db_dir}: {names}"
        )
    return candidates.pop()


def _resolve_installed_hmm_names(installed_hmms, parsed_json):
    names = installed_hmms.split(',') if ',' in installed_hmms else [installed_hmms]
    if 'all_prot' in names:
        names = [
            db['name'] for db in parsed_json['db_urls']
            if db['molecule_type'] == 'protein' and db['installed']
        ]
    return names


def validate_gpu_configuration(mappings, installed_hmm_names, parsed_json,
                               threads, macsyfinder_enabled):
    """Reject explicit GPU mappings that cannot be consumed exactly once."""
    if not mappings:
        return
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise GPUConfigurationError(
            "--threads must be a positive integer when --gpu-manifest is used"
        )
    if macsyfinder_enabled:
        raise GPUConfigurationError(
            "--write_macsyfinder cannot be combined with --gpu-manifest"
        )

    requested_names = list(installed_hmm_names or ())
    seen = set()
    duplicates = set()
    for name in requested_names:
        if name in seen:
            duplicates.add(name)
        seen.add(name)
    if duplicates:
        raise GPUConfigurationError(
            "duplicate installed HMM database(s) with GPU mappings: "
            + ', '.join(sorted(duplicates))
        )

    requested = set(requested_names)
    eligible = {
        db['name'] for db in parsed_json['db_urls']
        if (db.get('name') in requested
            and db.get('installed')
            and db.get('molecule_type') == 'protein'
            and db.get('installation_dir'))
    }
    unused = sorted(set(mappings) - eligible)
    if unused:
        raise GPUConfigurationError(
            "unused --gpu-manifest mapping(s): " + ', '.join(unused)
        )


def preflight_gpu_databases(mappings, installed_hmm_names, parsed_json,
                            all_sequences, threads, gpu_metrics=None):
    """Load every attested mapped database and initialize one target batch."""
    if not mappings:
        return {}, None, False

    preflight_started = time.perf_counter()
    from plan7_gpu import ProfileSession, SequenceBatch, load_pressed_profiles
    from plan7_gpu.pressed_manifest import validate_pressed_manifest

    database_specs = {}
    for db_name in installed_hmm_names:
        manifest_path = mappings.get(db_name)
        if manifest_path is None:
            continue
        database = next(
            item for item in parsed_json['db_urls'] if item['name'] == db_name
        )
        pressed_base = discover_pressed_base(database['installation_dir'])
        validate_pressed_manifest(pressed_base, manifest_path)
        database_specs[db_name] = (pressed_base, manifest_path)

    databases = {}
    for db_name, (pressed_base, manifest_path) in database_specs.items():
        load_started = time.perf_counter()
        pairs = load_pressed_profiles(pressed_base, manifest=manifest_path)
        if gpu_metrics is not None:
            gpu_metrics[db_name].profile_load_seconds += (
                time.perf_counter() - load_started
            )
        databases[db_name] = (
            pressed_base,
            pairs,
            None,
        )

    postfilter = gpu_postfilter_available()
    batch = None
    try:
        batch_started = time.perf_counter()
        batch = SequenceBatch(
            all_sequences,
            alphabet=pyhmmer.easel.Alphabet.amino(),
        )
        batch_seconds = time.perf_counter() - batch_started
        if gpu_metrics is not None:
            for metrics in gpu_metrics.values():
                metrics.target_batch_seconds = batch_seconds
        # A fresh producer thread begins on CUDA ordinal 0. Until plan7_gpu
        # exposes a scoped device bind, keep nonzero batches on the existing
        # same-thread path instead of creating an unusable ProfileSession.
        # Build workers retire before ProfileSession returns. Keeping the
        # persistent selection pool at zero makes eager multi-database
        # sessions threadless and selections synchronous before continuation.
        session_supported = (
            postfilter
            and gpu_profile_forward_available()
            and batch.memory_snapshot['device_ordinal'] == 0
            and callable(
                getattr(batch, '_postfilter_forward_selection', None)
            )
        )
        if session_supported:
            for db_name, (pressed_base, pairs, _) in tuple(databases.items()):
                if not pairs:
                    continue
                session_started = time.perf_counter()
                session = ProfileSession(
                    pairs,
                    build_workers=threads,
                    selection_workers=0,
                )
                databases[db_name] = (
                    pressed_base,
                    pairs,
                    session,
                )
                if gpu_metrics is not None:
                    gpu_metrics[db_name].session_build_seconds += (
                        time.perf_counter() - session_started
                    )
        if gpu_metrics is not None:
            preflight_seconds = time.perf_counter() - preflight_started
            for metrics in gpu_metrics.values():
                metrics.preflight_seconds = preflight_seconds
        return databases, batch, postfilter
    except BaseException:
        for _, _, session in databases.values():
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    pass
        if batch is not None:
            try:
                batch.close()
            except BaseException:
                pass
        raise


def gpu_postfilter_available():
    """Return whether plan7_gpu can resume HMMER after exact GPU filters."""
    from plan7_gpu import _pipeline

    probe = getattr(_pipeline, '_filter_scores_seam_available', None)
    return callable(probe) and probe() is True


def gpu_profile_forward_available():
    """Return whether selection generation can include exact CUDA Forward."""
    from plan7_gpu import SequenceBatch, _pipeline

    probe = getattr(
        _pipeline, '_filter_and_forward_scores_seam_available', None
    )
    return (
        callable(probe)
        and probe() is True
        and hasattr(SequenceBatch, '_postfilter_forward_selection')
    )


def has_thresholds(x):
    """Check if an HMM has any bitscore cutoffs available."""
    return (x.cutoffs.gathering_available() or
            x.cutoffs.noise_available() or
            x.cutoffs.trusted_available())


def write_macsyfinder_hit(hits, macsyfinder_dir, hmm_name_to_filename=None):
    """Write one HMM's search results as a hmmsearch-format text file for MacSyFinder.

    MacSyFinder's ``--previous-run`` expects per-gene ``.search_hmm.out`` files
    inside an ``hmmer_results/`` directory.  This function writes a minimal but
    parser-compatible file from a pyhmmer ``TopHits`` object.

    Called once per HMM per input FASTA file.  When ``prot_in`` is a directory
    with multiple ``.faa`` files, the same HMM file is appended to across
    successive FASTA files.  The header is written only on first call; the
    ``//`` end-of-query marker is added by ``finalize_macsyfinder_files()``.

    Parameters
    ----------
    hits : pyhmmer.plan7.TopHits
        Results of searching one HMM against the sequence database.
    macsyfinder_dir : str
        Path to the output directory (will contain ``hmmer_results/``).
    hmm_name_to_filename : dict, optional
        Mapping from HMM internal NAME to the HMM filename stem.
        When provided, the output file is named by the filename stem
        (which matches what MacSyFinder expects) rather than the
        internal NAME field (which may differ).
    """
    hmm_name = hits.query.name
    hmm_length = hits.query.M

    # Use filename stem if mapping is available, fall back to internal NAME
    file_stem = hmm_name
    if hmm_name_to_filename and hmm_name in hmm_name_to_filename:
        file_stem = hmm_name_to_filename[hmm_name]

    hmmer_dir = os.path.join(macsyfinder_dir, "hmmer_results")
    os.makedirs(hmmer_dir, exist_ok=True)
    out_path = os.path.join(hmmer_dir, f"{file_stem}.search_hmm.out")

    file_exists = os.path.exists(out_path)

    with open(out_path, "a") as fh:
        # Write header only on first call for this HMM
        if not file_exists:
            fh.write("# hmmsearch :: search profile(s) against a sequence database\n")
            fh.write("# HMMER 3.4 (pyhmmer); http://hmmer.org/\n")
            fh.write("# - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n")
            fh.write(f"Query:       {hmm_name}  [M={hmm_length}]\n\n")

        for hit in hits:
            if not hit.included:
                continue
            hit_name = hit.name
            hit_desc = hit.description if hit.description else ""

            fh.write(f">> {hit_name}\n")
            fh.write("   #    score  bias  c-Evalue  i-Evalue hmmfrom  hmm to"
                     "    alifrom  ali to    envfrom  env to     acc\n")
            fh.write(" ---   ------ ----- --------- --------- ------- -------"
                     "    ------- -------    ------- -------    ----\n")

            for dom_idx, domain in enumerate(hit.domains.reported, start=1):
                aln = domain.alignment
                h_from = aln.hmm_from if aln else 0
                h_to   = aln.hmm_to   if aln else 0
                t_from = aln.target_from if aln else domain.env_from
                t_to   = aln.target_to   if aln else domain.env_to

                fh.write(
                    f"  {dom_idx:>3d} ! {domain.score:>7.1f} {domain.bias:>5.1f}"
                    f"  {domain.c_evalue:>9.2e}  {domain.i_evalue:>9.2e}"
                    f"  {h_from:>7d} {h_to:>7d} .."
                    f"  {t_from:>7d} {t_to:>7d} .."
                    f"  {domain.env_from:>7d} {domain.env_to:>7d} .. 0.00\n"
                )

            fh.write("\n")


def finalize_macsyfinder_files(macsyfinder_dir):
    """Append ``//`` end-of-query markers to all MacSyFinder hmmsearch output files.

    Must be called once after all ``write_macsyfinder_hit()`` calls are complete.
    """
    hmmer_dir = os.path.join(macsyfinder_dir, "hmmer_results")
    if not os.path.isdir(hmmer_dir):
        return
    for fname in os.listdir(hmmer_dir):
        if fname.endswith(".search_hmm.out"):
            fpath = os.path.join(hmmer_dir, fname)
            with open(fpath, "a") as fh:
                fh.write("//\n")

def extract_sequences(results_or_ids, protein_dict_or_outdir, outdir=None):
    """Extract hit sequences and write per-HMM FASTAs.

    Supports two calling conventions:
      - New: extract_sequences(hit_ids_by_hmm, protein_dict, outdir)
        where hit_ids_by_hmm is dict[str, set[str]]
      - Legacy (scan.py): extract_sequences(results_dataframes_dict, outdir)
        where results_dataframes_dict is dict[str, DataFrame]
    """
    import pandas as pd

    if outdir is None:
        # Legacy call: extract_sequences(results_dataframes_dict, outdir)
        # results_or_ids is a dict of DataFrames, protein_dict_or_outdir is outdir
        outdir = protein_dict_or_outdir
        fastas_dir = os.path.join(outdir, 'fastas')
        os.makedirs(fastas_dir, exist_ok=True)
        for genome_file, df in results_or_ids.items():
            for hmm_name in df['hmm_name'].unique():
                ids = df[df['hmm_name'] == hmm_name]['sequence_id'].tolist()
                hits_fasta = os.path.join(fastas_dir, f"{hmm_name}.faa")
                with open(hits_fasta, 'a') as fh:
                    # Legacy path: re-read from disk (scan.py doesn't keep seqs in memory)
                    with pyhmmer.easel.SequenceFile(genome_file, digital=True,
                                                     alphabet=pyhmmer.easel.Alphabet.amino()) as sf:
                        for seq in sf:
                            if seq.name in ids:
                                text_seq = seq.textize()
                                fh.write(f">{text_seq.name}\n{text_seq.sequence}\n")
        return

    # New call: extract_sequences(hit_ids_by_hmm, protein_dict, outdir)
    hit_ids_by_hmm = results_or_ids
    protein_dict = protein_dict_or_outdir
    fastas_dir = os.path.join(outdir, 'fastas')
    os.makedirs(fastas_dir, exist_ok=True)

    # Build a flat name → sequence lookup (once, not per-HMM)
    seq_lookup = {}
    for sequences in protein_dict.values():
        for seq in sequences:
            seq_lookup[seq.name] = seq

    for hmm_name, seq_ids in hit_ids_by_hmm.items():
        hits_fasta = os.path.join(fastas_dir, f"{hmm_name}.faa")
        with open(hits_fasta, 'w') as fh:
            for sid in seq_ids:
                seq = seq_lookup.get(sid)
                if seq is not None:
                    text_seq = seq.textize()
                    fh.write(f">{text_seq.name}\n{text_seq.sequence}\n")


def _consume_gpu_candidate_chunk(spec, candidates, total_chunks, threads, fh,
                                 gpu_hmmsearch, gpu_metrics):
    chunk_index, hmm_chunk, _, kwargs = spec
    print(f"  Chunk {chunk_index}/{total_chunks} "
          f"({len(hmm_chunk)} HMMs)...", end="", flush=True)
    hit_iterator = None
    hits = None
    started = time.perf_counter()
    try:
        hit_iterator = gpu_hmmsearch(
            hmm_chunk,
            candidates,
            cpus=threads,
            postfilter=True,
            **kwargs,
        )
        for hits in hit_iterator:
            process_hits_to_file(hits, fh)
    except BaseException:
        if hit_iterator is not None:
            close = getattr(hit_iterator, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException:
                    pass
        raise
    finally:
        finished = time.perf_counter()
        if gpu_metrics is not None:
            gpu_metrics.continuation_seconds += finished - started
        hit_iterator = None
        hits = None
    if gpu_metrics is not None:
        gpu_metrics.consumed_chunk_count += 1
    print(" done")
    return started, finished


def _generate_gpu_profile_candidates(sequence_batch, selection, kwargs):
    """Run the same bounded CUDA stages as the live post-filter product path."""
    return sequence_batch._postfilter_forward_selection(
        selection,
        kwargs.get('F1', 0.02),
        kwargs.get('F2', 0.001),
        kwargs.get('F3', 0.00001),
        kwargs.get('bias_filter', True),
    )


def _run_gpu_profile_serial(chunks, profile_session, sequence_batch,
                            threads, fh, gpu_metrics=None):
    """Run sealed GPU-through-Forward generation as the serial control."""
    from plan7_gpu.astra_search import hmmsearch as gpu_hmmsearch

    if gpu_metrics is not None:
        statistics = profile_session.statistics
        gpu_metrics.chunk_count += len(chunks)
        gpu_metrics.profile_worker_count = statistics['worker_count']
        gpu_metrics.profile_build_worker_count = statistics.get(
            'build_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_selection_worker_count = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_host_bytes = statistics['host_bytes']
    pipeline_started = time.perf_counter()
    try:
        for spec in chunks:
            selection_started = time.perf_counter()
            selection = profile_session.select(spec[2])
            if gpu_metrics is not None:
                gpu_metrics.selection_seconds += (
                    time.perf_counter() - selection_started
                )
            generation_started = time.perf_counter()
            try:
                candidates = _generate_gpu_profile_candidates(
                    sequence_batch,
                    selection,
                    spec[3],
                )
            except BaseException:
                try:
                    selection.close()
                except BaseException:
                    pass
                raise
            selection.close()
            generation_finished = time.perf_counter()
            if gpu_metrics is not None:
                gpu_metrics.generated_chunk_count += 1
                gpu_metrics.generation_seconds += (
                    generation_finished - generation_started
                )
            _consume_gpu_candidate_chunk(
                spec,
                candidates,
                len(chunks),
                threads,
                fh,
                gpu_hmmsearch,
                gpu_metrics,
            )
            candidates = None
    finally:
        if gpu_metrics is not None:
            gpu_metrics.pipeline_wall_seconds += (
                time.perf_counter() - pipeline_started
            )


def _run_gpu_profile_pipeline(chunks, profile_session, sequence_batch,
                              threads, fh, gpu_metrics=None):
    """Overlap one sealed GPU-through-Forward batch with one CPU consumer."""
    if not chunks:
        return

    from plan7_gpu.astra_search import hmmsearch as gpu_hmmsearch

    if gpu_metrics is not None:
        statistics = profile_session.statistics
        gpu_metrics.chunk_count += len(chunks)
        gpu_metrics.profile_worker_count = statistics['worker_count']
        gpu_metrics.profile_build_worker_count = statistics.get(
            'build_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_selection_worker_count = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        gpu_metrics.profile_host_bytes = statistics['host_bytes']

    def generate(spec, selection):
        started = time.perf_counter()
        try:
            candidates = _generate_gpu_profile_candidates(
                sequence_batch,
                selection,
                spec[3],
            )
        except BaseException:
            try:
                selection.close()
            except BaseException:
                pass
            raise
        selection.close()
        return candidates, started, time.perf_counter()

    pipeline_started = time.perf_counter()
    executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix='astra-gpu-generate',
    )
    pending_future = None
    pending_error = None
    pending_selection = None
    def start_generation(spec):
        selection_started = time.perf_counter()
        try:
            selection = profile_session.select(spec[2])
        except BaseException as error:
            if gpu_metrics is not None:
                gpu_metrics.selection_seconds += (
                    time.perf_counter() - selection_started
                )
            return None, error, None
        if gpu_metrics is not None:
            gpu_metrics.selection_seconds += (
                time.perf_counter() - selection_started
            )
        try:
            return executor.submit(generate, spec, selection), None, selection
        except BaseException as error:
            try:
                selection.close()
            except BaseException:
                pass
            return None, error, None

    pending_future, pending_error, pending_selection = start_generation(chunks[0])
    previous_consumption = None
    try:
        for position, spec in enumerate(chunks):
            if pending_error is not None:
                raise pending_error
            completed_future = pending_future
            pending_future = None
            pending_selection = None
            ready_without_wait = completed_future.done()
            wait_started = time.perf_counter()
            candidates, generation_started, generation_finished = (
                completed_future.result()
            )
            wait_seconds = time.perf_counter() - wait_started
            if gpu_metrics is not None:
                gpu_metrics.generated_chunk_count += 1
                gpu_metrics.ready_without_wait_count += int(ready_without_wait)
                gpu_metrics.generation_seconds += (
                    generation_finished - generation_started
                )
                gpu_metrics.generation_wait_seconds += wait_seconds
                if position == 0:
                    gpu_metrics.initial_generation_wait_seconds += wait_seconds
                else:
                    gpu_metrics.pipeline_stall_seconds += wait_seconds
                if previous_consumption is not None:
                    overlap_start = max(
                        generation_started, previous_consumption[0]
                    )
                    overlap_finish = min(
                        generation_finished, previous_consumption[1]
                    )
                    gpu_metrics.overlap_seconds += max(
                        0.0, overlap_finish - overlap_start
                    )

            # Preselection is synchronous and the session has zero pack
            # workers, so its copy cannot contend with CPU continuation.
            # A prefetch failure is deferred until this ready chunk is emitted.
            if position + 1 < len(chunks):
                (
                    pending_future,
                    pending_error,
                    pending_selection,
                ) = start_generation(chunks[position + 1])

            previous_consumption = _consume_gpu_candidate_chunk(
                spec,
                candidates,
                len(chunks),
                threads,
                fh,
                gpu_hmmsearch,
                gpu_metrics,
            )
            candidates = None
    finally:
        if pending_future is not None:
            cancelled = pending_future.cancel()
            if cancelled and pending_selection is not None:
                try:
                    pending_selection.close()
                except BaseException:
                    pass
        active_error = sys.exc_info()[0] is not None
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            if not active_error:
                raise
        if gpu_metrics is not None:
            gpu_metrics.pipeline_wall_seconds += (
                time.perf_counter() - pipeline_started
            )



def hmmsearch(protein_dict, hmms, threads, options, db_name=None,
              macsyfinder_dir=None, hmm_name_to_filename=None,
              all_sequences=None, gpu_sequence_batch=None,
              gpu_postfilter=None, gpu_profile_session=None,
              gpu_metrics=None, gpu_profile_overlap=True):
    hmmsearch_kwargs = define_kwargs(options)

    if gpu_sequence_batch is not None and macsyfinder_dir is not None:
        raise GPUConfigurationError(
            "MacSyFinder output is unavailable for an explicitly GPU-mapped database"
        )
    if gpu_postfilter is None:
        gpu_postfilter = (
            gpu_postfilter_available() if gpu_sequence_batch is not None else False
        )
    elif type(gpu_postfilter) is not bool:
        raise TypeError("gpu_postfilter must be bool or None")
    if gpu_postfilter and gpu_sequence_batch is None:
        raise GPUConfigurationError(
            "exact GPU post-filter mode requires a GPU sequence batch"
        )
    if gpu_profile_session is not None:
        if gpu_sequence_batch is None or not gpu_postfilter:
            raise GPUConfigurationError(
                "GPU profile sessions require exact post-filter mode"
            )
        if gpu_profile_session.closed:
            raise GPUConfigurationError("GPU profile session is closed")
        if len(gpu_profile_session) != len(hmms):
            raise GPUConfigurationError(
                "GPU profile session does not cover the supplied profiles"
            )
    if gpu_metrics is not None and not isinstance(gpu_metrics, GPUOverlapMetrics):
        raise TypeError("gpu_metrics must be GPUOverlapMetrics or None")
    if type(gpu_profile_overlap) is not bool:
        raise TypeError("gpu_profile_overlap must be bool")
    profile_overlap_enabled = False
    continuation_threads = threads
    if gpu_profile_session is not None:
        (
            profile_overlap_enabled,
            producer_slots,
            continuation_threads,
        ) = gpu_profile_worker_allocation(threads, gpu_profile_overlap)
        statistics = gpu_profile_session.statistics
        selection_workers = statistics.get(
            'selection_worker_count', statistics['worker_count']
        )
        if selection_workers:
            raise GPUConfigurationError(
                "GPU profile sessions require zero persistent selection "
                "workers"
            )
        if gpu_metrics is not None:
            gpu_metrics.requested_thread_count = threads
            gpu_metrics.producer_slot_count = producer_slots
            gpu_metrics.continuation_worker_count = continuation_threads
            gpu_metrics.profile_overlap_enabled = profile_overlap_enabled

    # Always write to temp files — bulk mode is faster and avoids
    # keeping huge result lists in memory.  The per-genome loop is
    # only needed when MacSyFinder output requires per-genome provenance.
    tmp_dir = os.path.join(options['outdir'], 'tmp_results')
    os.makedirs(tmp_dir, exist_ok=True)

    def cutoff_available(query, cutoff):
        if gpu_sequence_batch is not None:
            # PressedProfilePair.cutoffs is the immutable snapshot captured
            # while the manifest-authenticated pressed streams were pinned.
            return getattr(query.cutoffs, cutoff) is not None
        return getattr(query.cutoffs, f"{cutoff}_available")()

    def get_best_cutoff(query):
        if options['cascade']:
            cutoff_order = [
                hmmsearch_kwargs.get('preferred_cutoff', 'trusted'),
                'trusted', 'gathering', 'noise'
            ]
            for cutoff in cutoff_order:
                if cutoff_available(query, cutoff):
                    return cutoff
        elif 'bit_cutoffs' in hmmsearch_kwargs:
            if cutoff_available(query, hmmsearch_kwargs['bit_cutoffs']):
                return hmmsearch_kwargs['bit_cutoffs']
        return None

    # Pre-compute HMM groups ONCE — grouping depends only on HMM cutoff
    # availability, not on per-genome data.  Previously this was inside the
    # per-genome loop, wasting len(hmms) * len(protein_dict) iterations.
    hmm_groups = {}
    for hmm_index, hmm in enumerate(hmms):
        best_cutoff = get_best_cutoff(hmm)
        hmm_groups.setdefault(best_cutoff, []).append((hmm_index, hmm))

    # Build per-group kwargs once (avoids re-copying per genome)
    group_kwargs_list = []
    for cutoff, indexed_hmm_group in hmm_groups.items():
        kwargs = hmmsearch_kwargs.copy()
        if cutoff:
            kwargs['bit_cutoffs'] = cutoff
        else:
            # No bitscore threshold available for these HMMs.
            # In cascade mode, fall back to E-value 1e-15 (the intended
            # cascade behavior) instead of pyhmmer's permissive default (10.0).
            if 'bit_cutoffs' in kwargs:
                del kwargs['bit_cutoffs']
            if options['cascade']:
                kwargs.setdefault('E', 1e-15)

        # Remove internal-only keys before passing to pyhmmer
        kwargs.pop('preferred_cutoff', None)
        group_kwargs_list.append((
            [hmm for _, hmm in indexed_hmm_group],
            tuple(index for index, _ in indexed_hmm_group),
            kwargs,
        ))

    # For large datasets without MacSyFinder output, flatten all sequences
    # and search once against the full pool.  This turns N_genomes * N_chunks
    # pyhmmer.hmmsearch() calls into just N_chunks calls — e.g. 54 instead of
    # 192,456 for KOFAM on DPANN (3,564 genomes × 54 chunks).
    bulk_mode = not macsyfinder_dir

    HEADER = ("sequence_id\thmm_name\tbitscore\tevalue\tc_evalue\ti_evalue\t"
              "env_from\tenv_to\tdom_bitscore\tali_from\tali_to\thmm_from\thmm_to\n")

    if bulk_mode:
        # Use pre-flattened list if provided, otherwise flatten now
        if all_sequences is None:
            all_sequences = []
            for sequences in protein_dict.values():
                all_sequences.extend(sequences)
        print(f"Bulk search: {len(all_sequences)} sequences × {len(hmms)} HMMs "
              f"({sum(len(g) for g, _, _ in group_kwargs_list)} grouped)")

        hmm_chunk_size = HMM_CHUNK_SIZE
        if gpu_sequence_batch is not None:
            hmm_chunk_size = gpu_hmm_chunk_size(len(all_sequences))

        # Single output file — keep handle open across all chunks
        out_file = os.path.join(tmp_dir, "bulk_results.tsv")
        total_chunks = sum(
            (len(g) + hmm_chunk_size - 1) // hmm_chunk_size
            for g, _, _ in group_kwargs_list
        )
        chunk_idx = 0
        with open(out_file, 'w') as fh:
            fh.write(HEADER)
            chunks = []
            for hmm_group, group_indices, kwargs in group_kwargs_list:
                for chunk_start in range(0, len(hmm_group), hmm_chunk_size):
                    hmm_chunk = hmm_group[
                        chunk_start:chunk_start + hmm_chunk_size
                    ]
                    chunk_indices = group_indices[
                        chunk_start:chunk_start + hmm_chunk_size
                    ]
                    chunk_idx += 1
                    chunks.append((
                        chunk_idx,
                        hmm_chunk,
                        chunk_indices,
                        kwargs,
                    ))

            if gpu_profile_session is not None:
                profile_runner = (
                    _run_gpu_profile_pipeline
                    if profile_overlap_enabled
                    else _run_gpu_profile_serial
                )
                profile_runner(
                    chunks,
                    gpu_profile_session,
                    gpu_sequence_batch,
                    continuation_threads,
                    fh,
                    gpu_metrics,
                )
            else:
                for chunk_index, hmm_chunk, _, kwargs in chunks:
                    print(f"  Chunk {chunk_index}/{total_chunks} "
                          f"({len(hmm_chunk)} HMMs)...", end="", flush=True)
                    if gpu_sequence_batch is None:
                        hit_iterator = pyhmmer.hmmsearch(
                            hmm_chunk, all_sequences, cpus=threads, **kwargs
                        )
                    else:
                        from plan7_gpu.astra_search import hmmsearch as gpu_hmmsearch
                        if gpu_postfilter:
                            hit_iterator = gpu_hmmsearch(
                                hmm_chunk, gpu_sequence_batch, cpus=threads,
                                postfilter=True, **kwargs
                            )
                        else:
                            hit_iterator = gpu_hmmsearch(
                                hmm_chunk, gpu_sequence_batch, cpus=threads, **kwargs
                            )
                    try:
                        for hits in hit_iterator:
                            process_hits_to_file(hits, fh)
                    except BaseException:
                        if gpu_sequence_batch is not None:
                            close = getattr(hit_iterator, "close", None)
                            if close is not None:
                                try:
                                    close()
                                except BaseException:
                                    pass
                        raise
                    finally:
                        if gpu_sequence_batch is not None:
                            hit_iterator = None
                            hits = None
                    if gpu_sequence_batch is None:
                        gc.collect()
                    print(" done")

        gc.collect()

    else:
        # Per-genome loop — only used when MacSyFinder output is needed
        # (requires per-genome provenance tracking).
        for fasta_file, sequences in tqdm(protein_dict.items()):
            safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.basename(fasta_file))
            tmp_file = os.path.join(tmp_dir, f"{safe_filename}_results.tsv")
            with open(tmp_file, 'w') as fh:
                fh.write(HEADER)
                for hmm_group, _, kwargs in group_kwargs_list:
                    for chunk_start in range(0, len(hmm_group), HMM_CHUNK_SIZE):
                        hmm_chunk = hmm_group[chunk_start:chunk_start + HMM_CHUNK_SIZE]
                        for hits in pyhmmer.hmmsearch(hmm_chunk, sequences,
                                                      cpus=threads, **kwargs):
                            process_hits_to_file(hits, fh)
                            write_macsyfinder_hit(hits, macsyfinder_dir, hmm_name_to_filename)
            gc.collect()

    return tmp_dir


def process_hits_to_file(hits, fh):
    """Write hits to an already-open file handle *fh*."""
    cog = hits.query.name
    for hit in hits:
        if hit.included:
            hit_name = hit.name
            full_bitscore = hit.score
            full_evalue = hit.evalue
            for domain in hit.domains.reported:
                aln = domain.alignment
                ali_from = aln.target_from if aln else domain.env_from
                ali_to = aln.target_to if aln else domain.env_to
                hmm_from = aln.hmm_from if aln else ""
                hmm_to = aln.hmm_to if aln else ""
                fh.write(f"{hit_name}\t{cog}\t{full_bitscore:.2f}\t{full_evalue:.2e}\t{domain.c_evalue:.2e}\t"
                         f"{domain.i_evalue:.2e}\t{domain.env_from}\t{domain.env_to}\t{domain.score:.2f}\t"
                         f"{ali_from}\t{ali_to}\t{hmm_from}\t{hmm_to}\n")


def extract_sequences_from_tmp(tmp_dir, protein_dict, outdir):
    """Read hit IDs from temp result files and extract sequences from memory."""
    # Collect hit IDs per HMM from all result files (no pandas needed)
    hit_ids_by_hmm = {}
    for filename in os.listdir(tmp_dir):
        if filename.endswith('_results.tsv'):
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path) as f:
                f.readline()  # skip header
                for line in f:
                    parts = line.split('\t', 3)  # only need first two columns
                    seq_id, hmm_name = parts[0], parts[1]
                    hit_ids_by_hmm.setdefault(hmm_name, set()).add(seq_id)

    if hit_ids_by_hmm:
        extract_sequences(hit_ids_by_hmm, protein_dict, outdir)
        print(f"Extracted sequences for {len(hit_ids_by_hmm)} HMMs → {outdir}/fastas/")

def cleanup_temp_files(temp_dir):
    shutil.rmtree(temp_dir)
    print(f"Temporary files removed from {temp_dir}")



def parse_single_hmm(hmm_path):
    """Single-file parser for fallback when no pressed DB exists."""
    with pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
        return hmm_file.read()

def _find_pressed_db(db_dir):
    """Check if a pressed HMM database exists in *db_dir*.

    Returns the base path (without extension) if all four pressed files
    (``.h3m``, ``.h3i``, ``.h3f``, ``.h3p``) exist, otherwise ``None``.
    """
    db_name = os.path.basename(db_dir.rstrip('/'))
    pressed_base = os.path.join(db_dir, db_name)
    if all(os.path.exists(f"{pressed_base}.{ext}") for ext in ('h3m', 'h3i', 'h3f', 'h3p')):
        return pressed_base
    return None


def parse_hmms(hmm_in):
    #Checks first whether HMMs are provided as a single file or as a directory.

    hmms = []  # Initialize an empty list to store parsed HMMs
    # Mapping from HMM internal NAME → filename stem, for MacSyFinder compat.
    # Many HMM databases (DefenseFinder, TXSScan) have HMM files whose internal
    # NAME field differs from the filename. MacSyFinder expects filenames, so
    # we track the mapping here and pass it through to write_macsyfinder_hit().
    hmm_name_to_filename = {}
    print("Parsing HMMs...")
    t0 = time.perf_counter()

    # Check if hmm_in is a directory or a single file
    if os.path.isdir(hmm_in):
        if not os.listdir(hmm_in):
            print("hmm_in directory is empty.")
            logging.info('hmm_in directory is empty.')
            sys.exit(1)

        # Prefer pressed database if available (~50x faster than individual files)
        pressed_base = _find_pressed_db(hmm_in)
        if pressed_base:
            print(f"  Loading from pressed database: {pressed_base}")
            with pyhmmer.plan7.HMMFile(pressed_base) as hmm_file:
                hmms = list(hmm_file)
            elapsed = time.perf_counter() - t0
            print(f"HMMs parsed: {len(hmms)} models in {elapsed:.1f}s (pressed DB)")
            return list(hmms), hmm_name_to_filename

        hmm_files = list(filter(lambda x: x.endswith(('.hmm', '.HMM')), os.listdir(hmm_in)))
        if len(hmm_files) == 0:
            print("No .hmm files found in directory.")
            logging.info('No .hmm files found in directory.')
            sys.exit(1)
        elif len(hmm_files) == 1:
            #Only one HMM file in input directory
            #Get full path to file
            hmm_path = os.path.join(hmm_in, hmm_files[0])
            with pyhmmer.plan7.HMMFile(hmm_path) as hmm_file:
                #Works in case of single-model or multi-model HMM file
                hmms = list(hmm_file)

        else:
            hmm_paths = [os.path.join(hmm_in, hmm_file) for hmm_file in hmm_files]

            #I have tried!! Every possible method! To parallelize this!
            #It does not work. SINGLE THREADED IT IS!
            hmms = list(tqdm(map(parse_single_hmm, hmm_paths)))

            # Build internal NAME → filename stem mapping for MacSyFinder compat
            for hmm_file, hmm_obj in zip(hmm_files, hmms):
                filename_stem = hmm_file.rsplit('.hmm', 1)[0].rsplit('.HMM', 1)[0]
                internal_name = hmm_obj.name
                if internal_name != filename_stem:
                    hmm_name_to_filename[internal_name] = filename_stem

            if hmm_name_to_filename:
                print(f"  {len(hmm_name_to_filename)} HMMs have internal NAME != filename (will use filename for MacSyFinder)")

    elif os.path.isfile(hmm_in):
        if os.path.getsize(hmm_in) == 0:
            print("hmm_in file is empty.")
            logging.info('hmm_in file is empty.')
            sys.exit(1)
        # Parse the single HMM file; handles multi-model files
        with pyhmmer.plan7.HMMFile(hmm_in) as hmm_file:
            hmms = list(hmm_file)
    else:
        print("Invalid HMM input.")
        logging.info("Invalid HMM input.")
        print("If you used pre-installed HMMs, check hmm_databases.json")
        logging.info("If you used pre-installed HMMs, check hmm_databases.json")
        print("Which is located in the databases directory.")
        logging.info("Which is located in the databases directory.")

        print("Thing that threw the error: {}".format(hmm_in))
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    print(f"HMMs parsed: {len(hmms)} models in {elapsed:.1f}s")

    return list(hmms), hmm_name_to_filename

def process_fasta(fasta_file):
    # Function to handle each file for parallelism
    with pyhmmer.easel.SequenceFile(fasta_file, digital=True, alphabet=pyhmmer.easel.Alphabet.amino()) as seq_file:
        sequences = seq_file.read_block()
    return fasta_file, sequences

def parse_protein_input(prot_in, threads):
    print("Parsing protein input sequences...")
    protein_dict = {}  # Initialize an empty dictionary to store parsed proteins
    
    # Check if prot_in is a directory or a single file
    if os.path.isdir(prot_in):
        if not os.listdir(prot_in):
            print("prot_in directory is empty.")
            logging.info("prot_in directory is empty.")
            sys.exit(1)

        # Initialize an empty dictionary to hold protein sequences
        protein_dict = {}

        fasta_paths = [os.path.join(prot_in, x) for x in os.listdir(prot_in)]

        # pyhmmer sequence objects can't pickle (no ProcessPoolExecutor), but
        # the GIL is released during pyhmmer's C-level I/O, so threads work.
        # For small file counts the overhead isn't worth it; threshold at 8.
        if len(fasta_paths) >= 8:
            n_workers = min(threads, len(fasta_paths))
            print(f"  Loading {len(fasta_paths)} files with {n_workers} threads...")
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(tqdm(executor.map(process_fasta, fasta_paths),
                                    total=len(fasta_paths)))
        else:
            results = list(map(process_fasta, tqdm(fasta_paths)))

        # Populate the protein_dict
        for fasta_path, sequences in results:
            protein_dict[fasta_path] = sequences
    elif os.path.isfile(prot_in):
        if os.path.getsize(prot_in) == 0:
            print("prot_in file is empty.")
            logging.info("prot_in file is empty.")
            sys.exit(1)
        # Parse the single protein FASTA file
        with pyhmmer.easel.SequenceFile(prot_in, digital=True) as seq_file:
            sequences = seq_file.read_block()
        protein_dict[prot_in] = sequences
    else:
        print("Invalid input for prot_in.")
        logging.info("Invalid input for prot_in.")
        sys.exit(1)
    
    return protein_dict

def define_kwargs(options):
    kwargs = {}
    
    if options['cascade']:
        # Cascade mode: per-HMM adaptive thresholds.  HMM grouping in
        # hmmsearch() assigns the best available bitscore cutoff to each
        # profile (preferred → trusted → gathering → noise), falling back
        # to E-value 1e-15 for profiles with no thresholds at all.
        # Do NOT set bit_cutoffs here — it is set per-group in hmmsearch().
        if options['cut_tc']:
            kwargs['preferred_cutoff'] = 'trusted'
        elif options['cut_ga']:
            kwargs['preferred_cutoff'] = 'gathering'
        elif options['cut_nc']:
            kwargs['preferred_cutoff'] = 'noise'
        else:
            kwargs['preferred_cutoff'] = 'trusted'
    elif options['cut_ga']:
        kwargs['bit_cutoffs'] = 'gathering'
    elif options['cut_nc']:
        kwargs['bit_cutoffs'] = 'noise'
    elif options['cut_tc']:
        kwargs['bit_cutoffs'] = 'trusted'


    #Numerical threshold parameters
    if options['bitscore'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['bitscore'], float):
            try:
                kwargs['T'] = float(options['bitscore'])
            except ValueError:
                print("Error: bitscore threshold must be a float or castable as a float.")
                logging.info("Error: bitscore threshold must be a float or castable as a float.")

    if options['domE'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['domE'], float):
            try:
                kwargs['domE'] = float(options['domE'])
            except ValueError:
                print("Error: domE must be a float or castable to float.")
                logging.info("Error: domE must be a float or castable to float.")

    if options['domT'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['domT'], float):
            try:
                kwargs['domT'] = float(options['domT'])
            except ValueError:
                print("Error: domT must be a float or castable to float.")
                logging.info("Error: domT must be a float or castable to float.")

    if options['incE'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incE'], float):
            try:
                kwargs['incE'] = float(options['incE'])
            except ValueError:
                print("Error: domT must be a float or castable to float.")
                logging.error("Error: domT must be a float or castable to float.")

    if options['incT'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incT'], float):
            try:
                kwargs['incT'] = float(options['incT'])
            except ValueError:
                print("Error: incT must be a float or castable to float.")
                logging.error("Error: incT must be a float or castable to float.")

    if options['incdomE'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incdomE'], float):
            try:
                kwargs['incdomE'] = float(options['incdomE'])
            except ValueError:
                print("Error: incdomE must be a float or castable to float.")
                logging.error("Error: incdomE must be a float or castable to float.")

    if options['incdomT'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['incdomT'], float):
            try:
                kwargs['incdomT'] = float(options['incdomT'])
            except ValueError:
                print("Error: incdomT must be a float or castable to float.")
                logging.error("Error: incdomT must be a float or castable to float.")

    if options['evalue'] is not None:
        #Make sure it's the right format, or castable as such!
        if not isinstance(options['evalue'], float):
            try:
                kwargs['E'] = float(options['evalue'])
            except ValueError:
                print("Error: evalue must be a float or castable to float.")
                logging.error("Error: evalue must be a float or castable to float.")

    return kwargs

def combine_results(tmp_dir, output_file):
    """Combine temp result files into a single output file.

    If the temp directory contains a single file (bulk mode), it is simply
    moved to *output_file* — no copying needed.  For multiple files
    (MacSyFinder per-genome mode) the files are streamed together.
    """
    tmp_files = sorted(
        f for f in os.listdir(tmp_dir) if f.endswith('_results.tsv')
    )
    if not tmp_files:
        print("No results found to combine.")
        return

    if len(tmp_files) == 1:
        # Bulk mode — single file, just move it
        src = os.path.join(tmp_dir, tmp_files[0])
        shutil.move(src, output_file)
        print(f"Results → {output_file}")
        return

    # Multiple files (MacSyFinder per-genome mode) — stream-combine
    total_rows = 0
    header_written = False

    with open(output_file, 'w') as out:
        for filename in tqdm(tmp_files, desc="Combining"):
            file_path = os.path.join(tmp_dir, filename)
            with open(file_path) as inp:
                header = inp.readline()
                if not header_written:
                    out.write(header)
                    header_written = True
                for line in inp:
                    out.write(line)
                    total_rows += 1

    print(f"Combined {total_rows:,} hits from {len(tmp_files)} files → {output_file}")

def _write_macsyfinder_conf(macsyfinder_dir, prot_in):
    """Write a minimal macsyfinder.conf for --previous-run compatibility.

    If *prot_in* is a directory, concatenates all .faa files into a single
    FASTA inside *macsyfinder_dir* so MacSyFinder can index it.
    """
    conf_path = os.path.join(macsyfinder_dir, "macsyfinder.conf")
    abs_prot = os.path.abspath(prot_in)

    if os.path.isdir(abs_prot):
        faa_files = sorted(
            f for f in os.listdir(abs_prot)
            if f.endswith(('.faa', '.fa', '.fasta'))
        )
        if len(faa_files) == 1:
            sequence_db = os.path.join(abs_prot, faa_files[0])
        else:
            # Concatenate all FASTA files into one
            concat_path = os.path.join(macsyfinder_dir, "all_proteins.faa")
            with open(concat_path, "w") as out:
                for faa in faa_files:
                    with open(os.path.join(abs_prot, faa)) as inp:
                        out.write(inp.read())
            sequence_db = concat_path
    else:
        sequence_db = abs_prot

    with open(conf_path, "w") as fh:
        fh.write("[base]\n")
        fh.write(f"sequence_db = {sequence_db}\n")
        fh.write("db_type = ordered_replicon\n")
        fh.write("hmmer = hmmsearch\n\n")
        fh.write("[hmmer]\n")
        fh.write("e_value_search = 0.1\n")
    print(f"MacSyFinder config written to {conf_path}")


def main(args):
    t1 = time.time()
    hmm_in = args.hmm_in
    prot_in = args.prot_in
    outdir = args.outdir
    gpu_manifests = parse_gpu_manifest_mappings(
        getattr(args, 'gpu_manifest', ())
    )
    gpu_parsed_json = None
    gpu_installed_hmm_names = None
    log_file_path = os.path.join(outdir, 'astra_search_log.txt')

    # --- Ribosomal-protein marker mode ---------------------------------
    # ``--16rp``/``--15rp`` are stored under non-identifier dests, so read
    # them via getattr.  RP mode points hmm_in at the RP16 marker set and
    # triggers genome-aware extraction after the search (see end of main).
    rp16_mode = getattr(args, '16rp', False)
    rp15_mode = getattr(args, '15rp', False)
    if rp15_mode:
        msg = ("--15rp (archaea-only 15-marker set) is not yet implemented. "
               "Use --16rp, which covers Bacteria and Archaea.")
        print(msg)
        logging.error(msg)
        sys.exit(1)
    synteny_threshold = getattr(args, 'synteny', None)
    if rp16_mode:
        if hmm_in is not None or args.installed_hmms is not None:
            print("--16rp uses its own marker set; ignoring "
                  "--hmm_in/--installed_hmms.")
        try:
            hmm_in = rp16_module.rp16_hmm_dir()
        except FileNotFoundError as exc:
            print(exc)
            logging.error(str(exc))
            sys.exit(1)
        args.installed_hmms = None
        # The markers all carry GA cutoffs; default to them when the
        # user did not request any explicit threshold.
        if not (args.cut_ga or args.cut_nc or args.cut_tc or args.cascade
                or args.evalue or args.bitscore):
            print("--16rp: no threshold specified, defaulting to --cut_ga.")
            args.cut_ga = True
    elif synteny_threshold is not None:
        print("--synteny is only used with --16rp/--15rp; ignoring.")
        synteny_threshold = None

    if gpu_manifests:
        if args.installed_hmms is None:
            gpu_parsed_json = {'db_urls': []}
            gpu_installed_hmm_names = []
        else:
            gpu_parsed_json = initialize.load_config()
            gpu_installed_hmm_names = _resolve_installed_hmm_names(
                args.installed_hmms, gpu_parsed_json
            )
        validate_gpu_configuration(
            gpu_manifests,
            gpu_installed_hmm_names,
            gpu_parsed_json,
            args.threads,
            getattr(args, 'write_macsyfinder', False),
        )

    hmmsearch_options = {
        "cascade": args.cascade,
        "cut_ga": args.cut_ga,
        "cut_nc": args.cut_nc,
        "cut_tc": args.cut_tc,
        "evalue": args.evalue,
        "bitscore": args.bitscore,
        "domE": args.domE,
        "domT": args.domT,
        "incE": args.incE,
        "incT": args.incT,
        "incdomE": args.incdomE,
        "incdomT": args.incdomT,
        "outdir": outdir,
        "meta": args.meta
    }

    if hmm_in is None and args.installed_hmms is None:
        error_out = "Either a user-provided or pre-installed HMM database must be specified."
        print(error_out)
        logging.error(error_out)
        sys.exit(1)

    protein_dict = parse_protein_input(prot_in, args.threads)

    # Pre-flatten sequences once for all database searches (avoids
    # re-allocating for each DB in multi-DB runs).
    all_sequences = []
    for seqs in protein_dict.values():
        all_sequences.extend(seqs)
    print(f"Total sequences loaded: {len(all_sequences)}")

    # Free protein_dict if we don't need per-file provenance.
    # write_seqs needs it for sequence extraction; MacSyFinder needs it
    # for per-genome loop.  Otherwise it's dead weight.
    needs_protein_dict = (args.write_seqs or rp16_mode
                          or getattr(args, 'write_macsyfinder', False))
    if not needs_protein_dict:
        del protein_dict
        gc.collect()
        protein_dict = None  # keep the name bound for the code paths that check it

    gpu_databases = {}
    gpu_sequence_batch = None
    gpu_postfilter = False
    gpu_metrics_by_db = None
    if gpu_manifests:
        # This check must happen before constructing a CUDA SequenceBatch or
        # creating any result path. With one HMM, a larger target set already
        # exceeds the bounded profile-by-target candidate matrix.
        gpu_hmm_chunk_size(len(all_sequences))
        if os.environ.get(GPU_TIMING_ENV) == '1':
            gpu_metrics_by_db = {
                db_name: GPUOverlapMetrics()
                for db_name in gpu_installed_hmm_names
                if db_name in gpu_manifests
            }
        gpu_databases, gpu_sequence_batch, gpu_postfilter = preflight_gpu_databases(
            gpu_manifests,
            gpu_installed_hmm_names,
            gpu_parsed_json,
            all_sequences,
            args.threads,
            gpu_metrics_by_db,
        )

    try:
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            if args.write_seqs:
                os.makedirs(os.path.join(outdir, 'fastas'))

        # Create temporary directory for results only after every explicit GPU
        # database and the shared target batch have completed preflight.
        tmp_results_dir = os.path.join(outdir, 'tmp_results')
        os.makedirs(tmp_results_dir, exist_ok=True)

        logging.basicConfig(filename=log_file_path, level=logging.INFO,
                            format='%(asctime)s %(levelname)s: %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    except BaseException:
        for _, _, session in gpu_databases.values():
            if session is not None:
                try:
                    session.close()
                except BaseException:
                    pass
        if gpu_sequence_batch is not None:
            gpu_sequence_batch.close()
        raise

    # MacSyFinder-compatible output directory (per-HMM hmmsearch text files)
    macsyfinder_dir = None
    if getattr(args, 'write_macsyfinder', False):
        macsyfinder_dir = os.path.join(outdir, 'macsyfinder_compat')
        hmmer_results_dir = os.path.join(macsyfinder_dir, 'hmmer_results')
        # Clean previous run's files to avoid stale appends
        if os.path.isdir(hmmer_results_dir):
            shutil.rmtree(hmmer_results_dir)
        os.makedirs(hmmer_results_dir, exist_ok=True)
        print(f"MacSyFinder-compatible output enabled → {macsyfinder_dir}/")
        logging.info(f"MacSyFinder-compatible output enabled → {macsyfinder_dir}/")

    try:
        if hmm_in is not None:
            print("Searching with user-provided HMM(s)...")
            logging.info("Searching with user-provided HMM(s)...")
            user_hmms, user_name_map = parse_hmms(hmm_in)
            results = hmmsearch(protein_dict, user_hmms, args.threads, hmmsearch_options,
                                macsyfinder_dir=macsyfinder_dir, hmm_name_to_filename=user_name_map,
                                all_sequences=all_sequences)
            if args.write_seqs:
                extract_sequences_from_tmp(results, protein_dict, outdir)
            hits_tsv = os.path.join(outdir,
                                    'rp16_raw_hits.tsv' if rp16_mode else 'user_hmms_hits_df.tsv')
            combine_results(results, hits_tsv)
            if rp16_mode:
                rp16_module.process(hits_tsv, protein_dict, outdir, synteny_threshold)
            del user_hmms

        if args.installed_hmms is not None:
            installed_hmm_names = args.installed_hmms.split(',') if ',' in args.installed_hmms else [args.installed_hmms]
            print(f"Searching with pre-installed HMMs: {', '.join(installed_hmm_names)}")
            logging.info(f"Searching with pre-installed HMMs: {', '.join(installed_hmm_names)}")

            parsed_json = (
                gpu_parsed_json
                if gpu_parsed_json is not None
                else initialize.load_config()
            )

            if 'all_prot' in installed_hmm_names:
                installed_hmm_names = [db['name'] for db in parsed_json['db_urls'] if db['molecule_type'] == 'protein' and db['installed']]

            for hmm_db in installed_hmm_names:
                installed_hmm_in = next((item for item in parsed_json['db_urls'] if item["name"] == hmm_db), None)
                if installed_hmm_in is not None:
                    installation_dir = installed_hmm_in['installation_dir']
                    manifest_path = gpu_manifests.get(hmm_db)
                    if manifest_path is None:
                        db_hmms, db_name_map = parse_hmms(installation_dir)
                        tmp_dir = hmmsearch(
                            protein_dict, db_hmms, args.threads, hmmsearch_options,
                            hmm_db, macsyfinder_dir=macsyfinder_dir,
                            hmm_name_to_filename=db_name_map,
                            all_sequences=all_sequences,
                        )
                    else:
                        (
                            pressed_base,
                            db_hmms,
                            gpu_profile_session,
                        ) = gpu_databases[hmm_db]
                        try:
                            gpu_profile_overlap_requested = (
                                os.environ.get(GPU_SERIAL_ENV) != '1'
                            )
                            if gpu_profile_session is not None:
                                gpu_profile_overlap, _, _ = (
                                    gpu_profile_worker_allocation(
                                        args.threads,
                                        gpu_profile_overlap_requested,
                                    )
                                )
                                scheduling = (
                                    "overlap" if gpu_profile_overlap else "serial"
                                )
                                gpu_mode = (
                                    f"exact GPU-through-Forward {scheduling}"
                                )
                            elif gpu_postfilter:
                                gpu_profile_overlap = False
                                gpu_mode = "exact post-filter + Forward"
                            else:
                                gpu_profile_overlap = False
                                gpu_mode = "legacy SSV"
                            print(
                                f"  GPU search for {hmm_db} ({gpu_mode}): "
                                f"{pressed_base}"
                            )
                            logging.info(
                                f"GPU search for {hmm_db} ({gpu_mode}): "
                                f"{pressed_base}"
                            )
                            db_metrics = (
                                gpu_metrics_by_db[hmm_db]
                                if gpu_metrics_by_db is not None
                                else None
                            )
                            tmp_dir = hmmsearch(
                                protein_dict, db_hmms, args.threads,
                                hmmsearch_options, hmm_db,
                                all_sequences=all_sequences,
                                gpu_sequence_batch=gpu_sequence_batch,
                                gpu_postfilter=gpu_postfilter,
                                gpu_profile_session=gpu_profile_session,
                                gpu_metrics=(
                                    db_metrics
                                    if gpu_profile_session is not None
                                    else None
                                ),
                                gpu_profile_overlap=gpu_profile_overlap,
                            )
                            if (
                                db_metrics is not None
                                and gpu_profile_session is not None
                            ):
                                timing = db_metrics.snapshot()
                                timing_line = (
                                    f"  GPU-through-Forward {scheduling} timing: "
                                    f"phase-workers=build:"
                                    f"{timing['profile_build_worker_count']}, "
                                    f"search=selection:"
                                    f"{timing['profile_selection_worker_count']}"
                                    f"/producer:{timing['producer_slot_count']}"
                                    f"/continuation:"
                                    f"{timing['continuation_worker_count']}"
                                    f"/{timing['requested_thread_count']}, "
                                    f"preflight={timing['preflight_seconds']:.3f}s, "
                                    f"selection={timing['selection_seconds']:.3f}s, "
                                    f"generation={timing['generation_seconds']:.3f}s, "
                                    f"wait={timing['generation_wait_seconds']:.3f}s, "
                                    f"CPU/output={timing['continuation_seconds']:.3f}s, "
                                    f"measured-overlap={timing['overlap_seconds']:.3f}s"
                                )
                                print(timing_line)
                                logging.info(timing_line.strip())
                        finally:
                            active_gpu_error = sys.exc_info()[0] is not None
                            session_close_error = None
                            if gpu_profile_session is not None:
                                try:
                                    gpu_profile_session.close()
                                except BaseException as error:
                                    session_close_error = error
                            gpu_databases.pop(hmm_db, None)
                            if (
                                session_close_error is not None
                                and not active_gpu_error
                            ):
                                raise session_close_error
                    if args.write_seqs:
                        extract_sequences_from_tmp(tmp_dir, protein_dict, outdir)
                    combine_results(tmp_dir, os.path.join(outdir, f'{hmm_db}_hits_df.tsv'))
                    del db_hmms
                    gc.collect()
                else:
                    print(f"No installation_dir specified for db {hmm_db}")
                    logging.info(f"No installation_dir specified for db {hmm_db}")
    finally:
        active_error = sys.exc_info()[0] is not None
        cleanup_error = None
        for _, _, session in gpu_databases.values():
            if session is not None:
                try:
                    session.close()
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
        if gpu_sequence_batch is not None:
            try:
                gpu_sequence_batch.close()
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None and not active_error:
            raise cleanup_error

    # Write MacSyFinder config file and finalize hmmsearch output if enabled
    if macsyfinder_dir:
        finalize_macsyfinder_files(macsyfinder_dir)
        _write_macsyfinder_conf(macsyfinder_dir, prot_in)

    # Clean up temporary directory if it exists
    if os.path.exists(os.path.join(outdir, 'tmp_results')):
        cleanup_temp_files(os.path.join(outdir, 'tmp_results'))

    time_printout = f"Process took {time.time()-t1} seconds."
    print(time_printout)
    logging.info(time_printout)

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description="ASTRA search tool")
    args = parser.parse_args()
    main(args)
