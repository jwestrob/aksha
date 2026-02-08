"""Memory/time benchmark for Aksha + PyHMMER.

Runs hmmsearch on a chunked stream of HMMs against a given FASTA,
logging wall time and RSS to CSV for quick tuning.

Usage:
    python scripts/bench_mem.py \
        --hmms ~/.local/share/aksha/PFAM/Pfam-A.hmm \
        --sequences dummy_dataset/Burkholderiales_bacterium_RIFCSPHIGHO2_01_FULL_64_960.contigs.faa \
        --chunk-size 1000 --threads 8 --max-chunks 3 --out bench.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import psutil
import pyhmmer

from aksha.parsers import parse_sequences
from aksha.thresholds import build_search_kwargs
from aksha.types import MoleculeType, ThresholdOptions


def iter_hmm_chunks(path: Path, chunk_size: int) -> Iterator[List[pyhmmer.plan7.HMM]]:
    """Yield HMMs from a monolithic file in chunks."""
    with pyhmmer.plan7.HMMFile(path) as f:
        while True:
            chunk: List[pyhmmer.plan7.HMM] = []
            try:
                for _ in range(chunk_size):
                    chunk.append(f.read())
            except EOFError:
                pass
            if not chunk:
                break
            yield chunk


def rss_mb() -> float:
    """Return current RSS in MB."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def run_once(
    hmm_chunk: List[pyhmmer.plan7.HMM],
    sequences: pyhmmer.easel.DigitalSequenceBlock,
    kwargs: dict,
    threads: int,
) -> Tuple[float, float]:
    """Run hmmsearch for one chunk; return (wall_time, rss_peak_delta)."""
    rss_before = rss_mb()
    t0 = time.perf_counter()
    for hits in pyhmmer.hmmsearch(hmm_chunk, sequences, cpus=threads, **kwargs):
        # exhaust iterator to force computation
        _ = len(hits)
    wall = time.perf_counter() - t0
    rss_after = rss_mb()
    return wall, rss_after - rss_before


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark Aksha/PyHMMER memory usage.")
    ap.add_argument("--hmms", required=True, type=Path, help="Path to monolithic HMM file")
    ap.add_argument("--sequences", required=True, type=Path, help="Protein FASTA/FAA")
    ap.add_argument("--chunk-size", type=int, default=1000, help="HMMs per batch")
    ap.add_argument("--threads", type=int, default=0, help="CPUs for PyHMMER (0=all)")
    ap.add_argument("--max-chunks", type=int, default=3, help="Limit processed chunks for speed")
    ap.add_argument("--out", type=Path, default=Path("bench.csv"), help="CSV output path")
    ap.add_argument("--cut-ga", action="store_true", help="Use GA thresholds")
    args = ap.parse_args()

    threads = args.threads if args.threads > 0 else (os.cpu_count() or 1)
    thresholds = ThresholdOptions(cut_ga=args.cut_ga)
    kwargs = build_search_kwargs(thresholds)

    seq_dict = parse_sequences(args.sequences, molecule_type=MoleculeType.PROTEIN, show_progress=False)
    sequences = next(iter(seq_dict.values()))

    out_exists = args.out.exists()
    with args.out.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if not out_exists:
            writer.writerow(
                [
                    "hmms",
                    "sequences",
                    "chunk_size",
                    "threads",
                    "chunk_index",
                    "wall_seconds",
                    "rss_delta_mb",
                ]
            )

        for idx, chunk in enumerate(iter_hmm_chunks(args.hmms, args.chunk_size)):
            if idx >= args.max_chunks:
                break
            wall, rss_delta = run_once(chunk, sequences, kwargs, threads)
            writer.writerow(
                [
                    args.hmms,
                    args.sequences,
                    args.chunk_size,
                    threads,
                    idx,
                    f"{wall:.3f}",
                    f"{rss_delta:.2f}",
                ]
            )
            print(
                f"chunk {idx}: wall {wall:.2f}s, rss +{rss_delta:.1f} MB "
                f"(chunk_size={args.chunk_size}, threads={threads})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
