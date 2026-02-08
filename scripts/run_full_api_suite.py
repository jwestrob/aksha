"""Exercise all public API entry points on bundled data.

Runs:
  - search (protein, PFAM with cut_ga)
  - scan  (protein, toy HMM)
  - phmmer (toy vs dummy_dataset)
  - jackhmmer (toy vs dummy_dataset)
  - nhmmer (nucleotide, barrnap)

This uses the installed PFAM and barrnap databases. Designed as an API
smoke/health check; adjust threads as needed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from aksha import (
    search,
    scan,
    phmmer,
    jackhmmer,
    nhmmer,
    ThresholdOptions,
    get_registry,
)


def run_search(outdir: Path, threads: int) -> None:
    target_dir = outdir / "search"
    target_dir.mkdir(parents=True, exist_ok=True)

    res = search(
        sequences="dummy_dataset",
        hmms="PFAM",
        thresholds=ThresholdOptions(cut_ga=True),
        threads=threads,
        output_dir=target_dir,
    )
    path = res.to_csv(target_dir / "search_results.tsv")
    print(f"[search] hits={len(res)} wrote={path}")


def run_scan(outdir: Path, threads: int) -> None:
    target_dir = outdir / "scan"
    target_dir.mkdir(parents=True, exist_ok=True)

    res = scan(
        sequences="dummy_dataset",
        hmms="tests/fixtures/small.hmm",
        thresholds=ThresholdOptions(),
        threads=threads,
        output_dir=target_dir,
    )
    path = res.to_csv(target_dir / "scan_results.tsv")
    print(f"[scan] hits={len(res)} wrote={path}")


def run_phmmer(outdir: Path, threads: int) -> None:
    target_dir = outdir / "phmmer"
    target_dir.mkdir(parents=True, exist_ok=True)

    res = phmmer(
        query="tests/fixtures/small.faa",
        target="dummy_dataset",
        thresholds=ThresholdOptions(),
        threads=threads,
        output_dir=target_dir,
    )
    path = res.to_csv(target_dir / "phmmer_results.tsv")
    print(f"[phmmer] hits={len(res)} wrote={path}")


def run_jackhmmer(outdir: Path, threads: int) -> None:
    target_dir = outdir / "jackhmmer"
    target_dir.mkdir(parents=True, exist_ok=True)

    res = jackhmmer(
        query="tests/fixtures/small.faa",
        target="dummy_dataset",
        thresholds=ThresholdOptions(),
        threads=threads,
        max_iterations=2,
        output_dir=target_dir,
    )
    path = res.to_csv(target_dir / "jackhmmer_results.tsv")
    print(f"[jackhmmer] hits={len(res)} wrote={path}")


def run_nhmmer(outdir: Path, threads: int) -> None:
    target_dir = outdir / "nhmmer"
    target_dir.mkdir(parents=True, exist_ok=True)

    res = nhmmer(
        sequences="dummy_genomes",
        hmms="barrnap",
        thresholds=ThresholdOptions(cut_ga=True),
        threads=threads,
        output_dir=target_dir,
    )
    path = res.to_csv(target_dir / "nhmmer_results.tsv")
    print(f"[nhmmer] hits={len(res)} wrote={path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Aksha API suite on bundled data")
    parser.add_argument("-t", "--threads", type=int, default=6, help="CPU threads")
    parser.add_argument("-o", "--output", type=Path, help="Output directory (default: tempdir)")
    args = parser.parse_args(argv)

    registry = get_registry()
    if registry.get_path("PFAM") is None:
        print("PFAM not installed; install with `aksha database install PFAM`", flush=True)
        return 1
    if registry.get_path("barrnap") is None:
        print("barrnap not installed; install with `aksha database install barrnap`", flush=True)
        return 1

    base = args.output or Path(TemporaryDirectory().name)
    base.mkdir(parents=True, exist_ok=True)

    run_search(base, args.threads)
    run_scan(base, args.threads)
    run_phmmer(base, args.threads)
    run_jackhmmer(base, args.threads)
    run_nhmmer(base, args.threads)

    print(f"All outputs in {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
