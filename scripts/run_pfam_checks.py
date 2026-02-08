"""Run a representative API check using the installed PFAM database.

This is not a unit test; it exercises the public API against the real PFAM HMMs
and the bundled dummy_dataset. Runtime depends on CPU/thread count.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from aksha import search, ThresholdOptions, get_registry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run PFAM search API check")
    parser.add_argument(
        "--sequences",
        default="dummy_dataset",
        help="Protein sequences (file or directory). Default: dummy_dataset",
    )
    parser.add_argument(
        "--threads",
        "-t",
        type=int,
        default=4,
        help="CPU threads (default: 4)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional output TSV path; if omitted, a temp file is used",
    )
    args = parser.parse_args(argv)

    registry = get_registry()
    pfam_path = registry.get_path("PFAM")
    if pfam_path is None:
        print("PFAM is not installed. Install with: aksha database install PFAM", file=sys.stderr)
        return 1

    out_path: Path
    if args.output:
        out_path = args.output
    else:
        tmpdir = TemporaryDirectory()
        out_path = Path(tmpdir.name) / "pfam_search.tsv"

    print(f"Using PFAM at {pfam_path}")
    print(f"Sequences: {args.sequences}")
    print(f"Threads: {args.threads}")

    result = search(
        sequences=args.sequences,
        hmms="PFAM",
        thresholds=ThresholdOptions(cut_ga=True),
        threads=args.threads,
        output_dir=out_path.parent if out_path.suffix else out_path,
        show_progress=True,
    )

    result.to_csv(out_path if out_path.suffix else out_path / "pfam_search.tsv")

    df = result.to_dataframe()
    print(f"Total hits: {len(df)}")
    print(df.head())

    return 0


if __name__ == "__main__":
    sys.exit(main())
