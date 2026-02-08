"""Command-line interface for Aksha."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from aksha.types import ThresholdOptions, MoleculeType


def main(argv: Optional[list[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001
        logging.error(str(exc))
        if getattr(args, "verbose", False):
            raise
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="aksha",
        description="Aksha: Scalable HMM-based sequence search",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_search_parser(subparsers)
    _add_scan_parser(subparsers)
    _add_phmmer_parser(subparsers)
    _add_jackhmmer_parser(subparsers)
    _add_nhmmer_parser(subparsers)
    _add_database_parser(subparsers)

    return parser


def _add_threshold_args(parser: argparse.ArgumentParser) -> None:
    """Add threshold arguments to parser."""
    group = parser.add_argument_group("thresholds")
    group.add_argument("--cut_ga", action="store_true", help="Use gathering thresholds")
    group.add_argument("--cut_tc", action="store_true", help="Use trusted cutoffs")
    group.add_argument("--cut_nc", action="store_true", help="Use noise cutoffs")
    group.add_argument("--cascade", action="store_true", help="Try available cutoffs in order")
    group.add_argument("-E", "--evalue", type=float, help="E-value threshold")
    group.add_argument("-T", "--bitscore", type=float, help="Bitscore threshold")
    group.add_argument("--domE", type=float, help="Domain E-value threshold")
    group.add_argument("--domT", type=float, help="Domain bitscore threshold")
    group.add_argument("--incE", type=float, help="Inclusion E-value threshold")
    group.add_argument("--incT", type=float, help="Inclusion bitscore threshold")
    group.add_argument("--incdomE", type=float, help="Domain inclusion E-value threshold")
    group.add_argument("--incdomT", type=float, help="Domain inclusion bitscore threshold")


def _threshold_opts_from_args(args: argparse.Namespace) -> ThresholdOptions:
    """Build ThresholdOptions from parsed args."""
    return ThresholdOptions(
        cut_ga=getattr(args, "cut_ga", False),
        cut_tc=getattr(args, "cut_tc", False),
        cut_nc=getattr(args, "cut_nc", False),
        cascade=getattr(args, "cascade", False),
        evalue=getattr(args, "evalue", None),
        bitscore=getattr(args, "bitscore", None),
        dom_evalue=getattr(args, "domE", None),
        dom_bitscore=getattr(args, "domT", None),
        inc_evalue=getattr(args, "incE", None),
        inc_bitscore=getattr(args, "incT", None),
        inc_dom_evalue=getattr(args, "incdomE", None),
        inc_dom_bitscore=getattr(args, "incdomT", None),
    )


def _resolve_output(output: str) -> tuple[Path, Optional[Path]]:
    """Parse -o into (output_dir for ResultCollector, final_csv_path or None).

    If output ends with a file extension (e.g. .tsv), output_dir is its parent
    and the csv path is the file itself.  Otherwise output is treated as a
    directory and csv path is deferred (caller picks the filename).
    """
    p = Path(output)
    if p.suffix:
        return p.parent, p
    return p, None


def _write_result(result, output_path: Path, csv_path: Optional[Path], default_name: str) -> None:
    """Write search result to CSV and print summary."""
    from aksha.types import SearchResult

    if csv_path:
        result.to_csv(csv_path)
    else:
        output_path.mkdir(exist_ok=True)
        result.to_csv(output_path / default_name)

    print(f"Found {len(result)} hits")


def _add_search_parser(subparsers) -> None:
    """Add search subcommand."""
    parser = subparsers.add_parser("search", help="Search sequences with HMM profiles")
    parser.add_argument("--sequences", "-s", required=True, help="Protein sequences (file or directory)")
    parser.add_argument("--hmms", "-H", required=True, help="HMM profiles (file, directory, or database name)")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads (0=auto)")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_search)


def _cmd_search(args: argparse.Namespace) -> int:
    """Execute search command."""
    from aksha.search import search

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = search(
        sequences=args.sequences,
        hmms=args.hmms,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "search_results.tsv")
    return 0


def _add_scan_parser(subparsers) -> None:
    """Add scan subcommand."""
    parser = subparsers.add_parser("scan", help="Scan sequences against HMM database")
    parser.add_argument("--sequences", "-s", required=True, help="Protein sequences")
    parser.add_argument("--hmms", "-H", required=True, help="HMM database")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_scan)


def _cmd_scan(args: argparse.Namespace) -> int:
    """Execute scan command."""
    from aksha.scan import scan

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = scan(
        sequences=args.sequences,
        hmms=args.hmms,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "scan_results.tsv")
    return 0


def _add_phmmer_parser(subparsers) -> None:
    """Add phmmer subcommand."""
    parser = subparsers.add_parser("phmmer", help="Protein vs protein search")
    parser.add_argument("--query", "-q", required=True, help="Query sequences")
    parser.add_argument("--target", "-d", required=True, help="Target database")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_phmmer)


def _cmd_phmmer(args: argparse.Namespace) -> int:
    """Execute phmmer command."""
    from aksha.phmmer import phmmer

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = phmmer(
        query=args.query,
        target=args.target,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "phmmer_results.tsv")
    return 0


def _add_jackhmmer_parser(subparsers) -> None:
    """Add jackhmmer subcommand."""
    parser = subparsers.add_parser("jackhmmer", help="Iterative protein search")
    parser.add_argument("--query", "-q", required=True, help="Query sequences")
    parser.add_argument("--target", "-d", required=True, help="Target database")
    parser.add_argument("--output", "-o", required=True, help="Output file")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    parser.add_argument("--iterations", "-N", type=int, default=5, help="Max iterations")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_jackhmmer)


def _cmd_jackhmmer(args: argparse.Namespace) -> int:
    """Execute jackhmmer command."""
    from aksha.jackhmmer import jackhmmer

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = jackhmmer(
        query=args.query,
        target=args.target,
        thresholds=thresholds,
        threads=args.threads,
        max_iterations=args.iterations,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "jackhmmer_results.tsv")
    return 0


def _add_nhmmer_parser(subparsers) -> None:
    """Add nhmmer subcommand."""
    parser = subparsers.add_parser("nhmmer", help="Nucleotide HMM search")
    parser.add_argument("--sequences", "-s", required=True, help="Nucleotide sequences")
    parser.add_argument("--hmms", "-H", required=True, help="Nucleotide HMMs")
    parser.add_argument("--output", "-o", required=True, help="Output file or directory")
    parser.add_argument("--threads", "-t", type=int, default=0, help="CPU threads")
    _add_threshold_args(parser)
    parser.set_defaults(func=_cmd_nhmmer)


def _cmd_nhmmer(args: argparse.Namespace) -> int:
    """Execute nhmmer command."""
    from aksha.nhmmer import nhmmer

    thresholds = _threshold_opts_from_args(args)
    output_dir, csv_path = _resolve_output(args.output)

    result = nhmmer(
        sequences=args.sequences,
        hmms=args.hmms,
        thresholds=thresholds,
        threads=args.threads,
        output_dir=output_dir,
    )

    _write_result(result, output_dir, csv_path, "nhmmer_results.tsv")
    return 0


def _add_database_parser(subparsers) -> None:
    """Add database management subcommand."""
    parser = subparsers.add_parser("database", aliases=["db"], help="Database management")
    db_subparsers = parser.add_subparsers(dest="db_command")

    list_parser = db_subparsers.add_parser("list", help="List databases")
    list_parser.add_argument("--installed", action="store_true", help="Show only installed")
    list_parser.add_argument("--protein", action="store_true", help="Show only protein databases")
    list_parser.add_argument("--nucleotide", action="store_true", help="Show only nucleotide databases")
    list_parser.set_defaults(func=_cmd_db_list)

    install_parser = db_subparsers.add_parser("install", help="Install database")
    install_parser.add_argument("names", nargs="+", help="Database names to install")
    install_parser.add_argument("--force", action="store_true", help="Reinstall if exists")
    install_parser.set_defaults(func=_cmd_db_install)

    uninstall_parser = db_subparsers.add_parser("uninstall", help="Uninstall database")
    uninstall_parser.add_argument("names", nargs="+", help="Database names to uninstall")
    uninstall_parser.set_defaults(func=_cmd_db_uninstall)

    register_parser = db_subparsers.add_parser("register", help="Register custom database")
    register_parser.add_argument("name", help="Database name")
    register_parser.add_argument("path", help="Path to HMM file or directory")
    register_parser.add_argument("--type", choices=["protein", "nucleotide"], default="protein")
    register_parser.add_argument("--citation", default="", help="Citation")
    register_parser.set_defaults(func=_cmd_db_register)

    parser.set_defaults(func=lambda args: parser.print_help() or 1)


def _cmd_db_list(args: argparse.Namespace) -> int:
    """List databases."""
    from aksha.databases import list_databases

    mol_type = None
    if args.protein:
        mol_type = MoleculeType.PROTEIN
    elif args.nucleotide:
        mol_type = MoleculeType.NUCLEOTIDE

    databases = list_databases(molecule_type=mol_type, installed_only=args.installed)

    if not databases:
        print("No databases found")
        return 0

    print(f"{'Name':<20} {'Type':<12} {'Installed':<10}")
    print("-" * 45)
    for db in databases:
        installed = "Yes" if db.installed else "No"
        print(f"{db.name:<20} {db.molecule_type.name.lower():<12} {installed:<10}")

    return 0


def _cmd_db_install(args: argparse.Namespace) -> int:
    """Install databases."""
    from aksha.databases import install_database

    for name in args.names:
        print(f"Installing {name}...")
        install_database(name, force=args.force)
        print(f"Installed {name}")

    return 0


def _cmd_db_uninstall(args: argparse.Namespace) -> int:
    """Uninstall databases."""
    from aksha.databases import uninstall_database

    for name in args.names:
        uninstall_database(name)
        print(f"Uninstalled {name}")

    return 0


def _cmd_db_register(args: argparse.Namespace) -> int:
    """Register custom database."""
    from aksha.databases import register_custom_database

    mol_type = MoleculeType.PROTEIN if args.type == "protein" else MoleculeType.NUCLEOTIDE
    register_custom_database(args.name, args.path, mol_type, args.citation)
    print(f"Registered {args.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
