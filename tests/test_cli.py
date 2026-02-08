"""CLI smoke tests to mirror common user workflows."""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cli_search_small(temp_dir):
    """`aksha search` should produce a TSV for small protein fixtures."""
    out_path = temp_dir / "cli_search.tsv"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "search",
        "--sequences",
        "tests/fixtures/small.faa",
        "--hmms",
        "tests/fixtures/small.hmm",
        "--output",
        str(out_path),
        "--threads",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    df = pd.read_csv(out_path, sep="\t")
    assert not df.empty
    assert {"sequence_id", "hmm_name", "bitscore"}.issubset(df.columns)


def test_cli_nhmmer_small(temp_dir):
    """`aksha nhmmer` should run on small nucleotide fixtures."""
    out_path = temp_dir / "cli_nhmmer.tsv"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "nhmmer",
        "--sequences",
        "tests/fixtures/small.fna",
        "--hmms",
        "tests/fixtures/small_nuc.hmm",
        "--output",
        str(out_path),
        "--threads",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    df = pd.read_csv(out_path, sep="\t")
    assert not df.empty
    assert {"sequence_id", "hmm_name", "bitscore"}.issubset(df.columns)


def test_cli_nhmmer_dir_output(temp_dir):
    """nhmmer should create a directory and write nhmmer_results.tsv when -o is a dir."""
    out_dir = temp_dir / "nhmmer_out"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "nhmmer",
        "--sequences",
        "tests/fixtures/small.fna",
        "--hmms",
        "tests/fixtures/small_nuc.hmm",
        "--output",
        str(out_dir),
        "--threads",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)

    out_file = out_dir / "nhmmer_results.tsv"
    df = pd.read_csv(out_file, sep="\t")
    assert out_file.exists()
    assert not df.empty


def test_cli_search_dir_output(temp_dir):
    """search should create directory output with search_results.tsv."""
    out_dir = temp_dir / "search_out"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "search",
        "--sequences",
        "tests/fixtures/small.faa",
        "--hmms",
        "tests/fixtures/small.hmm",
        "--output",
        str(out_dir),
        "--threads",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_file = out_dir / "search_results.tsv"
    df = pd.read_csv(out_file, sep="\t")
    assert out_file.exists()
    assert not df.empty


def test_cli_scan_dir_output(temp_dir):
    """scan should create directory output with scan_results.tsv."""
    out_dir = temp_dir / "scan_out"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "scan",
        "--sequences",
        "tests/fixtures/small.faa",
        "--hmms",
        "tests/fixtures/small.hmm",
        "--output",
        str(out_dir),
        "--threads",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_file = out_dir / "scan_results.tsv"
    df = pd.read_csv(out_file, sep="\t")
    assert out_file.exists()
    assert not df.empty


def test_cli_phmmer_dir_output(temp_dir):
    """phmmer should create directory output with phmmer_results.tsv."""
    out_dir = temp_dir / "phmmer_out"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "phmmer",
        "--query",
        "tests/fixtures/small.faa",
        "--target",
        "tests/fixtures/small.faa",
        "--output",
        str(out_dir),
        "--threads",
        "1",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_file = out_dir / "phmmer_results.tsv"
    df = pd.read_csv(out_file, sep="\t")
    assert out_file.exists()
    assert not df.empty


def test_cli_jackhmmer_dir_output(temp_dir):
    """jackhmmer should create directory output with jackhmmer_results.tsv."""
    out_dir = temp_dir / "jackhmmer_out"
    cmd = [
        sys.executable,
        "-m",
        "aksha.cli",
        "jackhmmer",
        "--query",
        "tests/fixtures/small.faa",
        "--target",
        "tests/fixtures/small.faa",
        "--output",
        str(out_dir),
        "--threads",
        "1",
        "--iterations",
        "2",
    ]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)
    out_file = out_dir / "jackhmmer_results.tsv"
    df = pd.read_csv(out_file, sep="\t")
    assert out_file.exists()
    assert not df.empty
