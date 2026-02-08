"""Database installation and management."""

from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Optional
from urllib.request import urlretrieve

import pandas as pd
import pyhmmer.plan7
import requests
from tqdm import tqdm

from aksha.types import MoleculeType, DatabaseInfo
from aksha.config import get_config, get_registry, AkshaConfig, DatabaseRegistry

logger = logging.getLogger(__name__)


class DownloadProgress(tqdm):
    """Progress bar for downloads."""

    def update_to(self, blocks: int = 1, block_size: int = 1, total_size: int = None):
        if total_size is not None:
            self.total = total_size
        self.update(blocks * block_size - self.n)


def list_databases(
    molecule_type: Optional[MoleculeType] = None,
    installed_only: bool = False,
) -> list[DatabaseInfo]:
    """List available databases."""
    registry = get_registry()
    return registry.list_available(molecule_type=molecule_type, installed_only=installed_only)


def install_database(
    name: str,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> Path:
    """Install a database from the registry."""
    config = get_config()
    registry = get_registry()

    db_info = registry.get(name)
    if db_info is None:
        raise ValueError(f"Unknown database: {name}")

    if db_info.installed and not force:
        logger.info("Database %s already installed", name)
        resolved = db_info.resolve_path(config.data_dir)
        if resolved:
            return resolved
        return config.data_dir / name

    if name == "KOFAM":
        return _install_kofam(config, registry, show_progress)

    return _install_standard(name, db_info, config, registry, show_progress)


def _install_standard(
    name: str,
    db_info: DatabaseInfo,
    config: AkshaConfig,
    registry: DatabaseRegistry,
    show_progress: bool,
) -> Path:
    """Standard database installation."""
    config.ensure_dirs()
    target_dir = config.data_dir / name
    target_dir.mkdir(exist_ok=True)

    url = db_info.url
    logger.info("Downloading %s from %s", name, url)

    if "github.com" in url:
        _download_github(url, target_dir, show_progress)
    else:
        _download_and_extract(url, target_dir, show_progress)

    registry.mark_installed(name, name)
    logger.info("Installed %s to %s", name, target_dir)
    return target_dir


def _download_and_extract(url: str, target_dir: Path, show_progress: bool) -> None:
    """Download and extract a file."""
    filename = url.split("/")[-1]
    download_path = target_dir / filename

    if show_progress:
        with DownloadProgress(unit="B", unit_scale=True, desc=filename) as pbar:
            urlretrieve(url, download_path, reporthook=pbar.update_to)
    else:
        urlretrieve(url, download_path)

    if tarfile.is_tarfile(download_path):
        logger.info("Extracting %s", filename)
        with tarfile.open(download_path, "r:*") as tar:
            tar.extractall(target_dir)
        download_path.unlink()
        _flatten_single_subdir(target_dir)
    elif filename.endswith(".gz") and not filename.endswith(".tar.gz"):
        logger.info("Decompressing %s", filename)
        output_path = download_path.with_suffix("")
        with gzip.open(download_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        download_path.unlink()


def _flatten_single_subdir(target_dir: Path) -> None:
    """If extraction created a single subdirectory, flatten it."""
    contents = list(target_dir.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        subdir = contents[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(target_dir))
        subdir.rmdir()


def _download_github(url: str, target_dir: Path, show_progress: bool) -> None:
    """Download files from a GitHub directory."""
    url = url.rstrip("/")
    if "/blob/" in url:
        raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        filename = raw_url.split("/")[-1]
        download_path = target_dir / filename
        urlretrieve(raw_url, download_path)
    else:
        api_url = url.replace("github.com", "api.github.com/repos").replace("/tree/master", "/contents")
        api_url = api_url.replace("/tree/main", "/contents")

        response = requests.get(api_url, timeout=30)
        response.raise_for_status()

        files = response.json()
        iterator = tqdm(files, desc="Downloading") if show_progress else files

        for file_info in iterator:
            if file_info["name"].lower().endswith(".hmm"):
                download_url = file_info["download_url"]
                download_path = target_dir / file_info["name"]
                urlretrieve(download_url, download_path)


def _install_kofam(
    config: AkshaConfig,
    registry: DatabaseRegistry,
    show_progress: bool,
) -> Path:
    """Install KOFAM with threshold injection."""
    name = "KOFAM"
    db_info = registry.get(name)

    config.ensure_dirs()
    target_dir = config.data_dir / name
    target_dir.mkdir(exist_ok=True)

    logger.info("Downloading KOFAM profiles")
    _download_and_extract(db_info.url, target_dir, show_progress)

    ko_list_url = "https://www.genome.jp/ftp/db/kofam/ko_list.gz"
    ko_list_path = config.data_dir / "ko_list.gz"

    logger.info("Downloading KOFAM thresholds")
    if show_progress:
        with DownloadProgress(unit="B", unit_scale=True, desc="ko_list.gz") as pbar:
            urlretrieve(ko_list_url, ko_list_path, reporthook=pbar.update_to)
    else:
        urlretrieve(ko_list_url, ko_list_path)

    with gzip.open(ko_list_path, "rb") as f_in:
        with open(ko_list_path.with_suffix(""), "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    ko_list_path.unlink()

    logger.info("Injecting thresholds into HMM files")
    ko_list = pd.read_csv(ko_list_path.with_suffix(""), sep="\t")

    for _, row in tqdm(ko_list.iterrows(), total=len(ko_list), desc="Injecting thresholds"):
        if row["threshold"] == "-":
            continue

        try:
            threshold = float(row["threshold"])
        except (ValueError, TypeError):
            continue

        hmm_path = target_dir / f"{row['knum']}.hmm"
        if hmm_path.exists():
            _inject_threshold(hmm_path, threshold)

    registry.mark_installed(name, name)
    logger.info("Installed KOFAM to %s", target_dir)
    return target_dir


def _inject_threshold(hmm_path: Path, threshold: float) -> None:
    """Inject threshold into HMM file."""
    try:
        with pyhmmer.plan7.HMMFile(hmm_path) as f:
            hmm = f.read()

        hmm.cutoffs.gathering = (threshold, threshold)
        hmm.cutoffs.trusted = (threshold, threshold)
        hmm.cutoffs.noise = (threshold, threshold)

        with open(hmm_path, "wb") as f:
            hmm.write(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to inject threshold into %s: %s", hmm_path, exc)


def uninstall_database(name: str) -> None:
    """Uninstall a database."""
    config = get_config()
    registry = get_registry()

    db_info = registry.get(name)
    if db_info is None or not db_info.installed:
        logger.warning("Database %s not installed", name)
        return

    db_path = db_info.resolve_path(config.data_dir)
    if db_path and db_path.exists():
        shutil.rmtree(db_path)
        logger.info("Removed %s", db_path)

    registry.mark_uninstalled(name)


def register_custom_database(
    name: str,
    path: str,
    molecule_type: MoleculeType,
    citation: str = "",
) -> None:
    """Register a custom database."""
    registry = get_registry()
    registry.register_custom(name, path, molecule_type, citation)
    logger.info("Registered custom database: %s", name)
