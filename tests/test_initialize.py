"""Database installer checks that need no network: KOFAM thresholds and registry, catalog URLs."""

import contextlib
import gzip
import io
import json
import tarfile

import pandas as pd
import pytest

from aksha import initialize as ini

KO_LIST_HEADER = ("knum\tthreshold\tscore_type\tprofile_type\tF-measure\tnseq\tnseq_used\talen\tmlen\t"
                  "eff_nseq\tre/pos\tdefinition\n")


def _ko_list_text(rows):
    return KO_LIST_HEADER + "".join(
        f"{knum}\t{threshold}\tfull\tall\t0.5\t10\t10\t100\t100\t1.0\t0.5\tdescription\n" for knum, threshold in rows)


def _string_inference(enabled):
    """pandas 3 reads text columns as str; pandas 2.2 can opt in with future.infer_string."""
    try:
        pd.get_option("future.infer_string")
    except (KeyError, pd.errors.OptionError):
        if enabled:
            pytest.skip("this pandas has no future.infer_string option")
        return contextlib.nullcontext()
    return pd.option_context("future.infer_string", enabled)


@pytest.mark.parametrize("infer_string", [False, True])
def test_ko_thresholds_are_floats_and_skip_dashes(tmp_path, infer_string):
    path = tmp_path / "ko_list"
    path.write_text(_ko_list_text([("K00001", "123.45"), ("K00002", "-"), ("K00003", "0")]))
    with _string_inference(infer_string):
        pairs = ini.ko_thresholds(str(path))
    assert pairs == [("K00001", 123.45), ("K00003", 0.0)]
    assert all(isinstance(threshold, float) for _, threshold in pairs)


def _fake_kofam_downloads(profile_names, ko_rows):
    def urlretrieve(url, filename, reporthook=None):
        if url.endswith("profiles.tar.gz"):
            with tarfile.open(filename, "w:gz") as tar:
                for name in profile_names:
                    data = b"HMMER3/f placeholder\n//\n"
                    info = tarfile.TarInfo(f"profiles/{name}")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        elif url.endswith("ko_list.gz"):
            with gzip.open(filename, "wt") as handle:
                handle.write(_ko_list_text(ko_rows))
        elif url.endswith("dbCAN-HMMdb-V14.txt"):
            with open(filename, "w") as handle:
                handle.write("HMMER3/b [3.0 | March 2010]\nNAME  example\n//\n")
        else:
            raise AssertionError(f"unexpected download {url}")
        return filename, None
    return urlretrieve


def _entry(name, url):
    return {"name": name, "url": url, "installed": False, "installation_dir": "", "version": None,
            "source_url": "", "installed_date": ""}


def _installer_sandbox(tmp_path, monkeypatch, downloads):
    """Config and database directories under tmp_path, fake downloads, no pressing; returns them."""
    config_dir, db_path = tmp_path / "config", tmp_path / "databases"
    config_dir.mkdir()
    registry = {"db_path": str(db_path), "db_urls": [
        _entry("KOFAM", "https://example.invalid/kofam/profiles.tar.gz"),
        _entry("dbCAN", "https://example.invalid/dbCAN-HMMdb-V14.txt")]}
    (config_dir / "hmm_databases.json").write_text(json.dumps(registry))
    monkeypatch.setattr(ini, "astra_config_dir", lambda: str(config_dir))
    monkeypatch.setattr(ini.urllib.request, "urlretrieve", downloads)
    monkeypatch.setattr(ini, "record_version", lambda db, url, version=None: db.update(version="test"))
    monkeypatch.setattr(ini, "press_hmm_database", lambda *args, **kwargs: None)
    thresholds = []
    monkeypatch.setattr(ini, "add_threshold", lambda path, threshold: thresholds.append((path, threshold)))
    return config_dir, db_path, thresholds


def _installed(config_dir):
    registry = json.loads((config_dir / "hmm_databases.json").read_text())
    return {db["name"]: db["installation_dir"] for db in registry["db_urls"] if db["installed"]}


def test_kofam_install_is_recorded_in_the_registry_aksha_reads(tmp_path, monkeypatch):
    config_dir, db_path, thresholds = _installer_sandbox(
        tmp_path, monkeypatch,
        _fake_kofam_downloads(["K00001.hmm", "K00002.hmm"], [("K00001", "50.5"), ("K00002", "-")]))

    ini.install_KOFAM()

    assert _installed(config_dir) == {"KOFAM": str(db_path / "KOFAM")}
    assert not (db_path / "hmm_databases.json").exists()
    assert thresholds == [(str(db_path / "KOFAM" / "K00001.hmm"), 50.5)]


def test_installs_in_one_command_keep_each_others_records(tmp_path, monkeypatch):
    # `aksha initialize --hmms KOFAM,dbCAN`: main() holds the registry from before KOFAM's install.
    config_dir, db_path, _ = _installer_sandbox(
        tmp_path, monkeypatch, _fake_kofam_downloads(["K00001.hmm"], [("K00001", "50.5")]))
    held = ini.initialize_config()
    ini.install_databases("KOFAM", held, str(db_path))
    ini.install_databases("dbCAN", held, str(db_path))
    assert _installed(config_dir) == {"KOFAM": str(db_path / "KOFAM"), "dbCAN": str(db_path / "dbCAN")}


def test_kofam_install_resumes_from_extracted_profiles_without_downloading(tmp_path, monkeypatch):
    fake = _fake_kofam_downloads([], [("K00001", "50.5")])

    def no_profile_download(url, filename, reporthook=None):
        assert not url.endswith("profiles.tar.gz"), "profiles were downloaded again"
        return fake(url, filename, reporthook)

    config_dir, db_path, thresholds = _installer_sandbox(tmp_path, monkeypatch, no_profile_download)
    (db_path / "KOFAM").mkdir(parents=True)
    (db_path / "KOFAM" / "K00001.hmm").write_text("HMMER3/f placeholder\n//\n")

    ini.install_KOFAM()

    assert _installed(config_dir) == {"KOFAM": str(db_path / "KOFAM")}
    assert thresholds == [(str(db_path / "KOFAM" / "K00001.hmm"), 50.5)]


def _catalog():
    with open(ini.os.path.join(ini.package_dir, "hmm_databases.json")) as handle:
        return {db["name"]: db for db in json.load(handle)["db_urls"]}


def test_catalog_points_dbcan_at_the_current_hmm_release():
    assert _catalog()["dbCAN"]["url"] == "https://pro.unl.edu/dbCAN2/download/Databases/V14/dbCAN-HMMdb-V14.txt"


def test_catalog_lists_cctyper_and_install_dispatches_to_its_installer(monkeypatch):
    assert _catalog()["CCTyper"]["url"].startswith("https://raw.githubusercontent.com/Russel88/CRISPRCasTyper/")
    called = []
    monkeypatch.setattr(ini, "install_CCTyper", lambda: called.append("CCTyper"))
    ini.install_databases("CCTyper")
    assert called == ["CCTyper"]


def test_a_download_that_is_not_hmm_text_stops_the_install(tmp_path):
    folder = tmp_path / "dbCAN"
    folder.mkdir()
    page = folder / "dbCAN-HMMdb-V14.txt"
    page.write_text("<!DOCTYPE html>\n<html><body>moved</body></html>\n")
    with pytest.raises(SystemExit, match="did not return an HMM file"):
        ini.require_hmm_download(str(page), "https://example.invalid/dbCAN-HMMdb-V14.txt")
    assert not folder.exists()

    folder.mkdir()
    hmm = folder / "dbCAN-HMMdb-V14.txt"
    hmm.write_text("HMMER3/b [3.0 | March 2010]\nNAME  example\n//\n")
    ini.require_hmm_download(str(hmm), "https://example.invalid/dbCAN-HMMdb-V14.txt")
    assert hmm.exists()
