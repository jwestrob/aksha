"""KOfam cutoffs follow ko_list's score_type: stamping, pressing, search semantics and re-stamping."""

import functools
import json
import math
import random

import astra_pyhmmer
import pytest

from aksha import initialize as ini
from aksha import rp16
from aksha import search

KO_LIST_HEADER = ("knum\tthreshold\tscore_type\tprofile_type\tF-measure\tnseq\tnseq_used\talen\tmlen\t"
                  "eff_nseq\tre/pos\tdefinition\n")
AMINO = astra_pyhmmer.easel.Alphabet.amino()
P = ini.PERMISSIVE_CUTOFF


def _ko_list(path, rows):
    path.write_text(KO_LIST_HEADER + "".join(
        f"{knum}\t{threshold}\t{score_type}\tall\t0.5\t10\t10\t100\t100\t1.0\t0.5\tdescription\n"
        for knum, threshold, score_type in rows))
    return path


def _core(seed=11, length=120):
    rng = random.Random(seed)
    return "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(length))


def _model(name, seed=11):
    builder = astra_pyhmmer.plan7.Builder(AMINO, seed=seed)
    background = astra_pyhmmer.plan7.Background(AMINO)
    sequence = astra_pyhmmer.easel.TextSequence(name=b"core", sequence=_core(seed)).digitize(AMINO)
    hmm, _, _ = builder.build(sequence, background)
    hmm.name = name.encode()
    return hmm


def _write(hmm, path, cutoffs=None):
    hmm = hmm.copy()
    if cutoffs is not None:
        hmm.cutoffs.gathering = hmm.cutoffs.trusted = hmm.cutoffs.noise = cutoffs
    with open(path, "wb") as handle:
        hmm.write(handle)
    return path


def _read(path):
    with astra_pyhmmer.plan7.HMMFile(str(path)) as handle:
        return handle.read()


def _targets():
    """One homolog of the model's core (30% substituted) inside short random flanks."""
    rng = random.Random(5)
    core = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") if rng.random() < 0.3 else c for c in _core())
    flank = "".join(rng.choice("ACDEFGHIKLMNPQRSTVWY") for _ in range(40))
    target = astra_pyhmmer.easel.TextSequence(name=b"target", sequence=flank + core + flank)
    return astra_pyhmmer.easel.DigitalSequenceBlock(AMINO, [target.digitize(AMINO)])


@functools.lru_cache(maxsize=None)
def _scores():
    """(full-sequence score, best domain score) of the target, searched without thresholds."""
    hits = next(iter(astra_pyhmmer.hmmer.hmmsearch(
        [_model("K90001")], _targets(), cpus=1, T=-1000.0, domT=-1000.0, incT=-1000.0, incdomT=-1000.0)))
    hit = hits[0]
    return hit.score, hit.best_domain.score


def _options(outdir, cascade=False):
    return {"cascade": cascade, "cut_ga": True, "cut_nc": False, "cut_tc": False, "evalue": None,
            "bitscore": None, "domE": None, "domT": None, "incE": None, "incT": None, "incdomE": None,
            "incdomT": None, "outdir": str(outdir), "meta": False}


def _rows(tmp_path, cutoffs, tag, hmms=None, cascade=False):
    """Aksha's output rows (sequence, model, full score, domain score) for a --cut_ga search."""
    if hmms is None:
        hmm = _model("K90001")
        hmm.cutoffs.gathering = cutoffs
        hmms = [hmm]
    outdir = tmp_path / tag
    search.hmmsearch({}, hmms, 1, _options(outdir, cascade), all_sequences=_targets())
    with open(outdir / "tmp_results" / "bulk_results.tsv") as handle:
        handle.readline()
        return [(f[0], f[1], float(f[2]), float(f[8])) for f in (line.split("\t") for line in handle)]


# --- Encoding --------------------------------------------------------------------------------

def test_threshold_goes_into_the_slot_score_type_names():
    assert ini.kofam_cutoffs(159.07, "full") == (159.07, P)
    assert ini.kofam_cutoffs("67.87", "domain") == (P, 67.87)
    with pytest.raises(ValueError, match="score_type"):
        ini.kofam_cutoffs(10.0, "-")


def test_permissive_cutoff_is_finite_and_distinct_from_hmmers_unset_marker():
    hmm = _model("K90001")
    del hmm.cutoffs.gathering
    unset = hmm.cutoffs.as_vector()[0]
    assert math.isfinite(P) and P != unset and P < -100


# --- Stamping ----------------------------------------------------------------------------------

@pytest.mark.parametrize("score_type, expected", [("full", (50.25, P)), ("domain", (P, 50.25))])
def test_add_threshold_stamps_ga_tc_and_nc_by_score_type(tmp_path, score_type, expected):
    path = _write(_model("K90001"), tmp_path / "K90001.hmm")
    assert ini.add_threshold(str(path), 50.25, score_type) is True
    cutoffs = _read(path).cutoffs
    for pair in (cutoffs.gathering, cutoffs.trusted, cutoffs.noise):
        assert pair == pytest.approx(expected)
    assert ini.add_threshold(str(path), 50.25, score_type) is False  # already stamped


def test_add_threshold_replaces_the_earlier_encoding(tmp_path):
    path = _write(_model("K90001"), tmp_path / "K90001.hmm", cutoffs=(50.25, 50.25))
    assert ini.add_threshold(str(path), 50.25, "full") is True
    assert _read(path).cutoffs.gathering == pytest.approx((50.25, P))


def test_add_threshold_leaves_missing_profiles_and_zero_thresholds_alone(tmp_path):
    missing = tmp_path / "K90009.hmm"
    assert ini.add_threshold(str(missing), 50.0, "full") is False
    assert not missing.exists()
    path = _write(_model("K90001"), tmp_path / "K90001.hmm")
    before = path.read_bytes()
    assert ini.add_threshold(str(path), 0.0, "full") is False
    assert path.read_bytes() == before
    assert _read(path).cutoffs.gathering is None


def test_permissive_cutoffs_survive_pressing_into_optimized_profiles(tmp_path):
    full = _model("K90001")
    full.cutoffs.gathering = ini.kofam_cutoffs(40.0, "full")
    domain = _model("K90002", seed=12)
    domain.cutoffs.gathering = ini.kofam_cutoffs(40.0, "domain")
    pressed = tmp_path / "pressed"
    astra_pyhmmer.hmmer.hmmpress([full, domain], str(pressed))
    with astra_pyhmmer.plan7.HMMFile(str(pressed)) as handle:
        profiles = list(handle.optimized_profiles())
    assert [p.cutoffs.gathering_available() for p in profiles] == [True, True]
    assert profiles[0].cutoffs.gathering == pytest.approx((40.0, P))
    assert profiles[1].cutoffs.gathering == pytest.approx((P, 40.0))


# --- Search semantics ----------------------------------------------------------------------------

def test_full_type_is_assigned_on_the_full_score_when_no_domain_reaches_the_threshold(tmp_path):
    # The controlled K02886 case: full score at or above the threshold, best domain below it.
    full, domain = _scores()
    assert full > domain  # HMMER's full score is at least the single domain's reconstruction
    threshold = (full + domain) / 2

    new = _rows(tmp_path, ini.kofam_cutoffs(threshold, "full"), "new")
    assert [row[:2] for row in new] == [("target", "K90001")]
    assert new[0][3] < threshold  # the passing sequence keeps its sub-threshold domain row
    assert _rows(tmp_path, (threshold, threshold), "old") == []  # earlier encoding drops it

    hmm = _model("K90001")
    hmm.cutoffs.gathering = ini.kofam_cutoffs(threshold, "full")
    pressed = tmp_path / "pressed"
    astra_pyhmmer.hmmer.hmmpress([hmm], str(pressed))
    with astra_pyhmmer.plan7.HMMFile(str(pressed)) as handle:
        from_pressed = list(handle)
    assert _rows(tmp_path, None, "pressed", hmms=from_pressed) == new
    assert _rows(tmp_path, ini.kofam_cutoffs(threshold, "full"), "cascade", cascade=True) == new

    assert _rows(tmp_path, ini.kofam_cutoffs(full + 1.0, "full"), "above-full") == []


def test_domain_type_is_decided_by_the_best_domain_score_alone(tmp_path):
    full, domain = _scores()
    # Mirror of the K02874 case: full score above the threshold, best domain below it.
    threshold = (full + domain) / 2
    assert _rows(tmp_path, ini.kofam_cutoffs(threshold, "domain"), "between") == []

    # Best domain above the threshold: assigned whatever the sequence slot would have required.
    threshold = domain - 1.0
    assert [r[:2] for r in _rows(tmp_path, ini.kofam_cutoffs(threshold, "domain"), "new")] == [
        ("target", "K90001")]
    assert _rows(tmp_path, (full + 1.0, threshold), "gated-on-full") == []


# --- RP16 install ---------------------------------------------------------------------------------

def _sandbox(tmp_path, monkeypatch, entries):
    config_dir, db_path = tmp_path / "config", tmp_path / "databases"
    config_dir.mkdir()
    db_path.mkdir()
    registry = {"db_path": str(db_path), "db_urls": entries}
    (config_dir / "hmm_databases.json").write_text(json.dumps(registry))
    monkeypatch.setattr(ini, "astra_config_dir", lambda: str(config_dir))
    return config_dir, db_path


def _registry_entry(config_dir, name):
    registry = json.loads((config_dir / "hmm_databases.json").read_text())
    return next(db for db in registry["db_urls"] if db["name"] == name)


def test_rp16_install_stamps_markers_by_score_type_from_a_legacy_kofam(tmp_path, monkeypatch):
    db_path = tmp_path / "databases"
    kofam_entry = {"name": "KOFAM", "url": "", "installed": True, "installation_dir": str(db_path / "KOFAM"),
                   "molecule_type": "protein", "version": "2026-07-30"}
    rp16_entry = {"name": "RP16", "url": "https://example.invalid/profiles.tar.gz", "installed": False,
                  "installation_dir": "", "molecule_type": "protein"}
    config_dir, db_path = _sandbox(tmp_path, monkeypatch, [kofam_entry, rp16_entry])
    (db_path / "KOFAM").mkdir()
    kos = list(rp16.RP16_MARKERS)
    types = {ko: ("domain" if index % 2 else "full") for index, ko in enumerate(kos)}
    _ko_list(db_path / "ko_list", [(ko, "35.00", types[ko]) for ko in kos])
    for index, ko in enumerate(kos):
        _write(_model(ko, 100 + index), db_path / "KOFAM" / f"{ko}.hmm", cutoffs=(35.0, 35.0))

    ini.install_RP16()

    for ko in kos:
        stamped = _read(db_path / "RP16" / rp16.profile_filename(ko)).cutoffs.gathering
        assert stamped == pytest.approx(ini.kofam_cutoffs(35.0, types[ko]))
    assert _registry_entry(config_dir, "RP16")["cutoff_encoding"] == ini.KOFAM_CUTOFF_ENCODING
