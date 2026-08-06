"""16-ribosomal-protein (RP16) marker handling for ``astra search --16rp``.

The ``--16rp`` flag runs the RP16 marker set (16 KOfam KOs, installed with
``astra initialize --hmms RP16``) against a set of per-genome protein FASTAs
and produces, for each of the 16 markers, a single genome-labelled multifasta
containing the best-scoring hit per genome -- ready for per-marker alignment
and concatenation.

Genome provenance: GTDB rep protein FASTAs are named like
``GB_GCA_000007325.1_protein.faa`` while the protein headers are Prodigal-style
``{contig_accession}_{orf_index}``.  The genome accession therefore lives only
in the filename, so we recover ``seq_id -> genome`` from the input file each
sequence came from (``protein_dict`` is keyed by file path).

Synteny: Prodigal ORF indices encode gene order on a contig, so ``--synteny``
keeps only genomes whose markers form a syntenic block (markers on one contig,
no more than ``MAX_SYNTENY_GAP`` intervening ORFs between neighbours) covering
at least the requested fraction of the 16 markers.  For kept genomes, only the
markers inside that block are written to the per-marker fastas.
"""

import os
import logging

# KOfam KO (== HMM NAME field == pyhmmer ``hits.query.name``, the value
# written to the hits TSV) -> canonical ribosomal-protein label.
RP16_MARKERS = {
    "K02886": "L2",
    "K02906": "L3",
    "K02926": "L4",
    "K02931": "L5",
    "K02933": "L6",
    "K02874": "L14",
    "K02876": "L15",
    "K02878": "L16",
    "K02881": "L18",
    "K02890": "L22",
    "K02895": "L24",
    "K02982": "S3",
    "K02994": "S8",
    "K02946": "S10",
    "K02961": "S17",
    "K02965": "S19",
}

# Stable output order (large-subunit then small-subunit, ascending).
MARKER_ORDER = ["L2", "L3", "L4", "L5", "L6", "L14", "L15", "L16", "L18",
                "L22", "L24", "S3", "S8", "S10", "S17", "S19"]


# Max number of intervening ORFs allowed between two adjacent markers for them
# to count as part of the same syntenic block.
MAX_SYNTENY_GAP = 3

# Filename prefixes/suffixes stripped to recover the genome accession.
_GENOME_PREFIXES = ("GB_", "RS_")
_GENOME_SUFFIXES = ("_protein.faa", "_protein.fa", ".faa", ".fa", ".fasta")


def profile_filename(ko):
    """On-disk filename for a marker profile, e.g. ``Ribosomal_L2_K02886.hmm``."""
    return "Ribosomal_{}_{}.hmm".format(RP16_MARKERS[ko], ko)


def rp16_hmm_dir():
    """Absolute path to the installed RP16 marker set.

    RP16 is a regular Astra database (``astra initialize --hmms RP16``) rather
    than data vendored into the package, so it resolves through the same
    config as PFAM/KOFAM.
    """
    from astra import initialize

    parsed_json = initialize.load_config()
    for db in parsed_json['db_urls']:
        if db['name'] == 'RP16' and db['installed'] and db['installation_dir']:
            if os.path.isdir(db['installation_dir']):
                return db['installation_dir']
            break

    raise FileNotFoundError(
        "The RP16 marker set is not installed. Install it with:\n"
        "    astra initialize --hmms RP16")


def genome_id_from_path(path):
    """Recover a genome accession from a GTDB-style protein FASTA filename.

    ``/.../GB_GCA_000007325.1_protein.faa`` -> ``GCA_000007325.1``
    Falls back to the bare basename if no known prefix/suffix matches.
    """
    name = os.path.basename(path)
    for suf in _GENOME_SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    for pre in _GENOME_PREFIXES:
        if name.startswith(pre):
            name = name[len(pre):]
            break
    return name


def build_seqid_to_genome(protein_dict):
    """Map every protein sequence id to the genome (file) it came from."""
    seqid_to_genome = {}
    for path, sequences in protein_dict.items():
        genome = genome_id_from_path(path)
        for seq in sequences:
            seqid_to_genome[seq.name] = genome
    return seqid_to_genome


def parse_orf_locus(seq_id):
    """Split a Prodigal protein id into ``(contig, orf_index)``.

    ``AE009951.2_42`` -> ``("AE009951.2", 42)``.  Returns ``(seq_id, None)`` if
    the trailing field is not an integer (so such hits never join a block).
    """
    contig, sep, tail = seq_id.rpartition("_")
    if sep and tail.isdigit():
        return contig, int(tail)
    return seq_id, None


def select_best_hits(hits_tsv, seqid_to_genome):
    """Best-scoring hit per (genome, marker) from a combined hits TSV.

    Returns ``{genome: {marker: (seq_id, bitscore)}}``.
    """
    best = {}
    with open(hits_tsv) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            seq_id, hmm_name, bitscore = parts[0], parts[1], parts[2]
            marker = RP16_MARKERS.get(hmm_name)
            if marker is None:
                continue
            genome = seqid_to_genome.get(seq_id)
            if genome is None:
                continue
            try:
                score = float(bitscore)
            except ValueError:
                continue
            genome_hits = best.setdefault(genome, {})
            prev = genome_hits.get(marker)
            if prev is None or score > prev[1]:
                genome_hits[marker] = (seq_id, score)
    return best


def _largest_syntenic_block(markers):
    """Return the set of markers in the largest syntenic block for one genome.

    *markers* is ``{marker: (seq_id, score)}``.  Markers are grouped by contig
    and, within a contig, sorted by ORF index; a block is a maximal run whose
    neighbouring markers are separated by no more than ``MAX_SYNTENY_GAP``
    intervening ORFs.  The marker set of the single largest block is returned
    (markers whose ORF index could not be parsed are ignored).
    """
    by_contig = {}
    for marker, (seq_id, _score) in markers.items():
        contig, idx = parse_orf_locus(seq_id)
        if idx is None:
            continue
        by_contig.setdefault(contig, []).append((idx, marker))

    best_block = set()
    for contig, entries in by_contig.items():
        entries.sort()  # by ORF index
        run = [entries[0]]
        for prev, cur in zip(entries, entries[1:]):
            # intervening ORFs between the two markers
            if (cur[0] - prev[0] - 1) <= MAX_SYNTENY_GAP:
                run.append(cur)
            else:
                if len(run) > len(best_block):
                    best_block = {m for _i, m in run}
                run = [cur]
        if len(run) > len(best_block):
            best_block = {m for _i, m in run}
    return best_block


def _build_seq_lookup(protein_dict, needed_ids):
    """``seq_id -> TextSequence`` for the required ids only (memory-frugal)."""
    lookup = {}
    for sequences in protein_dict.values():
        for seq in sequences:
            if seq.name in needed_ids:
                lookup[seq.name] = seq.textize()
    return lookup


def write_outputs(best_hits, protein_dict, outdir, synteny_threshold=None):
    """Write the RP16 fastas and summary tables.

    Produces in *outdir*:
      - ``rp16_fastas/{marker}.faa`` -- one genome-labelled multifasta/marker
      - ``rp16_presence.tsv`` -- genome x marker bitscore matrix + summary cols
      - ``rp16_hits.tsv`` -- one row per (genome, marker) best hit

    The syntenic-block size and membership are computed and recorded for every
    genome regardless of *synteny_threshold*, so the search is lossless and the
    threshold can be (re)applied downstream from the tables.  *synteny_threshold*
    governs only which genomes/markers reach the per-marker fastas: when set,
    only genomes whose block covers >= the fraction are written, and only the
    markers inside that block; when ``None``, every best hit is written.
    """
    # Real largest syntenic block per genome (always computed).
    block_by_genome = {g: _largest_syntenic_block(m) for g, m in best_hits.items()}
    block_sizes = {g: len(b) for g, b in block_by_genome.items()}

    if synteny_threshold is not None:
        min_markers = synteny_threshold * len(MARKER_ORDER)
        kept_hits = {g: {m: best_hits[g][m] for m in block_by_genome[g]}
                     for g in best_hits if block_sizes[g] >= min_markers}
    else:
        kept_hits = best_hits

    fastas_dir = os.path.join(outdir, "rp16_fastas")
    os.makedirs(fastas_dir, exist_ok=True)

    # Sequences we actually need to emit (only the kept genomes/markers).
    needed_ids = {seq_id for markers in kept_hits.values()
                  for (seq_id, _s) in markers.values()}
    seq_lookup = _build_seq_lookup(protein_dict, needed_ids)

    kept_genomes = sorted(kept_hits)
    for marker in MARKER_ORDER:
        out_path = os.path.join(fastas_dir, f"{marker}.faa")
        with open(out_path, "w") as fh:
            for genome in kept_genomes:
                hit = kept_hits[genome].get(marker)
                if hit is None:
                    continue
                seq = seq_lookup.get(hit[0])
                if seq is not None:
                    fh.write(f">{genome}\n{seq.sequence}\n")

    # Presence/absence + bitscore matrix over ALL genomes with any hit.
    presence_path = os.path.join(outdir, "rp16_presence.tsv")
    with open(presence_path, "w") as fh:
        cols = ["genome"] + MARKER_ORDER + ["n_markers", "syntenic_block", "included"]
        fh.write("\t".join(cols) + "\n")
        for genome in sorted(best_hits):
            markers = best_hits[genome]
            row = [genome]
            for marker in MARKER_ORDER:
                hit = markers.get(marker)
                row.append(f"{hit[1]:.2f}" if hit else "")
            included = "1" if genome in kept_hits else "0"
            row += [str(len(markers)), str(block_sizes.get(genome, 0)), included]
            fh.write("\t".join(row) + "\n")

    # Per-hit long table (all best hits), flagged by real syntenic-block membership.
    hits_path = os.path.join(outdir, "rp16_hits.tsv")
    with open(hits_path, "w") as fh:
        fh.write("genome\tmarker\tseq_id\tcontig\torf_index\tbitscore\tin_syntenic_block\n")
        for genome in sorted(best_hits):
            block_markers = block_by_genome[genome]
            for marker in MARKER_ORDER:
                hit = best_hits[genome].get(marker)
                if hit is None:
                    continue
                seq_id, score = hit
                contig, idx = parse_orf_locus(seq_id)
                in_block = "1" if marker in block_markers else "0"
                fh.write(f"{genome}\t{marker}\t{seq_id}\t{contig}\t"
                         f"{'' if idx is None else idx}\t{score:.2f}\t{in_block}\n")

    n_in = len(kept_genomes)
    n_total = len(best_hits)
    msg = (f"RP16: {n_total} genomes with >=1 marker; "
           f"{n_in} genomes written to rp16_fastas/")
    if synteny_threshold is not None:
        msg += f" (synteny >= {synteny_threshold:.0%})"
    print(msg)
    logging.info(msg)


def process(hits_tsv, protein_dict, outdir, synteny_threshold=None):
    """End-to-end RP16 post-processing from a combined hits TSV."""
    seqid_to_genome = build_seqid_to_genome(protein_dict)
    best_hits = select_best_hits(hits_tsv, seqid_to_genome)
    write_outputs(best_hits, protein_dict, outdir, synteny_threshold)
