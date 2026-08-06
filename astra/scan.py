
import gc
import logging
import os
import sys
import time
import shutil

import pyhmmer
from tqdm import tqdm

from astra import initialize
from astra.search import (
    extract_sequences,
    has_thresholds,
    parse_hmms,
    parse_protein_input,
)


HEADER = ("sequence_id\thmm_name\tbitscore\tevalue\tc_evalue\ti_evalue\t"
          "env_from\tenv_to\tdom_bitscore\n")


def process_scan_hits(hits, fh):
    """Write hmmscan hits to an open file handle.

    In hmmscan, query is the sequence and hit.name is the HMM.
    """
    seq_name = hits.query.name
    for hit in hits:
        if hit.included:
            hmm_name = hit.name
            full_bitscore = hit.score
            full_evalue = hit.evalue
            for domain in hit.domains.reported:
                fh.write(f"{seq_name}\t{hmm_name}\t{full_bitscore:.2f}\t{full_evalue:.2e}\t"
                         f"{domain.c_evalue:.2e}\t{domain.i_evalue:.2e}\t"
                         f"{domain.env_from}\t{domain.env_to}\t{domain.score:.2f}\n")


def hmmscan(all_sequences, hmms, threads, options, outdir, db_name=None):
    """Run hmmscan: scan sequences against an HMM library.

    Returns the path to the temp results directory.
    """
    hmmscan_kwargs = define_kwargs(options)

    tmp_dir = os.path.join(outdir, 'tmp_results')
    os.makedirs(tmp_dir, exist_ok=True)

    # Separate thresholded vs unthresholded HMMs if bit_cutoffs requested
    hmms_with_thresh = None
    hmms_without_thresh = None
    bit_cutoff = None

    if 'bit_cutoffs' in hmmscan_kwargs:
        if db_name in ('PFAM', 'FOAM'):
            # These DBs have thresholds for every HMM
            hmms_with_thresh = hmms
            bit_cutoff = hmmscan_kwargs.pop('bit_cutoffs')
        else:
            # Mixed DB — split by threshold availability
            print("Separating thresholded and non-thresholded HMMs...")
            hmms_with_thresh = [h for h in hmms if has_thresholds(h)]
            hmms_without_thresh = [h for h in hmms if not has_thresholds(h)]
            bit_cutoff = hmmscan_kwargs.pop('bit_cutoffs')

            if not hmms_with_thresh:
                print("No HMMs have the requested thresholds — using e-value/bitscore only.")
                hmms_with_thresh = None
            if not hmms_without_thresh:
                hmms_without_thresh = None
    else:
        hmms_without_thresh = hmms

    print(f"Scanning {len(all_sequences)} sequences against {len(hmms)} HMMs ({threads} threads)...")

    out_file = os.path.join(tmp_dir, "scan_results.tsv")
    with open(out_file, 'w') as fh:
        fh.write(HEADER)

        if hmms_with_thresh:
            print(f"  Thresholded pass ({len(hmms_with_thresh)} HMMs, bit_cutoffs={bit_cutoff})...")
            for hits in pyhmmer.hmmscan(all_sequences, hmms_with_thresh,
                                         cpus=threads, bit_cutoffs=bit_cutoff):
                process_scan_hits(hits, fh)

        if hmms_without_thresh:
            print(f"  Unthresholded pass ({len(hmms_without_thresh)} HMMs)...")
            for hits in pyhmmer.hmmscan(all_sequences, hmms_without_thresh,
                                         cpus=threads, **hmmscan_kwargs):
                process_scan_hits(hits, fh)

    gc.collect()
    return tmp_dir


def combine_results(tmp_dir, output_file):
    """Move or combine temp result files into the final output."""
    tmp_files = sorted(f for f in os.listdir(tmp_dir) if f.endswith('_results.tsv'))
    if not tmp_files:
        print("No results found to combine.")
        return

    if len(tmp_files) == 1:
        shutil.move(os.path.join(tmp_dir, tmp_files[0]), output_file)
        print(f"Results → {output_file}")
        return

    total_rows = 0
    header_written = False
    with open(output_file, 'w') as out:
        for filename in tmp_files:
            with open(os.path.join(tmp_dir, filename)) as inp:
                header = inp.readline()
                if not header_written:
                    out.write(header)
                    header_written = True
                for line in inp:
                    out.write(line)
                    total_rows += 1
    print(f"Combined {total_rows:,} hits from {len(tmp_files)} files → {output_file}")


def define_kwargs(options):
    kwargs = {}

    if options['cut_ga']:
        kwargs['bit_cutoffs'] = 'gathering'
    elif options['cut_nc']:
        kwargs['bit_cutoffs'] = 'noise'
    elif options['cut_tc']:
        kwargs['bit_cutoffs'] = 'trusted'

    if options['bitscore'] is not None:
        try:
            kwargs['T'] = float(options['bitscore'])
        except (ValueError, TypeError):
            print("Error: bitscore threshold must be a float.")

    if options['domE'] is not None:
        try:
            kwargs['domE'] = float(options['domE'])
        except (ValueError, TypeError):
            print("Error: domE must be a float.")

    if options['domT'] is not None:
        try:
            kwargs['domT'] = float(options['domT'])
        except (ValueError, TypeError):
            print("Error: domT must be a float.")

    if options['incE'] is not None:
        try:
            kwargs['incE'] = float(options['incE'])
        except (ValueError, TypeError):
            print("Error: incE must be a float.")

    if options['incT'] is not None:
        try:
            kwargs['incT'] = float(options['incT'])
        except (ValueError, TypeError):
            print("Error: incT must be a float.")

    if options['incdomE'] is not None:
        try:
            kwargs['incdomE'] = float(options['incdomE'])
        except (ValueError, TypeError):
            print("Error: incdomE must be a float.")

    if options['incdomT'] is not None:
        try:
            kwargs['incdomT'] = float(options['incdomT'])
        except (ValueError, TypeError):
            print("Error: incdomT must be a float.")

    if options['evalue'] is not None:
        try:
            kwargs['E'] = float(options['evalue'])
        except (ValueError, TypeError):
            print("Error: evalue must be a float.")

    return kwargs


def main(args):
    t1 = time.time()
    hmm_in = args.hmm_in
    prot_in = args.prot_in
    outdir = args.outdir
    meta = args.meta
    threads = args.threads
    log_file_path = os.path.join(outdir, 'astra_scan_log.txt')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    hmmscan_options = {
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
    }

    if hmm_in is None and args.installed_hmms is None:
        print("Either a user-provided or pre-installed HMM database must be specified.")
        sys.exit(1)

    protein_dict = parse_protein_input(prot_in, threads)

    # Pre-flatten sequences once
    all_sequences = []
    for seqs in protein_dict.values():
        all_sequences.extend(seqs)
    print(f"Total sequences loaded: {len(all_sequences)}")

    # Free protein_dict — scan doesn't need per-file provenance after flattening
    del protein_dict
    gc.collect()

    if hmm_in is not None:
        print("Scanning with user-provided HMM(s)...")
        logging.info("Scanning with user-provided HMM(s)...")
        user_hmms, _ = parse_hmms(args.hmm_in)
        tmp_dir = hmmscan(all_sequences, user_hmms, threads, hmmscan_options, outdir)
        combine_results(tmp_dir, os.path.join(outdir, 'all_hits_df.tsv'))
        del user_hmms

    if args.installed_hmms is not None:
        installed_hmm_names = args.installed_hmms.split(',') if ',' in args.installed_hmms else [args.installed_hmms]
        print(f"Scanning with pre-installed HMMs: {', '.join(installed_hmm_names)}")
        logging.info(f"Scanning with pre-installed HMMs: {', '.join(installed_hmm_names)}")

        parsed_json = initialize.load_config()

        if 'all_prot' in installed_hmm_names:
            installed_hmm_names = [db['name'] for db in parsed_json['db_urls']
                                   if db['molecule_type'] == 'protein' and db['installed']]

        for hmm_db in installed_hmm_names:
            installed_hmm_in = next((item for item in parsed_json['db_urls'] if item["name"] == hmm_db), None)
            if installed_hmm_in is not None:
                installation_dir = installed_hmm_in['installation_dir']
                db_hmms, _ = parse_hmms(installation_dir)
                tmp_dir = hmmscan(all_sequences, db_hmms, threads, hmmscan_options, outdir, hmm_db)
                combine_results(tmp_dir, os.path.join(outdir, f'{hmm_db}_hits_df.tsv'))
                del db_hmms
                gc.collect()
            else:
                print(f"No installation_dir specified for db {hmm_db}")
                logging.info(f"No installation_dir specified for db {hmm_db}")

    # Clean up temp directory
    tmp_results_path = os.path.join(outdir, 'tmp_results')
    if os.path.exists(tmp_results_path):
        shutil.rmtree(tmp_results_path)
        print(f"Temporary files removed from {tmp_results_path}")

    time_printout = f"Process took {time.time()-t1} seconds."
    print(time_printout)
    logging.info(time_printout)
