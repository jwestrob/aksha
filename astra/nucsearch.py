
import gc
import logging
import os
import sys
import time
import shutil

import pyhmmer
from tqdm import tqdm

from astra import initialize
from astra.search import parse_hmms


HEADER = "sequence_id\thmm_name\tbitscore\tevalue\tenv_from\tenv_to\n"


def process_nhmmer_hits(hits, fh):
    """Write nhmmer hits to an open file handle."""
    hmm_name = hits.query.name
    for hit in hits:
        if hit.included:
            fh.write(f"{hit.name}\t{hmm_name}\t{hit.score:.2f}\t{hit.evalue:.2e}\t"
                     f"{hit.env_from}\t{hit.env_to}\n")


def nucsearch(all_sequences, hmms, threads, options, outdir, db_name=None):
    """Run nhmmer: search nucleotide HMMs against nucleotide sequences.

    Returns the path to the temp results directory.
    """
    nucsearch_kwargs = define_kwargs(options)

    tmp_dir = os.path.join(outdir, 'tmp_results')
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Nucleotide search: {len(all_sequences)} sequences × {len(hmms)} HMMs "
          f"({threads} threads)...")

    out_file = os.path.join(tmp_dir, "nhmmer_results.tsv")
    with open(out_file, 'w') as fh:
        fh.write(HEADER)
        for hits in pyhmmer.nhmmer(hmms, all_sequences, cpus=threads, **nucsearch_kwargs):
            process_nhmmer_hits(hits, fh)

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


def parse_nuc_input(nuc_in, threads):
    """Parse nucleotide FASTA input (file or directory)."""
    print("Parsing nucleotide input sequences...")
    nuc_dict = {}

    if os.path.isdir(nuc_in):
        if not os.listdir(nuc_in):
            print("nuc_in directory is empty.")
            sys.exit(1)

        fasta_paths = [os.path.join(nuc_in, x) for x in os.listdir(nuc_in)]
        for fasta_file in tqdm(fasta_paths):
            with pyhmmer.easel.SequenceFile(fasta_file, digital=True,
                                             alphabet=pyhmmer.easel.Alphabet.dna()) as seq_file:
                nuc_dict[fasta_file] = seq_file.read_block()
    elif os.path.isfile(nuc_in):
        if os.path.getsize(nuc_in) == 0:
            print("nuc_in file is empty.")
            sys.exit(1)
        with pyhmmer.easel.SequenceFile(nuc_in, digital=True,
                                         alphabet=pyhmmer.easel.Alphabet.dna()) as seq_file:
            nuc_dict[nuc_in] = seq_file.read_block()
    else:
        print(f"Invalid input for nuc_in: {nuc_in}")
        sys.exit(1)

    return nuc_dict


def define_kwargs(options):
    kwargs = {}

    if options.get('cut_ga'):
        kwargs['cut_ga'] = True
    elif options.get('cut_nc'):
        kwargs['cut_nc'] = True
    elif options.get('cut_tc'):
        kwargs['cut_tc'] = True

    if options.get('bitscore') is not None:
        try:
            kwargs['T'] = float(options['bitscore'])
        except (ValueError, TypeError):
            print("Error: bitscore must be a float.")

    if options.get('evalue') is not None:
        try:
            kwargs['E'] = float(options['evalue'])
        except (ValueError, TypeError):
            print("Error: evalue must be a float.")

    return kwargs


def main(args):
    t1 = time.time()
    nuc_in = args.nuc_in
    outdir = args.outdir
    threads = args.threads
    log_file_path = os.path.join(outdir, 'astra_nucsearch_log.txt')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    options = {
        "cut_ga": getattr(args, 'cut_ga', False),
        "cut_nc": getattr(args, 'cut_nc', False),
        "cut_tc": getattr(args, 'cut_tc', False),
        "evalue": getattr(args, 'evalue', None),
        "bitscore": getattr(args, 'bitscore', None),
    }

    hmm_in = getattr(args, 'hmm_in', None)
    installed_hmms = getattr(args, 'installed_hmms', None)

    if hmm_in is None and installed_hmms is None:
        print("Either a user-provided or pre-installed HMM database must be specified.")
        sys.exit(1)

    nuc_dict = parse_nuc_input(nuc_in, threads)

    # Pre-flatten sequences once
    all_sequences = []
    for seqs in nuc_dict.values():
        all_sequences.extend(seqs)
    print(f"Total sequences loaded: {len(all_sequences)}")

    # Free nuc_dict — not needed after flattening
    del nuc_dict
    gc.collect()

    if hmm_in is not None:
        print("Searching with user-provided HMM(s)...")
        logging.info("Searching with user-provided HMM(s)...")
        user_hmms, _ = parse_hmms(hmm_in)
        tmp_dir = nucsearch(all_sequences, user_hmms, threads, options, outdir)
        combine_results(tmp_dir, os.path.join(outdir, 'user_hmms_hits_df.tsv'))
        del user_hmms

    if installed_hmms is not None:
        installed_hmm_names = installed_hmms.split(',') if ',' in installed_hmms else [installed_hmms]
        print(f"Searching with pre-installed HMMs: {', '.join(installed_hmm_names)}")
        logging.info(f"Searching with pre-installed HMMs: {', '.join(installed_hmm_names)}")

        parsed_json = initialize.load_config()

        if 'all_nuc' in installed_hmm_names:
            installed_hmm_names = [db['name'] for db in parsed_json['db_urls']
                                   if db['molecule_type'] == 'nucleotide' and db['installed']]

        for hmm_db in installed_hmm_names:
            installed_hmm_in = next((item for item in parsed_json['db_urls'] if item["name"] == hmm_db), None)
            if installed_hmm_in is not None:
                installation_dir = installed_hmm_in['installation_dir']
                db_hmms, _ = parse_hmms(installation_dir)
                tmp_dir = nucsearch(all_sequences, db_hmms, threads, options, outdir, hmm_db)
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
