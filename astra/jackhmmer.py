
import logging
import os
import sys
import time

import pyhmmer


HEADER = "sequence_id\tevalue\tenv_from\tenv_to\tbitscore\n"


def main(args):
    t1 = time.time()
    query_file = args.query_seqs
    database_file = args.subject_seqs
    threads = args.threads
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    log_file_path = os.path.join(outdir, 'astra_jackhmmer_log.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    print("Reading query sequences...")
    with pyhmmer.easel.SequenceFile(query_file, digital=True) as sf:
        query = sf.read_block()

    print("Reading target database...")
    with pyhmmer.easel.SequenceFile(database_file, digital=True) as sf:
        sequence_db = sf.read_block()

    print(f"Running jackhmmer: {len(query)} queries × {len(sequence_db)} targets ({threads} threads)...")
    logging.info(f"jackhmmer: {len(query)} queries × {len(sequence_db)} targets")

    out_file = os.path.join(outdir, 'jackhmmer_results.tsv')
    total_hits = 0
    with open(out_file, 'w') as fh:
        fh.write(HEADER)
        for hits in pyhmmer.hmmer.jackhmmer(query, sequence_db, cpus=threads):
            for hit in hits:
                if hit.included and not hit.duplicate:
                    for domain in hit.domains.reported:
                        fh.write(f"{hit.name}\t{hit.evalue:.2e}\t"
                                 f"{domain.env_from}\t{domain.env_to}\t{hit.score:.2f}\n")
                        total_hits += 1

    print(f"Results: {total_hits} hits → {out_file}")
    time_printout = f"Process took {time.time()-t1:.2f} seconds."
    print(time_printout)
    logging.info(time_printout)
