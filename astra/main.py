import argparse
from astra import search, initialize, nucsearch, phmmer, jackhmmer, scan

class CustomHelpFormatter(argparse.HelpFormatter):
    def _format_action(self, action):
        parts = super()._format_action(action)
        if action.nargs == argparse.PARSER:
            parts = "\n" + parts + '\n'
        return parts

def main():
    parser = argparse.ArgumentParser(description='Astra: A scalable tool for various sequence searching tasks.', formatter_class=CustomHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='\nAvailable subroutines:')






    ##############
    # INITIALIZE #
    ##############
    
    # Astra initialize sub-command
    parser_initialize = subparsers.add_parser('initialize', help='Downloads provided HMM databases.')
    parser_initialize.add_argument('--show_installed', action='store_true', help='Show installed databases.')
    parser_initialize.add_argument('--show_available', action='store_true', help='Show available databases.')
    parser_initialize.add_argument('--hmms', nargs='?', const='all', default=None, help='Install specified HMM databases. Use comma-separated values for multiple databases.')
    parser_initialize.add_argument('--press', action='store_true', default=False, help='Press installed databases for fast loading. Optionally specify --hmms to press only specific databases.')
    parser_initialize.add_argument('--force', action='store_true', default=False, help='Reinstall the databases named by --hmms even if they are already installed, replacing them with the current upstream release.')


    ##########
    # SEARCH #
    ##########


    # Astra search sub-command
    parser_search = subparsers.add_parser('search', description='Search protein sequences with HMM profiles.')
    parser_search.add_argument('--hmm_in', default=None, help='Input file/directory for HMM profiles.')
    parser_search.add_argument('--prot_in', required=True, help='Input file for protein sequences.')
    parser_search.add_argument('--outdir', required=True, help='Output directory for results.')
    parser_search.add_argument('--installed_hmms', default=None, type=str, help='Comma-separated list of installed HMM databases to use. If you specify a database that is not installed, Astra will not utilize it here. Go install it with initialize')

    #16rp 
    parser_search.add_argument('--16rp', action='store_true', default=False, help='Retrieve 16 ribosomal protein markers for concatenated phylogenetic analysis [BACTERIA + ARCHAEA]. Requires the RP16 database (astra initialize --hmms RP16).')
    parser_search.add_argument('--15rp', action='store_true', default=False, help='Retrieve 15 ribosomal protein markers for concatenated phylogenetic analysis [ARCHAEA ONLY]')
    parser_search.add_argument('--synteny', type=float, default=None, help='[16/15RP ONLY] Percentage of RP markers that must be present in a syntenic block (max gap 3 ORFs) to include the genome in the final alignment. (e.g. 0.5)')

    # Optional arguments
    parser_search.add_argument("--evalue", type=str, default=None, help="Custom e-value threshold for HMM search.")
    parser_search.add_argument("--bitscore", type=str, default=None, help="Custom bitscore threshold for HMM search.")

    #Inclusion thresholds
    parser_search.add_argument("--incE", type=str, default=None, help="The per-target E-value threshold for including a hit.")
    parser_search.add_argument("--incT", type=str, default=None, help="The per-target bitscore threshold for including a hit.")

    #Domain-specific thresholds
    parser_search.add_argument("--domE", type=str, default=None, help="The per-domain E-value threshold for reporting a hit.")
    parser_search.add_argument("--domT", type=str, default=None, help="The per-domain bitscore threshold for reporting a hit.")

    #Domain-specific inclusion thresholds
    parser_search.add_argument("--incdomE", type=str, default=None, help="The per-domain E-value threshold for including a hit.")
    parser_search.add_argument("--incdomT", type=str, default=None, help="The per-domain bitscore threshold for including a hit.")
 

    # Boolean flags
    parser_search.add_argument("--cut_ga", action="store_true", default=False, help="Use built-in GA thresholds. Default: False")
    parser_search.add_argument("--cut_nc", action="store_true", default=False, help="Use built-in NC thresholds. Default: False")
    parser_search.add_argument("--cut_tc", action="store_true", default=False, help="Use built-in TC thresholds. Default: False")   
    parser_search.add_argument("--cascade", action="store_true", default=False, help="Use whatever bitscore threshold is available; using simultaneously with e.g. --cut_ga will default to GA cutoffs where available, but take whatever other threshold is included if that is not an option.")

    parser_search.add_argument('--meta', action='store_true',default=False, help='Indicates input files are metagenomes; changes behavior to compensate for large input file size. See docs for details')
    parser_search.add_argument('--individual_results', action='store_true', default=False, help='Indicates a large number of input files; dont store everything in memory and write results per input file, then concatenate')
    parser_search.add_argument("--write_seqs", action="store_true", default=False, help="Obtain sequences for each HMM and write them to a folder within 'outdir'. Default: False")
    parser_search.add_argument('--threads', type=int, help="Number of threads to use for HMMsearch. Default behavior: Choose appropriate number of threads based on psutil.cpu_count and number of query sequences", default=1)
    parser_search.add_argument('--write_macsyfinder', action='store_true', default=False,
        help="Write per-HMM hmmsearch-format output files for MacSyFinder --previous-run compatibility. "
             "Files are written to {outdir}/macsyfinder_compat/hmmer_results/")

    ##########
    #  SCAN  #
    ##########

    # Astrasearch sub-command
    parser_scan = subparsers.add_parser('scan', help='Scan protein sequences with HMM profiles.')
    parser_scan.add_argument('--hmm_in', default=None, help='Input file/directory for HMM profiles.')
    parser_scan.add_argument('--prot_in', required=True, help='Input file for protein sequences.')
    parser_scan.add_argument('--outdir', required=True, help='Output directory for results.')
    parser_scan.add_argument('--installed_hmms', default=None, type=str, help='Comma-separated list of installed HMM databases to use. If you specify a database that is not installed, Astra will not utilize it here. Go install it with initialize')

    # Optional arguments
    parser_scan.add_argument("--evalue", type=str, default=None, help="Custom e-value threshold for HMM search.")
    parser_scan.add_argument("--bitscore", type=str, default=None, help="Custom bitscore threshold for HMM search.")

    #Inclusion thresholds
    parser_scan.add_argument("--incE", type=str, default=None, help="The per-target E-value threshold for including a hit.")
    parser_scan.add_argument("--incT", type=str, default=None, help="The per-target bitscore threshold for including a hit.")

    #Domain-specific thresholds
    parser_scan.add_argument("--domE", type=str, default=None, help="The per-domain E-value threshold for reporting a hit.")
    parser_scan.add_argument("--domT", type=str, default=None, help="The per-domain bitscore threshold for reporting a hit.")

    #Domain-specific inclusion thresholds
    parser_scan.add_argument("--incdomE", type=str, default=None, help="The per-domain E-value threshold for including a hit.")
    parser_scan.add_argument("--incdomT", type=str, default=None, help="The per-domain bitscore threshold for including a hit.")
 

    # Boolean flags
    parser_scan.add_argument("--cut_ga", action="store_true", default=False, help="Use built-in GA thresholds. Default: False")
    parser_scan.add_argument("--cut_nc", action="store_true", default=False, help="Use built-in NC thresholds. Default: False")
    parser_scan.add_argument("--cut_tc", action="store_true", default=False, help="Use built-in TC thresholds. Default: False")   
    parser_scan.add_argument("--Z", type=str, default=None, help="Number of sequences in input set; for conditioning scan evalues. Default behavior: Obtain from HMM file")
    
    parser_scan.add_argument('--meta', action='store_true',default=False, help='Indicates input files are metagenomes; changes behavior to compensate for large input file size. See docs for details')
    parser_scan.add_argument("--write_seqs", action="store_true", default=False, help="Obtain sequences for each HMM and write them to a folder within 'outdir'. Default: False")
    parser_scan.add_argument('--threads', type=int, help="Number of threads to use for HMMsearch. Default behavior: Choose appropriate number of threads based on psutil.cpu_count and number of query sequences", default=0) 
    # ... other arguments for scan

    # Astra nucsearch sub-command
    parser_nucsearch = subparsers.add_parser('nucsearch', help='Performs nhmmer search on nucleotide sequences.')
    parser_nucsearch.add_argument('--nuc_in', required=True, help='Input nucleotide FASTA file or directory.')
    parser_nucsearch.add_argument('--hmm_in', default=None, help='Input file/directory for nucleotide HMM profiles.')
    parser_nucsearch.add_argument('--installed_hmms', default=None, type=str, help='Comma-separated list of installed nucleotide HMM databases.')
    parser_nucsearch.add_argument('--outdir', required=True, help='Output directory for results.')
    parser_nucsearch.add_argument('--threads', type=int, default=1, help='Number of threads. Default=1.')
    parser_nucsearch.add_argument("--evalue", type=str, default=None, help="E-value threshold.")
    parser_nucsearch.add_argument("--bitscore", type=str, default=None, help="Bitscore threshold.")
    parser_nucsearch.add_argument("--cut_ga", action="store_true", default=False, help="Use built-in GA thresholds.")
    parser_nucsearch.add_argument("--cut_nc", action="store_true", default=False, help="Use built-in NC thresholds.")
    parser_nucsearch.add_argument("--cut_tc", action="store_true", default=False, help="Use built-in TC thresholds.")

    # Astra phmmer sub-command
    parser_phmmer = subparsers.add_parser('phmmer', help='Performs phmmer search.')
    parser_phmmer.add_argument('--query_seqs', required=True, help='Query sequences for jackhmmer. [FASTA]')
    parser_phmmer.add_argument('--subject_seqs', required=True, help='Subject sequences (database) for jackhmmer. [FASTA]')
    parser_phmmer.add_argument('--threads', type=int, default=1, help='Number of threads to use. Default=1.')
    parser_phmmer.add_argument('--outdir', required=True, help='Output directory for results.')
    # ... other arguments for phmmer

    # Astra jackhmmer sub-command
    parser_jackhmmer = subparsers.add_parser('jackhmmer', help='Performs jackhmmer search. Accepts either nucleotide or amino acid input, but the query and subject sequence alphabets must match.')
    parser_jackhmmer.add_argument('--query_seqs', required=True, help='Query sequences for jackhmmer. [FASTA]')
    parser_jackhmmer.add_argument('--subject_seqs', required=True, help='Subject sequences (database) for jackhmmer. [FASTA]')
    parser_jackhmmer.add_argument('--threads', type=int, default=1, help='Number of threads to use. Default=1.')
    parser_jackhmmer.add_argument('--outdir', required=True, help='Output directory for results.')


    args = parser.parse_args()

    if args.command == 'search':
        search.main(args)
    elif args.command == 'scan':
        scan.main(args)
    elif args.command == 'initialize':
        initialize.main(args)
    elif args.command == 'nucsearch':
        nucsearch.main(args)
    elif args.command == 'phmmer':
        phmmer.main(args)
    elif args.command == 'jackhmmer':
        jackhmmer.main(args)
    else:
        print("Invalid command. Use -h or --help for guidance.")

if __name__ == '__main__':
    main()
