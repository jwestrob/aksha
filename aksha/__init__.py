# aksha/__init__.py

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Aksha: A suite of sequence analysis tools.')
    parser.add_argument('command', choices=['search', 'initialize', 'nucsearch', 'phmmer', 'jackhmmer'],
                        help='Subcommand to run')

    # Parse only the first argument to determine the command
    args, remaining_args = parser.parse_known_args()

    if args.command == 'search':
        from . import search
        search.main(remaining_args)
    elif args.command == 'initialize':
        from . import initialize
        initialize.main(remaining_args)
    elif args.command == 'nucsearch':
        from . import nucsearch
        nucsearch.main(remaining_args)
    elif args.command == 'phmmer':
        from . import phmmer
        phmmer.main(remaining_args)
    elif args.command == 'jackhmmer':
        from . import jackhmmer
        jackhmmer.main(remaining_args)
