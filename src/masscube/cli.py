# cli module for masscube

import argparse
from . import untargeted_metabolomics_workflow, generate_sample_table

# Modify the main function to accommodate multiple commands
def main():
    parser = argparse.ArgumentParser(description="MassCube CLI Tools")
    subparsers = parser.add_subparsers(help='commands')

    # untargeted metabolomics parser
    parser_um = subparsers.add_parser('untargeted-metabolomics', help='Run untargeted metabolomics')
    parser_um.add_argument('--path', help='Path to the project directory')
    parser_um.set_defaults(func=untargeted_metabolomics_workflow)

    # generate sample table parser
    parser_gst = subparsers.add_parser('generate-sample-table', help='Generate sample table')
    parser_gst.add_argument('--path', help='Path to the project directory')
    parser_gst.set_defaults(func=generate_sample_table)

    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()

