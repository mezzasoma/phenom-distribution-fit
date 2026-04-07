import argparse
from utils.samples_processing import (save_bilby_gaussian_prior_appendix,
                                      save_plot_correlation_matrix)
from utils.path_utils import create_and_set_outdir

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Save prior appendix for bilby."
        )
    )
    parser.add_argument(
        "path_to_run",
        type=str,
        help="Path to the run directory.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    path_to_run = args.path_to_run

    outdir = create_and_set_outdir("bilby_prior")
    save_bilby_gaussian_prior_appendix(path_to_run, outdir)
    save_plot_correlation_matrix(path_to_run, outdir)

if __name__ == "__main__":
    main()