"""
Plot the results of the coverage experiments.
"""

# STD
import argparse
from typing import List

# EXT
import dill
import matplotlib.pyplot as plt
import numpy as np


def plot_coverage_results(result_files: List[str], save_path: str):
    """
    Plot the results of the coverage experiments.

    Parameters
    ----------
    result_files: List[str]
        List of paths to the result files.
    output_file: str
        Path to the output file.
    """
    # Load results
    results = []
    for result_file in result_files:
        with open(result_file, "rb") as f:
            results.append(dill.load(f))

    a = 3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-files",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None
    )
    args = parser.parse_args()

    plot_coverage_results(args.result_files, args.save_path)
