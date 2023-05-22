"""
Plot the results of the coverage experiments.
"""

# STD
import argparse
import re
from typing import List, Iterable, Dict, Optional

# EXT
import dill
import matplotlib.pyplot as plt
import numpy as np

# CONST
LABEL_MAPPING = {
    "avg_distances": "Average distances",
    "avg_weights": "Average weights",
    "avg_conformity_scores": "Average conformity scores",
    "all_n_effs": "Effective sample sizes",
    "all_q_hats": "Calibrated quantiles",
    "all_set_sizes": "Prediction set sizes",
}  # Map results to x-labels for nicer plots


def plot_coverage_results(
    result_files: List[str], plot_by: str, save_path: str,
    plot_results: Iterable[str] = (
        "coverage", "avg_distances", "avg_weights", "avg_conformity_scores", "all_n_effs", "all_q_hats",
        "all_set_sizes"
    )
):
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
    results = {}
    for result_file in result_files:

        # Parse parameters from file name
        neighbors, temperature, alpha = re.compile(
            r".+?/\d{2}-\d{2}-\d{4}_\(\d{2}:\d{2}:\d{2}\)_\w{4}_\w+?_(\d+)_(\d+\.\d+)_(\d\.\d+)\.pkl"
        ).match(result_file).groups()

        key_map = {
            "neighbors": int(neighbors),
            "temperature": float(temperature),
            "alpha": float(alpha)
        }

        with open(result_file, "rb") as f:
            results[key_map[plot_by]] = dill.load(f)

    for plot_result in plot_results:

        # Extract data
        data = {key: value[plot_result] for key, value in results.items()}

        if plot_result == "coverage":
            plot_bar_chart(data, x_label=plot_by, y_label="Coverage", img_path=f"{save_path}/coverage.pdf")
            continue

        # Replace infs for quantiles
        if plot_result == "all_q_hats":
            data = {key: np.where(np.isinf(value), 1.01, value) for key, value in data.items()}

        # Plot
        plot_histogram(
            data, xlabel=LABEL_MAPPING[plot_result], img_path=f"{save_path}/{plot_result}.pdf"
        )


def plot_histogram(data: Dict[str, List[int]], xlabel: str, num_bins: int = 20, img_path: Optional[str] = None):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.grid(axis="both", which="major", linestyle=":", color="grey")

    for name, d in data.items():
        plt.hist(d, num_bins, alpha=0.5, label=name)

    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()

    if img_path is not None:
        plt.savefig(img_path, dpi=300)

    else:
        plt.show()

    plt.close()


def plot_bar_chart(data, x_label, y_label, img_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.grid(axis="both", which="major", linestyle=":", color="grey")

    plt.bar(range(len(data)), list(data.values()), alpha=0.55,
            align='center', width=0.5)

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(list(data.keys()))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    if img_path is not None:
        plt.savefig(img_path, dpi=300)

    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-files",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--plot-by",
        type=str,
        choices=["neighbors", "alpha", "temperature"]
    )
    args = parser.parse_args()

    plot_coverage_results(args.result_files, args.plot_by, args.save_path)
