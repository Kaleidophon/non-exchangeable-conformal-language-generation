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

        if plot_result == "coverage_percentage":
            plot_bar_chart(data, x_label=plot_by, y_label="Coverage", img_path=f"{save_path}/coverage.pdf")
            continue

        elif plot_result == "coverage":
            set_sizes = {
                key: value["all_set_sizes"] for key, value in results.items()
            }

            for key in set_sizes:
                plot_conditional_converage(
                    data[key], set_sizes[key],
                    x_label="Set Size", y_label="Coverage", img_path=f"{save_path}/conditional_coverage_{key}.pdf"
                )
            continue

        # Replace infs for quantiles
        if plot_result == "all_q_hats":
            data = {key: np.where(np.isinf(value), 1.01, value) for key, value in data.items()}

        # Plot
        plot_histogram(
            data, xlabel=LABEL_MAPPING[plot_result], img_path=f"{save_path}/{plot_result}.pdf"
        )


def plot_histogram(data: Dict[str, List[int]], xlabel: str, num_bins: int = 75, img_path: Optional[str] = None):
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


def plot_conditional_converage(coverage, set_sizes, x_label, y_label, num_bins=75, max_set_size: Optional[int] = None, img_path=None):
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.grid(axis="both", which="major", linestyle=":", color="grey")

    if max_set_size is None:
        max_set_size = max(set_sizes)

    step = max_set_size / num_bins
    bins = np.arange(1, max_set_size + step, step)

    max_bin_size = 0
    ax2.set_ylim(0, 5000)  # TODO: Set automatically

    bin_indices = np.digitize(set_sizes, bins, right=True)

    # Plot coverage per bin
    bin_coverages = [
        np.mean(np.array(coverage)[bin_indices == i]) for i in range(1, len(bins))
    ]
    ax1.plot(
        bins[1:], bin_coverages,
        label="Conditional Coverage", linestyle="--", marker="o", alpha=0.6, markersize=5, linewidth=2
    )

    # Plot number of points ber bin
    bin_sizes = [
        np.sum((bin_indices == i).astype(int)) for i in range(1, len(bins))
    ]
    ax2.bar(
        bins[1:], bin_sizes,
        alpha=0.35, label="Number of Points", width=1200, align='center', color="indianred"
    )

    if max(bin_sizes) > max_bin_size:
        max_bin_size = max(bin_sizes)


    # Set max bin size for secondary y axis

    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Coverage")
    ax2.set_ylabel("Number of Points")
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
