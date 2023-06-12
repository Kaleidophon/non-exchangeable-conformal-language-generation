"""
Plot the results of the coverage experiments.
"""

# STD
import argparse
from functools import reduce
from operator import add
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
        neighbors, temperature, alpha, distance = re.compile(
            r".+?/\d{2}-\d{2}-\d{4}_\(\d{2}:\d{2}:\d{2}\)_\w+?_\w+?_(\d+)_(\d+\.\d+)_(\d\.\d+)(_.+)?\.pkl"
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
                flattened_set_sizes = list(reduce(add, set_sizes[key]))  # Flatten
                flattened_data = list(reduce(add, data[key]))  # Flatten

                expected_coverage_gap = plot_conditional_coverage(
                    flattened_data, flattened_set_sizes, bin_width=1200.0,
                    x_label="Set Size", y_label="Coverage", img_path=f"{save_path}/conditional_coverage_{key}.pdf"
                )

                print(f"{key}: Expected coverage gap: {expected_coverage_gap}")

                # Plot zoomed-in version where we focus on the first 100 set sizes
                filtered_set_sizes, filtered_data = zip(*[
                    (set_size, coverage) for set_size, coverage in zip(flattened_set_sizes, flattened_data)
                    if set_size <= 100
                ])

                plot_conditional_coverage(
                    list(filtered_data), list(filtered_set_sizes), num_bins=100, bin_width=100 / 180,
                    x_label="Set Size", y_label="Coverage", img_path=f"{save_path}/zoomed_conditional_coverage_{key}.pdf"
                )

            continue

        # Replace infs for quantiles
        if plot_result == "all_q_hats":
            data = {key: np.where(np.isinf(value), 1.01, value) for key, value in data.items()}

        # Plot
        if plot_result == "all_set_sizes":
            data = {key: list(reduce(add, value)) for key, value in data.items()}  # Flatten

        plot_histogram(
            data, xlabel=LABEL_MAPPING[plot_result], img_path=f"{save_path}/{plot_result}.pdf"
        )

    # Plot coverage and set sizes over time
    for key in results:
        plot_set_sizes_over_time(results[key]["all_set_sizes"], img_path=f"{save_path}/sizes_over_time_{key}.pdf")
        plot_coverages_over_time(results[key]["coverage"], img_path=f"{save_path}/coverages_over_time_{key}.pdf")


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


def plot_conditional_coverage(
    coverage, set_sizes, x_label, y_label, alpha = 0.1, num_bins=75, bin_width: float = 1200.0,
    max_set_size: Optional[int] = None, img_path=None
):
    fig = plt.figure(figsize=(7, 3.5))
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax1.grid(axis="both", which="major", linestyle=":", color="grey")

    if max_set_size is None:
        max_set_size = max(set_sizes)

    step = max_set_size / num_bins
    bins = np.arange(1, max_set_size + step, step)

    max_bin_size = 0

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
        alpha=0.45, label="Number of Points", width=bin_width, align='center', color="indianred"
    )

    if max(bin_sizes) > max_bin_size:
        max_bin_size = max(bin_sizes)

    ax2.set_ylim(0, max_bin_size * 1.1)  # TODO: Set automatically

    # Set max bin size for secondary y axis

    ax1.set_xlabel(x_label, fontsize=14)
    ax1.set_ylabel("Coverage", fontsize=14)
    ax2.set_ylabel("Number of Points", fontsize=14)

    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.tick_params(axis='both', which='minor', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='minor', labelsize=12)

    plt.tight_layout()

    if img_path is not None:
        plt.savefig(img_path, dpi=300)

    else:
        plt.show()

    # Compute expected coverage gap
    num_points = sum(bin_sizes)
    bin_coverages = np.array(bin_coverages)
    bin_coverages[np.isnan(bin_coverages)] = 0
    expected_coverage_gap = np.sum(bin_sizes / num_points * np.abs(1 - alpha - np.array(bin_coverages)))

    return expected_coverage_gap


def plot_set_sizes_over_time(set_sizes, img_path: Optional[str] = None, cutoff: int = 200):

    max_time_step = max([len(s) for s in set_sizes])

    x = np.arange(1, max_time_step + 1)

    # Put data into a nicer format
    set_sizes_matrix = np.zeros((len(set_sizes), max_time_step))

    for i, s in enumerate(set_sizes):
        set_sizes_matrix[i, :len(s)] = s

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax1.grid(axis="both", which="major", linestyle=":", color="grey")

    non_zero_entries = np.sum((set_sizes_matrix != 0).astype(int), axis=0)
    means = np.sum(set_sizes_matrix, axis=0) / non_zero_entries
    std_devs = np.sqrt(np.sum((set_sizes_matrix - means[None, :])**2, axis=0) / non_zero_entries)

    # Apply cut-off
    means = means[:cutoff]
    std_devs = std_devs[:cutoff]
    x = x[:cutoff]

    ax1.plot(x, means, label="Mean Set Size", linestyle="-", alpha=0.5, linewidth=2)
    ax1.fill_between(x, np.max(means - std_devs, 0), means + std_devs, alpha=0.15, color="blue")

    if img_path is not None:
        plt.savefig(img_path, dpi=300)

    else:
        plt.show()


def plot_coverages_over_time(coverage, img_path: Optional[str] = None, cutoff: int = 200):

    max_time_step = max([len(s) for s in coverage])

    x = np.arange(1, max_time_step + 1)

    # Put data into a nicer format
    coverage_matrix = np.zeros((len(coverage), max_time_step)) - 1

    for i, s in enumerate(coverage):
        coverage_matrix[i, :len(s)] = s

    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.gca()
    ax1.grid(axis="both", which="major", linestyle=":", color="grey")

    non_negative_entries = np.sum((coverage_matrix != -1).astype(int), axis=0)
    coverage_matrix[coverage_matrix == -1] = 0
    means = np.sum(coverage_matrix, axis=0) / non_negative_entries

    # Apply cut-off
    means = means[:cutoff]
    x = x[:cutoff]

    ax1.plot(x, means, label="Mean Coverage", linestyle="-", alpha=0.5, linewidth=2)
    #ax1.fill_between(x, np.max(means - std_devs, 0), means + std_devs, alpha=0.15, color="blue")

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
