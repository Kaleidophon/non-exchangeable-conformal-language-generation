"""
Plot results of the shift experiments.
"""

# STD
import argparse
import os
import re
from typing import List, Optional

# EXT
import dill
import matplotlib.pyplot as plt
import numpy as np

# CONST
MARKERS = {
    "nucleus_sampling": ("o", 8),
    "conformal_nucleus_sampling": ("^", 10),
    "non_exchangeable_conformal_nucleus_sampling": ("*", 12),
}
MAX_SET_SIZES = {
    "openwebtext": 50272
}
PLOT_TITLES = {
    "all_coverage": "Coverage",
    "all_set_sizes": "Set Size as % of Vocabulary",
}
METHOD_LABELS = {
    "nucleus_sampling": "Nucleus Sampling",
    "conformal_nucleus_sampling": "Conf. Sampling",
    "non_exchangeable_conformal_nucleus_sampling": "Non-Ex. CS",
}


def plot_coverage_results(
    result_files: List[str],
    save_path: Optional[str] = None
):
    results = {}

    # Load result files:
    for file_path in result_files:
        dataset, method = re.compile(
            r".+?/\d{2}-\d{2}-\d{4}_\(\d{2}:\d{2}:\d{2}\)_(\w+?)_(\w+_?)+_adaptive_.*\.pkl"
        ).match(file_path).groups()

        with open(file_path, "rb") as file:
            results[method] = dill.load(file)

    # Preprocess results
    for method in results:
        for noise in results[method]["all_coverage"]:
            results[method]["all_coverage"][noise] = np.mean(results[method]["all_coverage"][noise])
            results[method]["all_set_sizes"][noise] = np.mean(
                np.array(results[method]["all_set_sizes"][noise]) / MAX_SET_SIZES[dataset]
            )

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(10, 2))
    num_noises = len(results[list(results.keys())[0]])
    noises = list(results[list(results.keys())[0]]["all_coverage"].keys())
    noise_labels = [
        "No noise" if noise is None else str(noise[1])
        for noise in noises
    ]

    for key, ax in zip(["all_coverage", "all_set_sizes"], axes):
        ax.grid(axis="both", which="major", linestyle=":", color="grey")
        ax.set_title(PLOT_TITLES[key], fontsize=12)
        x = np.linspace(1, num_noises, num_noises)

        if key == "all_coverage":
            ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.8)

        for method in results:
            ax.plot(
                x,
                [results[method][key][noise] for noise in noises],
                marker=MARKERS[method][0],
                markersize=MARKERS[method][1],
                label=METHOD_LABELS[method],
                alpha=0.75
            )

        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels)

    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

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

    args = parser.parse_args()

    if args.save_path is not None:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

    plot_coverage_results(args.result_files, args.save_path)