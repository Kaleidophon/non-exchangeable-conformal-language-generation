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
MARKERS_AND_COLORS = {
    "nucleus_sampling": ("o", 8, "firebrick"),
    "conformal_nucleus_sampling": ("^", 10, "forestgreen"),
    "non_exchangeable_conformal_nucleus_sampling": ("*", 12, "royalblue"),
}
MAX_SET_SIZES = {
    "openwebtext": 50272,
    "deen": 128112,
    "jaen": 128112,
}
PLOT_TITLES = {
    "all_coverage": "Coverage",
    "all_set_sizes": "Set Size as \% of Vocabulary",
    "all_q_hats": r"$\hat{q}$",
}
METHOD_LABELS = {
    "nucleus_sampling": "Nucleus Sampling",
    "conformal_nucleus_sampling": "Conformal Nucleus Sampling",
    "non_exchangeable_conformal_nucleus_sampling": "Non-Exchangeable Conformal Nucleus Sampling",
}


def plot_coverage_results(
    coverage_result_files: List[str],
    generation_result_files: List[str],
    save_path: Optional[str] = None
):
    results = {}

    # Load result files:
    for file_path in coverage_result_files:
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
            results[method]["all_q_hats"][noise] = np.mean(results[method]["all_q_hats"][noise])

    # Load generation results
    perf_key = "mauve" if dataset == "openwebtext" else "bleu"
    for file_path in generation_result_files:
        dataset, method = re.compile(
            r".+?/\d{2}-\d{2}-\d{4}_\(\d{2}:\d{2}:\d{2}\)_(\w+?)_(\w+_?)+_adaptive_.*\.pkl"
        ).match(file_path).groups()

        with open(file_path, "rb") as file:
            generation_results = dill.load(file)
            results[method]["all_generation_results"] = {
                key: val_dict[perf_key]
                for key, val_dict in generation_results.items()
                if key != "method"
            }

    # Plot results
    plt.rcParams['text.usetex'] = True
    fig, axes = plt.subplots(1, 4, figsize=(10, 2))
    noises = list(results[list(results.keys())[0]]["all_coverage"].keys())
    num_noises = len(noises)
    noise_labels = [
        "None" if noise is None else str(noise[1])
        for noise in noises
    ]

    for key, ax in zip(["all_coverage", "all_set_sizes", "all_q_hats", "all_generation_results"], axes):
        ax.grid(axis="both", which="major", linestyle=":", color="grey")

        if key == "all_generation_results":
            title = perf_key.upper()

        else:
            title = PLOT_TITLES[key]

        ax.set_title(title, fontsize=12)
        x = np.linspace(1, num_noises, num_noises)

        if key == "all_coverage":
            ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.8)

        for method in results:
            marker, size, color = MARKERS_AND_COLORS[method]

            ax.plot(
                x,
                [results[method][key][noise] for noise in noises],
                marker=marker,
                markersize=size,
                label=METHOD_LABELS[method] if ax == axes[-1] else None,
                alpha=0.75,
                color=color
            )

        ax.set_xticks(x)
        ax.set_xticklabels(noise_labels)

    legend = fig.legend(
        loc="lower center",
        ncol=5,
        fontsize=12,
        bbox_to_anchor=(0.5, -0.14),
        handlelength=0.65,
        columnspacing=0.8
    )

    fig.tight_layout()

    if not save_path:
        plt.show()

    else:
        plt.savefig(
            save_path,
            bbox_extra_artists=(legend,),
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coverage-result-files",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--generation-result-files",
        type=str,
        nargs="+"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None
    )

    args = parser.parse_args()

    plot_coverage_results(args.coverage_result_files, args.generation_result_files, args.save_path)