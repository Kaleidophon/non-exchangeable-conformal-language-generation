"""
Run an ablation study that investigates the impact of the number of neighbors.
"""

# STD
import argparse
from datetime import datetime
import os
from typing import Tuple, Optional, List

# EXT
from codecarbon import OfflineEmissionsTracker
import dill
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import M2M100PreTrainedModel

# PROJECT
from src.custom_types import Device
from src.conformal import ConformalCalibrator
from src.data import load_data
from src.datastore import DataStore
from src.defaults import (
    BATCH_SIZE, DATASETS, MODEL_IDENTIFIER, DATA_DIR, EMISSION_DIR, PROJECT_NAME, RESULT_DIR,
    TEMPERATURE, NUM_NEIGHBORS, MODEL_HIDDEN_SIZES, SEED, HF_RESOURCES, DATASET_TASKS, ALPHA
)
from src.utils import shard_model

# GLOBALS
SECRET_IMPORTED = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Knockknock support
try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass


def run_alpha_ablation_study(
    model_identifier: str,
    neighbor_nums: List[float],
    dataset: str,
    batch_size: int,
    conformity_method: str,
    distance_type: str,
    temperature: float,
    alpha: float,
    datastore_dir: str,
    num_centroids: int,
    code_size: int,
    num_probes: int,
    use_quantization: bool,
    device: Device,
    data_dir: str,
    result_dir: str,
    ignore_token_ids: Tuple[int, ...] = (1, 2),
    sharding: Optional[List[int]] = None,
):
    """
    Parameters
    ----------
    model_identifier: str
        Model to be used for the experiments. If model_path is specified, the model will be loaded from there.
    neighbor_nums: List[float]
        Number of neigbors to test.
    dataset: str
        Dataset to be used for the experiments.
    batch_size: int
        Batch size to be used for the experiments.
    conformity_method: Optional[str]
        Type of non-conformity score to be used. Has to be either "simple" or "adaptive".
    distance_type: str
        Type of distance to be used for the experiments.
    temperature: float
        Temperature used for the datastore retrieval.
    alpha: float
        Used to set the 1 - alpha desired confidence level for conformal methods. Default is None.
    datastore_dir: str
        Directory the datastore is stored in.
    num_centroids : int
        Number of centroids to use for clustering during the quantization process.
    code_size : int
        Number of bytes in quantized codes.
    num_probes: int
        Number of coarse-level quantizers.
    use_quantization: bool
        Flag to indicate whether quantization should be used.
    device: Device
        Device to be used for the experiments.
    data_dir: str
        Path to the data directory.
    result_dir: str
        Path to the directory where the results should be saved.
    ignore_token_ids: Tuple[int, ...]
        Token IDs tio ignore during experiments.
    sharding: Optional[List[int]]
        Indices of GPUs a shared model should be distributed across. Defaults to None, in which case no sharding is used.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the results of the experiments.
    """
    model_class, config_class, tokenizer_class = HF_RESOURCES[model_identifier]

    # Initialize model
    if sharding is None:
        model = model_class.from_pretrained(model_identifier).to(device)

    # Shard models onto different GPUs
    else:
        model = shard_model(model_identifier, sharding, model_class=model_class, config_class=config_class).to(device)

    model.eval()
    task = DATASET_TASKS[dataset]

    if task == "mt":
        src_lang, tgt_lang = DATASETS[dataset]
        tokenizer = tokenizer_class.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)

    else:
        tokenizer = tokenizer_class.from_pretrained(model_identifier)

    # Load test data
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        load_splits=("test",),
        padding="max_length",
        max_length=256,
        truncation=True
    )
    data_loader = data_loaders["test"]
    model_hidden_size = MODEL_HIDDEN_SIZES[model.config.name_or_path]

    # Load calibration data
    # Init conformal calibrator
    for num_neighbors in neighbor_nums:
        calibrator = ConformalCalibrator(
            alpha=alpha, temperature=temperature, device=device
        )

        data_store = DataStore(
            key_dim=model_hidden_size, value_dim=1,
            distance_type=distance_type,
            num_centroids=num_centroids, code_size=code_size,
            num_probes=num_probes, use_quantization=use_quantization,
            device=device
        )  # Create empty data store
        data_store.load(datastore_dir)  # Load in contents

        # Use calibration data store to test coverage on test set
        # Also collect the following statistics and save them with dill to plot later:
        # - Prediction set sizes
        # - Distances and weights
        # - Conformity scores
        # - Found quantiles q hat
        # - The effective sample size
        all_set_sizes = []
        coverage = []
        avg_distances = []
        avg_weights = []
        avg_conformity_scores = []
        all_n_effs = []
        all_q_hats = []

        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
                # Get input and target
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                decoder_input_ids = batch["decoder_input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)

                forward_kwargs = {
                    "output_hidden_states": True,
                    "return_dict": True,
                }

                if isinstance(model, M2M100PreTrainedModel):
                    forward_kwargs["decoder_input_ids"] = decoder_input_ids

                # Generate outputs
                with torch.no_grad():
                    outputs = model.forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        **forward_kwargs
                    )

                if isinstance(model, M2M100PreTrainedModel):
                    hidden_states = outputs.decoder_hidden_states[-1]

                else:
                    hidden_states = outputs.hidden_states[-1]

                predictions = F.softmax(outputs.logits, dim=-1)

                # Reshape and filter out ignore tokens
                hidden_states = rearrange(hidden_states, "b s h -> (b s) h")
                predictions = rearrange(predictions, "b s c -> (b s) c")
                input_ids = rearrange(input_ids, "b s -> (b s)")
                labels = rearrange(labels, "b s -> (b s)")
                mask = torch.all(
                    torch.stack([input_ids != ignore_id for ignore_id in ignore_token_ids], dim=0), dim=0
                ).to(device)
                hidden_states = hidden_states[mask]

                if distance_type == "inner_product":
                    hidden_states /= model_hidden_size ** 0.25

                elif distance_type == "cosine":
                    hidden_states = F.normalize(hidden_states, p=2, dim=-1)

                if len(hidden_states) == 0:
                    continue

                predictions = predictions[mask]
                labels = labels[mask]

                # Run the non-exchangeable conformal prediction
                distances, conformity_scores = [], []

                # This can be hard on memory so we do it in batches
                bbatch_size = 1  # TODO: Debug batch_size
                for i in range(0, len(hidden_states), bbatch_size):
                    batch_distances, batch_conformity_scores = data_store.search_k(
                        hidden_states[i:i + bbatch_size, :], k=num_neighbors
                    )
                    distances.append(batch_distances)
                    conformity_scores.append(batch_conformity_scores)

                distances = torch.cat(distances, dim=0)
                conformity_scores = torch.cat(conformity_scores, dim=0).squeeze(-1)

                weights = calibrator.compute_weights(distances)
                conformal_results = calibrator.compute_q_hat(
                    weights, conformity_scores
                )
                q_hat = conformal_results["q_hat"]
                prediction_sets, set_sizes = calibrator.get_prediction_sets(conformity_method, predictions, q_hat)

                all_set_sizes.append(list(set_sizes))

                # Evaluate
                label_probs = prediction_sets.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                is_covered = list((label_probs > 0).float().cpu().numpy())
                coverage.append(is_covered)

                # Add results for this batch
                avg_distances += list(distances.mean(dim=-1).cpu().numpy())
                avg_weights += list(weights.mean(dim=-1).cpu().numpy())
                avg_conformity_scores += list(conformity_scores.mean(dim=-1).cpu().numpy())
                all_n_effs += list(conformal_results["n_eff"].cpu().numpy())
                all_q_hats += list(q_hat.cpu().numpy())

            # Save results
            flattened_coverage = [cov for seq_coverage in coverage for cov in seq_coverage]
            flattened_set_sizes = [size for seq_set_sizes in all_set_sizes for size in seq_set_sizes]
            coverage_percentage = np.mean(flattened_coverage)
            print(f"Coverage: {coverage_percentage:.4f}")
            compute_results(
                coverage=flattened_coverage, set_sizes=flattened_set_sizes, alpha=alpha,
                max_set_size=predictions.shape[-1]
            )

            results = {
                "coverage": coverage,
                "coverage_percentage": coverage_percentage,
                "all_set_sizes": all_set_sizes,
                "avg_distances": avg_distances,
                "avg_weights": avg_weights,
                "avg_conformity_scores": avg_conformity_scores,
                "all_n_effs": all_n_effs,
                "all_q_hats": all_q_hats,
            }

            timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))
            file_name = f"ablation_{num_neighbors}_{timestamp}_{dataset}_{conformity_method}_{alpha}_{temperature}_{distance_type}.pkl"

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            with open(os.path.join(result_dir, file_name), "wb") as result_file:
                dill.dump(results, result_file)


def compute_results(coverage, set_sizes, alpha = 0.1, num_bins = 75, max_set_size = None):

    if max_set_size is None:
        max_set_size = max(set_sizes)

    step = max_set_size / num_bins
    bins = np.arange(1, max_set_size + step, step)

    bin_indices = np.digitize(set_sizes, bins, right=True)

    bin_coverages = [
        np.mean(np.array(coverage)[bin_indices == i]) for i in range(1, len(bins))
    ]

    # Plot number of points ber bin
    bin_sizes = [
        np.sum((bin_indices == i).astype(int)) for i in range(1, len(bins))
    ]
    # Compute average set size
    print("Average set size:", np.mean(set_sizes))
    print("Average set (%):", np.mean(set_sizes) / max_set_size)

    # Compute expected coverage gap
    num_points = sum(bin_sizes)
    bin_coverages = np.array(bin_coverages)
    bin_sizes = np.array(bin_sizes)
    mask = ~np.isnan(bin_coverages)
    bin_coverages = bin_coverages[mask]
    bin_sizes = bin_sizes[mask]
    cmp = np.zeros(len(bin_coverages))
    gaps = 1 - alpha - np.array(bin_coverages)
    expected_coverage_gap = np.sum(bin_sizes / num_points * np.max(np.stack([cmp, gaps]), 0), axis=0)
    print("Expected coverage gap:", expected_coverage_gap)

    # Compute size-stratified coverage
    ssc = np.min(bin_coverages, axis=0)
    print("Size-stratified coverage:", ssc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--neighbor-nums",
        type=int,
        nargs="+",
        default=[10, 25, 50, 75, 100, 200, 300, 500]
    )
    parser.add_argument(
        "--save-dir",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=tuple(DATASETS.keys())
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA
    )
    parser.add_argument(
        "--model-identifier",
        type=str,
        default=MODEL_IDENTIFIER
    )
    parser.add_argument(
        "--datastore-dir",
        type=str,
    )
    parser.add_argument(
        "--distance-type",
        type=str,
        default="inner_product",
        choices=("inner_product", "l2", "cosine")
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--conformity-method",
        type=str,
        choices=("simple", "adaptive"),
        default="adaptive"
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--num-centroids",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=TEMPERATURE
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1
    )
    parser.add_argument(
        "--code-size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=2048
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=NUM_NEIGHBORS
    )
    parser.add_argument(
        "--sharding",
        type=int,
        nargs="+",
        default=None
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    tracker = None

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name=PROJECT_NAME,
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
            log_level="error"
        )
        tracker.start()

    try:
        run_alpha_ablation_study(
            model_identifier=args.model_identifier,
            neighbor_nums=args.neighbor_nums,
            dataset=args.dataset,
            batch_size=args.batch_size,
            conformity_method=args.conformity_method,
            distance_type=args.distance_type,
            temperature=args.temperature,
            alpha=args.alpha,
            datastore_dir=args.datastore_dir,
            use_quantization=args.use_quantization,
            num_centroids=args.num_centroids,
            code_size=args.code_size,
            num_probes=args.num_probes,
            device=args.device,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            sharding=args.sharding
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e

    finally:
        if tracker is not None:
            tracker.stop()
