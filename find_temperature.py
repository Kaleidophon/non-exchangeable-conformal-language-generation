"""
Determine the ideal temperature parameter for a model by tuning it on the calibration set.
"""

# STD
import argparse
from datetime import datetime
import os
from typing import Optional, Dict, Tuple

# EXT
from codecarbon import OfflineEmissionsTracker
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import wandb

# PROJECT
from src.data import load_data
from src.conformal import ConformalCalibrator
from src.custom_types import Device
from src.datastore import DataStore

# CONST
DATA_DIR = "./data/wmt22"
MODEL_DIR = "./models/"
RESULT_DIR = "./results"
EMISSION_DIR = "./emissions"
MODEL_IDENTIFIER = "facebook/m2m100_418M"
PROJECT_NAME = "nlg-conformal-risk-control"
# Map available language pairs to language identifiers for tokenizer
DATASETS = {
    "deen": ("de", "en"),
    "jaen": ("ja", "en")
}

# DEFAULTS
SEED = 1234
BATCH_SIZE = 4
ALPHA = 0.1
TEMPERATURE = 1
NUM_NEIGHBORS = 100
NUM_ATTEMPTS = 20
SEARCH_SPACE = (0.1, 10)


# GLOBALS
SECRET_IMPORTED = False

# Knockknock support
try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass

# CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def run_experiments(
    model_identifier: str,
    dataset: str,
    batch_size: int,
    conformity_method: str,
    alpha: float,
    num_attempts: int,
    search_space: Tuple[float, float],
    num_neighbors: int,
    num_centroids: int,
    code_size: int,
    num_probes: int,
    use_quantization: bool,
    device: Device,
    data_dir: str,
    result_dir: str,
    datastore_dir: str,
    ignore_token_ids: Tuple[int] = (1, 2),  # TODO: Double-check this default
):
    """
    Run experiments for conformal risk control in NLG.

    Parameters
    ----------
    model_identifier: str
        Model to be used for the experiments. If model_path is specified, the model will be loaded from there.
    dataset: str
        Dataset to be used for the experiments.
    batch_size: int
        Batch size to be used for the experiments.
    alpha: float
        Pre-defined confidence level for the experiments.
    device: Device
        Device to be used for the experiments.
    data_dir: str
        Path to the data directory.
    result_dir: str
        Path to the directory where the results should be saved.


    Returns
    -------
    Dict[str, float]
        Dictionary containing the results of the experiments.
    """
    # Load or init model
    model = M2M100ForConditionalGeneration.from_pretrained(model_identifier).to(device)
    model.eval()
    tokenizer = M2M100Tokenizer.from_pretrained(model_identifier)

    # Load test data
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        load_splits=("dev", ),
        padding="max_length",
        max_length=256,
        truncation=True
    )
    data_loader = data_loaders["dev"]

    # Load calibration data
    data_store = DataStore(
        key_dim=model.config.d_model, value_dim=1,
        num_centroids=num_centroids, code_size=code_size,
        num_probes=num_probes, use_quantization=use_quantization,
        device=device
    )  # Create empty data store
    data_store.load(datastore_dir)  # Load in contents

    min_temp, max_temp = search_space
    best_temperature = -1
    best_coverage = np.inf
    min_error = np.inf

    for attempt in range(num_attempts):

        temperature = (min_temp + max_temp) / 2

        # Init conformal calibrator
        calibrator = ConformalCalibrator(
            data_store,
            alpha=alpha, temperature=temperature, device=device
        )

        with torch.no_grad():
            coverage = []

            for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

                # TODO: Fast debug
                if i > 5:
                    break

                # Get input and target
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                decoder_input_ids = batch["decoder_input_ids"].to(model.device)
                labels = batch["labels"].to(model.device)

                # Generate outputs
                outputs = model.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    decoder_input_ids=decoder_input_ids,
                    return_dict=True,
                )
                decoder_states = outputs.decoder_hidden_states[-1]
                predictions = F.softmax(outputs.logits, dim=-1)

                # Reshape and filter out ignore tokens
                decoder_states = rearrange(decoder_states, "b s h -> (b s) h")
                predictions = rearrange(predictions, "b s c -> (b s) c")
                input_ids = rearrange(input_ids, "b s -> (b s)")
                labels = rearrange(labels, "b s -> (b s)")
                mask = torch.all(
                    torch.stack([input_ids != ignore_id for ignore_id in ignore_token_ids], dim=0), dim=0
                ).to(device)
                decoder_states = decoder_states[mask] / model.config.d_model ** 0.25
                predictions = predictions[mask]
                labels = labels[mask]

                # This can be hard on memory so we do it in batches
                bbatch_size = 1  # TODO: Debug batch_size
                for i in range(0, len(decoder_states), bbatch_size):

                    batch_distances, batch_conformity_scores = data_store.search_k(
                        decoder_states[i:i+bbatch_size, :], k=num_neighbors
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

                # Evaluate
                label_probs = prediction_sets.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                is_covered = list((label_probs > 0).float().cpu().numpy())
                coverage += is_covered

        # Compute coverage and adjust temperature
        coverage = np.mean(coverage)
        error = 1 - alpha - coverage
        abs_error = np.abs(error)

        print(f"Attempt: {i+1} | Temperature: {temperature:.4f}, Coverage: {coverage:.4f}, Error: {error:.4f}")

        if error < min_error:
            min_error = error
            best_coverage = coverage
            best_temperature = temperature

        # Coverage is too small, decrease temperature
        if error > 0:
            max_temp = temperature

        # Coverage is too large, increase temperature
        else:
            min_temp = temperature

    print(f"Best temperature after {num_attempts}: {best_temperature:.4f} with coverage {best_coverage:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_IDENTIFIER
    )
    parser.add_argument(
        "--datastore-dir",
        type=str,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=tuple(DATASETS.keys())
    )
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=NUM_ATTEMPTS
    )
    parser.add_argument(
        "--search-space",
        type=float,
        nargs=2,
        default=SEARCH_SPACE
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
        "--temperature",
        type=float,
        default=TEMPERATURE
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=NUM_NEIGHBORS
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=ALPHA
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
        "--code-size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=2048
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--knock", action="store_true", default=False)
    parser.add_argument("--wandb", action="store_true", default=False)
    args = parser.parse_args()

    tracker = None
    wandb_run = None

    if args.wandb:
        wandb_run = wandb.init(
            project=PROJECT_NAME,
            tags=[args.dataset, args.model],
            settings=wandb.Settings(start_method="fork"),
            group=args.dataset
        )

    if args.track_emissions:
        timestamp = str(datetime.now().strftime("%d-%m-%Y (%H:%M:%S)"))
        emissions_path = os.path.join(args.emission_dir, timestamp)
        os.mkdir(emissions_path)
        tracker = OfflineEmissionsTracker(
            project_name=PROJECT_NAME,
            country_iso_code=COUNTRY_CODE,
            output_dir=emissions_path,
        )
        tracker.start()

    try:
        # Run experiments
        run_experiments(
            model_identifier=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            conformity_method=args.conformity_method,
            alpha=args.alpha,
            temperature=args.temperature,
            num_neighbors=args.num_neighbors,
            use_quantization=args.use_quantization,
            num_centroids=args.num_centroids,
            code_size=args.code_size,
            num_probes=args.num_probes,
            datastore_dir=args.datastore_dir,
            device=args.device,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e

    finally:
        if tracker is not None:
            tracker.stop()