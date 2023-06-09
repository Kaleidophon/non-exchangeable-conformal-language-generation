"""
Determine the ideal temperature parameter for a model by tuning it on the calibration set.
"""

# STD
import argparse
from datetime import datetime
import os
from typing import Optional, Dict, Tuple, List

# EXT
from codecarbon import OfflineEmissionsTracker
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from transformers import M2M100PreTrainedModel
from tqdm import tqdm
import wandb

# PROJECT
from src.data import load_data
from src.defaults import (
    BATCH_SIZE, DATASETS, MODEL_IDENTIFIER, DATA_DIR, EMISSION_DIR, PROJECT_NAME, RESULT_DIR,
    STEP_SIZE, SEARCH_SPACE, NUM_BATCHES, TEMPERATURE, ALPHA, NUM_ATTEMPTS, NUM_NEIGHBORS, HF_RESOURCES, DATASET_TASKS
)
from src.conformal import ConformalCalibrator
from src.custom_types import Device
from src.datastore import DataStore
from src.utils import shard_model

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


def find_temperature(
    model_identifier: str,
    dataset: str,
    batch_size: int,
    conformity_method: str,
    distance_type: str,
    alpha: float,
    num_attempts: int,
    search_space: Tuple[float, float],
    num_batches: int,
    num_neighbors: int,
    num_centroids: int,
    code_size: int,
    num_probes: int,
    use_quantization: bool,
    device: Device,
    data_dir: str,
    datastore_dir: str,
    ignore_token_ids: Tuple[int] = (1, 2),
    sharding: Optional[List[int]] = None,
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
        tokenizer = model_class.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)

    else:
        tokenizer = tokenizer_class.from_pretrained(model_identifier)

    # Load test data
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        load_splits=("dev", ),
        padding="max_length",
        max_length=256,
        truncation=True
    )
    data_loader = data_loaders["dev"]

    try:
        model_hidden_size = model.config.d_model

    except:
        model_hidden_size = int(model.config.hidden_size / 2)

    # Load calibration data
    data_store = DataStore(
        key_dim=model_hidden_size, value_dim=1,
        distance_type=distance_type,
        num_centroids=num_centroids, code_size=code_size,
        num_probes=num_probes, use_quantization=use_quantization,
        device=device
    )  # Create empty data store
    data_store.load(datastore_dir)  # Load in contents

    min_temp, max_temp = search_space
    best_temperature = -1
    best_coverage = np.inf
    min_error = np.inf

    temperature = float(np.random.uniform(*search_space, size=1)[0])

    for attempt in range(num_attempts):

        with torch.no_grad():

            # Init conformal calibrator
            calibrator = ConformalCalibrator(
                alpha=alpha, temperature=temperature, distance_type=distance_type, device=device
            )

            coverage = []

            for i, batch in tqdm(enumerate(data_loader), total=num_batches):

                if i > num_batches - 1:
                    break

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
                    forward_kwargs["decoder_inputs_ids"] = decoder_input_ids

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
                    hidden_states /= model.config.d_model ** 0.25

                elif distance_type == "cosine":
                    hidden_states = F.normalize(hidden_states, p=2, dim=-1)

                predictions = predictions[mask]
                labels = labels[mask]

                # Run the non-exchangeable conformal prediction
                distances, conformity_scores = [], []

                # This can be hard on memory so we do it in batches
                bbatch_size = 1  # TODO: Debug batch_size
                for i in range(0, len(hidden_states), bbatch_size):

                    batch_distances, batch_conformity_scores = data_store.search_k(
                        hidden_states[i:i+bbatch_size, :], k=num_neighbors
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

        print(f"Attempt: {attempt+1} | Temperature: {temperature:.4f}, Coverage: {coverage:.4f}, Error: {error:.4f}")

        if error < min_error:
            min_error = error
            best_coverage = coverage
            best_temperature = temperature

        elif error == min_error and temperature < best_temperature:
            best_temperature = temperature

        temperature = temperature + STEP_SIZE * float(np.sign(error) * np.random.randn() * (max_temp - min_temp))

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
        "--distance-type",
        type=str,
        default="inner_product",
        choices=("inner_product", "l2", "cosine")
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
        "--num-batches",
        type=int,
        default=NUM_BATCHES
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
    parser.add_argument(
        "--sharding",
        type=int,
        nargs="+",
        default=None
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
            log_level="error"
        )
        tracker.start()

    try:
        # Run experiments
        find_temperature(
            model_identifier=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            conformity_method=args.conformity_method,
            distance_type=args.distance_type,
            alpha=args.alpha,
            num_attempts=args.num_attempts,
            search_space=args.search_space,
            num_batches=args.num_batches,
            num_neighbors=args.num_neighbors,
            use_quantization=args.use_quantization,
            num_centroids=args.num_centroids,
            code_size=args.code_size,
            num_probes=args.num_probes,
            datastore_dir=args.datastore_dir,
            device=args.device,
            data_dir=args.data_dir,
            sharding=args.sharding,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e

    finally:
        if tracker is not None:
            tracker.stop()