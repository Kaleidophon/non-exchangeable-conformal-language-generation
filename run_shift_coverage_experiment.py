"""
Perform hallucinations experiment, where we compare we first let the model generate freely, and then feed the same tokens
back into the decoder, but restricting the attention on the source side. We then compare the two set sizes.
"""

# STD
import argparse
from collections import defaultdict
from datetime import datetime
import os
from typing import Optional, Tuple, List

# EXT
from codecarbon import OfflineEmissionsTracker
import dill
from einops import rearrange
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import M2M100PreTrainedModel, OPTPreTrainedModel

# PROJECT
from src.conformal import ConformalCalibrator, ConformalLogitProcessor
from src.custom_types import Device
from src.data import load_data
from src.datastore import DataStore
from src.defaults import (
    BATCH_SIZE, SEQUENCE_LENGTH, SEED, DATASETS, MODEL_IDENTIFIER, DATA_DIR, EMISSION_DIR, PROJECT_NAME,
    ALPHA, TEMPERATURE, NUM_NEIGHBORS, RESULT_DIR, MODEL_HIDDEN_SIZES, HF_RESOURCES, DATASET_TASKS
)
from src.utils import shard_model

# GLOBALS
SECRET_IMPORTED = False

# Knockknock support
try:
    from secret import COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass


def perform_shift_experiment(
    model_identifier: str,
    dataset: str,
    batch_size: int,
    method: str,
    temperature: float,
    data_dir: str,
    result_dir: str,
    # Arguments for conformal sampling methods
    alpha: Optional[float] = None,
    data_store: Optional[DataStore] = None,
    distance_type: Optional[str] = None,
    conformity_score: Optional[str] = None,
    num_neighbors: Optional[int] = None,
    # Other arguments
    seed: int = SEED,
    device: Device = "cpu",
    ignore_token_ids: Tuple[int] = (1, 2),
    sharding: Optional[List[int]] = None,
):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Load data and model
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

    data_loader = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=("test",)
    )["test"]
    model_hidden_size = MODEL_HIDDEN_SIZES[model.config.name_or_path]

    calibrator = ConformalCalibrator(
        alpha=alpha, temperature=temperature, device=device
    )

    if method in ("conformal_nucleus_sampling", "non_conformal_nucleus_sampling"):
        if method == "conformal_nucleus_sampling":
            logit_processor = ConformalLogitProcessor(
                alpha, conformity_score, data_store, calibrator
            )

    # Define noise parameters
    noise_parameters = [
        None, (0, 0.025), (0, 0.05), (0, 0.075), (0, 0.1)
    ]

    # Define which information to save
    all_set_sizes = defaultdict(list)
    all_coverage = defaultdict(list)
    all_distances = defaultdict(list)
    all_weights = defaultdict(list)
    all_q_hats = defaultdict(list)

    for noise_params in noise_parameters:

        if noise_params is not None and isinstance(model, M2M100PreTrainedModel):

            mean, std = noise_params
            mean = torch.FloatTensor([mean]).to(device)
            std = torch.sqrt(torch.FloatTensor([std]).to(device))

            # Patch functions to add noise
            # For M2M100 models, patch the encoder function
            def m2m100_forward_hook(*args):
                _, _, output = args

                decoder_input = output[0]
                decoder_input = decoder_input + torch.randn(decoder_input.shape).to(device) * std + mean
                output.last_hidden_state = decoder_input

                return output

            model.model.encoder.register_forward_hook(m2m100_forward_hook)

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

            forward_kwargs = {
                "attention_mask": batch["attention_mask"].to(device),
                "output_hidden_states": True,
                "return_dict": True,
            }

            with torch.no_grad():

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                # For OPT model, add noise to embeddings
                if noise_params is not None and isinstance(model, OPTPreTrainedModel):
                    mean, std = noise_params
                    mean = torch.FloatTensor([mean]).to(device)
                    std = torch.FloatTensor([std]).to(device)

                    embeds = model.model.decoder.embed_tokens(input_ids)
                    embeds = embeds + torch.randn(embeds.shape).to(device) * std + mean
                    forward_kwargs["inputs_embeds"] = embeds

                elif isinstance(model, OPTPreTrainedModel):
                    forward_kwargs["input_ids"] = input_ids

                elif isinstance(model, M2M100PreTrainedModel):
                    decoder_token_ids = batch["decoder_input_ids"].to(device)
                    forward_kwargs["decoder_input_ids"] = decoder_token_ids
                    forward_kwargs["input_ids"] = input_ids

                outputs = model.forward(**forward_kwargs)

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

            # Apply one of three different methods here:
            #   1. "Classic" nucleus sampling: Include everything in the prediction set that corresponds to 1 - alpha
            #       cumulative probability mass
            #   2. Conformal nucleus sampling: Include everything in the prediction set according to the q hat
            #       corresponding to the current entropy bin.
            #   3. Non-exchangeable conformal nucleus sampling: Retrieve neighbors and compute q hat based on their
            #       weighted conformity scores.

            # Run the classic nucleus sampling
            if method == "nucleus_sampling":
                top_p = q_hat = torch.FloatTensor([1 - alpha]).repeat(predictions.shape[0])

                prediction_sets, set_sizes = calibrator.get_prediction_sets(
                    conformity_score, predictions, q_hat=top_p
                )

            elif method == "conformal_nucleus_sampling":
                q_hat = logit_processor.get_q_hats(predictions)

                prediction_sets, set_sizes = calibrator.get_prediction_sets(
                    conformity_score, predictions, q_hat=q_hat
                )

            # Run the non-exchangeable conformal prediction
            elif method == "non_exchangeable_conformal_nucleus_sampling":
                set_sizes = []
                q_hat = []
                distances = []
                weights = []
                prediction_sets = []

                # This can be hard on memory so we do it in batches
                for i in range(0, len(hidden_states), 1):
                    batch_distances, batch_conformity_scores = data_store.search_k(
                        hidden_states[i:i + 1, :], k=num_neighbors
                    )

                    batch_weights = calibrator.compute_weights(batch_distances)
                    conformal_results = calibrator.compute_q_hat(
                        batch_weights, batch_conformity_scores
                    )
                    batch_q_hat = conformal_results["q_hat"]
                    batch_prediction_sets, batch_set_sizes = calibrator.get_prediction_sets(
                        conformity_score, predictions[i, :].unsqueeze(0), batch_q_hat
                    )

                    prediction_sets.append(batch_prediction_sets)
                    set_sizes.append(batch_set_sizes)
                    q_hat.append(batch_q_hat)
                    distances.append(batch_distances)
                    weights.append(batch_weights)

                prediction_sets = torch.cat(prediction_sets, dim=0)
                set_sizes = np.concatenate(set_sizes, axis=0)
                q_hat = torch.cat(q_hat, dim=0)
                distances = torch.cat(distances, dim=0)
                weights = torch.cat(weights, dim=0)

            # Evaluate
            label_probs = prediction_sets.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            is_covered = list((label_probs > 0).float().cpu().numpy())
            all_coverage[noise_params] += is_covered
            all_q_hats[noise_params] += list(q_hat.cpu().numpy())
            all_set_sizes[noise_params] += list(set_sizes)

            # Add results for this batch
            if method == "non_exchangeable_conformal_nucleus_sampling":
                all_distances[noise_params] += list(distances.mean(dim=-1).cpu().numpy())
                all_weights[noise_params] += list(weights.mean(dim=-1).cpu().numpy())


    # Collect results
    results = {
        "method": method,
        "all_coverage": dict(all_coverage),
        "all_q_hats": dict(all_q_hats),
        "all_set_sizes": dict(all_set_sizes),
    }

    if method == "non_exchangeable_conformal_nucleus_sampling":
        results["all_distances"] = dict(all_distances)
        results["all_weights"] = dict(all_weights)

    # Save results to pickle
    file_name = f"{timestamp}_{dataset}_{method}_{conformity_score}_{alpha}_shift.pkl"

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(os.path.join(result_dir, file_name), "wb") as result_file:
        dill.dump(results, result_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_IDENTIFIER
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
        "--method",
        type=str,
        default="non_exchangeable_conformal_nucleus_sampling",
        choices=("non_exchangeable_conformal_nucleus_sampling", "conformal_nucleus_sampling", "nucleus_sampling")
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--conformity-score",
        type=str,
        default="adaptive",
        choices=("simple", "adaptive")
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
        "--alpha",
        type=float,
        default=ALPHA
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

        data_store = DataStore(
            key_dim=MODEL_HIDDEN_SIZES[args.model_identifier], value_dim=1,
            distance_type=args.distance_type,
            num_centroids=args.num_centroids, code_size=args.code_size,
            num_probes=args.num_probes, use_quantization=args.use_quantization,
            device=args.device
        )  # Create empty data store
        data_store.load(args.datastore_dir)  # Load in contents

        perform_shift_experiment(
            model_identifier=args.model_identifier,
            dataset=args.dataset,
            batch_size=args.batch_size,
            method=args.method,
            temperature=args.temperature,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            alpha=args.alpha,
            num_neighbors=args.num_neighbors,
            data_store=data_store,
            distance_type=args.distance_type,
            conformity_score=args.conformity_score,
            seed=args.seed,
            device=args.device,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e

    finally:
        if tracker is not None:
            tracker.stop()
