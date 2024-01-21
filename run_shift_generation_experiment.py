"""
Perform distributional shift experiment, where we observe generation quality under different magnitudes of noise added to model
embeddings.
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
import numpy as np
import torch
import types
from tqdm import tqdm
from transformers import M2M100PreTrainedModel, OPTPreTrainedModel
from transformers.generation import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput

# PROJECT
from src.conformal import ConformalCalibrator, ConformalLogitProcessor, NonExchangeableConformalLogitProcessor
from src.custom_types import Device
from src.data import load_data, SUFFIX
from src.datastore import DataStore
from src.defaults import (
    BATCH_SIZE, SEQUENCE_LENGTH, SEED, DATASETS, MODEL_IDENTIFIER, DATA_DIR, EMISSION_DIR, PROJECT_NAME,
    ALPHA, TEMPERATURE, NUM_NEIGHBORS, RESULT_DIR, MODEL_HIDDEN_SIZES, HF_RESOURCES, DATASET_TASKS
)
from src.evaluation import evaluate_translation_model, evaluate_generation_model
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
    softmax_temperature: float,
    # Arguments for conformal sampling methods
    alpha: Optional[float] = None,
    data_store: Optional[DataStore] = None,
    distance_type: Optional[str] = None,
    conformity_score: Optional[str] = None,
    num_neighbors: Optional[int] = None,
    # Other arguments
    seed: int = SEED,
    device: Device = "cpu",
    sharding: Optional[List[int]] = None,
):
    """
    Perform the distributional shift experiment for generation quality.

    Parameters
    ----------
    model_identifier: str
        Model to be used for the experiments. If model_path is specified, the model will be loaded from there.
    dataset: str
        Dataset to be used for the experiments.
    batch_size: int
        Batch size to be used for the experiments.
    method: str
        Type of non-conformity score to be used. Has to be either "simple" or "adaptive".
    temperature: float
        Temperature used for the datastore retrieval.
    data_dir: str
        Path to the data directory.
    result_dir: str
        Path to the directory where the results should be saved.
    alpha: float
        Used to set the 1 - alpha desired confidence level for conformal methods. Default is None.
    data_store: Optional[DataStore]
        Loaded datastore for nearest neighbor retrieval. Default is None.
    data_store: Optional[DataStore]
        Loaded datastore for nearest neighbor retrieval. Default is None.
    distance_type: Optional[str]
        Distance type to be used for retrieval. Either has to be "inner_product", "cosine" or "l2". Default is None.
    conformity_score: Optional[str]
        Type of non-conformity score to be used. Has to be either "simple" or "adaptive". Default is None.
    num_neighbors: Optional[int]
        Number of neighbors used for retrieval. Default is None.
    seed: int
        Set random seed used for replicability. Defaults to SEED.
    device: Device
        Device the model lives on.
    ignore_token_ids: Tuple[int, ...]
        Token IDs to ignore during experiments.
    sharding: Optional[List[int]]
        Indices of GPUs a shared model should be distributed across. Defaults to None, in which case no sharding is used.
    """
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
        load_splits=("test",),
        use_ravfogel_prompt=(task == "lm")
    )["test"]
    model_hidden_size = MODEL_HIDDEN_SIZES[model.config.name_or_path]

    if softmax_temperature is None:
        softmax_temperature = 1

    generation_config = {
        "max_length": model.config.max_length,
        "do_sample": True,
        "num_beams": 1,
        "temperature": softmax_temperature,
        "early_stopping": True,
    }

    if task == "mt":
        generation_config["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_lang)
        generation_config["num_beams"] = 1

    # That is actually not how Ravfogel et al. generate
    else:
        del generation_config["max_length"]
        generation_config["max_new_tokens"] = 350

    # ### Add custom arguments to geeneration config depending on method being used ###
    if method == "nucleus_sampling":
        generation_config["top_p"] = 1 - alpha

    # Prepare conformal sampling methods
    elif method in ("conformal_nucleus_sampling", "non_exchangeable_nucleus_sampling"):
        assert data_store is not None, "Data store must be provided for conformal sampling methods"
        assert alpha is not None, "alpha must be specified for conformal sampling methods"

        calibrator = ConformalCalibrator(
            alpha=alpha, temperature=temperature, device=device
        )

        # To assess the impact of the weights / neighbor retrieval, also test a variant with constant weights
        if method == "constant_non_exchangeable_nucleus_sampling":
            dummy_compute_weights = lambda distances: torch.ones_like(distances)
            calibrator.compute_weights = types.MethodType(dummy_compute_weights, calibrator)

        # Init logit processor
        if method in ("non_exchangeable_nucleus_sampling", "constant_non_exchangeable_nucleus_sampling"):
            generation_config["output_hidden_states"] = True
            generation_config["return_dict_in_generate"] = True

            logit_processor = NonExchangeableConformalLogitProcessor(
                data_store=data_store, conformity_score=conformity_score,
                calibrator=calibrator, distance_type=distance_type,
                num_neighbors=num_neighbors
            )
            model = logit_processor.patch_model(model)

        # For the conformal nucleus sampling, just compute one global quantile
        elif method == "conformal_nucleus_sampling":

            logit_processor = ConformalLogitProcessor(
                alpha=alpha,
                conformity_score=conformity_score,
                calibrator=calibrator,
                data_store=data_store,
            )

            del data_store

        generation_config["logits_processor"] = [logit_processor]

    # Generate translations according to specified method
    generations = defaultdict(list)

    # Define noise parameters
    noise_parameters = [
        None, (0, 0.025), (0, 0.05), (0, 0.075), (0, 0.1)
    ]

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

                outputs = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    **generation_config
                )

                # For the non-exchangeable conformal sampling
                if isinstance(outputs, SampleEncoderDecoderOutput) or isinstance(outputs, SampleDecoderOnlyOutput):
                    outputs = outputs.sequences

                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                # Truncate to 200 words for language modeling (Ravfogel et al. setup)
                if task == "lm":
                    outputs = [" ".join(out.split()[:200]) for out in outputs]

                generations[noise_params] += outputs

    del data_loader  # Delete data loader to free up memory
    del model  # Delete model to free up memory

    if task == "mt":
        src_abbr = src_lang[:2]
        tgt_abbr = tgt_lang[:2]
        source_file = f"{data_dir}/{dataset}/test.{SUFFIX[src_abbr]}"
        reference_path = f"{data_dir}/{dataset}/test.{SUFFIX[tgt_abbr]}"

        generation_results = {
            noise_params: evaluate_translation_model(
                noise_generations, source_file, reference_path,
                device=device, metrics=("bleu",), use_mbr=False
            )
            for noise_params, noise_generations in generations.items()
        }

    elif task == "lm":
        reference_path = f"{data_dir}/{dataset}/references.txt"

        generation_results = {
            noise_params: evaluate_generation_model(
                noise_generations, reference_path, device=device, metrics=("mauve",)
            )
            for noise_params, noise_generations in generations.items()
        }

    # Collect results
    results = {
        "method": method,
        **generation_results,
    }

    # Save results to pickle
    file_name = f"{timestamp}_{dataset}_{method}_{conformity_score}_{alpha}_shift_generations.pkl"

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
        "--softmax-temperature",
        type=float,
        default=1
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
            softmax_temperature=args.softmax_temperature,
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
