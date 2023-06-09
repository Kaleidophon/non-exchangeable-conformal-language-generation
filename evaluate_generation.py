"""
Evaluate the generations produced by different models.
In this case, we test the following methods:

- Regular beam search
- Top-k sampling
- Nucleus sampling
- Conformal nucleus sampling
- Non-exchangeable nucleus sampling
"""

# STD
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import types
from typing import Optional, Tuple, List

# EXT
from codecarbon import OfflineEmissionsTracker
import numpy as np
import torch
from tqdm import tqdm
from transformers.generation import SampleEncoderDecoderOutput

# PROJECT
from src.data import load_data, SUFFIX
from src.defaults import (
    DATA_DIR, RESULT_DIR, EMISSION_DIR, MODEL_IDENTIFIER, PROJECT_NAME, SEED, BATCH_SIZE, DATASETS,
    GENERATION_METHODS, ALPHA, TEMPERATURE, NUM_NEIGHBORS, NUM_BEAMS, TOP_P, TOP_K, SEQUENCE_LENGTH, HF_RESOURCES,
    DATASET_TASKS
)
from src.conformal import ConformalCalibrator, ConformalLogitProcessor, NonExchangeableConformalLogitProcessor
from src.custom_types import Device
from src.datastore import DataStore
from src.evaluation import evaluate_model
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


def evaluate_generations(
    model_identifier: str,
    dataset: str,
    batch_size: int,
    generation_method: str,
    temperature: float,
    data_dir: str,
    result_dir: str,
    evaluation_metrics: Tuple[str, ...] = ("bleu", "chrf", "comet"),
    # Arguments for common sampling methods
    num_beams: Optional[int] = None,
    num_samples: Optional[int] = None,
    softmax_temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    # Arguments for conformal sampling methods
    alpha: Optional[float] = None,
    data_store: Optional[DataStore] = None,
    distance_type: Optional[str] = None,
    conformity_score: Optional[str] = None,
    num_neighbors: Optional[int] = None,
    # Other arguments
    seed: int = SEED,
    device: Device = "cpu",
    sharding: Optional[List[int]] = None
):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Load data and model
    task = DATASET_TASKS[dataset]
    model_class, config_class, tokenizer_class = HF_RESOURCES[model_identifier]

    if task == "mt":
        src_lang, tgt_lang = DATASETS[dataset]
        tokenizer = model_class.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)

    else:
        tokenizer = tokenizer_class.from_pretrained(model_identifier)

    # Initialize model
    if sharding is None:
        model = model_class.from_pretrained(model_identifier).to(device)

    # Shard models onto different GPUs
    else:
        model = shard_model(model_identifier, sharding, model_class=model_class, config_class=config_class).to(device)

    model.eval()
    data_loader = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=("test",)
    )["test"]

    if softmax_temperature is None:
        softmax_temperature = 1

    if num_samples is None:
        num_samples = 1

    generation_config = {
        "max_length": model.config.max_length,
        "early_stopping": True,
        "forced_bos_token_id": tokenizer.get_lang_id(tgt_lang),
        "do_sample": True,
        "num_beams": 1,
        "temperature": softmax_temperature,
    }

    # ### Add custom arguments to geeneration config depending on method being used ###
    if generation_method == "beam_search":
        assert num_beams is not None, "num_beams must be specified for beam search"
        generation_config["num_beams"] = num_beams
        generation_config["do_sample"] = False

    elif generation_method == "greedy":
        generation_config["num_beams"] = 1
        generation_config["do_sample"] = False

    elif generation_method == "top_k_sampling":
        assert top_k is not None, "top_k must be specified for top-k sampling"
        generation_config["top_k"] = top_k

    elif generation_method == "nucleus_sampling":
        assert top_p is not None, "top_p must be specified for nucleus sampling"
        generation_config["top_p"] = top_p

    # Prepare conformal sampling methods
    elif generation_method in ("conformal_nucleus_sampling", "non_exchangeable_nucleus_sampling"):
        assert data_store is not None, "Data store must be provided for conformal sampling methods"
        assert alpha is not None, "alpha must be specified for conformal sampling methods"

        calibrator = ConformalCalibrator(
            alpha=alpha, temperature=temperature, device=device
        )

        # To assess the impact of the weights / neighbor retrieval, also test a variant with constant weights
        if generation_method == "constant_non_exchangeable_nucleus_sampling":
            dummy_compute_weights = lambda distances: torch.ones_like(distances)
            calibrator.compute_weights = types.MethodType(dummy_compute_weights, calibrator)

        # Init logit processor
        if generation_method in ("non_exchangeable_nucleus_sampling", "constant_non_exchangeable_nucleus_sampling"):
            generation_config["output_hidden_states"] = True
            generation_config["return_dict_in_generate"] = True

            logit_processor = NonExchangeableConformalLogitProcessor(
                data_store=data_store, conformity_score=conformity_score,
                calibrator=calibrator, distance_type=distance_type,
                num_neighbors=num_neighbors
            )
            model = logit_processor.patch_model(model)

        # For the conformal nucleus sampling, just compute one global quantile
        elif generation_method == "conformal_nucleus_sampling":

            logit_processor = ConformalLogitProcessor(
                alpha=alpha,
                conformity_score=conformity_score,
                calibrator=calibrator,
                data_store=data_store,
            )

            del data_store

        generation_config["logits_processor"] = [logit_processor]

    # Generate translations according to specified method
    translations = [[] for _ in range(num_samples)]

    for batch in tqdm(data_loader, total=len(data_loader)):
        for n in range(num_samples):
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generation_config
            )

            # For the non-exchangeable conformal sampling
            if isinstance(outputs, SampleEncoderDecoderOutput):
                outputs = outputs.sequences

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            translations[n] += outputs

    # Generate results
    src_abbr = src_lang[:2]
    tgt_abbr = tgt_lang[:2]
    source_file = f"{data_dir}/{dataset}/test.{SUFFIX[src_abbr]}"
    reference_file = f"{data_dir}/{dataset}/test.{SUFFIX[tgt_abbr]}"
    partial_results = [
        evaluate_model(translations[n], source_file, reference_file, metrics=evaluation_metrics)
        for n in range(num_samples)
    ]

    if num_samples == 1:
        results = partial_results[0]

    else:
        results = defaultdict(list)

        for result in partial_results:
            for key, value in result.items():
                results[key].append(value)

    # Compute mean and std dev
    print_results = {
        f"{key}_summary": f"{np.mean(value):.2f} \pm {np.std(value):.2f}"
        for key, value in results.items()
    }
    results.update(print_results)

    print(results)

    # Save results to path
    result_path = f"{result_dir}/{timestamp}_{model_identifier.replace('/', '_')}_{generation_method}_results.txt"
    with open(result_path, "w") as results_file:
        results_file.write(json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
        '--evaluation-metrics',
        nargs='+',
        type=str,
        default=("bleu", "chrf", "comet"),
        choices=("bleu", "chrf", "comet")
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--generation-method",
        type=str,
        choices=GENERATION_METHODS
    )
    parser.add_argument(
        "--conformity-score",
        type=str,
        default="adaptive",
        choices=("simple", "adaptive")
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=NUM_BEAMS
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P
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
        data_store = None

        if args.generation_method in ("conformal_nucleus_sampling", "non_exchangeable_nucleus_sampling"):
            model_class, _, _ = HF_RESOURCES[args.model_identifier]
            model = model_class.from_pretrained(args.model_identifier)
            data_store = DataStore(
                key_dim=model.config.d_model, value_dim=1,
                distance_type=args.distance_type,
                num_centroids=args.num_centroids, code_size=args.code_size,
                num_probes=args.num_probes, use_quantization=args.use_quantization,
                device=args.device
            )  # Create empty data store
            del model  # Free memory
            data_store.load(args.datastore_dir)  # Load in contents

        evaluate_generations(
            model_identifier=args.model_identifier,
            dataset=args.dataset,
            batch_size=args.batch_size,
            generation_method=args.generation_method,
            temperature=args.temperature,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            evaluation_metrics=args.evaluation_metrics,
            num_beams=args.num_beams,
            num_samples=args.num_samples,
            softmax_temperature=args.softmax_temperature,
            top_k=args.top_k,
            top_p=args.top_k,
            alpha=args.alpha,
            num_neighbors=args.num_neighbors,
            data_store=data_store,
            distance_type=args.distance_type,
            conformity_score=args.conformity_score,
            seed=args.seed,
            device=args.device,
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
