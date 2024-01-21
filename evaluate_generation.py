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
import codecs
from collections import defaultdict
from datetime import datetime
import json
import os
import types
from typing import Optional, Tuple, List

# EXT
from codecarbon import OfflineEmissionsTracker
from einops import rearrange
import numpy as np
import torch
from tqdm import tqdm
from transformers.generation import SampleEncoderDecoderOutput, SampleDecoderOnlyOutput

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
from src.evaluation import evaluate_translation_model, evaluate_generation_model
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
    """
    Evaluate the generations of different methods.

    Parameters
    ----------
    model_identifier: str
        Huggingface hub identifier for the target model.
    dataset: str
        Name of the target dataset.
    batch_size: int
        Batch size to be used.
    generation_method: str
        Generation method to be used. Should be either "beam_search", "greedy", "top_k_sampling", "nucleus_sampling",
        "conformal_nucleus_sampling" or "non_exchangeable_nucleus_sampling".
    temperature: float
        Temperature used for the datastore retrieval.
    data_dir: str
        Directory the dataset is stored in.
    result_dir: str
        Directory the results should be saved to.
    evaluation_metrics: Tuple[str, ...]
        Evaluation metrics used. Defaults to ("bleu", "chrf", "comet").
    num_beams: Optional[int]
        Number of beams used for beam search. Defaults to None.
    num_samples: Optional[int]
        Number of samples to be generated per input. Defaults to None, in which case a single sample is used.
    softmax_temperature: Optional[float]
        Softmax temperature used for generation. Defaults to None, in which case 1 is used.
    top_k: Optional[int]
        k used for top-k sampling. Defaults to None.
    top_p: Optional[float]
        p used for top_p sampling. Defaults to None.
    alpha: float
        Used to set the 1 - alpha desired confidence level for conformal methods. Default is None.
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
    sharding: Optional[List[int]]
        Indices of GPUs a shared model should be distributed across. Defaults to None, in which case no sharding is used.
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Load data and model
    task = DATASET_TASKS[dataset]
    model_class, config_class, tokenizer_class = HF_RESOURCES[model_identifier]

    if task == "mt":
        src_lang, tgt_lang = DATASETS[dataset]
        tokenizer = tokenizer_class.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)

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
        load_splits=("test",),
        use_ravfogel_prompt=(task == "lm")
    )["test"]

    if softmax_temperature is None:
        softmax_temperature = 1

    if num_samples is None:
        num_samples = 1

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
    generations = [[] for _ in range(num_samples)]

    import time
    generation_times = []
    sequence_generation_times = []

    for batch in tqdm(data_loader, total=len(data_loader)):
        for n in range(num_samples):

            start = time.time()
            outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generation_config
            )
            end = time.time()

            # For the non-exchangeable conformal sampling
            if isinstance(outputs, SampleEncoderDecoderOutput) or isinstance(outputs, SampleDecoderOnlyOutput):
                outputs = outputs.sequences

            token_ids = rearrange(outputs, "b s -> (b s)")
            token_ids = token_ids[token_ids != tokenizer.pad_token_id]
            generation_times.extend([(end - start) / len(token_ids)] * len(token_ids))
            sequence_generation_times.extend([end - start] * outputs.shape[0])

            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            # Truncate to 200 words for language modeling (Ravfogel et al. setup)
            if task == "lm":
                outputs = [" ".join(out.split()[:200]) for out in outputs]

            generations[n] += outputs

    print(f"Average speed {np.mean(generation_times):.3f} per token.")
    print(f"Average speed {np.mean(sequence_generation_times):.3f} per sequence.")

    del data_loader  # Delete data loader to free up memory
    del model  # Delete model to free up memory

    # Generate results
    if task == "mt":
        src_abbr = src_lang[:2]
        tgt_abbr = tgt_lang[:2]
        source_file = f"{data_dir}/{dataset}/test.{SUFFIX[src_abbr]}"
        reference_path = f"{data_dir}/{dataset}/test.{SUFFIX[tgt_abbr]}"
        partial_results = [
            evaluate_translation_model(
                generations[n], source_file, reference_path, metrics=evaluation_metrics
            )
            for n in range(num_samples)
        ]

    elif task == "lm":
        reference_path = f"{data_dir}/{dataset}/references.txt"
        partial_results = [
            evaluate_generation_model(generations[n], reference_path, metrics=evaluation_metrics, device=device)
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
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(result_path, "w") as results_file:
        results_file.write(json.dumps(results))

    # Write generations to file
    num_generations = len(generations[0])
    generations_path = f"{result_dir}/{timestamp}_{model_identifier.replace('/', '_')}_{generation_method}_generations.txt"

    with codecs.open(reference_path, "r", "utf-8") as reference_file:
        reference_lines = reference_file.readlines()

    with codecs.open(generations_path, "w", "utf-8") as generations_file:
        for i in range(num_generations):
            generations_file.write("Reference:\t" + reference_lines[i])

            for n in range(num_samples):
                generations_file.write(f"Generation {n+1}:\t{generations[n][i]}\n")

            generations_file.write("\n")


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
        choices=("bleu", "chrf", "comet", "mauve", "bleurt", "bert_score")
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

            try:
                model_hidden_size = model.config.d_model

            except:
                model_hidden_size = int(model.config.hidden_size / 2)

            data_store = DataStore(
                key_dim=model_hidden_size, value_dim=1,
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
