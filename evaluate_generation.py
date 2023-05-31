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
from datetime import datetime
import json
import os
from typing import Optional, Tuple

# EXT
from codecarbon import OfflineEmissionsTracker
import numpy as np
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers.generation import SampleEncoderDecoderOutput

# PROJECT
from src.data import load_data, SUFFIX
from src.conformal import ConformalCalibrator, ConformalLogitProcessor, NonExchangeableConformalLogitProcessor
from src.custom_types import Device
from src.datastore import DataStore
from src.evaluation import evaluate_model

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
GENERATION_METHODS = (
    "beam_search", "top_k_sampling", "nucleus_sampling", "conformal_nucleus_sampling",
    "non_exchangeable_nucleus_sampling"
)

# DEFAULTS
SEED = 1234
BATCH_SIZE = 4
ALPHA = 0.1
TEMPERATURE = 1
NUM_NEIGHBORS = 100
SEQUENCE_LENGTH = 128
NUM_BEAMS = 5
TOP_P = 0.9
TOP_K = 10

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
):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Load data and model
    src_lang, tgt_lang = DATASETS[dataset]
    model = M2M100ForConditionalGeneration.from_pretrained(model_identifier).to(device)
    model.eval()
    tokenizer = M2M100Tokenizer.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)
    data_loader = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=("test",)
    )["test"]

    generation_config = {
        "max_length": model.config.max_length,
        "early_stopping": True,
        "forced_bos_token_id": tokenizer.get_lang_id(tgt_lang),
        "do_sample": True,
        "num_beams": 1,
        "top_k": model.config.vocab_size,
    }

    # ### Add custom arguments to geeneration config depending on method being used ###
    if generation_method == "beam_search":
        assert num_beams is not None, "num_beams must be specified for beam search"
        generation_config["num_beams"] = num_beams
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

        # Init conformal calibrator
        if generation_method == "non_exchangeable_nucleus_sampling":
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
            conformity_scores = data_store.value_tensor.cpu().numpy()
            del data_store  # Free memory

            N = len(conformity_scores)
            q_level = np.ceil((N + 1) * (1 - alpha)) / N
            q_hat = torch.FloatTensor([np.quantile(conformity_scores, q_level, method='higher')], device=device)
            q_hat = q_hat.repeat(batch_size)
            logit_processor = ConformalLogitProcessor(q_hat, conformity_score, calibrator)

        generation_config["logits_processor"] = [logit_processor]

    # Generate translations according to specified method
    translations = []

    for batch in data_loader:
        outputs = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            **generation_config
        )

        # For the non-exchangeable conformal sampling
        if isinstance(outputs, SampleEncoderDecoderOutput):
            outputs = outputs.sequences

        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        translations += outputs

    # Generate results
    src_abbr = src_lang[:2]
    tgt_abbr = tgt_lang[:2]
    source_file = f"{data_dir}/{dataset}/test.{SUFFIX[src_abbr]}"
    reference_file = f"{data_dir}/{dataset}/test.{SUFFIX[tgt_abbr]}"
    results = evaluate_model(translations, source_file, reference_file, metrics=evaluation_metrics)
    print(results)

    # Save results to path
    with open(f"{result_dir}/{timestamp}_results.txt", "w") as results_file:
        results_file.write(json.dumps(results))


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
        choices=("inner_product", "l2")
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
        "--top-k",
        type=int,
        default=TOP_K
    )
    parser.add_argument(
        "--top-p",
        type=int,
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
        )
        tracker.start()

    try:
        data_store = None

        if args.generation_method in ("conformal_nucleus_sampling", "non_exchangeable_nucleus_sampling"):
            model = M2M100ForConditionalGeneration.from_pretrained(args.model_identifier)
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
            top_k=args.top_k,
            top_p=args.top_k,
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
