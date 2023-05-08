"""
Create the datastore for a model on a specified dataset.
"""

# STD
import argparse
from datetime import datetime
import os

# EXT
from codecarbon import OfflineEmissionsTracker
import numpy as np
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# PROJECT
from src.data import load_data
from src.datastore import CONFORMITY_SCORES, build_calibration_data
from src.custom_types import Device

# CONST
DATA_DIR = "./data/wmt22"
MODEL_DIR = "./models"
EMISSION_DIR = "./emissions"
PROJECT_NAME = "nlg-conformal-risk-control"
MODEL_IDENTIFIER = "facebook/mbart-large-50-many-to-many-mmt"
# Map available language pairs to language identifiers for tokenizer
DATASETS = {
    "deen": ("de_DE", "en_XX"),
    "jaen": ("ja_XX", "en_XX")
}

# DEFAULTS
SEED = 1234
BATCH_SIZE = 4
SEQUENCE_LENGTH = 128
NUM_TRAINING_STEPS = 40000
NUM_WARMUP_STEPS = 2500
LEARNING_RATE = 3e-05
BETAS = (0.9, 0.98)
VALIDATION_INTERVAL = 1000
WEIGHT_DECAY = 0

# GLOBALS
SECRET_IMPORTED = False
os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from secret import COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass


def create_datastore(
    model_identifier: str,
    dataset: str,
    batch_size: int,
    conformity_score: str,
    num_centroids: int,
    code_size: int,
    num_probes: int,
    use_quantization: bool,
    device: Device,
    seed: int,
    data_dir: str,
    save_dir: str,
):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load data
    src_lang, tgt_lang = DATASETS[dataset]
    tokenizer = MBart50TokenizerFast.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=("dev", )
    )

    # Initialize model
    model = MBartForConditionalGeneration.from_pretrained(model_identifier).to(device)
    model.eval()

    # Populate data score
    data_store = build_calibration_data(
        model, data_loaders["dev"], conformity_score,
        use_quantization=use_quantization,
        num_centroids=num_centroids,
        code_size=code_size,
        num_probes=num_probes,
    )

    # Save datastore
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data_store.save(save_dir)


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
        "--batch-size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--conformity-score",
        type=str,
        default="adaptive",
        choices=CONFORMITY_SCORES
    )
    parser.add_argument(
        "--use-quantization",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--num-centroids",
        type=int,
        default=4096
    )
    parser.add_argument(
        "--code-size",
        type=int,
        default=64
    )
    parser.add_argument(
        "--num-probes",
        type=int,
        default=32
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
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
        create_datastore(
            model_identifier=args.model,
            dataset=args.dataset,
            batch_size=BATCH_SIZE,
            conformity_score=args.conformity_score,
            use_quantization=args.use_quantization,
            num_centroids=args.num_centroids,
            code_size=args.code_size,
            num_probes=args.num_probes,
            device=args.device,
            seed=args.seed,
            data_dir=args.data_dir,
            save_dir=args.save_dir,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e