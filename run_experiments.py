"""
Conduct experiments for conformal risk control in NLG.
"""

# STD
import argparse
from datetime import datetime
import os
from typing import Optional, Dict

# EXT
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import torch
from torch.utils.data import DataLoader
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import wandb

# PROJECT
from src.data import load_data, SUFFIX
from src.types import Device, WandBRun
from src.conformal import build_calibration_data, CalibrationData, get_optimal_k

# CONST
DATA_DIR = "./data/wmt22"
MODEL_DIR = "./models/"
RESULT_DIR = "./results"
EMISSION_DIR = "./emissions"
MODEL_IDENTIFIER = "facebook/mbart-large-50-many-to-many-mmt"
PROJECT_NAME = "nlg-conformal-risk-control"
# Map available language pairs to language identifiers for tokenizer
DATASETS = {
    "deen": ("de_DE", "en_XX"),
    "jaen": ("ja_XX", "en_XX")
}

CALIBRATION_DATA_PATH = "./data/calibration/calibration_data.npy"

# DEFAULTS
SEED = 1234
BATCH_SIZE = 6  # TODO: Debug 64
NUM_BEAMS = 4  # TODO: Debug
ALPHA = 0.9

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
    num_beams: int,
    alpha: float,
    device: Device,
    data_dir: str,
    result_dir: str,
    source_path: str,
    references_path: str,
    model_path: Optional[str] = None,
    calibration_data_path: Optional[str] = None,
    wandb_run: Optional[WandBRun] = None,
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
    num_beams: int
        Number of beams to be used for the experiments.
    alpha: float
        Pre-defined confidence level for the experiments.
    device: Device
        Device to be used for the experiments.
    data_dir: str
        Path to the data directory.
    result_dir: str
        Path to the directory where the results should be saved.
    source_path: str
        Path to the file with translation sources.
    references_path: str
        Path to the file with references translations.
    calibration_data_path: Optional[str]
        Path to the calibration data. If not None, the calibration data will be loaded from the data directory.
    wandb_run: Optional[WandBRun]
        WandB run to be used for the experiments.

    Returns
    -------
    Dict[str, float]
        Dictionary containing the results of the experiments.
    """
    # Load or init model
    if model_path is not None:
        model = MBartForConditionalGeneration.from_pretrained(model_path)

    else:
        model = MBartForConditionalGeneration.from_pretrained(model_identifier)

    tokenizer = MBart50TokenizerFast.from_pretrained(model_identifier)

    # Load data
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        load_splits=("dev", "test"),
        padding="max_length",
        max_length=256,
        truncation=True
    )

    # Create generations on dev set, score them and create datastructure save the scores for easy access
    calibration_data = None
    if calibration_data_path is not None:
        if os.path.exists(calibration_data_path):
            calibration_data = CalibrationData.load(calibration_data_path)

    if calibration_data is None:
        calibration_data = build_calibration_data(
            model=model,
            tokenizer=tokenizer,
            data_loader=data_loaders["dev"],
            num_beams=num_beams,
            source_path=source_path,
            references_path=references_path
        )

        if calibration_data_path is not None:
            calibration_data.save(calibration_data_path)

    # Conduct experiments on test
    optimal_k = get_optimal_k(calibration_data, alpha)

    # Gather results and report them
    del data_loaders["dev"]



def test_found_k(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    k: int,
    test_loader: DataLoader
) -> Dict[str, float]:
    ... # TOOD: Implement




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_IDENTIFIER
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--calibration-data-path",
        type=str,
        default=CALIBRATION_DATA_PATH
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=tuple(DATASETS.keys())
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=BATCH_SIZE
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=NUM_BEAMS
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=ALPHA
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

    # Apply decorator
    if args.knock:
        if not SECRET_IMPORTED:
            raise ImportError(
                "secret.py wasn't found, please rename secret_template.py and fill in the information."
            )

        run_experiments = telegram_sender(
            token=TELEGRAM_API_TOKEN, chat_id=TELEGRAM_CHAT_ID
        )(run_experiments)

    # Define paths for translation data based on dataset chosen
    src_lang, tgt_lang = args.dataset[:2], args.dataset[2:]
    source_path = os.path.join(args.data_dir, args.dataset, f"dev.{SUFFIX[src_lang]}")
    references_path = os.path.join(args.data_dir, args.dataset, f"dev.{SUFFIX[tgt_lang]}")

    try:
        # Run experiments
        run_experiments(
            model_identifier=args.model,
            dataset=args.dataset,
            batch_size=args.batch_size,
            num_beams=args.num_beams,
            alpha=args.alpha,
            device=args.device,
            data_dir=args.data_dir,
            result_dir=args.result_dir,
            source_path=source_path,
            references_path=references_path,
            model_path=args.model_path,
            calibration_data_path=args.calibration_data_path,
            wandb_run=wandb_run
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e
