"""
Finetune model on a given dataset to use for later experiments.
"""

# STD
import argparse
from datetime import datetime
import json
import os
from typing import Optional

# EXT
import torch
from transformers.optimization import get_inverse_sqrt_schedule
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from codecarbon import OfflineEmissionsTracker
from knockknock import telegram_sender
import numpy as np
import wandb

# PROJECT
from src.evaluation import evaluate_model
from src.data import load_data, SUFFIX
from src.custom_types import Device, WandBRun

# CONST
DATA_DIR = "./data/wmt22"
MODEL_DIR = "./models"
EMISSION_DIR = "./emissions"
MODEL_IDENTIFIER = "facebook/mbart-large-50-many-to-many-mmt"
PROJECT_NAME = "nlg-conformal-risk-control"
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

# Knockknock support
try:
    from secret import TELEGRAM_API_TOKEN, TELEGRAM_CHAT_ID, COUNTRY_CODE

    SECRET_IMPORTED = True

except ImportError:
    pass

# CUDA
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# Original fairseq command for training:
# --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
# --lr-scheduler inverse_sqrt --lr 3e-05 --warmup-updates 2500 --max-update 40000 \
# --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0


def finetune_model(
    model_identifier: str,
    dataset: str,
    batch_size: int,
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    device: Device,
    seed: int,
    data_dir: str,
    model_dir: str,
    wandb_run: Optional[WandBRun] = None,
):
    """
    Finetune model on a given dataset.

    Parameters
    ----------
    model_identifier: str
        Identifier of the model to finetune.
    dataset: str
        Name of the dataset to finetune on.
    batch_size: int
        Batch size.
    learning_rate: float
        Learning rate.
    num_warmup_steps: int
        Number of warmup steps for inverse square root learning rate scheduler.
    num_training_steps: int
        Number of training steps.
    device: Device
        Device to use for training.
    seed: int
        Seed to use for training.
    data_dir: str
        Path to directory where data is stored.
    model_dir: str
        Path to directory where model is stored.
    wandb_run: Optional[WandBRun]
        WandB run to use for logging.
    """
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Load data
    src_lang, tgt_lang = DATASETS[dataset]
    tokenizer = MBart50TokenizerFast.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=["train", "dev"]
    )

    # Initialize model
    model = MBartForConditionalGeneration.from_pretrained(model_identifier).to(device)

    # Finetune model
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, betas=BETAS, weight_decay=WEIGHT_DECAY, eps=1e-06
    )
    scheduler = get_inverse_sqrt_schedule(
        optimizer=optimizer, num_warmup_steps=num_warmup_steps
    )

    for step, batch in enumerate(data_loaders["train"]):

        if step == num_training_steps:
            break

        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if wandb_run is not None:
            wandb_run.log(
                {
                    "batch_loss": loss.detach().cpu().item(),
                    "batch_learning_rate": optimizer.param_groups[0]['lr']
                }
            )

        if step % VALIDATION_INTERVAL == 0 and step > 0 and wandb_run is not None:
            with torch.no_grad():
                model.eval()
                # Get validation loss
                val_batch = next(iter(data_loaders["dev"]))
                val_outputs = model(**val_batch)
                val_loss = val_outputs.loss

                # Get validation scores
                val_results = evaluate_model(
                    model=model,
                    data_loader=data_loaders["dev"],
                    tokenizer=tokenizer,
                    source_file=f"{data_dir}/{dataset}/dev.{SUFFIX[src_lang[:2]]}",
                    reference_file=f"{data_dir}/{dataset}/dev.{SUFFIX[tgt_lang[:2]]}",
                    metrics=("bleu", "chrf")
                )

                model.train()

            val_results = {
                "val_loss": val_loss.detach().cpu().item(),
                **val_results
            }
            print(val_results)

            wandb_run.log(val_results)

    # Save model
    result_dir = f"{model_dir}/{dataset}_{timestamp}"
    os.mkdir(result_dir)
    torch.save(model.state_dict(), f"{result_dir}/model.pt")

    del data_loaders

    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=["test"]
    )

    # Evaluate model
    src_abbr = src_lang[:2]
    tgt_abbr = tgt_lang[:2]
    test_results = evaluate_model(
        model=model,
        data_loader=data_loaders["test"],
        tokenizer=tokenizer,
        source_file=f"{data_dir}/{dataset}/test.{SUFFIX[src_abbr]}",
        reference_file=f"{data_dir}/{dataset}/test.{SUFFIX[tgt_abbr]}",
    )

    # Return results for knockkock and result file
    results = {
        "model_identifier": model_identifier,
        "dataset": dataset,
        **test_results
    }

    # Write results to file
    with open(f"{result_dir}/results.json", "w") as f:
        json.dump(results, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_IDENTIFIER
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
        "--learning-rate",
        type=float,
        default=LEARNING_RATE
    )
    parser.add_argument(
        "--num-warmup-steps",
        type=int,
        default=NUM_WARMUP_STEPS
    )
    parser.add_argument(
        "--num-training-steps",
        type=int,
        default=NUM_TRAINING_STEPS
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=VALIDATION_INTERVAL
    )
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--model-dir", type=str, default=MODEL_DIR)
    parser.add_argument("--emission-dir", type=str, default=EMISSION_DIR)
    parser.add_argument("--track-emissions", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
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
        )(finetune_model)

    try:
        finetune_model(
            model_identifier=args.model,
            dataset=args.dataset,
            batch_size=BATCH_SIZE,
            learning_rate=args.learning_rate,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.num_training_steps,
            device=args.device,
            seed=args.seed,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            wandb_run=wandb_run,
        )

    except Exception as e:
        # Save data from emission tracker in any case
        if tracker is not None:
            tracker.stop()

        raise e
