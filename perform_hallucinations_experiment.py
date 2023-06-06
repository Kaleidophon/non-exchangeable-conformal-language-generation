"""
Perform hallucinations experiment, where we compare we first let the model generate freely, and then feed the same tokens
back into the decoder, but restricting the attention on the source side. We then compare the two set sizes.
"""

# STD
import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
from typing import Optional, Tuple

# EXT
from codecarbon import OfflineEmissionsTracker
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# PROJECT
from src.conformal import ConformalCalibrator, NonExchangeableConformalLogitProcessor
from src.custom_types import Device
from src.data import load_data
from src.datastore import DataStore

# CONST
DATA_DIR = "./data/wmt22"
MODEL_DIR = "./models"
RESULT_DIR = "./results"
EMISSION_DIR = "./emissions"
PROJECT_NAME = "nlg-conformal-risk-control"
MODEL_IDENTIFIER = "facebook/m2m100_418M"
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
SEQUENCE_LENGTH = 128


def perform_hallucinations_experiment(
    model_identifier: str,
    model: M2M100ForConditionalGeneration,
    dataset: str,
    batch_size: int,
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
):
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    timestamp = str(datetime.now().strftime("%d-%m-%Y_(%H:%M:%S)"))

    # Load data and model
    src_lang, tgt_lang = DATASETS[dataset]
    model.eval()
    tokenizer = M2M100Tokenizer.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)
    data_loaders = load_data(
        dataset, tokenizer, batch_size, device, data_dir,
        padding="max_length",
        max_length=SEQUENCE_LENGTH,
        truncation=True,
        load_splits=("dev", "test")
    )

    generation_config = {
        "max_length": model.config.max_length,
        "early_stopping": True,
        "forced_bos_token_id": tokenizer.get_lang_id(tgt_lang),
        "do_sample": True,
        "num_beams": 1,
    }

    calibrator = ConformalCalibrator(
        alpha=alpha, temperature=temperature, device=device
    )

    # Init logit processor
    generation_config["output_hidden_states"] = True
    generation_config["return_dict_in_generate"] = True

    logit_processor = NonExchangeableConformalLogitProcessor(
        data_store=data_store, conformity_score=conformity_score,
        calibrator=calibrator, distance_type=distance_type,
        num_neighbors=num_neighbors,
        store_set_sizes=True
    )
    model = logit_processor.patch_model(model)
    generation_config["logits_processor"] = [logit_processor]

    # Record set sizes for generations and hallucinations on the validation set
    hallucination_mask = torch.zeros(model.config.decoder_layers, model.config.decoder_attention_heads)
    normal_set_sizes = defaultdict(list)
    hallucination_set_sizes = defaultdict(list)

    for split in ("dev", "test"):
        for batch in tqdm(data_loaders[split], total=len(data_loaders[split])):

            generate_outputs = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **generation_config
            )
            batch_normal_sizes = logit_processor.last_set_sizes
            batch_normal_sizes = [torch.LongTensor(sizes) for sizes in batch_normal_sizes]
            batch_normal_sizes = torch.stack(batch_normal_sizes, dim=1)
            logit_processor.last_set_sizes = []

            # Feed in the same tokens, but restrict attention on the source side
            decoder_token_ids = generate_outputs.sequences

            with torch.no_grad():
                forward_outputs = model.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_token_ids,
                    cross_attn_head_mask=hallucination_mask,
                    output_hidden_states=True,
                    return_dict=True
                )

            decoder_token_ids_wo_bos_eos = decoder_token_ids[:, 1:-1]
            decoder_token_ids = decoder_token_ids
            batch_normal_sizes = batch_normal_sizes

            decoder_states = forward_outputs.decoder_hidden_states[-1]
            predictions = F.softmax(forward_outputs.logits, dim=-1)

            # Reshape and filter out ignore tokens
            mask = torch.all(
                torch.stack([decoder_token_ids != ignore_id for ignore_id in ignore_token_ids], dim=0), dim=0
            ).to(device)
            set_size_mask = torch.all(
                torch.stack([decoder_token_ids_wo_bos_eos != ignore_id for ignore_id in ignore_token_ids], dim=0), dim=0
            ).to(device)
            decoder_states = decoder_states[mask]

            batch_normal_sizes = batch_normal_sizes[set_size_mask]  # Ignore BOS and last token sets

            if distance_type == "inner_product":
                decoder_states /= model.config.d_model ** 0.25

            elif distance_type == "cosine":
                decoder_states = F.normalize(decoder_states, p=2, dim=-1)

            predictions = predictions[mask]

            # Run the non-exchangeable conformal prediction
            distances, conformity_scores = [], []

            # This can be hard on memory so we do it in batches
            bbatch_size = 1
            for i in range(0, len(decoder_states), bbatch_size):
                batch_distances, batch_conformity_scores = data_store.search_k(
                    decoder_states[i:i + bbatch_size, :], k=num_neighbors
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
            _, batch_hallucination_sizes = calibrator.get_prediction_sets(conformity_score, predictions, q_hat)
            batch_hallucination_sizes = torch.LongTensor(batch_hallucination_sizes)

            # Resplit again into the original sequences
            lengths = set_size_mask.long().sum(dim=-1).tolist()
            normal_set_sizes[split].extend(list(torch.split(batch_normal_sizes, lengths)))
            hallucination_set_sizes[split].extend(list(torch.split(batch_hallucination_sizes, lengths)))

    # Compute average treatment effect
    diffs = []

    for seq_normal, seq_hallucinatory in zip(normal_set_sizes["test"], hallucination_set_sizes["test"]):
        diff = seq_hallucinatory.float() - seq_normal.float()
        diffs.append(diff)

    ate = torch.cat(diffs).mean().item()
    print(f"ATE: {ate:.4f}")

    # Fit time step models
    max_time_step = max([len(s) for s in normal_set_sizes["dev"]])
    num_seqs = len(normal_set_sizes["dev"])

    # Put data into a nicer format
    coverage_matrix_normal = np.zeros((num_seqs, max_time_step)) - 1
    coverage_matrix_hallucinatory = np.zeros((num_seqs, max_time_step)) - 1

    for i in range(num_seqs):
        coverage_matrix_normal[i, :len(normal_set_sizes["dev"][i])] = normal_set_sizes["dev"][i]
        coverage_matrix_hallucinatory[i, :len(hallucination_set_sizes["dev"][i])] = hallucination_set_sizes["dev"][i]

    # Get means and variances
    non_negative_entries_normal = np.sum((coverage_matrix_normal != -1).astype(int), axis=0)
    coverage_matrix_normal[coverage_matrix_normal == -1] = 0
    means_normal = np.sum(coverage_matrix_normal, axis=0) / non_negative_entries_normal
    std_devs_normal = np.sqrt(
        np.sum((coverage_matrix_normal - means_normal[None, :]) ** 2, axis=0) / non_negative_entries_normal
    ) / np.sqrt(non_negative_entries_normal)

    non_negative_entries_hallucinatory = np.sum((coverage_matrix_hallucinatory != -1).astype(int), axis=0)
    coverage_matrix_hallucinatory[coverage_matrix_hallucinatory == -1] = 0
    means_hallucinatory = np.sum(coverage_matrix_hallucinatory, axis=0) / non_negative_entries_hallucinatory
    std_devs_hallucinatory = np.sqrt(
        np.sum((coverage_matrix_hallucinatory - means_hallucinatory[None, :]) ** 2, axis=0)
        / non_negative_entries_hallucinatory
    ) / np.sqrt(non_negative_entries_hallucinatory)
    def score_set_sizes(set_sizes, means, std_devs):
        return sum([
            stats.norm.logpdf(set_size.item(), loc=mean, scale=std_dev+ 1e-16)
            for set_size, mean, std_dev in zip(set_sizes, means, std_devs)
        ])

    # Compute Bayes factors
    bayes_factors_normal = [
        score_set_sizes(set_sizes_normal, means_hallucinatory, std_devs_hallucinatory) -
        score_set_sizes(set_sizes_normal, means_normal, std_devs_normal)  # Subtract since we are in log space
        for set_sizes_normal in normal_set_sizes["test"]
    ]

    bayes_factors_hallucinatory = [
        score_set_sizes(set_sizes_hallucinatory, means_hallucinatory, std_devs_hallucinatory) -
        score_set_sizes(set_sizes_hallucinatory, means_normal, std_devs_normal)  # Subtract since we are in log space
        for set_sizes_hallucinatory in hallucination_set_sizes["test"]
    ]

    # Put into nice table
    result_df = pd.DataFrame(
        columns=["Data", "Avg. Bayes Factor", "Accept H0", "Accept H1", "Undecided", "Type I rate", "Type II rate"]
    )
    result_df["Data"] = ["Normal", "Hallucinatory"]
    result_df.set_index("Data", inplace=True)

    # Compute occurrences and put into table
    normal_num_h0 = sum([bf <= -3 for bf in bayes_factors_normal])
    normal_num_h1 = sum([bf >= 3 for bf in bayes_factors_normal])
    normal_num_undecided = len(bayes_factors_normal) - normal_num_h0 - normal_num_h1
    normal_type1_rate = normal_num_h0 / len(bayes_factors_normal)  # FP rate - don't accept H0 since no hallucinations
    normal_type2_rate = normal_num_undecided / len(bayes_factors_normal)  # FN rate - should have accepted H1
    result_df.loc["Normal", "Avg. Bayes Factor"] = np.mean(bayes_factors_normal)
    result_df.loc["Normal", "Accept H0"] = normal_num_h0
    result_df.loc["Normal", "Accept H1"] = normal_num_h1
    result_df.loc["Normal", "Undecided"] = normal_num_undecided
    result_df.loc["Normal", "Type I rate"] = normal_type1_rate
    result_df.loc["Normal", "Type II rate"] = normal_type2_rate

    hallucinatory_num_h0 = sum([bf <= -3 for bf in bayes_factors_hallucinatory])
    hallucinatory_num_h1 = sum([bf >= 3 for bf in bayes_factors_hallucinatory])
    hallucinatory_num_undecided = len(bayes_factors_hallucinatory) - hallucinatory_num_h0 - hallucinatory_num_h1
    # FP rate - don't accept H1 we have hallucinations
    hallucinatory_type1_rate = hallucinatory_num_h1 / len(bayes_factors_hallucinatory)
    # FN rate - should have accepted H1
    hallucinatory_type2_rate = hallucinatory_num_undecided / len(bayes_factors_hallucinatory)
    result_df.loc["Hallucinatory", "Avg. Bayes Factor"] = np.mean(bayes_factors_hallucinatory)
    result_df.loc["Hallucinatory", "Accept H0"] = hallucinatory_num_h0
    result_df.loc["Hallucinatory", "Accept H1"] = hallucinatory_num_h1
    result_df.loc["Hallucinatory", "Undecided"] = hallucinatory_num_undecided
    result_df.loc["Hallucinatory", "Type I rate"] = hallucinatory_type1_rate
    result_df.loc["Hallucinatory", "Type II rate"] = hallucinatory_type2_rate

    print(result_df.to_markdown())

    # Save results to path
    with open(f"{result_dir}/{timestamp}_hallucinations_results.txt", "w") as results_file:
        results_file.write(f"ATE: {ate:.4f}\n\n")
        results_file.write(result_df.to_markdown())


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
        )
        tracker.start()

    try:
        data_store = None

        model = M2M100ForConditionalGeneration.from_pretrained(args.model_identifier)
        data_store = DataStore(
            key_dim=model.config.d_model, value_dim=1,
            distance_type=args.distance_type,
            num_centroids=args.num_centroids, code_size=args.code_size,
            num_probes=args.num_probes, use_quantization=args.use_quantization,
            device=args.device
        )  # Create empty data store
        data_store.load(args.datastore_dir)  # Load in contents

        perform_hallucinations_experiment(
            model_identifier=args.model_identifier,
            model=model,
            dataset=args.dataset,
            batch_size=args.batch_size,
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
