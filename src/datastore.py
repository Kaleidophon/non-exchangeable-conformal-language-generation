"""
Implementation of the datastore using FAISS. This implementation is heavily based on the implementation by
@tongyao-zhu available under https://github.com/tongyao-zhu/knn-mt-reimplement/blob/main/datastore.py.
"""

# STD
from collections import namedtuple
import dill
from functools import reduce
from operator import add
import os
from typing import Tuple

# EXT
from einops import rearrange
import evaluate
import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# PROJECT
from src.types import Device

# CONST
RAW_FEATURE_KEY_SUFFIX = ".pt"
RAW_FEATURE_VALUE_SUFFIX = "_values.pt"


class DataStore:
    """
    This class represents a datastore. It can be trained from raw features, saved and loaded from disk.
    During inference time, it can search given a query and return the normalized score for each token.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        alpha: float = 0.9,
        num_centroids: int = 4096,
        code_size: int = 64,
        num_probes: int = 32,
        use_quantization: bool = True,
        temperature: int = 10,
        device: Device = "cpu",
        index_file_name: str = "index.trained",
        token_ids_file_name: str = "token_ids.pt",
    ):
        """
        Set the necessary attributes. The number follow the original paper.

        Parameters
        ----------
        key_dim : int
            Dimensionality of the keys.
        value_dim: int
            Dimensionality of the values.
        num_centroids : int
            Number of centroids to use for clustering during the quantization process.
        code_size : int
            Number of bytes in quantized codes.
        num_probes: int
            Number of coarse-level quantizers.
        temperature : int
            Temperature as described in the paper.
        device : Device
            Device that the datastore lives on.
        index_file_name : str
            Name of the file to save the index to. Defaults to 'index.trained'.
        token_ids_file_name : str
            Name of the file to save the token ids to. Defaults to 'token_ids.pt'.
        """

        # Set attributes
        self.index_file_name = index_file_name
        self.token_ids_file_name = token_ids_file_name
        self.device = device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_centroids = num_centroids
        self.num_probes = num_probes
        self.code_size = code_size
        self.alpha = alpha

        # Init index
        if use_quantization:
            quantizer = faiss.IndexFlatIP(self.key_dim)
            self.index = faiss.IndexIVFPQ(quantizer, self.key_dim, self.num_centroids, self.code_size, 8)
            self.index.nprobe = 32

        else:
            self.index = faiss.IndexFlatIP(self.key_dim)

        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  # to avoid GPU memory issue
            resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(resources, 0, self.index, co)

        self.temperature = temperature  # temperature as described in the paper
        self.value_tensor = torch.empty((0, self.value_dim), dtype=torch.float16)

    def load(self, save_dir: str) -> None:
        """
        Load the pretrained FAISS index from a directory with the necessary components.

        Parameters
        ----------
        save_dir : str
            Directory containing index.trained (the trained index) and token_ids.pt (token ids sorted by index id).
        """
        self.index = faiss.read_index(os.path.join(save_dir, self.index_file_name))
        self.value_tensor = torch.tensor(
            torch.load(os.path.join(save_dir, self.token_ids_file_name))
        )

    def train_index(self, key_data: torch.FloatTensor, max_training_keys: int = 1000000) -> None:
        """
        Training the FAISS index. We will perform random sampling on the keys.
        :param key_store: a numpy array with shape (num_keys, dim_keys), each row is a key
        :return: None. The index attribute will be updated after training.
        """
        random_indices = np.random.choice(
            np.arange(len(key_data)),
            size=[min(max_training_keys, len(key_data))],
            replace=False,
        )
        self.index.train(key_data[random_indices, :])

        if self.device != "cpu":
            self.index = faiss.index_gpu_to_cpu(self.index)  # put back to CPU

    def add(self, keys: torch.FloatTensor, values: torch.FloatTensor) -> None:
        """
        Add the keys to the trained index.
        :param keys_to_add: a numpy array of shape (num_keys, keys_dim)
        :return: The index will be updated with the input keys.
        """
        self.index.add(keys)  # add vectors to the index
        self.value_tensor = torch.cat(
            [self.value_tensor, values.to(torch.float16)], dim=0
        )

    def save(self, output_dir: str) -> None:
        """
        Save the index and the index-to-token mapping in the output_dir.
        :param output_dir: The directory to save the results.
        :return: None. Results will be saved to output_dir.
        """
        try:
            # write the trained index
            faiss.write_index(self.index, os.path.join(output_dir, self.index_file_name))
        except Exception as e:
            raise IOError(f"Encountered error when writing FAISS index to {output_dir}: {e}")

        try:
            # save the index for token_ids
            torch.save(
                self.value_tensor, os.path.join(output_dir, self.token_ids_file_name)
            )
        except Exception as e:
            raise IOError(f"Encountered error when saving torch tensor to {output_dir}: {e}")

    def search_k(self, query: torch.tensor, k: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Search for the top K nearest neighbors, along with the distance.
        :param k: top k
        :param query: should have shape (num_queries, dim_keys).
        :return: scores: should have shape (num_queries, vocab_size), contains scores for each token for each entry
        """
        distances, indices = self.index.search(
            query, k
        )  # D, I will have shape (num_queries, k), containing the distance and the index
        values = self.value_tensor[indices, :]  # (num_queries, k)

        return distances, values


def build_calibration_data(
    model: MBartForConditionalGeneration,
    tokenizer: MBart50TokenizerFast,
    data_loader: DataLoader,
    num_beams: int,
    source_path: str,
    references_path: str,
    max_length: int = 64,  # TODO: Debug 256,
) -> DataStore:
    """
    Build calibration data from dev set using the specified model.

    Parameters
    ----------
    model: MBartForConditionalGeneration
        Model to be used for the experiments.
    tokenizer: MBart50TokenizerFast
        Tokenizer to be used for the experiments.
    data_loader: DataLoader
        Data loader to be used for the experiments.
    num_beams: int
        Number of beams to be used for the experiments.
    source_path: str
        Path to the source file. Used for evaluating translations.
    references_path: str
        Path to the references file. Used for evaluating translations.

    Returns
    -------
    CalibrationData
        Calibration data, including the losses (i.e. the quality) and log probabilities of the hypotheses.
    """
    # TODO: Rewrite
    calibration_data = DataStore(key_dim=model)
    comet_metric = evaluate.load('bleu')  # TODO: Change this back to comet

    model.eval()

    # Load source sentences and references
    with open(source_path, "r") as f:
        sources = [line.strip() for line in f.readlines()]

    # Load reference translations
    with open(references_path, "r") as f:
        references = [line.strip() for line in f.readlines()]

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        batch_size = len(batch["input_ids"])

        # Get input and target
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        # Generate hypotheses
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True,
            num_return_sequences=num_beams,
            return_dict_in_generate=True,
            output_scores=True,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
        )
        batch_translations = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        batch_scores = outputs.sequences_scores

        # Duplicate references and sources for this batch since we have every source represented multiple times (for
        # every beam) on the hypotheses
        batch_references = reduce(add, [
           [references[j]] * num_beams
           for j in range(i * batch_size, (i + 1) * batch_size)
        ])
        batch_sources = reduce(add, [
            [sources[j]] * num_beams
            for j in range(i * batch_size, (i + 1) * batch_size)
        ])
        comet_scores = torch.FloatTensor([
            comet_metric.compute(
                predictions=[trans], references=[ref],  # sources=batch_sources
            )["bleu"] / 100   # TODO: Change this back to comet
            for trans, ref, source in zip(batch_translations, batch_references, batch_sources)
        ])
        batch_losses = 1 - comet_scores

        # Do some reshaping
        batch_losses = rearrange(batch_losses, "(batch_size num_beams) -> batch_size num_beams", batch_size=batch_size)
        batch_scores = rearrange(batch_scores, "(batch_size num_beams) -> batch_size num_beams", batch_size=batch_size)

        # Add calibration points
        calibration_data.add_calibration_points(batch_losses, batch_scores)

    return calibration_data


if __name__ == "__main__":
    n_samples = 160000
    num_features = 64
    value_dim = 1
    dummy_data = torch.randn((n_samples, num_features))
    dummy_values = torch.randn((n_samples, value_dim))
    test_query = dummy_data.mean(dim=0).unsqueeze(0)

    datastore = DataStore(key_dim=num_features, value_dim=value_dim, use_quantization=False)
    datastore.train_index(dummy_data)
    datastore.add(dummy_data, dummy_values)
    print(datastore.search_k(test_query, 10))
