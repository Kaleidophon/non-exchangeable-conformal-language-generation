"""
Implementation of the datastore using FAISS. This implementation is heavily based on the implementation by
@tongyao-zhu available under https://github.com/tongyao-zhu/knn-mt-reimplement/blob/main/datastore.py.
"""

# STD
import os
from typing import Tuple

# EXT
from einops import rearrange
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# PROJECT
from src.conformal import simple_conformity_scores, adaptive_conformity_score
from src.custom_types import Device

# CONST
RAW_FEATURE_KEY_SUFFIX = ".pt"
RAW_FEATURE_VALUE_SUFFIX = "_values.pt"
CONFORMITY_SCORES = {
    "simple": simple_conformity_scores,
    "adaptive": adaptive_conformity_score
}
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
    data_loader: DataLoader,
    conformity_score: str = "adaptive",
    ignore_token_ids: Tuple[int] = (1, 2)
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
    assert conformity_score in ("simple", "adaptive"), f"Conformity score must be 'simple' or 'adaptive', but " \
                                                       f"'{conformity_score}' found."

    calibration_data = DataStore(key_dim=model.config.d_model, value_dim=1)

    model.eval()

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):

        # Get input and target
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        decoder_input_ids = batch["decoder_input_ids"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Generate hypotheses
        with torch.no_grad():
            outputs = model.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                decoder_input_ids=decoder_input_ids,
                return_dict=True,
            )
        decoder_states = outputs.decoder_hidden_states[-1]
        predictions = F.softmax(outputs.logits, dim=-1)

        # Reshape and filter out ignore tokens
        decoder_states = rearrange(decoder_states, "b s h -> (b s) h")
        predictions = rearrange(predictions, "b s c -> (b s) c")
        input_ids = rearrange(input_ids, "b s -> (b s)")
        labels = rearrange(labels, "b s -> (b s)")
        mask = torch.all(torch.stack([input_ids != ignore_id for ignore_id in ignore_token_ids], dim=0), dim=0)
        decoder_states = decoder_states[mask]
        predictions = predictions[mask]
        labels = labels[mask]

        # Compute non-conformity scores
        conformity_scores = CONFORMITY_SCORES[conformity_score](predictions, labels)
        print(conformity_scores)

        # Add calibration points
        calibration_data.add(decoder_states, conformity_scores)
        a = 3

    return calibration_data


if __name__ == "__main__":
    src_lang, tgt_lang = "de_DE", "en_XX"
    model_identifier = "facebook/mbart-large-50-many-to-many-mmt"
    dataset = "deen"
    BATCH_SIZE = 6  # TODO: Debug 64

    from src.data import load_data

    tokenizer = MBart50TokenizerFast.from_pretrained(model_identifier, src_lang=src_lang, tgt_lang=tgt_lang)
    data_loaders = load_data(
        dataset, tokenizer, BATCH_SIZE, "cpu", "../data/wmt22",
        padding="max_length",
        max_length=64,
        truncation=True,
        load_splits=["train", "dev"]
    )

    # Initialize model
    model = MBartForConditionalGeneration.from_pretrained(model_identifier)
    num_features = model.config.d_model

    # Populate datastore
    datastore = build_calibration_data(
        model,
        data_loaders["dev"],
    )

    """
    n_samples = 50
    value_dim = 1
    dummy_data = torch.randn((n_samples, num_features))
    dummy_values = torch.randn((n_samples, value_dim))
    test_query = dummy_data.mean(dim=0).unsqueeze(0)

    datastore = DataStore(key_dim=num_features, value_dim=value_dim, use_quantization=False)
    datastore.train_index(dummy_data)
    datastore.add(dummy_data, dummy_values)
    print(datastore.search_k(test_query, 10))
    """
