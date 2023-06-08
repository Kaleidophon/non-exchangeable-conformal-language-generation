"""
Implementation of the datastore using FAISS. This implementation is heavily based on the implementation by
@tongyao-zhu available under https://github.com/tongyao-zhu/knn-mt-reimplement/blob/main/datastore.py.
"""

# STD
import os
from typing import Tuple
import warnings

# EXT
from einops import rearrange
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedModel

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
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class DataStore:
    """
    This class represents a datastore. It can be trained from raw features, saved and loaded from disk.
    During inference time, it can search given a query and return the normalized score for each token.
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        distance_type: str = "inner_product",
        num_centroids: int = 2048,
        code_size: int = 64,
        num_probes: int = 32,
        use_quantization: bool = True,
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
        distance_type: str
            Type of distance measure being used. Either has to be "inner_product" or "l2".
        num_centroids : int
            Number of centroids to use for clustering during the quantization process.
        code_size : int
            Number of bytes in quantized codes.
        num_probes: int
            Number of coarse-level quantizers.
        device : Device
            Device that the datastore lives on.
        index_file_name : str
            Name of the file to save the index to. Defaults to 'index.trained'.
        token_ids_file_name : str
            Name of the file to save the token ids to. Defaults to 'token_ids.pt'.
        """
        assert distance_type in ("inner_product", "l2", "cosine"), \
            "Distance type has to be either 'inner_product', 'cosine' or 'l2'."

        # Set attributes
        self.index_file_name = index_file_name
        self.token_ids_file_name = token_ids_file_name
        self.device = device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_centroids = num_centroids
        self.num_probes = num_probes
        self.code_size = code_size
        self.distance_type = distance_type

        # Init index
        if distance_type in ("inner_product", "cosine"):
            self.index = faiss.IndexFlatIP(self.key_dim)

        elif distance_type == "l2":
            self.index = faiss.IndexFlatL2(self.key_dim)

        if use_quantization:
            self.index = faiss.IndexIVFPQ(self.index, self.key_dim, self.num_centroids, self.code_size, 8)
            self.index.nprobe = num_probes

        if self.device != "cpu":
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  # to avoid GPU memory issue
            resources = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(resources, 0, self.index, co)

        self.value_tensor = torch.empty((0, self.value_dim), dtype=torch.float16).to(device)

    def load(self, save_dir: str) -> None:
        """
        Load the pretrained FAISS index from a directory with the necessary components.

        Parameters
        ----------
        save_dir : str
            Directory containing index.trained (the trained index) and token_ids.pt (token ids sorted by index id).
        """
        self.index = faiss.read_index(os.path.join(save_dir, self.index_file_name))
        print(f"Loaded index with {self.index.ntotal} entries from disk.")
        self.value_tensor = torch.load(os.path.join(save_dir, self.token_ids_file_name), map_location=self.device)
        print(f"Loaded {self.value_tensor.shape[0]} values from disk.")

    def train_index(self, key_data: torch.FloatTensor, max_training_keys: int = 1000000) -> None:
        """
        Training the FAISS index. We will perform random sampling on the keys.
        :param key_store: a numpy array with shape (num_keys, dim_keys), each row is a key
        :return: None. The index attribute will be updated after training.
        """
        random_indices = np.random.choice(
            np.arange(key_data.shape[0]),
            size=[min(max_training_keys, len(key_data))],
            replace=False,
        )
        train_data = key_data[random_indices, :].cpu().numpy()
        self.index.train(train_data)

        if self.device != "cpu":
            self.index = faiss.index_gpu_to_cpu(self.index)  # put back to CPU

    def add(self, keys: torch.FloatTensor, values: torch.FloatTensor) -> None:
        """
        Add the keys to the trained index.
        :param keys_to_add: a numpy array of shape (num_keys, keys_dim)
        :return: The index will be updated with the input keys.
        """
        self.index.add(keys.cpu().numpy())  # add vectors to the index
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

    def search_k(self, query: torch.FloatTensor, k: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Search for the top K nearest neighbors, along with the distance.
        :param k: top k
        :param query: should have shape (num_queries, dim_keys).
        :return: scores: should have shape (num_queries, vocab_size), contains scores for each token for each entry
        """
        query = query.cpu().numpy()

        distances, indices = self.index.search(
            query, k
        )  # D, I will have shape (num_queries, k), containing the distance and the index

        distances, indices = torch.FloatTensor(distances), torch.LongTensor(indices)

        found_mask = indices != -1

        found_keys = found_mask.long().sum()

        if found_keys < k:
            warnings.warn(f"Fewer keys were found than requested ({found_keys} < {k}). Something might be going wrong.")

        elif found_keys == 0:
            raise ValueError("No matching keys found in the index.")

        distances = distances[found_mask].unsqueeze(0)
        indices = indices[found_mask]

        values = self.value_tensor[indices, :].squeeze(-1).unsqueeze(0)  # (num_queries, k)

        return distances.to(self.device), values.to(self.device)


def build_calibration_data(
    model: PreTrainedModel,
    data_loader: DataLoader,
    distance_type: str = "inner_product",
    conformity_score: str = "adaptive",
    ignore_token_ids: Tuple[int] = (1, 2),
    device: Device = "cpu",
    **datastore_kwargs,
) -> DataStore:
    """
    Build calibration data from dev set using the specified model.

    Parameters
    ----------
    model: PreTrainedModel
        Model to be used for the experiments.
    data_loader: DataLoader
        Data loader to be used for the experiments.
    distance_type: str
        Type of distance measure being used. Either has to be "inner_product" or "l2".

    Returns
    -------
    CalibrationData
        Calibration data, including the losses (i.e. the quality) and log probabilities of the hypotheses.
    """
    assert conformity_score in ("simple", "adaptive"), f"Conformity score must be 'simple' or 'adaptive', but " \
                                                       f"'{conformity_score}' found."

    calibration_data = DataStore(
        key_dim=model.config.d_model, value_dim=1, distance_type=distance_type, device=device, **datastore_kwargs
    )
    all_hidden = torch.empty((0, model.config.d_model), dtype=torch.float16).to(device)
    all_conformity_scores = torch.empty((0, 1), dtype=torch.float16).to(device)

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

        # Add to existing ones
        all_hidden = torch.cat([all_hidden, decoder_states], dim=0)
        all_conformity_scores = torch.cat([all_conformity_scores, conformity_scores], dim=0)

    # Normalize distances by dimensionality to make distance computation easier
    # We want to scale the inner products "transformer attention-style" by the square root of the hidden dimensionality,
    # but faiss only implements either L2 or inner product as distances. So instead, we can just scale the latents now
    # by the forth root of the dimensionality, which using the inner product later will amount to the same thing.
    num_latents, dim = all_hidden.shape

    if distance_type == "inner_product":
        all_hidden = all_hidden / dim ** 0.25

    # To search by cosine distance, we just use inner product search and normalize vectors beforehand
    # https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances#how-can-i-index-vectors-for-cosine-similarity
    elif distance_type == "cosine":
        all_hidden = F.normalize(all_hidden, dim=-1)

    # Train index
    mean = torch.mean(all_hidden, dim=0)
    std = torch.std(all_hidden, dim=0)
    print(f"Latent summary statistics: Dim={dim}, mean={mean}, std={std}")

    print("Training index...")
    calibration_data.train_index(all_hidden)

    # Add calibration points
    print(f"Adding {num_latents} data points to the index...")
    calibration_data.add(all_hidden, all_conformity_scores)

    return calibration_data
