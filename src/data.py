"""
Implement functions concerned with data loading and preprocessing.
"""

# EXT
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

# PROJECT
from src.custom_types import Tokenizer, Device

# CONSTANTS
SUFFIX = {
    "de": "deu",
    "en": "eng",
    "ja": "jpn"
}


class ParallelDataset(Dataset):
    def __init__(self, src_data, tgt_data, tokenizer: Tokenizer, device: Device, **tokenizer_kwargs):
        super().__init__()

        self.tgt_data = tgt_data
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_kwargs = tokenizer_kwargs

        # TODO: Remove in the future, just for debugging
        """
        line_break = 2250

        # Process data
        self.src_data = []
        for i, line in enumerate(src_data):
            if i > line_break:
                break

            self.src_data.append(self.tokenizer(line.strip(), return_tensors="pt", **tokenizer_kwargs))

        self.tgt_data = []
        for i, line in enumerate(tgt_data):
            if i > line_break:
                break

            self.tgt_data.append(self.tokenizer(line.strip(), return_tensors="pt", **tokenizer_kwargs))

        """

        self.src_data = [line.strip() for line in src_data]
        self.tgt_data = [line.srip() for line in tgt_data]

        self.length = len(self.src_data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        src = self.tokenizer(self.src_data[idx], return_tensors="pt", **self.tokenizer_kwargs)
        tgt = self.tokenizer(self.tgt_data[idx], return_tensors="pt", **self.tokenizer_kwargs)

        data = {
            "input_ids": src["input_ids"].squeeze(0).to(self.device),
            "attention_mask": src["attention_mask"].squeeze(0).to(self.device),
            "decoder_input_ids": tgt["input_ids"].squeeze(0).to(self.device),
            "labels": torch.cat([tgt["input_ids"].squeeze(0)[1:], torch.ones(1).long()]).to(self.device),
        }

        return data


def load_data(
    dataset_name: str,
    tokenizer: Tokenizer,
    batch_size: int,
    device: Device,
    data_dir: str,
    load_splits: Tuple[str, ...] = ("train", "dev", "test"),
    **tokenizer_kwargs
) -> DataLoader:
    """
    Load dataset and tokenize it.

    Parameters
    ----------
    dataset_name: str
        Name of the dataset to load.
    tokenizer: Tokenizer
        Tokenizer to use for tokenization.
    batch_size: int
        Batch size.
    device: Device
        Device to use for tokenization.
    data_dir: str
        Path to directory where data is stored.
    load_splits: Tuple[str, ...]
        Splits to load.

    Returns
    -------
    DataLoader
        DataLoader containing the dataset.
    """
    data_loaders = {}

    for split_name in load_splits:
        src_lang, tgt_lang = dataset_name[:2], dataset_name[2:]
        src_suffix, tgt_suffix = SUFFIX[src_lang], SUFFIX[tgt_lang]

        # Load splits
        src_split = (line for line in open(
            f"{data_dir}/{dataset_name}/{split_name}.{src_suffix}",
            "r"
        ))
        tgt_split = (line for line in open(
            f"{data_dir}/{dataset_name}/{split_name}.{tgt_suffix}",
            "r"
        ))

        dataset = ParallelDataset(src_split, tgt_split, tokenizer, device, **tokenizer_kwargs)

        # Collate into dataloader
        split_dl = DataLoader(
            dataset, batch_size=batch_size
        )
        del src_split, tgt_split, dataset

        data_loaders[split_name] = split_dl

    return data_loaders
