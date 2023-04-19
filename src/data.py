"""
Implement functions concerned with data loading and preprocessing.
"""

# EXT
import torch
from torch.utils.data import DataLoader, Dataset

# PROJECT
from src.types import Tokenizer, Device

# CONSTANTS
SUFFIX = {
    "de": "deu",
    "en": "eng",
    "ja": "jpn"
}


class ParallelDataset(Dataset):
    def __init__(self, src_data, tgt_data, tokenizer: Tokenizer, device: Device, **tokenizer_kwargs):
        super().__init__()
        self.src_data = src_data
        self.tgt_data = tgt_data
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_kwargs = tokenizer_kwargs

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src = self.src_data[idx]
        tgt = self.tgt_data[idx]

        src = self.tokenizer(src.strip(), return_tensors="pt", **self.tokenizer_kwargs)
        tgt = self.tokenizer(tgt.strip(), return_tensors="pt", **self.tokenizer_kwargs)

        data = {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "decoder_input_ids": tgt["input_ids"].squeeze(0),
            "labels": torch.cat([tgt["input_ids"].squeeze(0)[1:], torch.ones(1).long()]),
        }

        return data


def load_data(
    dataset_name: str,
    tokenizer: Tokenizer,
    batch_size: int,
    device: Device,
    data_dir: str,
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

    Returns
    -------
    DataLoader
        DataLoader containing the dataset.
    """
    data_loaders = {}

    for split_name in ["train", "dev", "test"]:
        src_lang, tgt_lang = dataset_name[:2], dataset_name[2:]
        src_suffix, tgt_suffix = SUFFIX[src_lang], SUFFIX[tgt_lang]

        # Load splits
        src_split = open(
            f"{data_dir}/{dataset_name}/{split_name}.{src_suffix}",
            "r"
        ).readlines()
        tgt_split = open(
            f"{data_dir}/{dataset_name}/{split_name}.{tgt_suffix}",
            "r"
        ).readlines()

        dataset = ParallelDataset(src_split, tgt_split, tokenizer, device, **tokenizer_kwargs)

        # Collate into dataloader
        split_dl = DataLoader(
            dataset, batch_size=batch_size, shuffle=split_name == "train"
        )

        data_loaders[split_name] = split_dl

    return data_loaders
