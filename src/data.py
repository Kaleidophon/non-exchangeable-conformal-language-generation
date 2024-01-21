"""
Implement functions concerned with data loading and preprocessing.
"""

# STD
from typing import Dict, List

# EXT
import codecs
import torch
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Union

# PROJECT
from src.custom_types import Tokenizer, Device
from src.defaults import DATASET_TASKS

# CONSTANTS
SUFFIX = {
    "de": "deu",
    "en": "eng",
    "ja": "jpn"
}


class ParallelDataset(Dataset):
    """
    A parallel dataset used for Machine translation datasets.
    """
    def __init__(
        self,
        src_data: List[str],
        tgt_data: List[str],
        tokenizer: Tokenizer,
        device: Device,
        **tokenizer_kwargs
    ):
        """
        Initialize a parallel dataset.

        Parameters
        ----------
        src_data: List[str]
            Source language sentences.
        tgt_data: List[str]
            Target language sentences.
        tokenizer: Tokenizer
            Model tokenizer.
        device: Device
            Device the batched inputs should be moved to.
        """
        super().__init__()

        self.tgt_data = tgt_data
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_kwargs = tokenizer_kwargs
        self.src_data = [line.strip() for line in src_data]
        self.tgt_data = [line.strip() for line in tgt_data]

        self.length = len(self.src_data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.LongTensor, torch.FloatTensor]]:
        src = self.tokenizer(self.src_data[idx], return_tensors="pt", **self.tokenizer_kwargs)
        tgt = self.tokenizer(self.tgt_data[idx], return_tensors="pt", **self.tokenizer_kwargs)

        data = {
            "input_ids": src["input_ids"].squeeze(0).to(self.device),
            "attention_mask": src["attention_mask"].squeeze(0).to(self.device),
            "decoder_input_ids": tgt["input_ids"].squeeze(0).to(self.device),
            "labels": torch.cat([tgt["input_ids"].squeeze(0)[1:], torch.ones(1).long()]).to(self.device),
        }

        return data


class TextDataset(Dataset):
    """
    A simple text dataset used for language modeling.
    """
    def __init__(self, data, tokenizer: Tokenizer, device: Device, use_ravfogel_prompt: bool = False, **tokenizer_kwargs):
        """
        Initialize a simple text dataset.

        Parameters
        ----------
        data: List[str]
            Sentences in the dataset.
        tokenizer: Tokenizer
            Used model tokenizer.
        device: Device
            Device the batched inputs should be moved to.
        use_ravfogel_prompt: bool
            Flag to indicate whether we should use a prompting setup like in Ravfogel et al. (2023). Default is False.
        """
        super().__init__()

        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_kwargs = tokenizer_kwargs
        self.data = [line.strip() for line in data]
        self.ravfogel_prompt = use_ravfogel_prompt
        self.length = len(self.data)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.LongTensor, torch.FloatTensor]]:

        data = self.data[idx]

        if self.ravfogel_prompt:
            self.tokenizer.padding_side = "left"

        tokenized = self.tokenizer(data, return_tensors="pt", **self.tokenizer_kwargs)

        if not self.ravfogel_prompt:
            data = {
                "input_ids": tokenized["input_ids"].squeeze(0)[:-1].to(self.device),
                "attention_mask": tokenized["attention_mask"].squeeze(0)[:-1].to(self.device),
                "decoder_input_ids": tokenized["input_ids"].squeeze(0)[1:].to(self.device),
                "labels": tokenized["input_ids"].squeeze(0)[1:].to(self.device),
            }

        # Use the prompt style by Ravfogel et al. (2023): 35 initial tokens followed by generation
        else:
            data = {
                "input_ids": tokenized["input_ids"].squeeze(0).to(self.device),
                "attention_mask": tokenized["attention_mask"].squeeze(0).to(self.device),
            }

        return data


def load_data(
    dataset_name: str,
    tokenizer: Tokenizer,
    batch_size: int,
    device: Device,
    data_dir: str,
    load_splits: Tuple[str, ...] = ("train", "dev", "test"),
    use_ravfogel_prompt: bool = False,
    **tokenizer_kwargs
) -> Dict[str, DataLoader]:
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
    use_ravfogel_prompt: bool
            Flag to indicate whether we should use a prompting setup like in Ravfogel et al. (2023). Default is False.

    Returns
    -------
    Dict[str, DataLoader]
        Dataloaders containing different splits of the dataset.
    """
    data_loaders = {}
    task_type = DATASET_TASKS[dataset_name]

    if task_type == "mt":

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

    elif task_type == "lm":

        if "dev" in load_splits:
            data_split = codecs.open(f"{data_dir}/{dataset_name}/dev.txt", "r", "utf-8").readlines()
            dev_dataset = TextDataset(data_split, tokenizer, device, **tokenizer_kwargs)
            dev_dl = DataLoader(
                dev_dataset, batch_size=batch_size
            )
            data_loaders["dev"] = dev_dl

        if "test" in load_splits:

            data_split = codecs.open(f"{data_dir}/{dataset_name}/test.txt", "r", "utf-8").readlines()
            data_split = "".join(data_split).split("</s>")[:-1]
            test_dataset = TextDataset(
                data_split, tokenizer, device, use_ravfogel_prompt=use_ravfogel_prompt, **tokenizer_kwargs
            )

            test_dl = DataLoader(
                test_dataset, batch_size=batch_size
            )
            data_loaders["test"] = test_dl

    return data_loaders
