"""
Define custom types for this project.
"""

# STD
from typing import Union

# EXT
import torch
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
import wandb

# TYPES
Device = Union[torch.device, str]
WandBRun = wandb.wandb_sdk.wandb_run.Run
Tokenizer = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]