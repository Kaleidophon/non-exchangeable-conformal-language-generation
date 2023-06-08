"""
Define some default values that are used by other scripts and bundle them here for consistency.
"""

# EXT
from transformers import (
    M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer, OPTForCausalLM, OPTConfig, AutoTokenizer
)

# Project-level defaults
DATA_DIR = "./data/wmt22"
RESULT_DIR = "./results"
EMISSION_DIR = "./emissions"
PROJECT_NAME = "nlg-conformal-risk-control"
MODEL_IDENTIFIER = "facebook/m2m100_418M"

# Map available language pairs to language identifiers for tokenizer
DATASETS = {
    "deen": ("de", "en"),
    "jaen": ("ja", "en"),
    "openwebtext": (),
}
DATASET_TASKS = {
    "deen": "mt",
    "jaen": "mt",
    "openwebtext": "lm",
}

# Experimental defaults
SEED = 1234
BATCH_SIZE = 4
SEQUENCE_LENGTH = 128
GENERATION_METHODS = (
    "beam_search", "greedy", "top_k_sampling", "nucleus_sampling", "conformal_nucleus_sampling",
    "non_exchangeable_nucleus_sampling", "constant_non_exchangeable_nucleus_sampling"
)

HF_RESOURCES = {
    "facebook/m2m100_418M": (M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer),
    "facebook/m2m100_1.2B": (M2M100ForConditionalGeneration, M2M100Config, M2M100Tokenizer),
    "facebook/opt-350m": (OPTForCausalLM, OPTConfig, AutoTokenizer),
    "facebook/opt-1.3B": (OPTForCausalLM, OPTConfig, AutoTokenizer),
}

MODEL_HIDDEN_SIZES = {
    "facebook/m2m100_418M": 1024,
    "facebook/m2m100_1.2B": 1024,
    "facebook/opt-350m": 1024,
    "facebook/opt-1.3B": 1024,
}

ALPHA = 0.1
TEMPERATURE = 1
NUM_NEIGHBORS = 100
NUM_BEAMS = 5
TOP_P = 0.9
TOP_K = 10
NUM_ATTEMPTS = 20
SEARCH_SPACE = (0.01, 25)
STEP_SIZE = 0.1
NUM_BATCHES = 10
