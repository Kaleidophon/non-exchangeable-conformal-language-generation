"""
Define some default values that are used by other scripts and bundle them here for consistency.
"""

# Project-level defaults
DATA_DIR = "./data/wmt22"
RESULT_DIR = "./results"
EMISSION_DIR = "./emissions"
PROJECT_NAME = "nlg-conformal-risk-control"
MODEL_IDENTIFIER = "facebook/m2m100_418M"

# Map available language pairs to language identifiers for tokenizer
DATASETS = {
    "deen": ("de", "en"),
    "jaen": ("ja", "en")
}

# Experimental defaults
SEED = 1234
BATCH_SIZE = 4
SEQUENCE_LENGTH = 128
GENERATION_METHODS = (
    "beam_search", "greedy", "top_k_sampling", "nucleus_sampling", "conformal_nucleus_sampling",
    "non_exchangeable_nucleus_sampling", "constant_non_exchangeable_nucleus_sampling"
)
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
