"""
Miscellaneous utility functions.
"""

# STD
from typing import List

# EXT
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from accelerate.utils.modeling import get_max_memory
from transformers import M2M100ForConditionalGeneration, M2M100Config, AutoModelForPreTraining


def shard_model(
    model_identifier: str,
    sharding_devices: List[str]
) -> M2M100ForConditionalGeneration:
    """
    Load model onto multiple GPUs at once.
    """
    max_memory = get_max_memory()
    max_memory = {
        device: max_memory[device]
        for device in max_memory.keys()
        if device in sharding_devices or device == "cpu"
    }

    config = M2M100Config.from_pretrained(model_identifier)

    with init_empty_weights():
        model = AutoModelForPreTraining.from_config(config)

    model.tie_weights()
    model = load_checkpoint_and_dispatch(
        model, model_identifier, device_map="auto", max_memory=max_memory
    )

    return model
