"""
Miscellaneous utility functions.
"""

# STD
from typing import Type, List

# EXT
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate.utils.modeling import get_max_memory
from transformers import PreTrainedModel, AutoModel, AutoConfig, OPTPreTrainedModel


def shard_model(
    model_identifier: str,
    sharding_devices: List[int],
    model_class: Type[PreTrainedModel],
    config_class: Type[AutoConfig]
) -> PreTrainedModel:
    """
    Load model onto multiple GPUs at once.
    """
    max_memory = get_max_memory()
    max_memory = {
        device: max_memory[device]
        for device in max_memory.keys()
        if device in sharding_devices or device == "cpu"
    }

    config = config_class.from_pretrained(model_identifier)

    with init_empty_weights():
        if "opt" in model_identifier:
            # Load into CPU first to infer device map
            model = model_class.from_pretrained(model_identifier)

        else:
            model = AutoModel.from_config(config)

    device_map = infer_auto_device_map(model, max_memory=max_memory)
    del model

    # Reload with device map
    model = model_class.from_pretrained(
        model_identifier, device_map=device_map, max_memory=max_memory
    )

    return model
