import torch
from trans_exp.utils.convert_huggingface_vit_to_timm import convert_vit_weights
from multimae.tools.multimae2vit_converter import multimae_to_vit


def load_weights(ckpt_path, model_name, device):
    """Load weights to the original vit model.

    Args:
        ckpt_path (str): path to the checkpoint.
        model_name (str): encoder vit model name.
        device (str): device to load weights.
    """
    if model_name == "multivit":
        state_dict = convert_multivit_weights(ckpt_path, device)
    elif model_name == "vit":
        state_dict = convert_vit_weights(model_name, ckpt_path, device)
    
    return state_dict


def convert_multivit_weights(ckpt_path, device="cuda"):
    """Convert trained weights using multivit as backbones.

    Args:
        ckpt_path (str): path to the checkpoints.
        device (str, optional): _description_. Defaults to "cuda".
    """
    model_weights = torch.load(ckpt_path, map_location=device)
    multimae_state_dict = {}

    for k, v in model_weights.items():
        if "embedding_layer.vit_model." in k:
            multimae_state_dict[k.replace("embedding_layer.vit_model.", "")] = v
    
    state_dict = multimae_to_vit(state_dict)

    return state_dict