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
    
    state_dict = convert_custom_weights(ckpt_path, device=device)
    
    if model_name == "multivit" or model_name == "multimae":
        state_dict = multimae_to_vit(state_dict)
    elif model_name == "vit":
        state_dict = convert_vit_weights(model_name, state_dict)
    
    return state_dict


def convert_custom_weights(ckpt_path, prefix=None, device="cuda"):
    """Convert trained weights to remove custom layer names.

    Args:
        ckpt_path (str): path to the checkpoints.
        prefix (str): the prefix of the weights name to be removed.
        device (str, optional): _description_. Defaults to "cuda".
    """
    model_weights = torch.load(ckpt_path, map_location=device)
    
    if "state_dict" in model_weights:
        model_weights = model_weights["state_dict"]
    else:
        print(model_weights.keys())
        print("not found key: state_dict")
        
    state_dict = {}
    
    if not prefix:
        prefix = "embedding_layer.vit_model."

    for k, v in model_weights.items():
        if prefix in k:
            state_dict[k.replace(prefix, "")] = v

    return state_dict