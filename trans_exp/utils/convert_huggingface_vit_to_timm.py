import os
import torch
import json
import timm
import requests
import argparse
from PIL import Image
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import (DeiTFeatureExtractor, ViTConfig, 
                          ViTFeatureExtractor, 
                          ViTForImageClassification, 
                          ViTModel)
from trans_exp.utils.model_utils import MODEL_CONFIG


# here we list all keys to be renamed (huggingface name on the left, timm name on the right)
def create_rename_keys(config, base_model=False):
    rename_keys = []
    for i in range(config.num_hidden_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append((f"vit.encoder.layer.{i}.layernorm_before.weight", f"blocks.{i}.norm1.weight"))
        rename_keys.append((f"vit.encoder.layer.{i}.layernorm_before.bias", f"blocks.{i}.norm1.bias"))
        rename_keys.append((f"vit.encoder.layer.{i}.attention.output.dense.weight", f"blocks.{i}.attn.proj.weight"))
        rename_keys.append((f"vit.encoder.layer.{i}.attention.output.dense.bias", f"blocks.{i}.attn.proj.bias"))
        rename_keys.append((f"vit.encoder.layer.{i}.layernorm_after.weight", f"blocks.{i}.norm2.weight"))
        rename_keys.append((f"vit.encoder.layer.{i}.layernorm_after.bias", f"blocks.{i}.norm2.bias"))
        rename_keys.append((f"vit.encoder.layer.{i}.intermediate.dense.weight", f"blocks.{i}.mlp.fc1.weight"))
        rename_keys.append((f"vit.encoder.layer.{i}.intermediate.dense.bias", f"blocks.{i}.mlp.fc1.bias"))
        rename_keys.append((f"vit.encoder.layer.{i}.output.dense.weight", f"blocks.{i}.mlp.fc2.weight"))
        rename_keys.append((f"vit.encoder.layer.{i}.output.dense.bias", f"blocks.{i}.mlp.fc2.bias"))

    # projection layer + position embeddings
    rename_keys.extend(
        [
            ("vit.embeddings.cls_token", "cls_token"),
            ("vit.embeddings.patch_embeddings.projection.weight", "patch_embed.proj.weight"),
            ("vit.embeddings.patch_embeddings.projection.bias", "patch_embed.proj.bias"),
            ("vit.embeddings.position_embeddings", "pos_embed"),
        ]
    )

    if base_model:
        # layernorm + pooler
        rename_keys.extend(
            [
                ("layernorm.weight","norm.weight"),
                ("layernorm.bias", "norm.bias"),
                ("pooler.dense.weight", "pre_logits.fc.weight"),
                ("pooler.dense.bias", "pre_logits.fc.bias"),
            ]
        )

        # if just the base model, we should remove "vit" from all keys that start with "vit"
        rename_keys = [(pair[0][4:], pair[1]) if pair[0].startswith("vit") else pair for pair in rename_keys]
    else:
        # layernorm + classification head
        rename_keys.extend(
            [
                ("vit.layernorm.weight","norm.weight"),
                ("vit.layernorm.bias", "norm.bias"),
                ("classifier.weight", "head.weight"),
                ("classifier.bias", "head.bias"),
            ]
        )

    return rename_keys


# we split up the matrix of each encoder layer into queries, keys and values
# concatenate them back to the qkv version.
def concat_qkv(state_dict, config, base_model=False):

    for i in range(config.num_hidden_layers):
        if base_model:
            prefix = ""
        else:
            prefix = "vit."
          
        # next, pop query, keys and values (in that order) from the state dict
        qw = state_dict.pop(f"{prefix}encoder.layer.{i}.attention.attention.query.weight")
        qb = state_dict.pop(f"{prefix}encoder.layer.{i}.attention.attention.query.bias") 
        kw = state_dict.pop(f"{prefix}encoder.layer.{i}.attention.attention.key.weight") 
        kb = state_dict.pop(f"{prefix}encoder.layer.{i}.attention.attention.key.bias")
        vw = state_dict.pop(f"{prefix}encoder.layer.{i}.attention.attention.value.weight")
        vb = state_dict.pop(f"{prefix}encoder.layer.{i}.attention.attention.value.bias")

        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        # concatenate dimension -> 0
        in_proj_weight = torch.cat([qw, kw, vw])
        in_proj_bias = torch.cat([qb, kb, vb])

        state_dict[f"blocks.{i}.attn.qkv.weight"] = in_proj_weight
        state_dict[f"blocks.{i}.attn.qkv.bias"] = in_proj_bias


def remove_classification_head_(state_dict):
    ignore_keys = ["classifier.weight", "classifier.bias"]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_vit_weights(vit_name, ckpt_path=None, device="cuda"):
    """Convert trained weights in the huggingface format to the timm format for visualization.

    Args:
        vit_name (str): name in the custom formatting. (encoder name)
        ckpt_path (str): path to the checkpoint stored in huggingface format.
        device (str): device to load the state dict.
    """
    if vit_name in MODEL_CONFIG.keys():
        vit_name = MODEL_CONFIG[vit_name]["name"]

    # define default ViT configuration
    config = ViTConfig()
    base_model = False 

    # dataset (ImageNet-21k only or also fine-tuned on ImageNet 2012), patch_size and image_size
    if vit_name[-5:] == "in21k":
        base_model = True
        config.patch_size = int(vit_name[-12:-10])
        config.image_size = int(vit_name[-9:-6])
    else:
        config.num_labels = 1000
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        config.patch_size = int(vit_name[-6:-4])
        config.image_size = int(vit_name[-3:])
    # size of the architecture
    if "deit" in vit_name:
        if vit_name[9:].startswith("tiny"):
            config.hidden_size = 192
            config.intermediate_size = 768
            config.num_hidden_layers = 12
            config.num_attention_heads = 3
        elif vit_name[9:].startswith("small"):
            config.hidden_size = 384
            config.intermediate_size = 1536
            config.num_hidden_layers = 12
            config.num_attention_heads = 6
        else:
            pass
    else:
        if vit_name[4:].startswith("small"):
            config.hidden_size = 768
            config.intermediate_size = 2304
            config.num_hidden_layers = 8
            config.num_attention_heads = 8
        elif vit_name[4:].startswith("base"):
            pass
        elif vit_name[4:].startswith("large"):
            config.hidden_size = 1024
            config.intermediate_size = 4096
            config.num_hidden_layers = 24
            config.num_attention_heads = 16
        elif vit_name[4:].startswith("huge"):
            config.hidden_size = 1280
            config.intermediate_size = 5120
            config.num_hidden_layers = 32
            config.num_attention_heads = 16

    # load HuggingFace model
    if vit_name[-5:] == "in21k":
        model = ViTModel(config).eval()
    else:
        model = ViTForImageClassification(config).eval()

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        
    state_dict = model.state_dict()

    if base_model:
        remove_classification_head_(state_dict)
    rename_keys = create_rename_keys(config, base_model)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    concat_qkv(state_dict, config, base_model)

    # load original model from timm
    timm_model = timm.create_model(vit_name, pretrained=False)

    # load state_dict of original model, remove and rename some keys
    timm_model.load_state_dict(state_dict)

    # Check outputs on an image, prepared by ViTFeatureExtractor/DeiTFeatureExtractor
    if "deit" in vit_name:
        feature_extractor = DeiTFeatureExtractor(size=config.image_size)
    else:
        feature_extractor = ViTFeatureExtractor(size=config.image_size)

    encoding = feature_extractor(images=prepare_img(), return_tensors="pt")
    pixel_values = encoding["pixel_values"]
    outputs = model(pixel_values)

    if base_model:
        timm_pooled_output = timm_model.forward_features(pixel_values)
        assert timm_pooled_output.shape == outputs.pooler_output.shape
        assert torch.allclose(timm_pooled_output, outputs.pooler_output, atol=1e-3)
    else:
        timm_logits = timm_model(pixel_values)
        assert timm_logits.shape == outputs.logits.shape
        assert torch.allclose(timm_logits, outputs.logits, atol=1e-3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--vit_name",
        default="vit_base_patch16_224",
        type=str,
        help="Name of the ViT timm model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", "-p", default=None, type=str, help="Path to the output PyTorch model directory."
    )

    args = parser.parse_args()
    convert_vit_weights(args.vit_name, args.pytorch_dump_folder_path)