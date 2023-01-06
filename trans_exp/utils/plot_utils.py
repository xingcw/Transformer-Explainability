import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from einops import rearrange

import matplotlib.pyplot as plt
import torch.nn  as nn
import torchvision.transforms as transforms
from torch.nn.functional import interpolate

from trans_exp.baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from trans_exp.baselines.ViT.ViT_explanation_generator import LRP
from trans_exp.utils.load_weights import load_weights
from pipelines.models.student import StudentPolicy

# create heatmap from mask on image
def show_cam_on_image(img, mask):
    """show attention heatmap on top of the raw images.

    Args:
        img (np.ndarray): original image
        mask (np.ndarray): attention weights
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    vis = heatmap + np.float32(img)
    vis = vis / np.max(vis)
    vis = cv2.cvtColor(np.array(np.uint8(255 * vis)), cv2.COLOR_RGB2BGR)
    
    return vis


def standardize(x: np.array):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def visualize(attr_gen: LRP, images, class_index=None, use_thresholding=False):
    """Get the visualizations.

    Args:
        attr_gen (LRP): _description_
        images (np.ndarray): images to be visualized.
        class_index (int, optional): _description_. Defaults to None.
        use_thresholding (bool, optional): _description_. Defaults to False.

    """
    trans_gen = attr_gen.generate_LRP(images, 
                                      method="transformer_attribution", 
                                      index=class_index).detach()
    trans_gen = rearrange(trans_gen, "(a b) (h w) -> a b h w", a=1, h=14)
    trans_gen = interpolate(trans_gen, scale_factor=16, mode='bilinear')
    trans_gen = trans_gen.data.cpu().numpy()
    
    new_trans_gen = []
    for t in trans_gen:
        t = standardize(t.squeeze())
        if use_thresholding:
            t = (t * 255).astype(np.uint8)
            ret, trans_gen = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t[t == 255] = 1
        new_trans_gen.append(t)
    
    image_trans_gen = [standardize(im.permute(1, 2, 0).data.cpu().numpy()) for im in images]
    
    return image_trans_gen, new_trans_gen


def make_attention_map_video(data_dir, env_id, model: StudentPolicy, transform, weight_dirs, save_dir, fps=10):
    """Create videos for attention maps visualization.

    Args:
        data_dir (str): path to the collected data.
        env_id (int): env id for data dir.
        model (nn.Module): models to vis.
        transform (transforms.Compose): torch transformations.
        save_dir (str): path to save video.
        fps (int, optional): video fps. Defaults to 10.
    """

    data_dir = data_dir / f"data/{env_id:03d}"
    save_dir = save_dir / f"vis/{data_dir.parent.parent.stem}_env_{env_id:03d}"
    os.makedirs(save_dir, exist_ok=True)
    
    weight_dirs = [None] + weight_dirs   # make a place holder for the pretrained weights
    
    original_img_path = save_dir / f"original_images"
    os.makedirs(original_img_path, exist_ok=True)
    
    video_path = save_dir / "comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(video_path), fourcc, 15, (224 * len(weight_dirs), 224))
    
    images_per_model, heatmaps_per_model = [], []
    
    for weight_path in weight_dirs:
        
        model.cuda()
        
        if weight_path:
            # load custom trained weights
            # NOTE: no weights for head.weight and head.bias in the custom weights
            weights = load_weights(weight_path, str(weight_path.stem).split("_")[0], "cuda")
            # print(weights.keys())
            # print(weights["blocks.0.attn.proj.weight"])
            model.load_state_dict(weights, strict=False)
            model.eval()
            # store the images for convenience
            save_img_path = save_dir / weight_path.stem
            os.makedirs(save_img_path, exist_ok=True)
        else:
            save_img_path = save_dir / "original_vit"
            os.makedirs(save_img_path, exist_ok=True)
            
        # for k, v in model.named_parameters():
        #     if k == "blocks.0.attn.proj.weight":
        #         print(v.shape)
        #         print(v)
            
        attr_gen = LRP(model)
        
        # batch inference is not supported yet
        num_steps, batch_size = 500, 1
        num_batch = np.ceil(num_steps / batch_size).astype(int)
        start_ids = [i * batch_size for i in range(num_batch)]
        end_ids = start_ids[1:] + [num_steps]
        
        images_all, heatmaps_all = [], []
        
        for start, end in tqdm(zip(start_ids, end_ids), total=num_batch):
            images = []
            for j in range(start, end):
                img_path = data_dir / f"{j:06d}.npz"
                image_np = np.load(img_path)["rgb"].squeeze()
                image = Image.fromarray(image_np)
                image = transform(image)
                images.append(image)

            images_cuda = torch.stack(images).cuda()
            images, heatmaps = visualize(attr_gen, images_cuda)
            images_all += images
            heatmaps_all += heatmaps
            
            del images_cuda
            
        images_per_model.append(images_all)
        heatmaps_per_model.append(heatmaps_all)
        
    images_per_model = torch.from_numpy(np.asarray(images_per_model))
    images_per_step = rearrange(images_per_model, "nm ns h w c -> ns nm h w c").numpy()
    heatmaps_per_model = torch.from_numpy(np.asarray(heatmaps_per_model))
    heatmaps_per_step = rearrange(heatmaps_per_model, "nm ns h w -> ns nm h w").numpy()
    
    for i, (images, heatmaps) in tqdm(enumerate(zip(images_per_step, heatmaps_per_step)), total=len(images_per_step)):
        
        masked_images = []
        for weight_dir, image, heatmap in zip(weight_dirs, images, heatmaps):
            vis = show_cam_on_image(image, heatmap)
            masked_images.append(vis)
            # cv2.imshow("original image", image)
            cv2.imwrite(str(original_img_path / f"{i:03d}.png"), image)
            # cv2.imshow("vis attention", vis)
            ckpt_name = "original_vit" if not weight_dir else weight_dir.stem
            cv2.imwrite(str(save_dir / ckpt_name / f"{i:03d}.png"), vis)
        
        concat_img = cv2.cvtColor(cv2.hconcat(masked_images), cv2.COLOR_BGR2RGB)
        cv2.imshow("comparison", concat_img)
        video.write(concat_img)
        cv2.waitKey(int(1000/fps))
            
    video.release()
    
    
if __name__ == "__main__":
    
    flightmare_path = Path(os.environ["FLIGHTMARE_PATH"])
    data_dir = flightmare_path / "flightpy/results/students/teacher_PPO_5/12-26-14-24-47"
    res_dir = flightmare_path / "flightpy/results/students/teacher_PPO_5/multivit_01-04-13-17-37"
    
    # get trained models to be visualized
    weight_dirs = []
    weight_path = res_dir / "model/multivit_ep00_data_data_024768.pth"   
    weight_dirs.append(weight_path)
    weight_path = res_dir / "model/multivit_ep95_data_data_2377728.pth"
    weight_dirs.append(weight_path)
    
    # initialize ViT pretrained
    model = vit_LRP(pretrained=True)
    
    # from dataset "12-26-14-24-47"
    normalize = transforms.Normalize(mean=[0.48236614, 0.5381361, 0.5536909], 
                                     std=[0.27358302, 0.2505713, 0.255988])
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
    ])
    
    make_attention_map_video(data_dir, 0, model, transform, weight_dirs, res_dir)
    