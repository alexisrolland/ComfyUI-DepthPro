import numpy as np
import os
import torch
from torchvision.transforms import Normalize
from tqdm import trange

import comfy.utils
import folder_paths
import model_management 

from . import depth_pro


class LoadDepthPro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "precision": (["fp16", "fp32"],),
                },
            }
    
    RETURN_TYPES = ("DEPTHPRO_MODEL",)
    RETURN_NAMES = ("DEPTHPRO_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "DepthPro"
    
    def load_model(self, precision):
        device = model_management.get_torch_device()
        dtype = torch.float16 if precision == "fp16" else torch.float32
        
        depth_model_path = os.path.join(folder_paths.models_dir, "depth", "ml-depth-pro")
        if not os.path.exists(depth_model_path):
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="spacepxl/ml-depth-pro",
                local_dir=depth_model_path,
                local_dir_use_symlinks=False,
                )
        
        depth_model_path = os.path.join(depth_model_path, "depth_pro.fp16.safetensors")
        model, transform = depth_pro.create_model_and_transforms(depth_model_path, device=device, precision=dtype)
        model.eval()
        
        model_dict = {
            "model": model,
            "device": device,
            "dtype": dtype,
            }
        
        return (model_dict,)


class RunDepthPro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "DEPTHPRO_MODEL": ("DEPTHPRO_MODEL",),
                "IMAGE": ("IMAGE",),
                "invert": ("BOOLEAN", {"default": True,}),
                "gamma": ("FLOAT", {"default": 0.35, "min": 0.01, "max": 100, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "LIST", "FLOAT",)
    RETURN_NAMES = ("IMAGE", "DEPTH", "FOCAL_LIST", "FOCAL_AVG",)
    FUNCTION = "estimate_depth"
    CATEGORY = "DepthPro"
    
    def estimate_depth(self, DEPTHPRO_MODEL, IMAGE, invert, gamma):
        model = DEPTHPRO_MODEL["model"]
        device = DEPTHPRO_MODEL["device"]
        dtype = DEPTHPRO_MODEL["dtype"]
        
        transform = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        rgb = IMAGE.unsqueeze(0) if len(IMAGE.shape) < 4 else IMAGE
        rgb = rgb.movedim(-1, 1) # BCHW
        
        depth = []
        focal_px = []
        
        pbar = comfy.utils.ProgressBar(rgb.size(0)) if comfy.utils.PROGRESS_BAR_ENABLED else None
        for i in trange(rgb.size(0)):
            rgb_image = rgb[i, :3].unsqueeze(0).to(device, dtype=dtype)
            rgb_image = transform(rgb_image)
            
            prediction = model.infer(rgb_image)
            depth.append(prediction["depth"].unsqueeze(-1))
            focal_px.append(prediction["focallength_px"].item())
            if pbar is not None: pbar.update(1)
        
        depth = torch.stack(depth, dim=0).repeat(1,1,1,3)
        focal_list = focal_px
        focal_avg = np.mean(focal_px)
        
        # Convert depth metrics to image
        relative_depth = 1 / (1 + depth.detach().clone())

        for i in range(relative_depth.size(0)):
            relative_depth[i] = relative_depth[i] - relative_depth[i].min()
            relative_depth[i] = relative_depth[i] / relative_depth[i].max()
        
        if not invert:
            relative_depth = 1 - relative_depth
        
        if gamma != 1:
            relative_depth = relative_depth ** (1 / gamma)

        return (relative_depth, depth, focal_list, focal_avg)
