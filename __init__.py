import os
import gc
import json
import torch
import torch.nn.functional as F

from .utils.cqdm import cqdm
from .utils.patch import apply_patch
from .utils.sam3_interactive import EasySAM3PointCollector
from modelscope.hub.snapshot_download import snapshot_download
from ultralytics.models.sam import (
    SAM3VideoSemanticPredictor,
    SAM3SemanticPredictor,
    SAM3VideoPredictor,
    SAM3Predictor
)

import folder_paths
import comfy.model_management as mm

apply_patch()
model_folder = os.path.join(folder_paths.models_dir, "sam3")
model_path = os.path.join(model_folder, "sam3.pt")

def parse_points_from_string(point_str, label_value=1):
    if not point_str:
        return None, None
    
    try:
        data = json.loads(point_str)
        if not data:
            return None, None
        
        pts = [[float(item['x']), float(item['y'])] for item in data]
        lbs = [label_value] * len(pts)
        
        return torch.tensor(pts), torch.tensor(lbs)
    except Exception as e:
        return None, None

def parse_points(pos_points_str, neg_points_str):
    p_pts, p_lbs = parse_points_from_string(pos_points_str, label_value=1)
    n_pts, n_lbs = parse_points_from_string(neg_points_str, label_value=0)
    
    custom_points = None
    custom_labels = None
    
    if p_pts is not None and n_pts is not None:
        custom_points = torch.cat([p_pts, n_pts], dim=0)
        custom_labels = torch.cat([p_lbs, n_lbs], dim=0)
    elif p_pts is not None:
        custom_points, custom_labels = p_pts, p_lbs
    elif n_pts is not None:
        custom_points, custom_labels = n_pts, n_lbs
    
    return custom_points, custom_labels

def get_mask_by_track_id(result, size, track_id=-1):
    if result.masks is None or len(result.masks) == 0:
        return torch.zeros(size, dtype=torch.float32, device="cpu")
    
    masks_tensor = result.masks.data 
    height, width = size
    mask_orig_size = F.interpolate(masks_tensor.unsqueeze(0), size=(height, width), mode="nearest")[0]
    
    if track_id == -1:
        merged = torch.any(mask_orig_size.bool(), dim=0).float()
        return merged
    else:
        if result.boxes is None or result.boxes.id is None:
            if track_id < len(mask_orig_size):
                return mask_orig_size[track_id]
            else:
                return torch.zeros(size, dtype=torch.float32, device="cpu")
            
        ids = result.boxes.id
        idx = (ids == track_id).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return torch.zeros(size, dtype=torch.float32, device="cpu")
        return mask_orig_size[idx[0]]

class EasySAM3Segment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "default":"",
                    "tooltip": "Supports multiple text prompts, separated by commas (,)"
                }),
                "threshold": ("FLOAT", {"default": 0.45, "min": 0.00, "max":1.00, "step":0.01}),
                "object_id": ("INT", {
                    "default": -1, "min": -1, "max": 1024, "step": 1,
                    "tooltip": "ID of the masked object, -1=all"
                }),
                "visualize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Output segmentation preview image"
                })
            },
            "optional": {
                "points": ("points", {"forceInput": True})
            }
        }
    
    FUNCTION = "main"
    CATEGORY = "EasySAM3"
    RETURN_TYPES = ("MASK", "IMAGE")
    DESCRIPTION = "Run SAM3 segmentation using Ultralytics."
    
    def main(predictor, image, prompt, threshold, object_id, visualize, points=[None, None]):
        if points == [None, None] and prompt == "":
            raise ValueError("You must provide any text or point prompts!")
                
        if not os.path.exists(model_path):
            print("[EasySAM3] Downloading SAM3 from modelscope...")
            _save_dir = snapshot_download(model_id="facebook/sam3", file_path='sam3.pt', local_dir=model_folder)
        
        num_frames, height, width, channels = image.shape
        prompt_mode = "text" if points == [None, None] else "point"
        print(f"[EasySAM3] Prompt mode: {prompt_mode}")
        
        prompts = prompt.split(",")
        sam_points, sam_labels = parse_points(points[0], points[1])
        source_input = image if num_frames > 1 else [img[..., ::-1] for img in (image * 255).byte().numpy()]
        
        if visualize:
            final_plot_tensor = torch.zeros((num_frames, height, width, 3), dtype=torch.float32, device="cpu")
        final_masks_tensor = torch.zeros((num_frames, height, width), dtype=torch.float32, device="cpu")
        overrides = dict(conf=threshold, task="segment", mode="predict", model=model_path, half=True, save=False, verbose=False)
        
        print("[EasySAM3] Loading SAM3...")
        try:
            
            if num_frames > 1:
                if prompt_mode == "text":
                    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
                else:
                    predictor = SAM3VideoPredictor(overrides=overrides)
            else:
                if prompt_mode == "text":
                    predictor = SAM3SemanticPredictor(overrides=overrides)
                else:
                    predictor = SAM3Predictor(overrides=overrides)
            predictor.setup_model(verbose=False)
            
            if prompt_mode == "text":
                results = predictor(source=source_input, text=prompts, stream=True)
            else:
                results = predictor(source=source_input, points=sam_points, labels=sam_labels, stream=True)
                
            print("[EasySAM3] Starting inference...")
            for index, result in cqdm(enumerate(results), total=num_frames):
                mask = get_mask_by_track_id(result, (height, width), track_id=object_id)
                final_masks_tensor[index] = mask
                
                if visualize:
                    plot_img = result.plot(show=False)
                    plot_img_rgb = plot_img[..., ::-1].copy()
                    plot_img_tensor = torch.from_numpy(plot_img_rgb).float() / 255.0
                    final_plot_tensor[index] = plot_img_tensor
        finally:
            if 'predictor' in locals():
                del predictor
            if 'source_input' in locals():
                del source_input
            gc.collect()
            mm.soft_empty_cache()
            
        if visualize:
            return (final_masks_tensor, final_plot_tensor)
        
        empty_image = torch.zeros((1, height, width, 3), dtype=torch.float32, device="cpu")
        return (final_masks_tensor, empty_image)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "EasySAM3Segment": EasySAM3Segment,
    "EasySAM3PointCollector": EasySAM3PointCollector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EasySAM3Segment": "EasySAM3 Segment",
    "EasySAM3PointCollector": "EasySAM3 Point Editor",
}