import math
import numpy as np
import torch
import cv2

import ultralytics.engine.predictor as engine_predictor
from ultralytics.data.loaders import SourceTypes
from ultralytics.models.sam import (
    SAM3VideoSemanticPredictor,
    SAM3SemanticPredictor,
    SAM3VideoPredictor,
    SAM3Predictor
)

class LoadTensorVideo:
    def __init__(self, tensor, batch: int = 1, vid_stride: int = 1):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
            
        if tensor.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got {tensor.ndim}D")
            
        if tensor.shape[1] in [1, 3]:
            tensor = np.transpose(tensor, (0, 2, 3, 1))
            
        if tensor.dtype in [np.float32, np.float64] and tensor.max() <= 1.0:
            tensor = (tensor * 255).astype(np.uint8)
        else:
            tensor = tensor.astype(np.uint8)
            
        self.tensor = tensor
        self.total_frames = self.tensor.shape[0]
        
        self.mode = "video"
        self.video_flag = [True]
        self.nf = 1
        self.bs = batch
        self.vid_stride = vid_stride
        self.frames = math.ceil(self.total_frames / self.vid_stride)
        self.mock_path = "mock.mp4"
        self.fps = 30
        
    def __iter__(self):
        self.frame_idx = 0
        self.frame = 0
        self.count = 0 
        return self
    
    def __next__(self) -> tuple[list[str], list[np.ndarray], list[str]]:
        paths, imgs, info = [], [], []
        
        while len(imgs) < self.bs:
            if self.frame_idx >= self.total_frames:
                if imgs:
                    return paths, imgs, info
                raise StopIteration
                
            im0 = self.tensor[self.frame_idx]
            if im0.shape[-1] == 3:
                im0 = im0[..., ::-1]
            im0 = np.ascontiguousarray(im0)
            
            self.frame += 1
            paths.append(self.mock_path)
            imgs.append(im0)
            info.append(f"video 1/1 (frame {self.frame}/{self.frames}) {self.mock_path}: ")
            self.frame_idx += self.vid_stride
            
        return paths, imgs, info
    
    def __len__(self) -> int:
        return math.ceil(self.frames / self.bs)

def get_patched_setup_source(original_method):
    def patched_setup_source(self, source):
        is_video_tensor = isinstance(source, (torch.Tensor, np.ndarray)) and len(source.shape) == 4
        
        if is_video_tensor:
            original_load = engine_predictor.load_inference_source
            
            def custom_load_inference_source(source, batch=1, vid_stride=1, buffer=False, channels=3, **kwargs):
                dataset = LoadTensorVideo(source, batch=batch, vid_stride=vid_stride)
                dataset.source_type = SourceTypes(stream=False, screenshot=False, from_img=False, tensor=True)
                return dataset
            
            engine_predictor.load_inference_source = custom_load_inference_source
            
            try:
                return original_method(self, source)
            finally:
                engine_predictor.load_inference_source = original_load
        else:
            return original_method(self, source)
            
    return patched_setup_source

def apply_patch():
    predictor_classes = [
        SAM3VideoSemanticPredictor,
        SAM3SemanticPredictor,
        SAM3VideoPredictor,
        SAM3Predictor
    ]
    
    for cls in predictor_classes:
        if hasattr(cls, "setup_source"):
            old_method = cls.setup_source
            if old_method.__name__ != "patched_setup_source":
                cls.setup_source = get_patched_setup_source(old_method)
                
    print(f"[EasySAM3] Successfully patched {len(predictor_classes)} SAM3 Predictors.")