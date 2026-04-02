import io
import json
import time
import base64
import numpy as np
from PIL import Image
from aiohttp import web
from server import PromptServer
import comfy.model_management as mm

class EasySAM3PointCollector:
    status_storage = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image": ("IMAGE",)},
            "hidden": {"unique_id": "UNIQUE_ID"}
        }
    
    FUNCTION = "main"
    CATEGORY = "EasySAM3"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos_points","neg_points")
    DESCRIPTION = 'Click "Continue" button after marking points to continue.'
    
    @classmethod
    def IS_CHANGED(cls, image, unique_id):
        return float("nan")

    def main(self, image, unique_id):
        img_base64 = self.tensor_to_base64(image)
        self.status_storage[unique_id] = {"status": "paused", "data": None}
        
        PromptServer.instance.send_sync("easysam3_show_image", {
            "node_id": unique_id, 
            "bg_image": img_base64
        })
        
        print(f"[EasySAM3] Node {unique_id} is waiting for user input...")
        while self.status_storage[unique_id]["status"] == "paused":
            mm.throw_exception_if_processing_interrupted()
            time.sleep(0.5)
            
        final_data = self.status_storage[unique_id]["data"]
        pos = json.dumps(final_data.get("positive", []))
        neg = json.dumps(final_data.get("negative", []))
        print(f"[EasySAM3] Positive points: {pos}")
        print(f"[EasySAM3] Negative points: {neg}")

        del self.status_storage[unique_id]
        return (pos, neg)

    def tensor_to_base64(self, tensor):
        img_array = (tensor[0].cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

@PromptServer.instance.routes.post("/sam3_point/continue/{node_id}")
async def handle_continue(request):
    node_id = request.match_info["node_id"].strip()
    post_data = await request.json()
    
    if node_id in EasySAM3PointCollector.status_storage:
        EasySAM3PointCollector.status_storage[node_id]["data"] = post_data.get("points")
        EasySAM3PointCollector.status_storage[node_id]["status"] = "continue"
        return web.json_response({"status": "ok"})
    
    return web.json_response({"status": "error", "message": "node not found"}, status=404)