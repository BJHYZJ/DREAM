import os
from pathlib import Path
from typing import List, Tuple, Type

import cv2
import numpy as np
import rerun as rr
import torch
import wget
from image_processors.image_processor import ImageProcessor
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, Owlv2ForObjectDetection

def _hf_hub_root() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("TRANSFORMERS_CACHE"):
        return Path(os.environ["TRANSFORMERS_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"

def _is_valid_snapshot(path: Path) -> bool:
    if not path.exists():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()
    return has_config and has_weights

def _find_local_hf_snapshot(repo_id: str):
    model_dir_name = "models--" + repo_id.replace("/", "--")
    base = _hf_hub_root() / model_dir_name
    snapshots = base / "snapshots"
    if not snapshots.exists():
        return None
    ref_main = base / "refs" / "main"
    if ref_main.exists():
        commit = ref_main.read_text().strip()
        candidate = snapshots / commit
        if _is_valid_snapshot(candidate):
            return str(candidate)
    candidates = sorted(
        [p for p in snapshots.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _is_valid_snapshot(candidate):
            return str(candidate)
    return None

class _LoadPlan:
    def __init__(self, load_target: str, local_only: bool, local_snapshot: str):
        self.load_target = load_target
        self.local_only = local_only
        self.local_snapshot = local_snapshot

def build_hf_load_plan(repo_id: str, friendly_name: str):
    local_snapshot = _find_local_hf_snapshot(repo_id)
    load_target = local_snapshot if local_snapshot is not None else repo_id
    local_only = local_snapshot is not None
    return _LoadPlan(load_target, local_only, local_snapshot)


class OWLSAMProcessor(ImageProcessor):
    def __init__(self, device="cuda"):
        super().__init__()
        print('Loading OWLSAMv2')
        self.device = device

        configuration = "google/owlv2-large-patch14-ensemble"
        load_plan = build_hf_load_plan(configuration, "OWL")
        self.processor = AutoProcessor.from_pretrained(
            load_plan.load_target, local_files_only=load_plan.local_only, use_fast=False
        )
        self.model = Owlv2ForObjectDetection.from_pretrained(
            load_plan.load_target, local_files_only=load_plan.local_only
        ).to(self.device)

        sam_checkpoint = f"../../checkpoints/sam2/sam2.1_hiera_small.pt"
        sam_config = "configs/sam2.1/sam2.1_hiera_s.yaml"
        if not os.path.exists(sam_checkpoint):
            wget.download(
                "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
                out=sam_checkpoint,
            )
        sam2_model = build_sam2(
            sam_config, sam_checkpoint, device=self.device, apply_postprocessing=False
        )
        self.sam_predictor = SAM2ImagePredictor(sam2_model)

    def detect_obj(
        self,
        image: Type[Image.Image],
        text: str = None,
        bbox: List[int] = None,
        visualize_box: bool = False,
        box_filename: str = None,
        visualize_mask: bool = False,
        mask_filename: str = None,
    ) -> Tuple[np.ndarray, List[int]]:
        # print("OWLSAM detection !!!")
        inputs = self.processor(text=[["a photo of a " + text]], images=image, return_tensors="pt")
        for input in inputs:
            inputs[input] = inputs[input].to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]]).to("cuda")
        results = self.processor.image_processor.post_process_object_detection(
            outputs, threshold=0.05, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) == 0:
            return None, None

        bounding_box = results["boxes"][torch.argmax(results["scores"])]

        bounding_boxes = bounding_box.unsqueeze(0)

        self.sam_predictor.set_image(image)
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None, point_labels=None, box=bounding_boxes, multimask_output=False
        )
        if len(masks) == 0:
            return None, None
        mask = torch.Tensor(masks).bool()[0]

        seg_mask = mask.detach().cpu().numpy()
        bbox = np.array(bounding_box.detach().cpu(), dtype=int)

        if visualize_mask:
            self.draw_bounding_box(image, bbox, box_filename)
            self.draw_mask_on_image(image, seg_mask, mask_filename)
            # if mask_filename is not None:
            #     rr.log(
            #         "object_detection_results",
            #         rr.Image(cv2.imread(mask_filename)[:, :, [2, 1, 0]]),
            #         static=True,
            #     )

        return seg_mask, bbox
