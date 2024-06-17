from collections import defaultdict
from typing import Dict, List
import json

import clip
import torch

from .paths import PROJECT_DIR


def load_clip(
    model_name: str, device: str
) -> tuple[torch._script.RecursiveScriptModule, torch.Compose]:
    """
    Load and return the vision encoder of the specified CLIP model and the corresponding preprocessing transforms.
    All but the final projection parameters are frozen. Only works for ViT models for now.

    Args:
        model_name (str): Must be one of OpenAI clip models. Call `clip.available_models()` to see options.
        device (str): "cpu" or "cuda"

    Returns:
        tuple[torch._script.RecursiveScriptModule, torch.Compose]: Jitted vision encoder and the image transforms.
    """
    model, preprocess = clip.load(model_name, device=device, jit=True)
    model = model.visual

    # freeze all
    for _, param in model.named_parameters():
        param.requires_grad = False

    # resnets don't have a projection layer
    if "ViT" in model_name:
        # unfreeze the final projection
        model.proj.requires_grad = True

    model.train()
    return model, preprocess


def fetch_annotations(
    paths: List[str] = [
        f"{PROJECT_DIR}/data/coco/annotations/captions_train2017.json",
        f"{PROJECT_DIR}/data/coco/annotations/captions_val2017.json",
    ],
) -> Dict[int, str]:
    source_captions = []
    for path in paths:
        source_captions.append(json.load(open(path)))

    caption_dictionary = {}
    annotations_by_id = defaultdict(list)

    for caption_dict in source_captions:
        for annotation in caption_dict["annotations"]:
            annotations_by_id[annotation["image_id"]].append(annotation["caption"])

        for item in caption_dict["images"]:
            cur_id = item["id"]
            captions = annotations_by_id[cur_id]
            caption_dictionary[cur_id] = " ".join(captions)

    return caption_dictionary
