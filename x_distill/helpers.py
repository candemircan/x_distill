from collections import defaultdict
from typing import Dict, List
import json

from .paths import PROJECT_DIR


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
