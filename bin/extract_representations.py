import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from x_distill.helpers import fetch_annotations
from x_distill.paths import DATA_DIR


def main(args):
    model_name = args.model_name
    token = args.token

    _ = torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    annotations = fetch_annotations()

    SAVE_DIR = f"{DATA_DIR}/{model_name}/{token}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    for image_id, annotation in tqdm(annotations.items()):
        tokenized = tokenizer(annotation, return_tensors="pt")
        input_ids = tokenized.input_ids.to(model.device)

        output = model(
            input_ids=input_ids, return_dict=True, output_hidden_states=True
        )["hidden_states"]
        output = torch.stack([block.cpu()[0] for block in output])

        if token == "pool":
            # Average activity across tokens for each block in the output list
            output = output.mean(dim=-1)
        else:  # take the last token
            output = output[:, -1]

        torch.save(output, os.path.join(SAVE_DIR, f"{image_id}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process annotations with a language model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-70B",
        help="Name or path of the pretrained model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        choices=["pool", "last"],
        default="last",
        help="Token pooling strategy.",
    )
    args = parser.parse_args()
    main(args)
