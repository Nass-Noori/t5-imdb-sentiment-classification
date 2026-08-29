"""Final, one-time evaluation on the held-out IMDB TEST set.

This is intentionally a separate script from train.py: the test set should
only be touched after model selection (checkpointing / early stopping) is
fully decided using the validation set. Running this script re-uses the
best checkpoint saved by train.py -- it does not retrain anything.
"""

import argparse
import json
import os

import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

from data import load_and_split_dataset, build_tokenized_dataset, build_dataloaders
from lora import apply_lora, freeze_base_model
from train import evaluate_loop
from utils import set_seed, load_config, get_device


def load_model_for_eval(config: dict, checkpoint_dir: str, device: torch.device):
    """Reconstruct the right model for eval. Full fine-tuning saved the whole
    model via save_pretrained(), so it loads directly. LoRA saved only the
    small adapter, so we rebuild base T5 + LoRA structure first, then load
    just the adapter weights on top.
    """
    tokenizer = T5TokenizerFast.from_pretrained(checkpoint_dir)

    if config.get("peft", {}).get("method") == "lora":
        model = T5ForConditionalGeneration.from_pretrained(config["model_name"])
        apply_lora(model, rank=config["peft"]["rank"], alpha=config["peft"]["alpha"])
        freeze_base_model(model)

        adapter_path = os.path.join(checkpoint_dir, "lora_adapter.pt")
        adapter_state = torch.load(adapter_path, map_location=device)
        missing, unexpected = model.load_state_dict(adapter_state, strict=False)
        # every "missing" key here should be a frozen base-model weight (expected,
        # since we only saved the adapter) -- but there should be zero "unexpected"
        # keys, since that would mean the adapter has params the model doesn't.
        assert len(unexpected) == 0, f"Unexpected keys when loading LoRA adapter: {unexpected}"
    else:
        model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir)

    return tokenizer, model.to(device)


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["seed"])
    device = get_device()

    checkpoint_dir = os.path.join(config["output_dir"], config["checkpoint_name"])
    tokenizer, model = load_model_for_eval(config, checkpoint_dir, device)

    dataset = load_and_split_dataset(config)
    dataset = build_tokenized_dataset(dataset, tokenizer, config)
    _train_loader, _val_loader, test_loader = build_dataloaders(dataset, tokenizer, config)

    test_metrics = evaluate_loop(model, test_loader, tokenizer, device)

    print("Final TEST set results (touched once, after model selection):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    results_path = os.path.join(config["output_dir"], "test_results.json")
    with open(results_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
