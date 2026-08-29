"""
Fine-tune T5 for IMDB sentiment as text-to-text generation.

"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from data import load_and_split_dataset, build_tokenized_dataset, build_dataloaders, id2label, label2id
from lora import lora_state_dict, count_parameters
from metrics import compute_metrics
from model import build_tokenizer, build_model
from utils import set_seed, load_config, get_device


def save_checkpoint(model, tokenizer, checkpoint_dir: str, config: dict) -> None:
    """Save the best-so-far model. For LoRA, only the small adapter weights
    are saved (base T5 weights are unchanged and can be re-downloaded), which
    is one of the practical wins of LoRA worth highlighting on a resume.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)

    if config.get("peft", {}).get("method") == "lora":
        adapter_path = os.path.join(checkpoint_dir, "lora_adapter.pt")
        torch.save(lora_state_dict(model), adapter_path)
    else:
        model.save_pretrained(checkpoint_dir)


def train_one_epoch(model, loader: DataLoader, optimizer, scheduler, grad_clip_norm: float, device) -> dict:
    model.train()
    batch_losses = []

    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()

        batch = batch.to(device)
        out = model(**batch)
        loss = out.loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        scheduler.step()

        batch_losses.append(loss.item())

    return {"train_loss": float(np.mean(batch_losses))}


@torch.no_grad()
def evaluate_loop(model, loader: DataLoader, tokenizer, device, max_gen_length: int = 5) -> dict:
    model.eval()

    all_true_ids, all_pred_ids = [], []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = batch.to(device)
        generated = model.generate(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            max_length=max_gen_length,
        )

        all_true_ids += batch.labels.detach().cpu().tolist()
        all_pred_ids += generated.detach().cpu().tolist()

    y_true = label2id(tokenizer.batch_decode(all_true_ids, skip_special_tokens=True))
    y_pred = label2id(tokenizer.batch_decode(all_pred_ids, skip_special_tokens=True))

    return compute_metrics(y_true, y_pred)


def main(config_path: str):
    config = load_config(config_path)
    set_seed(config["seed"])
    device = get_device()
    os.makedirs(config["output_dir"], exist_ok=True)

    tokenizer = build_tokenizer(config)
    dataset = load_and_split_dataset(config)
    dataset = build_tokenized_dataset(dataset, tokenizer, config)
    train_loader, val_loader, _test_loader = build_dataloaders(dataset, tokenizer, config)

    model = build_model(config).to(device)

    param_counts = count_parameters(model)
    print(
        f"Trainable params: {param_counts['trainable_params']:,} / "
        f"{param_counts['total_params']:,} ({param_counts['trainable_pct']}%)"
    )

    # only pass trainable params to the optimizer -- with LoRA this excludes
    # the frozen base weights, which is what actually makes LoRA cheaper
    # (smaller optimizer state, not just fewer gradients)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=config["learning_rate"]
    )

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history = []

    checkpoint_dir = os.path.join(config["output_dir"], config["checkpoint_name"])

    for epoch in range(config["epochs"]):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, config["grad_clip_norm"], device
        )
        val_metrics = evaluate_loop(model, val_loader, tokenizer, device)

        row = {"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items() if k != "confusion_matrix"}}
        history.append(row)
        print(
            f"epoch {epoch} | train_loss={row['train_loss']:.4f} | "
            f"val_accuracy={row['val_accuracy']:.4f} | val_f1={row['val_f1']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            save_checkpoint(model, tokenizer, checkpoint_dir, config)
            print(f"  -> new best val_accuracy={best_val_accuracy:.4f}, checkpoint saved to {checkpoint_dir}")
        else:
            epochs_without_improvement += 1
            print(f"  -> no improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= config["early_stopping_patience"]:
            print(f"Early stopping triggered after epoch {epoch} (patience={config['early_stopping_patience']})")
            break

    history_path = os.path.join(config["output_dir"], "train_history.json")
    with open(history_path, "w") as f:
        json.dump({"param_counts": param_counts, "epochs": history}, f, indent=2)

    print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best checkpoint saved at: {checkpoint_dir}")
    print(f"Training history saved at: {history_path}")
    print(f"\nRun `python src/evaluate.py --config {config_path}` to get the final TEST set score.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()
    main(args.config)
