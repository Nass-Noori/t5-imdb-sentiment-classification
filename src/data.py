"""Data loading and preprocessing for IMDB sentiment as a text-to-text task.

Key fix vs. the original notebook: the IMDB `test` split is held out and used
EXACTLY ONCE, at the very end of training, for final reporting. Model
selection during training (checkpointing / early stopping) uses a validation
split carved out of `train` instead. This avoids test-set leakage.
"""

from datasets import load_dataset, DatasetDict
from transformers import T5TokenizerFast, DataCollatorForSeq2Seq
import torch

LABEL_NAMES = ["negative", "positive"]
LABEL2ID = {"negative": 0, "positive": 1}


def id2label(ids):
    return [LABEL_NAMES[i] for i in ids]


def label2id(labels):
    # unknown/malformed generations (model didn't output "negative"/"positive")
    # are mapped to id 2 so they show up as wrong answers instead of crashing
    return [LABEL2ID.get(label, 2) for label in labels]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.replace("<br />", " ")
    return text


def load_and_split_dataset(config: dict) -> DatasetDict:
    """Load IMDB and split `train` into train/val. `test` is left untouched."""
    raw = load_dataset("imdb")
    raw.pop("unsupervised")

    split = raw["train"].train_test_split(
        test_size=config["val_split"],
        seed=config["seed"],
        shuffle=True,
    )

    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
            "test": raw["test"],  # untouched, used once at the very end
        }
    )


def build_tokenized_dataset(dataset: DatasetDict, tokenizer: T5TokenizerFast, config: dict) -> DatasetDict:
    def map_function(row):
        processed_input = [preprocess_text(text) for text in row["text"]]
        input_info = tokenizer(processed_input, truncation=True, max_length=config["max_length"])
        output_info = tokenizer(id2label(row["label"]))
        return {**input_info, "labels": output_info.input_ids}

    dataset = dataset.map(map_function, batched=True, remove_columns=["text", "label"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return dataset


def build_dataloaders(dataset: DatasetDict, tokenizer: T5TokenizerFast, config: dict):
    collate_fn = DataCollatorForSeq2Seq(tokenizer, return_tensors="pt", padding="longest")

    train_loader = torch.utils.data.DataLoader(
        dataset["train"],
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset["validation"],
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
