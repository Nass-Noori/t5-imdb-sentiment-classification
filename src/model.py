"""Model + tokenizer construction, kept separate so train.py / evaluate.py
can all load the same thing consistently.

Both full fine-tuning and LoRA fine-tuning go through this same function --
which method is used is controlled entirely by the `peft` section of the
config file, not by separate code paths. See configs/base.yaml (peft.method:
none) vs configs/lora.yaml (peft.method: lora).
"""

from transformers import T5TokenizerFast, T5ForConditionalGeneration

from lora import apply_lora, freeze_base_model


def build_tokenizer(config: dict) -> T5TokenizerFast:
    return T5TokenizerFast.from_pretrained(config["model_name"])


def build_model(config: dict) -> T5ForConditionalGeneration:
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

    peft_config = config.get("peft", {"method": "none"})
    if peft_config.get("method") == "lora":
        apply_lora(model, rank=peft_config["rank"], alpha=peft_config["alpha"])
        freeze_base_model(model)

    return model
