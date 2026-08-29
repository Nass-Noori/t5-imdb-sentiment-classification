# T5 IMDB Sentiment Classification — Full Fine-Tuning vs. LoRA

Fine-tunes [T5-small](https://huggingface.co/t5-small) on the [IMDB movie review dataset](https://huggingface.co/datasets/imdb), framing binary sentiment classification as **text-to-text generation**: instead of predicting a class index (0/1), the model is trained to generate the literal word `"positive"` or `"negative"` given a review as input. The same task is fine-tuned two ways — **full fine-tuning** (all ~60M parameters) and **LoRA** (a hand-implemented low-rank adapter, ~a few thousand trainable parameters) — using the exact same data pipeline, training loop, and evaluation code, so the two are directly comparable.

```
Input:  "this movie was completely a waste of time"
Output: "negative"
```

## Why frame classification as generation?

T5 is pretrained purely as a text-to-text model — every task it has ever seen (translation, summarization, QA) is framed as "input text in, output text out." Rather than bolting a classification head onto the encoder and throwing away that pretraining objective, this project keeps the model in its native generation mode and fine-tunes it to emit the label as a word.

The point of this project is to explore what fine-tuning a generative model looks like end-to-end: tokenization of both inputs _and_ targets, generation-based evaluation instead of a softmax + argmax, and the failure modes that come with it.

## Full fine-tuning vs. LoRA

Both methods share `src/train.py`, `src/evaluate.py`, `src/data.py`, and `src/metrics.py` — the only difference is which config file is passed in. `src/model.py` reads the `peft` section of the config and decides at build time whether to fine-tune every weight or freeze the base model and train only LoRA adapters:

|                  | Full fine-tuning (`configs/base.yaml`) | LoRA (`configs/lora.yaml`)                                                   |
| ---------------- | -------------------------------------- | ---------------------------------------------------------------------------- |
| Trainable params | 100% of T5-small (~60M)                | Only the injected rank-`r` A/B matrices on the `q`/`v` attention projections |
| Checkpoint saved | Full model (`model.save_pretrained`)   | Adapter weights only (`lora_adapter.pt`), a few hundred KB instead of ~230MB |
| Learning rate    | `1e-5`                                 | `1e-4` (LoRA tolerates a higher LR since far fewer params move)              |
| `src/lora.py`    | not used                               | implements `LoRALinear`, `apply_lora`, `freeze_base_model`                   |

The LoRA implementation follows [Hu et al., 2021](https://arxiv.org/abs/2106.09685): each targeted `nn.Linear` is wrapped with a frozen copy of itself plus a trainable low-rank update `(alpha/rank) * B(A(x))`, where `A` is Kaiming-initialized and `B` is zero-initialized so training starts exactly at the pretrained model's behavior. Only the `q` (query) and `v` (value) projections inside T5's self-attention blocks are adapted. `tests/test_lora.py` has sanity checks for this (zero-init no-op, correct freezing, correct target layers) — run with:

```bash
python tests/test_lora.py
```

## Project structure

```
t5-imdb-sentiment-classification/
├── configs/
│   ├── base.yaml           # full fine-tuning config
│   └── lora.yaml            # LoRA config (same schema, different `peft` section)
├── src/
│   ├── data.py              # dataset loading, train/val/test split, tokenization
│   ├── model.py              # tokenizer + model construction (branches on peft.method)
│   ├── lora.py                 # LoRA layer, injection, freezing, adapter (de)serialization
│   ├── metrics.py                # accuracy, precision, recall, F1, confusion matrix
│   ├── train.py                   # training loop: scheduler, checkpointing, early stopping
│   └── evaluate.py                 # one-time final evaluation on the held-out test set
├── tests/
│   └── test_lora.py         # sanity checks for the LoRA implementation
├── outputs/                # created at runtime: checkpoints, logs, results (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/t5-imdb-sentiment-classification.git
cd t5-imdb-sentiment-classification

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.10+. A CUDA-capable GPU is strongly recommended — T5-small full fine-tuning on 22,500 training reviews for even a few epochs is slow on CPU.

## Usage

**1. Train.** Trains on 90% of the IMDB train split, validates on the remaining 10%, and never touches the test set. Pick a config to choose the method:

```bash
python src/train.py --config configs/base.yaml   # full fine-tuning
python src/train.py --config configs/lora.yaml    # LoRA
```

This will:

- Print the trainable / total parameter count at startup (this is where the LoRA vs. full fine-tuning difference is immediately visible)
- Save the best checkpoint (by validation accuracy) to `outputs/full_finetune/best_model/` or `outputs/lora/best_model/`
- Save per-epoch metrics to `<output_dir>/train_history.json`
- Stop early if validation accuracy doesn't improve for `early_stopping_patience` epochs (default: 3)

**2. Evaluate on the test set.** Run exactly once per method, after training is finished, using the saved checkpoint — pass the matching config:

```bash
python src/evaluate.py --config configs/base.yaml
python src/evaluate.py --config configs/lora.yaml
```

This writes final accuracy / precision / recall / F1 / confusion matrix to `<output_dir>/test_results.json`.

## Configuration

All hyperparameters live in `configs/base.yaml` rather than being hardcoded, so a new experiment is a new config file, not a code change:

| Key                        | Meaning                                                         | Default                                   |
| -------------------------- | --------------------------------------------------------------- | ----------------------------------------- |
| `model_name`               | Pretrained checkpoint to fine-tune                              | `t5-small`                                |
| `seed`                     | Random seed for reproducibility                                 | `42`                                      |
| `max_length`               | Max input token length (reviews are truncated)                  | `256`                                     |
| `val_split`                | Fraction of `train` carved out for validation                   | `0.1`                                     |
| `batch_size`               | Training/eval batch size                                        | `32`                                      |
| `learning_rate`            | AdamW learning rate                                             | `1e-5`                                    |
| `epochs`                   | Max training epochs                                             | `10`                                      |
| `warmup_ratio`             | Fraction of steps spent on LR warmup                            | `0.1`                                     |
| `grad_clip_norm`           | Max gradient norm (clipping)                                    | `1.0`                                     |
| `early_stopping_patience`  | Epochs without improvement before stopping                      | `3`                                       |
| `peft.method`              | `none` (full fine-tuning) or `lora`                             | `none`                                    |
| `peft.rank` / `peft.alpha` | LoRA rank and scaling factor (only used if `peft.method: lora`) | `8` / `4.0`                               |
| `output_dir`               | Where checkpoints/logs are written                              | `outputs/full_finetune` or `outputs/lora` |

## Methodology notes

A few deliberate design decisions worth calling out, since they're the kind of thing that's easy to get wrong with this framing:

- **Proper train/val/test split.** The IMDB `test` split is only ever used once, by `evaluate.py`, after the model is fully trained and selected. All checkpointing and early-stopping decisions during training use a validation split carved out of `train` instead, to avoid test-set leakage.
- **Reproducibility.** `random`, NumPy, and PyTorch RNGs are all seeded at the start of every run.
- **Malformed generations.** Because the model _generates_ the label rather than predicting a class index, it can occasionally emit something that isn't exactly `"negative"` or `"positive"` — especially early in training. These are tracked explicitly as a `malformed_generation_rate` metric and treated as incorrect predictions (rather than silently dropped or crashing the metrics code), so the reported accuracy reflects true end-to-end performance, generation failures included.

## Results & Discussion

_Fill this in after running both configs on your machine — this comparison table is the single most important thing a reviewer will look at._

| Metric                    | Full fine-tuning (test) | LoRA (test) |
| ------------------------- | ----------------------- | ----------- |
| Accuracy                  | 0.8961                  | 0.8747      |
| Precision                 | 0.8984                  | 0.8807      |
| Recall                    | 0.8932                  | 0.8668      |
| F1                        | 0.8958                  | 0.8737      |
| Malformed generation rate | 8e-05                   | 0.00012     |
| Trainable params          | 60,506,624              | 294,912     |


Suggested things to note here once you have real numbers:

- The full fine-tuning method performs slightly better on every metric that is computed above. As shown, LoRA typically comes close to full fine-tuning on tasks like this, which is the whole point of the comparison.
- The number of trainable parameters is much smaller in LoRA than in full fine-tuning, which is where LoRA's practical advantage actually lies.


## License

[MIT](LICENSE) .
