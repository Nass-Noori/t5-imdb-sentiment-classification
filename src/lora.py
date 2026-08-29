"""LoRA (Low-Rank Adaptation) for T5 attention projections.

Reference: Hu et al., 2021 -- https://arxiv.org/abs/2106.09685

Only the query (`q`) and value (`v`) projection matrices inside T5's
self-attention blocks are adapted; the paper finds this captures most of the
benefit of LoRA while keeping the number of trainable parameters small. All
other weights (including `k` and `o` projections, embeddings, and
feed-forward layers) are frozen.
"""

import math

import torch
import torch.nn as nn

LORA_PARAM_MARKER = "lora_"  # substring used to identify LoRA params for freezing / saving


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank update.

    Forward pass: h = W x + (alpha / rank) * B(A(x))

    A is Kaiming-uniform initialized (same scheme PyTorch uses for a fresh
    nn.Linear). B is zero-initialized, which makes the LoRA branch a no-op
    at step 0 -- the wrapped model behaves identically to the pretrained
    model until gradient updates start moving B away from zero.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: float = 4.0):
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False  # frozen; never updated by the optimizer

        in_dim = base_layer.in_features
        out_dim = base_layer.out_features

        self.lora_A = nn.Linear(in_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_dim, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return base_out + lora_out


def apply_lora(model: nn.Module, rank: int = 8, alpha: float = 4.0, target_names=("q", "v")) -> int:
    """Recursively replace nn.Linear children named in `target_names` with LoRALinear.

    Returns the number of layers replaced.

    Raises RuntimeError if LoRA layers are already present (protects against
    silently double-wrapping the same model), or if nothing matched
    `target_names` (protects against a silent no-op if the architecture
    doesn't have layers with those names -- e.g. if you swap in a model
    other than T5).
    """
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            raise RuntimeError(
                "Model already has LoRA layers applied. Load a fresh base "
                "model before calling apply_lora() again."
            )

    num_replaced = 0

    def _recurse(module: nn.Module):
        nonlocal num_replaced
        for name, child in module.named_children():
            if isinstance(child, nn.Linear) and name in target_names:
                setattr(module, name, LoRALinear(child, rank=rank, alpha=alpha))
                num_replaced += 1
            else:
                _recurse(child)

    _recurse(model)

    if num_replaced == 0:
        raise RuntimeError(
            f"apply_lora() replaced 0 layers -- target_names={target_names} matched "
            "no nn.Linear submodules. Check the model architecture."
        )

    return num_replaced


def freeze_base_model(model: nn.Module) -> None:
    """Freeze every parameter except the LoRA A/B matrices."""
    for name, param in model.named_parameters():
        param.requires_grad = LORA_PARAM_MARKER in name


def lora_state_dict(model: nn.Module) -> dict:
    """Extract only the LoRA adapter weights -- a few hundred KB instead of
    the full ~230MB T5-small checkpoint, since the base weights are
    unchanged and don't need to be re-saved.
    """
    return {k: v for k, v in model.state_dict().items() if LORA_PARAM_MARKER in k}


def count_parameters(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4) if total else 0.0,
    }
