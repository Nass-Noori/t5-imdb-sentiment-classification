"""Sanity checks for the LoRA implementation in src/lora.py.

Run with: python tests/test_lora.py

Not a full pytest suite (kept dependency-free on purpose) -- these are the
few checks that actually matter for catching a subtly broken LoRA
implementation, which is very easy to get wrong silently (e.g. B not
actually zero-initialized, or freezing the wrong parameters).
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lora import LoRALinear, apply_lora, freeze_base_model, count_parameters  # noqa: E402


def test_zero_init_is_a_noop():
    """At initialization, B is zero, so LoRALinear must produce EXACTLY the
    same output as the wrapped base layer. If this fails, fine-tuning would
    start from a randomly perturbed version of the pretrained model instead
    of the pretrained model itself.
    """
    torch.manual_seed(0)
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, rank=4, alpha=2.0)

    x = torch.randn(3, 16)
    assert torch.allclose(base(x), lora(x)), "LoRA output should equal base output at init (B is zero)"
    print("PASS: zero-init LoRA is a no-op")


def test_base_weights_are_frozen():
    base = nn.Linear(16, 16)
    lora = LoRALinear(base, rank=4, alpha=2.0)
    assert not lora.base_layer.weight.requires_grad, "base_layer weights should be frozen"
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad
    print("PASS: base weights frozen, LoRA A/B trainable")


def test_apply_lora_only_targets_q_and_v():
    """Build a tiny fake attention block shaped like T5's, and confirm
    apply_lora only wraps q/v, leaving k/o untouched.
    """

    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8, bias=False)
            self.k = nn.Linear(8, 8, bias=False)
            self.v = nn.Linear(8, 8, bias=False)
            self.o = nn.Linear(8, 8, bias=False)

    model = FakeAttention()
    num_replaced = apply_lora(model, rank=2, alpha=1.0)

    assert num_replaced == 2, f"expected 2 layers replaced (q, v), got {num_replaced}"
    assert isinstance(model.q, LoRALinear)
    assert isinstance(model.v, LoRALinear)
    assert isinstance(model.k, nn.Linear) and not isinstance(model.k, LoRALinear)
    assert isinstance(model.o, nn.Linear) and not isinstance(model.o, LoRALinear)
    print("PASS: apply_lora targets only q/v")


def test_freeze_base_model_trainable_param_count():
    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8, bias=False)
            self.k = nn.Linear(8, 8, bias=False)
            self.v = nn.Linear(8, 8, bias=False)
            self.o = nn.Linear(8, 8, bias=False)

    model = FakeAttention()
    apply_lora(model, rank=2, alpha=1.0)
    freeze_base_model(model)

    counts = count_parameters(model)
    # q, v each contribute (8*2 + 2*8) = 32 trainable params -> 64 total
    assert counts["trainable_params"] == 64, counts
    assert counts["trainable_pct"] < 100, "LoRA should train a small fraction of total params"
    print(f"PASS: trainable param count correct ({counts['trainable_params']} / {counts['total_params']})")


def test_apply_lora_raises_on_double_wrap():
    class FakeAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(8, 8, bias=False)
            self.v = nn.Linear(8, 8, bias=False)

    model = FakeAttention()
    apply_lora(model, rank=2, alpha=1.0)
    try:
        apply_lora(model, rank=2, alpha=1.0)
        raise AssertionError("expected RuntimeError on double-wrapping")
    except RuntimeError:
        print("PASS: double-wrapping raises RuntimeError as expected")


if __name__ == "__main__":
    test_zero_init_is_a_noop()
    test_base_weights_are_frozen()
    test_apply_lora_only_targets_q_and_v()
    test_freeze_base_model_trainable_param_count()
    test_apply_lora_raises_on_double_wrap()
    print("\nAll LoRA sanity checks passed.")
