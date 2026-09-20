"""Training và Evaluation engine cho V2 RAG Captioner."""

from __future__ import annotations

from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.shared.utils import trainable_parameters


def _step(
    model: nn.Module,
    visual_features: torch.Tensor,     # [B, 197, 768]
    input_ids: torch.Tensor,           # [B, max_length]
    attention_mask: torch.Tensor,      # [B, max_length]
    k_ctx_tokens: torch.Tensor,        # [B, K, max_ctx_len]
    k_ctx_objects: torch.Tensor,       # [B, K, max_obj_len]
    k_ctx_relations: torch.Tensor,     # [B, K, max_rel_len]
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Một bước forward + tính loss. Trả về (logits, loss_sum, num_tokens)."""

    # Mảng mục tiêu: dịch phải 1 bước (bỏ <SOS> đầu)
    target_ids = input_ids[:, 1:]       # [B, max_length - 1]

    logits = model(
        visual_features=visual_features,
        k_ctx_tokens=k_ctx_tokens,
        k_ctx_objects=k_ctx_objects,
        k_ctx_relations=k_ctx_relations,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )   # [B, max_length - 1, vocab_size]

    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    num_tokens = (target_ids != pad_idx).sum()
    return logits, loss_sum, num_tokens


def train_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    pad_idx: int,
    accelerator: Accelerator,
    max_grad_norm: float = 1.0,
    show_progress: bool = True,
) -> float:
    model.train()

    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Training")

    for batch in iterator:
        visual_features, input_ids, attention_mask, k_ctx_tokens, k_ctx_objects, k_ctx_relations = batch

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            _, loss_sum, num_tokens = _step(
                model,
                visual_features, input_ids, attention_mask,
                k_ctx_tokens, k_ctx_objects, k_ctx_relations,
                pad_idx,
            )

        accelerator.backward(loss_sum)
        accelerator.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        optimizer.step()

        reduced_loss   = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss   += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    dataloader: Iterable,
    pad_idx: int,
    accelerator: Accelerator,
    show_progress: bool = True,
) -> float:
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Evaluating")

    for batch in iterator:
        visual_features, input_ids, attention_mask, k_ctx_tokens, k_ctx_objects, k_ctx_relations = batch

        with accelerator.autocast():
            _, loss_sum, num_tokens = _step(
                model,
                visual_features, input_ids, attention_mask,
                k_ctx_tokens, k_ctx_objects, k_ctx_relations,
                pad_idx,
            )

        reduced_loss   = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss   += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)

