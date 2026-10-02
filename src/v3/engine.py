"""Training và Evaluation engine cho V3 RAG Captioner."""

from __future__ import annotations

from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.shared.utils import trainable_parameters

def _step_v3(
    model: nn.Module,
    batch,
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''Một bước forward + tính loss. Trả về (logits, loss_sum, num_tokens).'''
    
    visual_features, input_ids, attention_mask, rag_input_ids, rag_attention_mask = batch

    # Teacher Forcing: Cắt đầu cắt đuôi
    tgt_input = input_ids[:, :-1]
    tgt_mask = attention_mask[:, :-1]
    target_ids = input_ids[:, 1:]

    logits = model(
        visual_features=visual_features,
        input_ids=tgt_input,
        attention_mask=tgt_mask,
        rag_input_ids=rag_input_ids,
        rag_attention_mask=rag_attention_mask
    )

    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    
    num_tokens = (target_ids != pad_idx).sum()
    
    return logits, loss_sum, num_tokens


def train_one_epoch_v3(
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
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Training V3")

    for batch in iterator:
        optimizer.zero_grad(set_to_none=True)
        
        with accelerator.autocast():
            _, loss_sum, num_tokens = _step_v3(model, batch, pad_idx)

        accelerator.backward(loss_sum)
        accelerator.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
        optimizer.step()

        reduced_loss   = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss   += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)


def evaluate_one_epoch_v3(
    model: nn.Module,
    dataloader: Iterable,
    pad_idx: int,
    accelerator: Accelerator,
    show_progress: bool = True,
) -> float:
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Evaluating V3")

    for batch in iterator:
        with torch.no_grad():
            with accelerator.autocast():
                _, loss_sum, num_tokens = _step_v3(model, batch, pad_idx)

        reduced_loss   = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss   += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)
