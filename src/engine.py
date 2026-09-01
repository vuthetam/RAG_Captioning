from __future__ import annotations

from typing import Iterable

import torch
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.utils import trainable_parameters


def _step(
    encoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    memory = encoder(images)

    decoder_input_ids = input_ids[:, :-1]
    decoder_attention_mask = attention_mask[:, :-1]
    target_ids = input_ids[:, 1:]

    logits = decoder(
        input_ids=decoder_input_ids,
        memory=memory,
        attention_mask=decoder_attention_mask,
    )

    loss_sum = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=pad_idx,
        reduction="sum",
    )
    num_tokens = (target_ids != pad_idx).sum()
    return logits, loss_sum, num_tokens



def train_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: Iterable,
    optimizer: torch.optim.Optimizer,
    pad_idx: int,
    accelerator: Accelerator,
    max_grad_norm: float = 1.0,
    show_progress: bool = False,
) -> float:
    encoder.train()
    decoder.train()

    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Training")

    for images, input_ids, attention_mask in iterator:
        images = images.to(accelerator.device)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        optimizer.zero_grad(set_to_none=True)
        with accelerator.autocast():
            _, loss_sum, num_tokens = _step(encoder, decoder, images, input_ids, attention_mask, pad_idx)

        accelerator.backward(loss_sum)
        accelerator.clip_grad_norm_(
            trainable_parameters(encoder, decoder),
            max_grad_norm,
        )

        optimizer.step()

        reduced_loss = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_one_epoch(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: Iterable,
    pad_idx: int,
    accelerator: Accelerator,
    show_progress: bool = False,
) -> float:
    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    total_tokens = 0
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Evaluating")

    for images, input_ids, attention_mask in iterator:
        images = images.to(accelerator.device)
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        with accelerator.autocast():
            _, loss_sum, num_tokens = _step(encoder, decoder, images, input_ids, attention_mask, pad_idx)

        reduced_loss = accelerator.reduce(loss_sum.detach(), reduction="sum")
        reduced_tokens = accelerator.reduce(num_tokens.detach(), reduction="sum")

        total_loss += reduced_loss.item()
        total_tokens += reduced_tokens.item()
        iterator.set_postfix(loss=total_loss / max(total_tokens, 1))

    return total_loss / max(total_tokens, 1)
