from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.vocabulary import Vocabulary


@torch.no_grad()
def beam_search(
    decoder: nn.Module,
    memory: torch.Tensor,
    vocab: Vocabulary,
    beam_size: int = 5,
    max_length: int = 40,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """Batched beam search given pre-computed encoder memory.

    Separating encoding from decoding lets the caller compute *memory*
    however needed (plain image features, retrieval-augmented context, etc.).

    Args:
        memory: (B, S, D) — encoder output, already on the correct device.

    Returns:
        Tensor of shape (B, max_length) — best token-ID sequences, padded
        with pad_idx after <eos>.
    """
    B, S, D = memory.size()
    device = memory.device
    k = beam_size
    V = len(vocab)

    # ── Expand memory for k beams per image ───────────────────────────────────
    # (B, S, D) → (B, k, S, D) → (B*k, S, D)
    memory = memory.unsqueeze(1).expand(B, k, S, D).reshape(B * k, S, D)

    # ── Initialise beams ──────────────────────────────────────────────────────
    # All beams start with <sos>; only beam-0 per image is active (score=0),
    # beams 1..k-1 start at -inf so they don't pollute the first topk.
    sequences = torch.full((B * k, 1), vocab.sos_idx(), dtype=torch.long, device=device)
    scores = torch.full((B * k,), float("-inf"), device=device)
    scores[torch.arange(B, device=device) * k] = 0.0

    eos_mask = torch.zeros(B * k, dtype=torch.bool, device=device)
    seq_lengths = torch.ones(B * k, device=device)

    # ── Decoding loop ─────────────────────────────────────────────────────────
    for _ in range(max_length - 1):
        logits = decoder(input_ids=sequences, memory=memory)           # (B*k, t, V)
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)            # (B*k, V)

        # Finished beams only extend with pad (score unchanged)
        log_probs[eos_mask] = float("-inf")
        log_probs[eos_mask, vocab.pad_idx()] = 0.0

        # Cumulative scores for all candidate next tokens
        next_scores = scores.unsqueeze(1) + log_probs                  # (B*k, V)

        # Per-image top-k: flatten k beams × V vocab into one dim
        top_scores, top_flat_indices = next_scores.view(B, k * V).topk(k, dim=1)  # (B, k)
        beam_indices  = top_flat_indices // V   # which of the k beams this came from
        token_indices = top_flat_indices % V    # which token was selected

        # Convert to global (B*k) indices
        global_indices = (
            torch.arange(B, device=device).unsqueeze(1) * k + beam_indices
        ).view(-1)                                                       # (B*k,)

        # Reorder sequences and append new tokens
        sequences = torch.cat(
            [sequences[global_indices], token_indices.view(-1, 1)], dim=1
        )                                                                # (B*k, t+1)
        scores   = top_scores.view(B * k)
        
        eos_mask = eos_mask[global_indices] | token_indices.view(-1).eq(vocab.eos_idx())
        
        seq_lengths = seq_lengths[global_indices]
        seq_lengths[~eos_mask] += 1

        if eos_mask.view(B, k).all():
            break

    # ── Pick best beam per image (with length penalty) ────────────────────────
    if length_penalty > 0.0:
        scores = scores / (seq_lengths ** length_penalty)
        
    scores = scores.view(B, k)
    best_indices = scores.argmax(dim=1)  # (B,)
    global_best_indices = torch.arange(B, device=device) * k + best_indices

    best = sequences[global_best_indices]  # (B, t)

    # Pad / truncate to exactly max_length
    t = best.size(1)
    if t < max_length:
        best = F.pad(best, (0, max_length - t), value=vocab.pad_idx())
    else:
        best = best[:, :max_length]

    return best  # (B, max_length)


@torch.no_grad()
def generate_captions(
    encoder: nn.Module,
    decoder: nn.Module,
    dataloader: Iterable,
    vocab: Vocabulary,
    beam_size: int,
    max_length: int,
    accelerator: Accelerator,
    length_penalty: float = 1.0,
    show_progress: bool = False,
) -> list[list[str]]:
    """Generate captions for every image in *dataloader* using batched beam search.

    Each Accelerate process handles its own shard of the dataloader;
    results are gathered to the main process via ``gather_for_metrics``.

    The encoding step is kept separate from ``beam_search`` so the caller can
    extend it with retrieval-augmented context before passing *memory* in.

    Args:
        dataloader: prepared DataLoader (each batch yields at least images as
                    its first element; extra elements such as input_ids are ignored).

    Returns:
        On the main process: list of decoded token lists (special tokens excluded),
        in dataset order.
        On other processes: empty list.
    """
    encoder.eval()
    decoder.eval()

    all_captions: list[list[str]] = []
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Generating")

    for images, *_ in iterator:
        images = images.to(accelerator.device)

        # Encode first — extend here with retrieval context if needed
        memory = encoder(images)                             # (B, S, D)

        # Beam search on this process's shard
        sequences = beam_search(
            decoder, memory, vocab, beam_size, max_length, length_penalty
        )                                                    # (B_local, max_length)

        # Gather across all processes, stripping dummy samples from the last batch
        gathered = accelerator.gather_for_metrics(sequences) # (B_total, max_length)

        if accelerator.is_main_process:
            for row in gathered.tolist():
                tokens = vocab.decode(row, skip_special_tokens=True)
                # Remove any residual pad tokens
                tokens = [t for t in tokens if t != vocab.pad_token]
                all_captions.append(tokens)

    return all_captions
