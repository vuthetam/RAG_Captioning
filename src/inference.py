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
    memory_key_padding_mask: torch.Tensor | None = None,
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
    
    if memory_key_padding_mask is not None:
        # (B, S) → (B, k, S) → (B*k, S)
        memory_key_padding_mask = memory_key_padding_mask.unsqueeze(1).expand(B, k, S).reshape(B * k, S)

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
        logits = decoder(
            input_ids=sequences,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask
        )                                                              # (B*k, t, V)
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
    model: nn.Module,
    dataloader: Iterable,
    vocab: Vocabulary,
    beam_size: int,
    max_length: int,
    accelerator: Accelerator,
    length_penalty: float = 1.0,
    show_progress: bool = False,
) -> dict[int, list[str]]:
    """Generate captions for every image in *dataloader* using batched beam search.

    Each Accelerate process handles its own shard of the dataloader;
    results are gathered to the main process via ``gather_for_metrics``.

    Args:
        dataloader: prepared DataLoader. Must yield (visual_inputs, ..., image_id)
                    where image_id is the last element.

    Returns:
        On the main process: dict mapping imgid -> list of decoded token strings.
        On other processes: empty dict.
    """
    model.eval()
    base_model = accelerator.unwrap_model(model)

    all_captions: dict[int, list[str]] = {}
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Generating")

    for batch in iterator:
        if len(batch) == 4:
            visual_inputs, rag_input_ids, rag_attention_mask, image_ids = batch
            rag_input_ids = rag_input_ids.to(accelerator.device)
            rag_attention_mask = rag_attention_mask.to(accelerator.device)
            has_rag = True
        else:
            visual_inputs, image_ids = batch
            has_rag = False

        visual_inputs = visual_inputs.to(accelerator.device)
        image_ids = image_ids.to(accelerator.device)

        with accelerator.autocast():
            if has_rag:
                memory, mem_mask = base_model.encode_memory(visual_inputs, rag_input_ids, rag_attention_mask)
            else:
                if base_model.use_precomputed_features:
                    memory = base_model.encode_features(visual_inputs)       # (B, S, D)
                else:
                    memory = base_model.encode_image(visual_inputs)          # (B, S, D)
                mem_mask = None

            sequences = beam_search(
                base_model.decoder, memory, vocab, beam_size, max_length, length_penalty, memory_key_padding_mask=mem_mask
            )                                                # (B_local, max_length)

        # Gather both generated sequences and their corresponding IDs
        gathered_seqs = accelerator.gather_for_metrics(sequences) # (B_total, max_length)
        gathered_ids = accelerator.gather_for_metrics(image_ids)  # (B_total,)

        if accelerator.is_main_process:
            for imgid_tensor, token_ids in zip(gathered_ids, gathered_seqs.tolist()):
                imgid = int(imgid_tensor.item())
                tokens = vocab.decode(token_ids, skip_special_tokens=True)
                tokens = [t for t in tokens if t != vocab.pad_token]
                all_captions[imgid] = tokens

    return all_captions
