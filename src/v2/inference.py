"""Inference engine cho V2 RAG Captioner — Beam Search + Caption Generation."""

from __future__ import annotations

from typing import Iterable

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from src.shared.vocabulary import Vocabulary
from src.shared.inference import beam_search  # Thuật toán Beam Search dùng chung V1/V2


@torch.no_grad()
def generate_captions(
    model,
    dataloader: Iterable,
    vocab: Vocabulary,
    beam_size: int,
    max_length: int,
    accelerator: Accelerator,
    length_penalty: float = 1.0,
    show_progress: bool = True,
) -> dict[int, list[str]]:
    """Sinh caption cho toàn bộ ảnh trong dataloader bằng Beam Search.

    Dataloader phải yield:
        visual_features, imgid, k_ctx_tokens, k_ctx_objects, k_ctx_relations

    Returns:
        dict imgid -> list[str] tokens (chỉ trả về ở main process).
    """
    model.eval()
    base_model = accelerator.unwrap_model(model)

    all_captions: dict[int, list[str]] = {}
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Generating")

    for batch in iterator:
        visual_features, image_ids, k_ctx_tokens, k_ctx_objects, k_ctx_relations = batch

        with accelerator.autocast():
            # Bước 1: Encode toàn bộ input thành fused_memory [B, 196, d_model]
            memory = base_model.encode_memory(
                visual_features, k_ctx_tokens, k_ctx_objects, k_ctx_relations
            )

            # Bước 2: Beam Search trên Decoder dùng fused_memory làm Key/Value
            sequences = beam_search(
                decoder=base_model.decoder,
                memory=memory,
                vocab=vocab,
                beam_size=beam_size,
                max_length=max_length,
                length_penalty=length_penalty,
            )   # [B, max_length]

        # Gather kết quả từ tất cả GPU về main process
        gathered_seqs = accelerator.gather_for_metrics(sequences)
        gathered_ids  = accelerator.gather_for_metrics(image_ids)

        if accelerator.is_main_process:
            for imgid_tensor, token_ids in zip(gathered_ids, gathered_seqs.tolist()):
                imgid  = int(imgid_tensor.item())
                tokens = vocab.decode(token_ids, skip_special_tokens=True)
                tokens = [t for t in tokens if t != vocab.pad_token]
                all_captions[imgid] = tokens

    return all_captions

