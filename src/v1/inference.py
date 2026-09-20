from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch import nn
from tqdm.auto import tqdm

from src.shared.vocabulary import Vocabulary


from src.shared.inference import beam_search  # noqa: F401  (re-export for backward compat)


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
