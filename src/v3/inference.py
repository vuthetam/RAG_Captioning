import torch
import torch.nn.functional as F
from torch import nn
from accelerate import Accelerator
from tqdm.auto import tqdm
from typing import Iterable

from src.shared.vocabulary import Vocabulary


@torch.no_grad()
def beam_search_v3(
    decoder: nn.Module,
    image_memory: torch.Tensor,
    rag_memory: torch.Tensor,
    vocab: Vocabulary,
    rag_attention_mask: torch.Tensor | None = None,
    beam_size: int = 5,
    max_length: int = 40,
    length_penalty: float = 1.0,
) -> torch.Tensor:
    """Thuật toán Beam Search chuyên biệt cho V3 (Dual Cross-Attention)."""
    B = image_memory.size(0)
    device = image_memory.device
    k = beam_size
    V = len(vocab)

    # 1. Nhân bản Memory lên k lần cho k beams
    S_img = image_memory.size(1)
    D = image_memory.size(2)
    image_memory = image_memory.unsqueeze(1).expand(B, k, S_img, D).reshape(B * k, S_img, D)

    S_rag = rag_memory.size(1)
    rag_memory = rag_memory.unsqueeze(1).expand(B, k, S_rag, D).reshape(B * k, S_rag, D)

    if rag_attention_mask is not None:
        rag_attention_mask = rag_attention_mask.unsqueeze(1).expand(B, k, S_rag).reshape(B * k, S_rag)

    # 2. Khởi tạo mảng lưu Sequences và Scores
    sequences = torch.full((B * k, 1), vocab.sos_idx(), dtype=torch.long, device=device)
    scores = torch.full((B * k,), float("-inf"), device=device)
    scores[torch.arange(B, device=device) * k] = 0.0

    eos_mask = torch.zeros(B * k, dtype=torch.bool, device=device)
    seq_lengths = torch.ones(B * k, device=device)

    # 3. Vòng lặp giải mã (Decoding Loop)
    for _ in range(max_length - 1):
        # Forward qua Dual Decoder
        logits = decoder(
            tgt_ids=sequences,
            image_memory=image_memory,
            rag_memory=rag_memory,
            tgt_attention_mask=None,
            rag_attention_mask=rag_attention_mask
        )
        log_probs = F.log_softmax(logits[:, -1, :], dim=-1)

        # Nếu beam nào đã có eos, ép pad_idx = 0.0, phần còn lại -inf
        log_probs[eos_mask] = float("-inf")
        log_probs[eos_mask, vocab.pad_idx()] = 0.0

        next_scores = scores.unsqueeze(1) + log_probs
        top_scores, top_flat_indices = next_scores.view(B, k * V).topk(k, dim=1)
        
        beam_indices = top_flat_indices // V
        token_indices = top_flat_indices % V

        global_indices = (torch.arange(B, device=device).unsqueeze(1) * k + beam_indices).view(-1)
        
        sequences = torch.cat([sequences[global_indices], token_indices.view(-1, 1)], dim=1)
        scores = top_scores.view(B * k)
        
        eos_mask = eos_mask[global_indices] | token_indices.view(-1).eq(vocab.eos_idx())
        seq_lengths = seq_lengths[global_indices]
        seq_lengths[~eos_mask] += 1

        if eos_mask.view(B, k).all():
            break

    # 4. Phạt độ dài (Length Penalty)
    if length_penalty > 0.0:
        scores = scores / (seq_lengths ** length_penalty)

    scores = scores.view(B, k)
    best_indices = scores.argmax(dim=1)
    global_best_indices = torch.arange(B, device=device) * k + best_indices
    best = sequences[global_best_indices]

    # Cắt / đệm cho đúng max_length
    t = best.size(1)
    if t < max_length:
        best = F.pad(best, (0, max_length - t), value=vocab.pad_idx())
    else:
        best = best[:, :max_length]

    return best


@torch.no_grad()
def generate_captions_v3(
    model,
    dataloader: Iterable,
    vocab: Vocabulary,
    beam_size: int,
    max_length: int,
    accelerator: Accelerator,
    length_penalty: float = 1.0,
    show_progress: bool = True,
) -> dict[int, list[str]]:
    
    model.eval()
    base_model = accelerator.unwrap_model(model)

    all_captions: dict[int, list[str]] = {}
    iterator = tqdm(dataloader, disable=not show_progress, leave=False, desc="Generating V3")

    for batch in iterator:
        visual_features, rag_input_ids, rag_attention_mask, image_ids = batch

        with accelerator.autocast():
            # Bước 1: Encode độc lập 2 luồng Memory
            image_memory, rag_memory, rag_memory_mask = base_model.encode_memory(
                visual_features, rag_input_ids, rag_attention_mask
            )

            # Bước 2: Beam Search chuyên biệt cho 2 luồng Memory
            sequences = beam_search_v3(
                decoder=base_model.decoder,
                image_memory=image_memory,
                rag_memory=rag_memory,
                vocab=vocab,
                rag_attention_mask=rag_memory_mask,
                beam_size=beam_size,
                max_length=max_length,
                length_penalty=length_penalty,
            )

        gathered_seqs = accelerator.gather_for_metrics(sequences)
        gathered_ids  = accelerator.gather_for_metrics(image_ids)

        if accelerator.is_main_process:
            for imgid_tensor, token_ids in zip(gathered_ids, gathered_seqs.tolist()):
                imgid  = int(imgid_tensor.item())
                tokens = vocab.decode(token_ids, skip_special_tokens=True)
                tokens = [t for t in tokens if t != vocab.pad_token]
                all_captions[imgid] = tokens

    return all_captions
