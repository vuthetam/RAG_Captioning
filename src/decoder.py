import math

import torch
from torch import Tensor, nn

from src.config import DMODEL, NHEADS, NLAYERS


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pos_encoding[:, : x.size(1), :]


class TransformerCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = DMODEL,
        nhead: int = NHEADS,
        num_layers: int = NLAYERS,
        dropout: float = 0.1,
        max_length: int = 40,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_fc = nn.Linear(d_model, vocab_size)

    def _generate_square_subsequent_mask(self, size: int, device: torch.device) -> Tensor:
        return torch.triu(
            torch.full((size, size), float("-inf"), device=device),
            diagonal=1,
        )

    def forward(
        self,
        input_ids: Tensor,
        memory: Tensor,
        attention_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        tgt_mask = self._generate_square_subsequent_mask(x.size(1), x.device)
        tgt_key_padding_mask = None
        if attention_mask is not None:
            tgt_key_padding_mask = attention_mask == 0

        out = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_fc(out)
