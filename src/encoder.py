import math
import torch
from torch import Tensor, nn
from transformers import CLIPVisionModel

from src.config import CTX_NLAYERS
from src.decoder import PositionalEncoding


class CLIPViTB16Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        # Support both plain CLIPVisionModel and wrappers that expose .vision_model.
        self.backbone = getattr(clip_model, "vision_model", clip_model)
        self.backbone.requires_grad_(False)
        self.output_dim = self.backbone.config.hidden_size

    def forward(self, images: Tensor) -> Tensor:
        self.backbone.eval()
        with torch.no_grad():
            hidden_states = self.backbone(pixel_values=images).last_hidden_state

        # Preserve CLS and patch tokens so downstream consumers can choose either.
        return hidden_states


class TextContextEncoder(nn.Module):
    """Encodes retrieved RAG text contexts using a shared embedding and TransformerEncoder."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float,
        max_ctx_length: int,
        embedding: nn.Embedding,
        num_layers: int = CTX_NLAYERS,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        
        # Use the shared embedding from the decoder
        self.embedding = embedding
        self.pos_encoding = PositionalEncoding(d_model, max_ctx_length)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, rag_input_ids: Tensor, rag_attention_mask: Tensor) -> Tensor:
        """
        Args:
            rag_input_ids: (B, L_ctx) token IDs of the concatenated RAG captions.
            rag_attention_mask: (B, L_ctx) binary mask, 1 for real tokens, 0 for padding.

        Returns:
            context_memory: (B, L_ctx, d_model) contextualized representations.
        """
        # Embed tokens and scale
        x = self.embedding(rag_input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # src_key_padding_mask for TransformerEncoder expects True for padding positions
        src_key_padding_mask = rag_attention_mask == 0

        # Encode context
        context_memory = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )
        return context_memory
