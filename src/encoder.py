import torch
from torch import Tensor, nn
from transformers import CLIPVisionModel

from src.config import CLIP_MODEL_NAME


class CLIPViTB16Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        clip_model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME)

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
