import torch
from torch import Tensor, nn
from transformers import CLIPVisionModel


class CLIPViTB16Encoder(nn.Module):
    MODEL_NAME = "openai/clip-vit-base-patch16"

    def __init__(self, d_model: int) -> None:
        super().__init__()
        clip_model = CLIPVisionModel.from_pretrained(self.MODEL_NAME)

        # Support both plain CLIPVisionModel and wrappers that expose .vision_model.
        self.backbone = getattr(clip_model, "vision_model", clip_model)
        self.backbone.requires_grad_(False)
        self.projection = nn.Linear(self.backbone.config.hidden_size, d_model)

    def forward(self, images: Tensor) -> Tensor:
        self.backbone.eval()
        with torch.no_grad():
            hidden_states = self.backbone(pixel_values=images).last_hidden_state

        # Drop CLS token, keep only patch tokens.
        patch_tokens = hidden_states[:, 1:, :]
        return self.projection(patch_tokens)
