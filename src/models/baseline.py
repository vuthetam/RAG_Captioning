from torch import Tensor, nn
from src.encoder import CLIPViTB16Encoder
from src.decoder import TransformerCaptionDecoder
from src.visual_projector import VisualProjector

class BaselineCaptioner(nn.Module):
    def __init__(self, vocab_size, d_model, nheads, nlayers, dropout, max_length, pad_idx):
        super().__init__()
        self.encoder = CLIPViTB16Encoder()
        self.visual_projector = VisualProjector(self.encoder.output_dim, d_model)
        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nheads,
            num_layers=nlayers,
            dropout=dropout,
            max_length=max_length,
            pad_idx=pad_idx
        )
        
    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """
        Dùng cho lúc Training (Teacher Forcing).
        Cắt token cuối của input_ids làm đầu vào cho decoder.
        """
        memory = self.encode_image(images, include_cls_token=include_cls_token)
        
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=memory,
            attention_mask=decoder_attention_mask,
        )
        return logits
    
    def encode_image(self, images: Tensor, include_cls_token: bool = False) -> Tensor:
        """Encode images with either all CLIP tokens or patch tokens only."""
        features = self.encoder(images)
        if not include_cls_token:
            features = features[:, 1:, :]
        return self.visual_projector(features)
