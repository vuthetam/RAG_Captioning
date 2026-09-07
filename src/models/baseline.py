from torch import Tensor, nn
from src.encoder import CLIPViTB16Encoder
from src.decoder import TransformerCaptionDecoder
from src.visual_projector import VisualProjector

class BaselineCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        nheads,
        nlayers,
        dropout,
        max_length,
        pad_idx,
        use_precomputed_features: bool = False,
        visual_feature_dim: int | None = None,
    ):
        super().__init__()
        self.use_precomputed_features = use_precomputed_features
        if use_precomputed_features:
            self.encoder = None
            encoder_output_dim = visual_feature_dim
        else:
            self.encoder = CLIPViTB16Encoder()
            encoder_output_dim = self.encoder.output_dim

        self.visual_projector = VisualProjector(encoder_output_dim, d_model)
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
        visual_inputs: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """
        Dùng cho lúc Training (Teacher Forcing).
        Cắt token cuối của input_ids làm đầu vào cho decoder.
        """
        if self.use_precomputed_features:
            memory = self.encode_features(visual_inputs, include_cls_token=include_cls_token)
        else:
            memory = self.encode_image(visual_inputs, include_cls_token=include_cls_token)
        
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=memory,
            attention_mask=decoder_attention_mask,
        )
        return logits
    
    def _project_features(self, features: Tensor, include_cls_token: bool = False) -> Tensor:
        """Drop the optional CLS token and project visual features."""
        if not include_cls_token:
            features = features[:, 1:, :]
        # H5 files are stored as float16; this also supports inference without AMP.
        features = features.to(dtype=self.visual_projector.projection.weight.dtype)
        return self.visual_projector(features)

    def encode_image(self, images: Tensor, include_cls_token: bool = False) -> Tensor:
        """Encode image pixels with CLIP, then project the visual tokens."""
        features = self.encoder(images)
        return self._project_features(features, include_cls_token=include_cls_token)

    def encode_features(self, features: Tensor, include_cls_token: bool = False) -> Tensor:
        """Project visual tokens that were pre-extracted and loaded from H5."""
        return self._project_features(features, include_cls_token=include_cls_token)
