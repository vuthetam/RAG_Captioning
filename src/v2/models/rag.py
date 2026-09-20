import torch
from torch import Tensor, nn

from src.v2.encoder import RagContextEncoder, RagFusionEncoder
from src.shared.decoder import TransformerCaptionDecoder


class RagModelV2(nn.Module):
    """
    Trùm cuối Model V2: Kết hợp Ảnh và RAG Text thông qua màng lọc Gated Residual Fusion.
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nheads: int = 8,
        nlayers: int = 4,
        dropout: float = 0.1,
        max_length: int = 40,
        pad_idx: int = 0,
        visual_feature_dim: int = 768, # Mặc định của CLIP ViT-B/16
    ):
        super().__init__()
        self.visual_feature_dim = visual_feature_dim
        self.d_model = d_model
        
        # 1. Image Projector: Ép chiều từ 768 (CLIP) về 512 (d_model)
        self.visual_projector = nn.Linear(visual_feature_dim, d_model)
        
        # 2. Decoder (Tự động khởi tạo nn.Embedding bộ từ vựng bên trong)
        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nheads,
            num_layers=nlayers,
            dropout=dropout,
            max_length=max_length,
            pad_idx=pad_idx,
        )
        
        # 3. Context Encoder (Dùng chung Embedding layer với Decoder)
        self.context_encoder = RagContextEncoder(
            d_model=d_model,
            nhead=nheads,
            embedding_layer=self.decoder.embedding,
            num_layers=2
        )
        
        # 4. Fusion Encoder (Soft Filter + Cổng Dynamic Gating)
        self.fusion_encoder = RagFusionEncoder(
            d_model=d_model,
            nhead=nheads
        )

    def forward(
        self,
        visual_features: Tensor,        # [B, 197, 768] (Chứa cả token CLS)
        k_ctx_tokens: Tensor,           # [B, K, max_ctx_len]
        k_ctx_objects: Tensor,          # [B, K, max_obj_len]
        k_ctx_relations: Tensor,        # [B, K, max_rel_len]
        input_ids: Tensor,              # [B, seq_len] (Câu caption gốc - Teacher Forcing)
        attention_mask: Tensor,         # [B, seq_len]
        include_cls_token: bool = False
    ) -> Tensor:
        
        # --- BƯỚC 1: XỬ LÝ ẢNH THÔ ---
        # Ép kiểu float32 (H5 lưu float16)
        visual_features = visual_features.float()
        
        # Bỏ đi token CLS ở vị trí số 0 (CLS luôn ở index 0 với mọi ViT backbone)
        if not include_cls_token:
            visual_features = visual_features[:, 1:, :]
            
        # Ép chiều: [B, 196, 768] -> [B, 196, 512]
        image_features = self.visual_projector(visual_features)
        
        # --- BƯỚC 2: XỬ LÝ CHỮ RAG ---
        # encoded_context: [B, 240, 512]
        # context_padding_mask: [B, 240]
        encoded_context, context_padding_mask = self.context_encoder(
            k_ctx_tokens, k_ctx_objects, k_ctx_relations
        )
        
        # --- BƯỚC 3: LAI TẠO (FUSION) ---
        # fused_memory: [B, 196, 512]
        fused_memory = self.fusion_encoder(
            image_features, encoded_context, context_padding_mask
        )
        
        # --- BƯỚC 4: GIẢI MÃ SINH CHỮ (DECODER) ---
        # Dịch trái 1 bước (cắt từ cuối cùng) để mớm cho Decoder
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=fused_memory,
            attention_mask=decoder_attention_mask,
        )
        
        return logits

    def encode_memory(
        self,
        visual_features: Tensor,        # [B, 197, 768]
        k_ctx_tokens: Tensor,           # [B, K, max_ctx_len]
        k_ctx_objects: Tensor,          # [B, K, max_obj_len]
        k_ctx_relations: Tensor,        # [B, K, max_rel_len]
        include_cls_token: bool = False,
    ) -> Tensor:
        """Trả về fused_memory [B, 196, d_model] để mớm cho Beam Search."""
        visual_features = visual_features.float()
        if not include_cls_token:
            visual_features = visual_features[:, 1:, :]
        image_features = self.visual_projector(visual_features)

        encoded_context, context_padding_mask = self.context_encoder(
            k_ctx_tokens, k_ctx_objects, k_ctx_relations
        )
        fused_memory = self.fusion_encoder(
            image_features, encoded_context, context_padding_mask
        )
        return fused_memory  # [B, 196, d_model]
