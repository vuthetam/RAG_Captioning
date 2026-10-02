import math
import torch
import torch.nn as nn
from torch import Tensor

# Import các mảnh ghép từ V3 và Shared
from src.shared.decoder import PositionalEncoding
from src.v3.decoder import DualCrossAttnDecoder


class RagTextEncoderV3(nn.Module):
    '''
    Text Encoder mã hóa K câu RAG một cách độc lập (FiD).
    Sử dụng PositionalEncoding từ shared để đếm vị trí chuẩn xác từ 0->L.
    '''
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int = 512, 
        nhead: int = 8, 
        num_layers: int = 2, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1,
        max_rag_len: int = 64
    ):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_rag_len)
        self.pos_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, input_ids_flat: Tensor, attention_mask_flat: Tensor) -> Tensor:
        # 1. Embedding + Positional Encoding
        x = self.embedding(input_ids_flat) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.pos_dropout(x)
        
        # 2. Xử lý Boolean Mask cho PyTorch (True ở vị trí cần bỏ qua PAD)
        src_key_padding_mask = None
        if attention_mask_flat is not None:
            src_key_padding_mask = (attention_mask_flat == 0)
            
        # 3. Transformer Encoder (Làm giàu ngữ cảnh nội bộ)
        out = self.transformer_encoder(
            src=x, 
            src_key_padding_mask=src_key_padding_mask
        )
        return out


class RagModelV3(nn.Module):
    '''Lắp ráp toàn bộ kiến trúc FiD Dual-Cross-Attention'''
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_length: int = 40,
        max_rag_len: int = 64,
        top_k: int = 4,
        visual_feature_dim: int = 768,
        include_cls_token: bool = False
    ) -> None:
        super().__init__()
        
        self.top_k = top_k
        self.max_rag_len = max_rag_len
        self.include_cls_token = include_cls_token
        
        # 1. Image Projector (Ép chiều CLIP 768 -> 512)
        self.visual_projector = nn.Linear(visual_feature_dim, d_model)
        
        # 2. RAG Text Encoder
        self.context_encoder = RagTextEncoderV3(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_rag_len=max_rag_len
        )
        
        # 3. Dual Cross-Attention Decoder
        self.decoder = DualCrossAttnDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_length=max_length
        )

    def encode_memory(
        self, 
        visual_features: Tensor, 
        rag_input_ids: Tensor, 
        rag_attention_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        '''
        Hàm nướng sẵn (Pre-compute) các ma trận Memory để phục vụ Beam Search Inference siêu tốc.
        - visual_features: [B, 197, 768]
        - rag_input_ids: [B, K, L]
        '''
        # 1. Luồng Image
        if not self.include_cls_token:
            visual_features = visual_features[:, 1:, :]
            
        image_memory = self.visual_projector(visual_features.float()) # [B, 196 (hoặc 197), D]
        
        # 2. Luồng RAG Text (FiD Logic)
        B, K, L = rag_input_ids.shape
        
        # Ép phẳng Batch * K để mã hóa độc lập (Tránh lai tạp)
        rag_input_ids_flat = rag_input_ids.view(B * K, L)
        rag_attn_mask_flat = rag_attention_mask.view(B * K, L)
        
        encoded_rag_flat = self.context_encoder(rag_input_ids_flat, rag_attn_mask_flat) # [B*K, L, D]
        
        # Duỗi ngang thành chuỗi dài dằng dặc [B, K*L] làm thức ăn cho Cross-Attention
        rag_memory = encoded_rag_flat.view(B, K * L, -1)
        rag_memory_mask = rag_attn_mask_flat.view(B, K * L)
        
        return image_memory, rag_memory, rag_memory_mask

    def forward(
        self, 
        visual_features: Tensor, 
        input_ids: Tensor, 
        attention_mask: Tensor, 
        rag_input_ids: Tensor, 
        rag_attention_mask: Tensor
    ) -> Tensor:
        '''
        Luồng Forward đầy đủ dành cho lúc Training.
        '''
        # Chuẩn bị 2 luồng Memory
        image_memory, rag_memory, rag_memory_mask = self.encode_memory(
            visual_features, rag_input_ids, rag_attention_mask
        )
        
        # Truyền qua Dual Decoder để sinh logits
        logits = self.decoder(
            tgt_ids=input_ids,
            image_memory=image_memory,
            rag_memory=rag_memory,
            tgt_attention_mask=attention_mask,
            rag_attention_mask=rag_memory_mask
        )
        
        return logits
