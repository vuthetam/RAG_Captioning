import math
import torch
import torch.nn as nn

from src.shared.decoder import PositionalEncoding


class RagContextEncoder(nn.Module):
    """
    Xử lý Text RAG: Nhận tokens, objects, relations -> Trả về mảng Key/Value chuẩn [B, K * (len), d_model].
    Sử dụng nn.TransformerEncoder để đan chéo ngữ cảnh nội bộ của từng câu.
    """
    def __init__(self, embedding_layer: nn.Embedding, d_model: int, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        # Tái sử dụng embedding layer chung của toàn Model
        self.embedding = embedding_layer
        self.pad_idx = embedding_layer.padding_idx
        
        # Positional Encoding (dành cho tối đa 100 từ trong mỗi context)
        self.pos_encoding = PositionalEncoding(d_model, max_len=100)
        
        # 2. Text Encoder (Self-Attention)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, k_ctx_tokens, k_ctx_objects, k_ctx_relations):
        """
        Inputs từ Dataset:
            k_ctx_tokens   : [B, K, max_ctx_len]
            k_ctx_objects  : [B, K, max_obj_len]
            k_ctx_relations: [B, K, max_rel_len]
        """
        B, K, _ = k_ctx_tokens.shape

        # 1. Nối chiều ngang -> [B, K, 60] (Giả sử 40 + 10 + 10 = 60)
        combined = torch.cat([k_ctx_tokens, k_ctx_objects, k_ctx_relations], dim=2)
        total_len = combined.shape[2]

        # 2. Ép shape thành [B*K, 60] để nhét vào Text Encoder (chống nhiễu chéo)
        combined_flat = combined.view(B * K, total_len)

        # 3. Tạo padding mask (True ở những ô là số pad_idx)
        # nn.MultiheadAttention mong đợi mask dạng [B*K, 60]
        padding_mask = (combined_flat == self.pad_idx)

        # 4. Mạ vàng bằng Embedding & Positional Encoding -> [B*K, 60, d_model]
        embedded = self.embedding(combined_flat) * math.sqrt(self.d_model)
        embedded = self.pos_encoding(embedded)

        # 5. Đi qua Text Encoder
        # PyTorch TransformerEncoderLayer với batch_first=True nhận [Batch, SeqLen, d_model]
        encoded_text = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask)

        # 6. Đập vách ngăn nặn lại thành [B, K * 60, d_model] cho Soft Filter
        encoded_context = encoded_text.view(B, K * total_len, self.d_model)
        
        # Mask cũng nặn lại thành [B, K * 60] tương ứng
        final_mask = padding_mask.view(B, K * total_len)

        return encoded_context, final_mask


class RagFusionEncoder(nn.Module):
    """
    Lai tạo Ảnh và Text RAG thông qua Soft Filter (Cross-Attention) và Dynamic Gating (Sigmoid).
    Đã được nâng cấp chuẩn kiến trúc Transformer với LayerNorm và FeedForward Network.
    """
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # 1. Soft Filter (Cross Attention)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        
        # 2. Dynamic Gating
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        # Bắt đầu với bias = 0.0 -> sigmoid(0) = 0.5 (Tỉ lệ 50/50 để gradient chảy qua tốt nhất)
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 0.0)
        self.sigmoid = nn.Sigmoid()
        
        # 3. Transformer Add-ons (LayerNorm, Dropout, FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, image_features, encoded_context, context_padding_mask):
        """
        Inputs:
            image_features       : [B, 196, d_model] (Query)
            encoded_context      : [B, 240, d_model] (Key/Value)
            context_padding_mask : [B, 240] (True ở vị trí là padding dư thừa)
        """
        # --- BƯỚC 1: CROSS ATTENTION ---
        c_attn, _ = self.cross_attn(
            query=image_features,
            key=encoded_context,
            value=encoded_context,
            key_padding_mask=context_padding_mask
        )
        c_attn = self.dropout(c_attn)

        # --- BƯỚC 2: DYNAMIC GATING (Residual) ---
        combined_features = torch.cat([image_features, c_attn], dim=-1)
        g = self.sigmoid(self.gate_linear(combined_features))
        
        fused = g * image_features + (1 - g) * c_attn
        
        # Norm 1
        fused = self.norm1(fused)

        # --- BƯỚC 3: FEED-FORWARD NETWORK ---
        ffn_out = self.ffn(fused)
        fused_memory = self.norm2(fused + self.dropout(ffn_out))
        
        return fused_memory

