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
    """
    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        
        # 1. Soft Filter (Cross Attention)
        # batch_first=True để nhận input [B, SeqLen, d_model]
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        # 2. Dynamic Gating (Context-Aware Sigmoid Gate)
        # Nhận đầu vào là [V ; C_filtered] -> kích thước là 2 * d_model
        self.gate_linear = nn.Linear(2 * d_model, d_model)
        
        # Zero-Initialization cho gate:
        # Bắt đầu với trọng số bằng 0, bias dương để lúc mới train Gate ưu tiên Ảnh (V) giống hệt V1.
        nn.init.zeros_(self.gate_linear.weight)
        nn.init.constant_(self.gate_linear.bias, 5.0)  # sigmoid(5) ~ 0.99
        self.sigmoid = nn.Sigmoid()

    def forward(self, image_features, encoded_context, context_padding_mask):
        """
        Inputs:
            image_features       : [B, 196, d_model] (Query) - Note: Đã là 3D từ Patch Features
            encoded_context      : [B, 240, d_model] (Key/Value)
            context_padding_mask : [B, 240] (True ở vị trí là padding dư thừa)
        """
        # --- BƯỚC 1: SOFT FILTER (Lọc từ khóa) ---
        # Q = Ảnh, K = V = Context
        # attn_output shape: [B, 196, d_model]
        c_filtered, _ = self.cross_attn(
            query=image_features,
            key=encoded_context,
            value=encoded_context,
            key_padding_mask=context_padding_mask
        )

        # --- BƯỚC 2: DYNAMIC GATING (Hòa trộn) ---
        # Nối Ảnh gốc (V) và Ảnh đã lọc (C_filtered) -> [B, 196, 2 * d_model]
        combined_features = torch.cat([image_features, c_filtered], dim=-1)
        
        # Tính toán cổng G (từ 0 đến 1)
        # gate shape: [B, 196, d_model]
        g = self.sigmoid(self.gate_linear(combined_features))
        
        # Hợp nhất: F = g * V + (1 - g) * C_filtered
        fused_memory = g * image_features + (1 - g) * c_filtered
        
        return fused_memory

