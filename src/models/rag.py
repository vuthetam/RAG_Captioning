import torch
from torch import Tensor

from src.models.baseline import BaselineCaptioner
from src.encoder import TextContextEncoder

class RAGCaptioner(BaselineCaptioner):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nheads: int,
        nlayers: int,
        dropout: float,
        max_length: int,
        pad_idx: int,
        use_precomputed_features: bool = False,
        visual_feature_dim: int | None = None,
        ctx_nlayers: int = 2,
        max_ctx_length: int = 80,
    ):
        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            nheads=nheads,
            nlayers=nlayers,
            dropout=dropout,
            max_length=max_length,
            pad_idx=pad_idx,
            use_precomputed_features=use_precomputed_features,
            visual_feature_dim=visual_feature_dim,
        )
        
        # Instantiate TextContextEncoder sharing the decoder's embedding
        self.context_encoder = TextContextEncoder(
            d_model=d_model,
            nhead=nheads,
            dropout=dropout,
            max_ctx_length=max_ctx_length,
            embedding=self.decoder.embedding,
            num_layers=ctx_nlayers,
        )

    def forward(
        self,
        visual_inputs: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        rag_input_ids: Tensor,
        rag_attention_mask: Tensor,
        include_cls_token: bool = False,
    ) -> Tensor:
        """
        Args:
            visual_inputs: Either image pixels or precomputed features.
            input_ids: Target token IDs (B, T)
            attention_mask: Target attention mask (B, T)
            rag_input_ids: RAG context token IDs (B, L_ctx)
            rag_attention_mask: RAG context attention mask (B, L_ctx)
            include_cls_token: Whether to keep CLS token in visual memory.
        """
        if self.use_precomputed_features:
            visual_memory = self.encode_features(visual_inputs, include_cls_token=include_cls_token)
        else:
            visual_memory = self.encode_image(visual_inputs, include_cls_token=include_cls_token)
            
        context_memory = self.context_encoder(rag_input_ids, rag_attention_mask)
        
        # Concat visual memory and context memory
        memory = torch.cat([visual_memory, context_memory], dim=1) # (B, 196+L_ctx, D)
        
        # Create memory_key_padding_mask
        B = visual_memory.size(0)
        V_len = visual_memory.size(1)
        # Visual tokens are always fully attended
        visual_pad_mask = torch.zeros(B, V_len, dtype=torch.bool, device=memory.device)
        # Context tokens: mask padding where attention_mask == 0
        context_pad_mask = (rag_attention_mask == 0)
        
        memory_key_padding_mask = torch.cat([visual_pad_mask, context_pad_mask], dim=1)
        
        # Prepare decoder inputs
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=memory,
            attention_mask=decoder_attention_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return logits
    
    def encode_memory(self, visual_inputs: Tensor, rag_input_ids: Tensor, rag_attention_mask: Tensor, include_cls_token: bool = False) -> tuple[Tensor, Tensor]:
        """Encode both visual and context representations into a single memory tensor for inference."""
        if self.use_precomputed_features:
            visual_memory = self.encode_features(visual_inputs, include_cls_token=include_cls_token)
        else:
            visual_memory = self.encode_image(visual_inputs, include_cls_token=include_cls_token)
            
        context_memory = self.context_encoder(rag_input_ids, rag_attention_mask)
        memory = torch.cat([visual_memory, context_memory], dim=1)
        
        B = visual_memory.size(0)
        V_len = visual_memory.size(1)
        visual_pad_mask = torch.zeros(B, V_len, dtype=torch.bool, device=memory.device)
        context_pad_mask = (rag_attention_mask == 0)
        memory_key_padding_mask = torch.cat([visual_pad_mask, context_pad_mask], dim=1)
        
        return memory, memory_key_padding_mask

