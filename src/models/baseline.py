import torch.nn as nn
from src.encoder import CLIPViTB16Encoder
from src.decoder import TransformerCaptionDecoder

class BaselineCaptioner(nn.Module):
    def __init__(self, vocab_size, d_model, nheads, nlayers, dropout, max_length, pad_idx):
        super().__init__()
        self.encoder = CLIPViTB16Encoder(d_model=d_model)
        self.decoder = TransformerCaptionDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nheads,
            num_layers=nlayers,
            dropout=dropout,
            max_length=max_length,
            pad_idx=pad_idx
        )
        
    def forward(self, images, input_ids, attention_mask):
        """
        Dùng cho lúc Training (Teacher Forcing).
        Cắt token cuối của input_ids làm đầu vào cho decoder.
        """
        memory = self.encoder(images)
        
        decoder_input_ids = input_ids[:, :-1]
        decoder_attention_mask = attention_mask[:, :-1]
        
        logits = self.decoder(
            input_ids=decoder_input_ids,
            memory=memory,
            attention_mask=decoder_attention_mask,
        )
        return logits
    
    def encode_image(self, images):
        """
        Dùng cho lúc Inference. Chỉ cần trích xuất memory 1 lần.
        """
        return self.encoder(images)
