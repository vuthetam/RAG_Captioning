# RAG-Augmented Memory cho Image Captioning

Tích hợp relevant captions từ RAG retrieval vào memory của encoder, để decoder cross-attention nhận được cả thông tin visual lẫn thông tin ngữ nghĩa từ các caption tương tự.

## Kiến trúc hiện tại (Baseline)

```
Image pixels → CLIP ViT → patch tokens (B, 197, 768)
                              ↓
                     VisualProjector (Linear 768→512)
                              ↓
                     memory (B, 196, 512)  ← bỏ CLS token
                              ↓
              TransformerDecoder cross-attention
                              ↓
                        logits (B, T, V)
```

- RAG retrieval đã có sẵn: mỗi ảnh có 8 retrieved captions (text strings) + scores, lưu trong `artifacts/rag/*.parquet`
- Dataset trả về `(visual_features, input_ids, attention_mask)` — **chưa** kèm RAG context

## Kiến trúc đề xuất (RAG-Augmented)

```
Image → CLIP ViT → patch tokens (B, 196, 512)
                              ↓
                     visual_memory ──────────────┐
                                                  │ concat dim=1
RAG captions (K câu) → normalize + tokenize      │
         ↓                                        │
    SharedEmbedding + PositionalEncoding           │
         ↓                                        │
    TransformerEncoder (self-attention, 2 layers)  │
         ↓                                        │
    context_memory (B, L_ctx, 512) ──────────────┘
                                                  ↓
                              enriched_memory (B, 196+L_ctx, 512)
                              memory_key_padding_mask (B, 196+L_ctx)
                                                  ↓
                              TransformerDecoder cross-attention
                                                  ↓
                                            logits (B, T, V)
```

> [!IMPORTANT]
> **Ý tưởng cốt lõi**: Concat visual tokens với context tokens dọc theo trục sequence → memory lớn hơn cho decoder cross-attend. Decoder **không cần thay đổi kiến trúc**, chỉ nhận memory dài hơn + padding mask.

---

## Proposed Changes

### Component 1: Text Context Encoder

#### [NEW] [context_encoder.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/src/context_encoder.py)

Module encode chuỗi RAG captions đã nối thành context memory:

```python
class TextContextEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout, max_ctx_length,
                 embedding: nn.Embedding):
        # embedding: reference từ decoder (shared, không copy)
        # pos_encoding: PositionalEncoding(d_model, max_ctx_length)
        # transformer_encoder: nn.TransformerEncoder(num_layers=num_layers)  ← CTX_NLAYERS từ config

    def forward(self, rag_input_ids, rag_attention_mask):
        # x = embedding(rag_input_ids) * sqrt(d_model)
        # x = pos_encoding(x)
        # src_key_padding_mask = (rag_attention_mask == 0)
        # x = transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # return x  → (B, L_ctx, d_model)
```

- Nhận `rag_input_ids` `(B, L_ctx)` — chuỗi đã nối K captions, tokenize sạch
- Dùng **shared embedding** reference từ decoder
- Positional Encoding riêng (max_len = `MAX_CTX_LENGTH`)
- TransformerEncoder 2 layers self-attention → contextual representation
- Output: `(B, L_ctx, d_model)` — context memory

---

### Component 2: RAG Captioner Model

#### [NEW] [rag.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/src/models/rag.py)

Model mới, tái sử dụng `VisualProjector` + `TransformerCaptionDecoder` từ baseline, thêm `TextContextEncoder`:

```python
class RAGCaptioner(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, nlayers, dropout, max_length,
                 pad_idx, use_precomputed_features, visual_feature_dim,
                 ctx_nlayers, max_ctx_length):
        # self.visual_projector = VisualProjector(...)
        # self.decoder = TransformerCaptionDecoder(...)
        # self.context_encoder = TextContextEncoder(
        #     embedding=self.decoder.embedding,  ← shared reference
        #     ...
        # )

    def forward(self, visual_inputs, input_ids, attention_mask,
                rag_input_ids, rag_attention_mask):
        # 1. visual_memory = project(visual_features)     → (B, 196, D)
        # 2. context_memory = context_encoder(rag_input_ids, rag_attention_mask)
        #                                                  → (B, L_ctx, D)
        # 3. memory = cat([visual_memory, context_memory], dim=1)
        #                                                  → (B, 196+L_ctx, D)
        # 4. memory_key_padding_mask = cat([
        #        zeros(B, 196),           ← visual tokens: không mask
        #        rag_attention_mask == 0   ← context tokens: mask padding
        #    ], dim=1)
        # 5. logits = decoder(input_ids, memory, attention_mask, memory_key_padding_mask)

    def encode_memory(self, visual_inputs, rag_input_ids, rag_attention_mask):
        # Cho inference — trả về (enriched_memory, memory_key_padding_mask)
        # beam_search sẽ gọi hàm này thay vì encode_image/encode_features
```

> [!NOTE]
> `TransformerCaptionDecoder.forward()` đã có param `memory_key_padding_mask` → decoder không cần sửa gì, chỉ truyền mask đúng để nó ignore padding tokens trong context.

---

### Component 3: Dataset cung cấp RAG context

#### [MODIFY] [dataset.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/src/dataset.py)

Thêm 2 class mới + hàm tiền xử lý RAG text:

**Hàm normalize RAG text:**
```python
import re, string

def normalize_caption(text: str) -> list[str]:
    """Lowercase + bỏ dấu câu + split → token list.
    Khớp pipeline tokenize Karpathy split."""
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text.split()
```

**Hàm encode chuỗi nối K captions:**
```python
def encode_rag_context(
    retrieved_texts: list[str],
    vocab: Vocabulary,
    max_ctx_length: int,
    top_k: int,
) -> tuple[Tensor, Tensor]:
    """
    Normalize + tokenize từng caption, nối bằng <eos>, encode thành tensor.

    Ví dụ với K=3:
      "A hand is cutting a cake." → [a, hand, is, cutting, a, cake]
      "A woman cuts cake."       → [a, woman, cuts, cake]
      "A person with a knife."   → [a, person, with, a, knife]

    Nối: [a, hand, is, cutting, a, cake, <eos>, a, woman, cuts, cake, <eos>,
           a, person, with, a, knife]

    Encode: [45, 102, 8, 233, 45, 789, 2, 45, 87, 445, 789, 2, 45, 201, 67, 45, 512]
    Pad tới max_ctx_length, tạo attention_mask.
    """
    all_tokens = []
    for caption in retrieved_texts[:top_k]:
        tokens = normalize_caption(caption)
        if all_tokens:                      # thêm separator trước caption thứ 2+
            all_tokens.append(vocab.eos_token)
        all_tokens.extend(tokens)

    # Cắt nếu quá dài
    if len(all_tokens) > max_ctx_length:
        all_tokens = all_tokens[:max_ctx_length]

    # Encode tokens → IDs
    token_ids = [vocab.token_to_idx(t) for t in all_tokens]
    attention_mask = [1] * len(token_ids)

    # Pad
    pad_len = max_ctx_length - len(token_ids)
    token_ids.extend([vocab.pad_idx()] * pad_len)
    attention_mask.extend([0] * pad_len)

    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)
```

**Class `RAGPrecomputedFeatureDataset`:**
```python
class RAGPrecomputedFeatureDataset(PrecomputedFeatureDataset):
    def __init__(self, df, vocab, features_path, rag_contexts_path,
                 max_length, max_ctx_length, top_k):
        super().__init__(df, vocab, features_path, max_length)
        rag_df = pd.read_parquet(rag_contexts_path)
        # Build lookup: imgid → retrieved_texts list
        self._rag_lookup = dict(zip(rag_df["imgid"], rag_df["retrieved_texts"]))
        self.max_ctx_length = max_ctx_length
        self.top_k = top_k

    def __getitem__(self, idx):
        visual_feature, input_ids, attention_mask = super().__getitem__(idx)
        imgid = int(self.df.iloc[idx]["imgid"])
        retrieved_texts = self._rag_lookup.get(imgid, [])
        rag_input_ids, rag_attention_mask = encode_rag_context(
            retrieved_texts, self.vocab, self.max_ctx_length, self.top_k
        )
        return visual_feature, input_ids, attention_mask, rag_input_ids, rag_attention_mask
```

**Class `RAGPrecomputedFeatureOnlyDataset`** (cho inference):
```python
class RAGPrecomputedFeatureOnlyDataset(PrecomputedFeatureOnlyDataset):
    def __init__(self, df, features_path, rag_contexts_path,
                 vocab, max_ctx_length, top_k):
        super().__init__(df, features_path)
        rag_df = pd.read_parquet(rag_contexts_path)
        self._rag_lookup = dict(zip(rag_df["imgid"], rag_df["retrieved_texts"]))
        self.vocab = vocab
        self.max_ctx_length = max_ctx_length
        self.top_k = top_k

    def __getitem__(self, idx):
        visual_feature, imgid = super().__getitem__(idx)
        retrieved_texts = self._rag_lookup.get(imgid, [])
        rag_input_ids, rag_attention_mask = encode_rag_context(
            retrieved_texts, self.vocab, self.max_ctx_length, self.top_k
        )
        return visual_feature, rag_input_ids, rag_attention_mask, imgid
```

---

### Component 4: Config mới

#### [MODIFY] [config.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/src/config.py)

Thêm hyperparameters cho RAG context encoder vào section 5:

```python
# RAG Context Encoder
CTX_NLAYERS = int(os.getenv("CTX_NLAYERS", "2"))       # Layers cho TextContextEncoder
MAX_CTX_LENGTH = int(os.getenv("MAX_CTX_LENGTH", "80")) # Max tokens sau khi nối K captions
TOP_K_CAPTIONS = int(os.getenv("TOP_K_CAPTIONS", "4"))  # Dùng 4 trong 8 retrieved captions
```

---

### Component 5: Engine hỗ trợ RAG inputs

#### [MODIFY] [engine.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/src/engine.py)

Sửa `_step`, `train_one_epoch`, `evaluate_one_epoch` để xử lý batch 5 phần tử:

```python
def _step(model, visual_inputs, input_ids, attention_mask, pad_idx, **kwargs):
    target_ids = input_ids[:, 1:]
    logits = model(visual_inputs, input_ids, attention_mask, **kwargs)
    # ... loss tính như cũ

def train_one_epoch(...):
    for batch in dataloader:
        if len(batch) == 5:
            visual_inputs, input_ids, attention_mask, rag_input_ids, rag_attention_mask = batch
            kwargs = {"rag_input_ids": rag_input_ids.to(device),
                      "rag_attention_mask": rag_attention_mask.to(device)}
        else:
            visual_inputs, input_ids, attention_mask = batch
            kwargs = {}
        _, loss_sum, num_tokens = _step(model, visual_inputs, input_ids, attention_mask, pad_idx, **kwargs)
```

Tương tự cho `evaluate_one_epoch`.

---

### Component 6: Inference hỗ trợ RAG *(chưa implement lần này)*

#### [MODIFY] [inference.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/src/inference.py)

> [!NOTE]
> Giữ plan để reference, sẽ implement sau khi training hoạt động ổn.

`beam_search` **không cần sửa** — nó đã nhận `memory` tensor sẵn rồi. Tuy nhiên cần sửa signature để nhận thêm `memory_key_padding_mask`:

```python
def beam_search(decoder, memory, vocab, beam_size, max_length, length_penalty,
                memory_key_padding_mask=None):
    # Expand memory_key_padding_mask giống cách expand memory cho k beams
    # Truyền vào decoder(..., memory_key_padding_mask=...)
```

Sửa `generate_captions` để xử lý RAG dataloader:

```python
for batch in dataloader:
    if len(batch) == 4:  # RAG: (visual, rag_ids, rag_mask, imgid)
        visual_inputs, rag_input_ids, rag_attention_mask, image_ids = batch
        memory, mem_mask = base_model.encode_memory(visual_inputs, rag_input_ids, rag_attention_mask)
    else:               # Baseline: (visual, imgid)
        visual_inputs, image_ids = batch
        memory = base_model.encode_features(visual_inputs)
        mem_mask = None

    sequences = beam_search(decoder, memory, vocab, beam_size, max_length,
                            length_penalty, memory_key_padding_mask=mem_mask)
```

---

### Component 7: Training script

#### [NEW] [train_rag.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/script/train_rag.py)

Script training mới cho RAG model, tương tự [train_precomputed_features.py](file:///home/tam/Link%20to%20workspace/ML/rag_captioning/script/train_precomputed_features.py) nhưng:
- Dùng `RAGPrecomputedFeatureDataset` (truyền thêm `rag_contexts_path`, `max_ctx_length`, `top_k`)
- Dùng `RAGCaptioner` thay vì `BaselineCaptioner`
- `RUN_MODE = 'rag'` → checkpoint lưu riêng tại `checkpoints/rag/`

---

## Tất cả quyết định đã xác nhận

| Quyết định | Chọn | Lý do |
|---|---|---|
| Embedding | **Shared** với decoder | Cùng vocab, cùng không gian biểu diễn, giảm ~5M params |
| Separator giữa K captions | **`<eos>`** | Đã có trong vocab, không đổi `vocab_size`, không phá checkpoint |
| Tokenize RAG texts | **Lowercase + bỏ dấu câu + split** | Khớp pipeline Karpathy split |
| `MAX_CTX_LENGTH` | **80** | 4 captions ≈ 50–55 tokens, dư margin. Memory tổng = 196 + 80 = 276 |
| `CTX_NLAYERS` | **2** | Lightweight, text đã mang semantic từ retrieval |
| `TOP_K_CAPTIONS` | **4** | Dùng 4 trong 8 retrieved captions |
| Warm-start | **Không** | Train từ đầu, checkpoint baseline và RAG tách riêng theo `RUN_MODE` |
| Inference | **Chưa implement** | Giữ plan, implement sau |

---

## Verification Plan

### Automated Tests
```bash
# Compile check
"/home/tam/Link to workspace/ML/.venv/bin/python" -m compileall src script

# Smoke test forward pass
"/home/tam/Link to workspace/ML/.venv/bin/python" -c "
from src.models.rag import RAGCaptioner
import torch
model = RAGCaptioner(vocab_size=5000, d_model=512, ...)
visual = torch.randn(2, 197, 768)
ids = torch.randint(0, 5000, (2, 20))
mask = torch.ones(2, 20)
rag_ids = torch.randint(0, 5000, (2, 80))
rag_mask = torch.ones(2, 80)
out = model(visual, ids, mask, rag_ids, rag_mask)
assert out.shape == (2, 19, 5000)
"

# Verify RAG dataset output shapes
"/home/tam/Link to workspace/ML/.venv/bin/python" -c "
from src.dataset import RAGPrecomputedFeatureDataset
# ... load data, check output tuple has 5 elements
"
```

