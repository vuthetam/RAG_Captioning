# V3 Implementation Plan: Dual Cross-Attention RAG Captioner

## Tổng quan kiến trúc

V3 cải tiến V2 theo 3 hướng đột phá:

1. **Prompt Template**: Chuyển đổi dữ liệu RAG (caption + objects + relations) thành văn bản tự nhiên có cấu trúc (thay vì nối Tensor thô). Sử dụng `vocab_v3` (có thêm dấu câu) để tokenize.
2. **Chiến lược FiD (Fusion-in-Decoder)**: Encode ĐỘC LẬP từng câu RAG truy hồi để tránh lai tạp tri thức và bảo toàn Positional Encoding. Sau đó gộp kết quả lại ở Decoder.
3. **Dual Cross-Attention Decoder**: Mỗi Decoder Layer có 2 luồng Cross-Attention độc lập (Image và RAG Text), kết hợp bằng Dynamic Gate linh động theo từng từ.

```text
Decoder Layer (mỗi layer — chuẩn Add & Norm):
    ┌─ Sub-layer 1: Self-Attn (causal) ──────────────────────────────┐
    │   x = norm1(x + dropout(self_attn(x, x, x, causal_mask)))      │
    └─────────────────────────────────────────────────────────────────┘
    ┌─ Sub-layer 2: Dual Cross-Attn + Gate ──────────────────────────┐
    │   attn_img = img_cross_attn(query=x, K/V=image_memory)         │
    │   attn_rag = rag_cross_attn(query=x, K/V=rag_memory)           │
    │   gate     = sigmoid(Linear([attn_img, attn_rag]))              │
    │   fused    = gate * attn_img + (1 - gate) * attn_rag           │
    │   x = norm2(x + dropout(fused))           ← Add & Norm         │
    └─────────────────────────────────────────────────────────────────┘
    ┌─ Sub-layer 3: FFN ──────────────────────────────────────────────┐
    │   x = norm3(x + dropout(ffn(x)))          ← Add & Norm         │
    └─────────────────────────────────────────────────────────────────┘
```

---

## Dữ liệu

- **Không thay đổi:** `build_kb.py`, `retrieve_rag_contexts.py`, các file `.parquet`, `.h5`.
- **Vocab:** Sử dụng `VOCAB_PATH` đã được cập nhật thêm các dấu câu `:`, `,`, `.`.

### Prompt Template (Tạo ra K dòng độc lập cho 1 ảnh):
```text
Dòng 1: "similar image shows: {cap1}. objects: {obj1}. actions: {act1}."
Dòng 2: "similar image shows: {cap2}. objects: {obj2}. actions: {act2}."
...
```

---

## Danh sách file cần tạo / chỉnh sửa

### Bước 1 — Dataset
| File | Ghi chú |
|:---|:---|
| `src/v3/dataset.py` | Ghép prompt thành K chuỗi rời rạc. Output `rag_tokens` dạng `[B, K, L]` |

Dataset trả về 5 phần tử:
```python
visual_features  # [B, 197, 768]    — Ảnh
input_ids        # [B, MAX_LEN]     — Caption gốc (để teacher forcing)
attention_mask   # [B, MAX_LEN]     — Mask caption gốc
rag_tokens       # [B, K, MAX_RAG]  — K chuỗi RAG ĐỘC LẬP
rag_attn_mask    # [B, K, MAX_RAG]  — Mask cho K chuỗi RAG
```

### Bước 2 — Decoder (QUAN TRỌNG NHẤT)
| File | Ghi chú |
|:---|:---|
| `src/v3/decoder.py` | Custom `DualCrossAttnDecoderLayer` + `DualCrossAttnDecoder` |

### Bước 3 — Model
| File | Ghi chú |
|:---|:---|
| `src/v3/models/__init__.py` | Empty |
| `src/v3/models/rag.py` | Lắp ráp: VisualProjector + RagContextEncoder + DualCrossAttnDecoder |

> **Lưu ý thiết kế RAG Encoder theo chuẩn FiD:**
> 1. Reshape `rag_tokens` thành `[B*K, MAX_RAG]` để **Encode Độc Lập**.
> 2. Đưa qua Embedding + Positional Encoding (Bảo toàn chuẩn vị trí 0->L cho từng câu, PAD không làm hỏng PE).
> 3. Đi qua Transformer Encoder (Self-Attention) -> Output: `[B*K, MAX_RAG, D]`. Không lo ảo giác lai tạp tri thức.
> 4. Xếp nối đuôi lại (Flatten): Reshape thành `rag_memory [B, K * MAX_RAG, D]` làm input cho Cross-Attention. (Mask cũng reshape tương tự).
> 5. **BỎ HOÀN TOÀN** phần Soft Filter (`RagFusionEncoder` của V2).

### Bước 4 — Engine
| File | Ghi chú |
|:---|:---|
| `src/v3/engine.py` | Unpack 5 phần tử, loss, backward |

### Bước 5 — Scripts
| File | Ghi chú |
|:---|:---|
| `script/v3/train_rag.py` | Tương tự V2, dùng VOCAB_PATH hiện tại |
| `script/v3/generate_rag_captions.py` | Tương tự V2 |

---

## Thứ tự triển khai

```text
[1] dataset.py  →  [2] decoder.py
                          ↓
[4] engine.py   ←  [3] models/rag.py
                          ↓
[5] train_rag.py + generate_rag_captions.py
```
