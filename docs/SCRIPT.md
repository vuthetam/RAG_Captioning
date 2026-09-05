# Thứ tự chạy script

Chạy các lệnh từ thư mục gốc của project. Cần có dataset MS COCO và cấu hình đường dẫn trong `.env` hoặc dùng giá trị mặc định trong `src/config.py`.

## 1. Chuẩn bị dữ liệu

```bash
python script/preprocess_dfs.py
```

Script đọc `dataset_coco.json`, tách dữ liệu thành train/validation/test và tạo:

- `artifacts/splits/train_df.parquet`
- `artifacts/splits/val_df.parquet`
- `artifacts/splits/test_df.parquet`

## 2. Tạo vocabulary

```bash
python script/build_vocab.py
```

Script đọc `train_df.parquet`, tạo vocabulary từ cột `tokens` với ngưỡng tần suất mặc định là `5`, rồi ghi:

```text
artifacts/vocab.json
```

## 3. Xây dựng Knowledge Base cho RAG

```bash
python script/build_kb.py
```

Script dùng caption của tập train để tạo text embedding bằng model `openai/clip-vit-large-patch14-336`, chuẩn hóa vector và lưu:

- `artifacts/kb/kb_text_index.faiss`
- `artifacts/kb/kb_metadata.parquet`

## 4. Retrieve context RAG

```bash
accelerate launch script/retrieve_rag_contexts.py
```

Script dùng ảnh làm truy vấn CLIP, tìm các caption tương tự trong FAISS và lưu tối đa `8` context cho mỗi ảnh vào:

- `artifacts/rag/train_rag_contexts.parquet`
- `artifacts/rag/val_rag_contexts.parquet`
- `artifacts/rag/test_rag_contexts.parquet`

Script đọc trực tiếp ảnh và sử dụng `kb_text_index.faiss` cùng `kb_metadata.parquet`.

## 5. Train, generate và evaluate

Thực hiện các notebook theo thứ tự:

1. `train.ipynb`: train `BaselineCaptioner`, lưu checkpoint vào `checkpoints/<RUN_MODE>/`.
2. `generate_caption.ipynb`: load `best_checkpoint.pth`, sinh caption cho test set bằng beam search và lưu prediction JSON.
3. `evaluate.ipynb`: đọc prediction và `test_df.parquet`, tính BLEU, METEOR, ROUGE-L, CIDEr, SPICE và hiển thị mẫu kết quả.

Các notebook hiện được thiết kế để chạy trong môi trường notebook/Kaggle. Khi chạy local, cần chỉnh các cell cấu hình đường dẫn và bảo đảm `main.py` được tạo đúng thư mục mà lệnh `accelerate launch` sử dụng.

## Cấu hình đường dẫn chính

Các biến được định nghĩa trong `src/config.py`:

| Biến | Mặc định |
| --- | --- |
| `DATASET_COCO_PATH` | `dataset/mscoco/dataset_coco.json` |
| `IMAGES_PATH` | `dataset/mscoco/images` |
| `ARTIFACTS_DIR` | `artifacts` |
| `CHECKPOINTS_DIR` | `checkpoints` |
| `RUN_MODE` | `baseline` |

Có thể ghi đè bằng biến môi trường hoặc file `.env`.
