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

## 5. Train, generate va evaluate bang visual features

Sau khi da co ca ba file H5, train va sinh caption khong can doc anh hay tai
CLIP vision encoder nua:

```bash
RUN_MODE=baseline_precomputed /home/tam/Link\ to\ workspace/ML/.venv/bin/accelerate launch script/train_precomputed_features.py
RUN_MODE=baseline_precomputed /home/tam/Link\ to\ workspace/ML/.venv/bin/accelerate launch script/generate_precomputed_captions.py
```

Hai lenh phai dung cung `RUN_MODE` de dung chung checkpoint. `RUN_MODE` moi
tranh resume nham checkpoint cu duoc train theo duong online-encoder. Feature
files can co format `imgids` va `features` nhu mo ta o `ARTIFACTS.md`; code se
kiem tra moi `imgid` trong split deu co feature truoc khi train.

Cuoi cung chay `evaluate.ipynb` de tinh BLEU, METEOR, ROUGE-L, CIDEr va SPICE
tu prediction JSON. Notebook nay khong encode anh; phan hien thi mau van can
`IMAGES_PATH` neu muon xem anh goc.

## Cấu hình đường dẫn chính

Các biến được định nghĩa trong `src/config.py`:

| Biến | Mặc định |
| --- | --- |
| `DATASET_COCO_PATH` | `dataset/mscoco/dataset_coco.json` |
| `IMAGES_PATH` | `dataset/mscoco/images` |
| `ARTIFACTS_DIR` | `artifacts` |
| `VISUAL_FEATURES_DIR` | `artifacts/visual_features` |
| `CHECKPOINTS_DIR` | `checkpoints` |
| `RUN_MODE` | `baseline` |

Có thể ghi đè bằng biến môi trường hoặc file `.env`.
