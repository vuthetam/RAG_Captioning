# Cấu trúc và nội dung artifacts

## Cấu trúc thư mục

```text
artifacts/
├── vocab.json
├── <RUN_MODE>_predictions.json
├── baseline/
│   └── baseline_predictions.json
├── splits/
│   ├── train_df.parquet
│   ├── val_df.parquet
│   └── test_df.parquet
├── kb/
│   ├── kb_text_index.faiss
│   └── kb_metadata.parquet
├── visual_features/
│   ├── train_visual_features.h5
│   ├── val_visual_features.h5
│   └── test_visual_features.h5
└── rag/
    ├── train_rag_contexts.parquet
    ├── val_rag_contexts.parquet
    └── test_rag_contexts.parquet
```

## 1. `splits/*.parquet`

Được tạo bởi `script/preprocess_dfs.py`.

- `train_df.parquet`: một dòng cho mỗi caption của ảnh thuộc split `train` hoặc `restval`.
- `val_df.parquet`: một dòng cho mỗi caption của ảnh thuộc split `val`.
- `test_df.parquet`: một dòng cho mỗi ảnh thuộc split `test`, gom 5 caption tham chiếu để đánh giá.

Các cột dùng trong pipeline:

| File | Cột | Ý nghĩa |
| --- | --- | --- |
| train & val | `imgid` | ID ảnh |
| train & val | `filepath`, `filename`, `split` | Thông tin ảnh/split |
| train & val | `tokens` | Caption đã tách token, dạng list |
| train & val | `raw` | Caption gốc, dạng chuỗi |
| test | `imgid` | ID ảnh, duy nhất |
| test | `filepath`, `filename`, `split` | Thông tin ảnh/split |
| test | `all_tokens` | Danh sách token của các caption tham chiếu |
| test | `all_raws` | Danh sách 5 caption tham chiếu |

## 2. `vocab.json`

Được tạo bởi `script/build_vocab.py` từ `train_df.parquet`. Đây là JSON state của `Vocabulary`, gồm:

```json
{
  "freq_threshold": 5,
  "pad_token": "<pad>",
  "sos_token": "<sos>",
  "eos_token": "<eos>",
  "unk_token": "<unk>",
  "special_tokens": ["<pad>", "<sos>", "<eos>", "<unk>"],
  "itos": ["..."],
  "stoi": {"token": 0}
}
```

`itos` ánh xạ index sang token; `stoi` ánh xạ token sang index.

## 3. `kb/kb_text_index.faiss`

Index FAISS kiểu `IndexFlatIP`. Mỗi vector là text embedding CLIP đã L2-normalize, vì vậy Inner Product tương đương cosine similarity. Vector ở vị trí `i` phải tương ứng với dòng `i` trong `kb_metadata.parquet`.

## 4. `kb/kb_metadata.parquet`

Được tạo cùng lúc với FAISS index từ caption của `train_df.parquet`. Gồm các cột:

- `imgid`: ID ảnh nguồn.
- `filepath`, `filename`: vị trí ảnh nguồn.
- `caption`: caption được mã hóa ở vị trí tương ứng trong FAISS index.

Khi retrieve, các caption có cùng `imgid` với ảnh truy vấn sẽ bị loại bỏ, sau đó giữ lại tối đa `8` caption có điểm cao nhất.

## 5. `visual_features/*_visual_features.h5`

Được tạo bởi `script/extract_visual_features.py`. Mặc định các file này nằm trong
`artifacts/visual_features`. Đặt `VISUAL_FEATURES_DIR` để ghi các file HDF5 vào
một thư mục khác. Mỗi `imgid` lưu output CLIP nguyên gốc gồm CLS token và patch
tokens; với model mặc định `openai/clip-vit-base-patch16`, shape là `(197, 768)`.
Khi train, có thể dùng đủ 197 token hoặc bỏ CLS qua `features[:, 1:, :]` để còn
196 patch tokens. Mỗi file HDF5 có hai dataset:

| Dataset | Shape | Nội dung |
| --- | --- | --- |
| `features` | `(N, 197, 768)` | CLIP features `float16`; `features[i]` là feature của `imgids[i]` |
| `imgids` | `(N,)` | Image ID `int64` ứng với từng row của `features` |

`features` được chunk theo từng ảnh để ưu tiên random access. Khi train qua
`script/train_precomputed_features.py`, dataframe van co mot dong cho moi
caption nhung feature duoc tra theo `imgid`.

## 6. `rag/*_rag_contexts.parquet`

Được tạo bởi `script/retrieve_rag_contexts.py`, mỗi dòng tương ứng với một ảnh duy nhất:

| Cột | Kiểu/nội dung |
| --- | --- |
| `imgid` | ID ảnh |
| `retrieved_texts` | list tối đa 8 caption được retrieve |
| `retrieved_scores` | list điểm cosine tương ứng với `retrieved_texts` |

Phần tử ở cùng vị trí trong hai list là một cặp caption/score.

## 7. `<RUN_MODE>_predictions.json`

Được tạo bởi `script/generate_precomputed_captions.py`. Theo `src/config.py`, file được ghi trực tiếp vào `artifacts/<RUN_MODE>_predictions.json`; với `RUN_MODE=baseline`, tên mặc định là `artifacts/baseline_predictions.json`.

Đây là JSON array, mỗi phần tử có dạng:

```json
{
  "imgid": 0,
  "caption": "a man riding a motorcycle down a dirt road"
}
```

`imgid` phải khớp với `test_df.parquet`; `caption` là caption dự đoán sau khi giải mã token.
