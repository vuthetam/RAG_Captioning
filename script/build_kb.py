import sys
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import faiss

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_DF_PATH, KB_MODEL_ID, KB_FAISS_INDEX_PATH, KB_METADATA_PATH


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang sử dụng device: {device}")

    # 1. Load CLIP Model
    print(f"Loading CLIP model ({KB_MODEL_ID})...")
    model = CLIPModel.from_pretrained(KB_MODEL_ID, torch_dtype=torch.float16).to(device)
    processor = CLIPProcessor.from_pretrained(KB_MODEL_ID)
    model.eval()


    # 3. Đọc tập Train và lọc lấy các cột cần thiết cho Metadata
    print("Đọc tập train_df...")
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    
    # Đảm bảo tương thích: tìm xem cột chứa văn bản tên là 'caption' hay 'raw'
    cap_col = 'caption' if 'caption' in train_df.columns else 'raw'
    
    # Chỉ giữ lại các cột cần thiết cho Metadata
    metadata_df = train_df[['imgid', 'filepath', 'filename', cap_col]].copy()
    metadata_df.rename(columns={cap_col: 'caption'}, inplace=True)
    
    print(f"Tổng số caption cần mã hóa: {len(metadata_df):,}")

    # 4. Khởi tạo FAISS Index
    # CLIP large có dimension = 768. IndexFlatIP dùng cho Cosine Similarity.
    d = model.config.text_config.hidden_size # Tự động lấy số chiều (768 với ViT-L)
    index = faiss.IndexFlatIP(d)

    # 5. Rút trích Vector theo Batch
    batch_size = 512
    captions = metadata_df['caption'].tolist()
    
    print("Bắt đầu trích xuất Text Embeddings...")
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="Encoding Captions", leave=False):
            batch_texts = captions[i : i + batch_size]
            
            # Tiền xử lý text
            inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Lấy text features
            # Lấy text features an toàn (tương thích mọi phiên bản transformers)
            text_features = model.get_text_features(**inputs)
            if not isinstance(text_features, torch.Tensor):
                if hasattr(text_features, "text_embeds"):
                    text_features = text_features.text_embeds
                elif hasattr(text_features, "pooler_output"):
                    text_features = model.text_projection(text_features.pooler_output)
                else:
                    text_features = text_features[0]
            
            # Chuẩn hóa (Normalize) vector để dùng Inner Product tính ra Cosine Similarity
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            
            # Chuyển về numpy float32 và nạp vào FAISS
            embeddings_np = text_features.cpu().numpy().astype('float32')
            index.add(embeddings_np)

    # 6. Lưu file xuống ổ cứng
    print("\nĐang lưu Knowledge Base xuống đĩa...")
    faiss.write_index(index, str(KB_FAISS_INDEX_PATH))
    metadata_df.to_parquet(KB_METADATA_PATH)
    
    print(f"HOÀN TẤT! Đã lưu tại:\n- {KB_FAISS_INDEX_PATH}\n- {KB_METADATA_PATH}")

if __name__ == "__main__":
    main()
