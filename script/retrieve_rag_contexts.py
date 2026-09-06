# Chạy bằng accelerate

import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel
import faiss
from PIL import Image
from accelerate import Accelerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
    TRAIN_RAG_CONTEXTS_PATH, VAL_RAG_CONTEXTS_PATH, TEST_RAG_CONTEXTS_PATH,
    KB_MODEL_ID, KB_FAISS_INDEX_PATH, KB_METADATA_PATH,
    IMAGES_DIR
)
from src.utils import extract_clip_features

TARGET_K = 8

class ImageQueryDataset(Dataset):
    def __init__(self, df, images_path, processor):
        self.imgids = df['imgid'].tolist()
        self.filepaths = df['filepath'].tolist()
        self.filenames = df['filename'].tolist()
        self.images_path = images_path
        self.processor = processor

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        path = self.images_path / self.filepaths[idx] / self.filenames[idx]
        with Image.open(path) as img:
            pixel_values = self.processor(images=img.convert("RGB"), return_tensors="pt")
            
        return {
            "imgid": int(self.imgids[idx]), 
            # Squeeze to remove batch dim added by processor
            "pixel_values": pixel_values["pixel_values"].squeeze(0)
        }

def process_and_retrieve(df_path, output_parquet_path, encoder, processor, index, kb_metadata, accelerator, batch_size=256):
    accelerator.print(f"\nĐang xử lý {df_path.name}...")
    
    df = pd.read_parquet(df_path)
    df_unique = df.drop_duplicates(subset=['imgid']).reset_index(drop=True)
    dataset = ImageQueryDataset(df_unique, IMAGES_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    dataloader = accelerator.prepare(dataloader)
    
    results = []
    
    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Retrieving {df_path.name}", leave=False)
    unwrap_encoder = accelerator.unwrap_model(encoder)

    for batch in pbar:
        imgids = batch["imgid"]
        pixel_values = batch["pixel_values"]
        
        with torch.no_grad():
            with accelerator.autocast():
                output = unwrap_encoder.get_image_features(pixel_values=pixel_values)
                image_features = extract_clip_features(output)
                
            # Chuẩn hóa vector về độ dài = 1
            image_features = F.normalize(image_features, p=2, dim=-1)
            
        # Gom Vector và ID từ 2 GPU về
        gathered_ids, gathered_features = accelerator.gather_for_metrics((imgids, image_features))
        
        if accelerator.is_main_process:
            gathered_features = gathered_features.cpu().numpy().astype('float32')
            gathered_ids = gathered_ids.cpu().numpy()
            
            # Truy vấn FAISS (Chạy trên CPU)
            distances, indices = index.search(gathered_features, TARGET_K+7)
            
            for idx, imgid in enumerate(gathered_ids):
                retrieved_ids = indices[idx]
                
                # Lấy ID gốc của ảnh, caption tương ứng dựa vào id của faiss
                retrieved_imgids = kb_metadata.iloc[retrieved_ids]['imgid'].tolist()
                raw_texts = kb_metadata.iloc[retrieved_ids]['caption'].tolist()
                raw_scores = distances[idx].tolist()
                
                valid_texts = []
                valid_scores = []
                
                for r_id, txt, score in zip(retrieved_imgids, raw_texts, raw_scores):
                    if r_id != imgid: # Loại bỏ các caption của chính bức ảnh truy xuất
                        valid_texts.append(txt)
                        valid_scores.append(score)
                
                # Cắt đúng Top TARGET_K câu xịn nhất (sau khi đã lọc)
                results.append({
                    'imgid': imgid,
                    'retrieved_texts': valid_texts[:TARGET_K],
                    'retrieved_scores': valid_scores[:TARGET_K]
                })

    # Sau khi chạy xong toàn bộ batch, GPU 0 lưu file
    if accelerator.is_main_process:
        results_df = pd.DataFrame(results)
        
        output_parquet_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_parquet(output_parquet_path)
        accelerator.print(f"Đã lưu kết quả tại {output_parquet_path}")

        
def main():
    accelerator = Accelerator(mixed_precision="fp16")
    
    accelerator.print(f"Khởi động môi trường Multi-GPU ({accelerator.num_processes} processes)")
    accelerator.print(f"Loading CLIP model ({KB_MODEL_ID})...")
    
    encoder = CLIPModel.from_pretrained(KB_MODEL_ID)
    encoder.eval()
    encoder = accelerator.prepare(encoder)
    
    processor = CLIPProcessor.from_pretrained(KB_MODEL_ID)
    
    accelerator.print("Đang tải FAISS Index và Knowledge Base Metadata...")
    # Khởi tạo None, chỉ GPU 0 cần load FAISS index để tiết kiệm RAM CPU
    index = None
    kb_metadata = None
    
    if accelerator.is_main_process:
        if not KB_FAISS_INDEX_PATH.exists():
            accelerator.print(f"Lỗi: Không tìm thấy {KB_FAISS_INDEX_PATH}")
            return
        index = faiss.read_index(str(KB_FAISS_INDEX_PATH))
        kb_metadata = pd.read_parquet(KB_METADATA_PATH)

    datasets = [
        (TRAIN_DF_PATH, TRAIN_RAG_CONTEXTS_PATH),
        (VAL_DF_PATH, VAL_RAG_CONTEXTS_PATH),
        (TEST_DF_PATH, TEST_RAG_CONTEXTS_PATH)
    ]
    
    for df_path, parquet_path in datasets:
        if df_path.exists():
            process_and_retrieve(df_path, parquet_path, encoder, processor, index, kb_metadata, accelerator)
        else:
            accelerator.print(f"Cảnh báo: Không tìm thấy {df_path}")

if __name__ == "__main__":
    main()
