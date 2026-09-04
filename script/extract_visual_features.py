# Chạy bằng accelerate

import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import h5py
from PIL import Image
from accelerate import Accelerator

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
    TRAIN_VISUAL_FEATURES_PATH, VAL_VISUAL_FEATURES_PATH, TEST_VISUAL_FEATURES_PATH,
    IMAGES_PATH, DMODEL
)
from src.encoder import CLIPViTB16Encoder
from src.dataset import create_clip_transform


class ImageFeatureDataset(Dataset):
    def __init__(self, df, images_path, transform):
        self.imgids = df['imgid'].tolist()
        self.filepaths = df['filepath'].tolist()
        self.filenames = df['filename'].tolist()
        self.images_path = images_path
        self.transform = transform

    def __len__(self):
        return len(self.imgids)

    def __getitem__(self, idx):
        path = self.images_path / self.filepaths[idx] / self.filenames[idx]
        with Image.open(path) as img:
            pixel_values = self.transform(img.convert("RGB"))
        
        return {
            "imgid": int(self.imgids[idx]), 
            "pixel_values": pixel_values
        }

def process_and_save(df_path, output_h5_path, encoder, transform, accelerator, batch_size=256):
    accelerator.print(f"\nĐang xử lý {df_path.name}...")
        
    df = pd.read_parquet(df_path)
    df_unique = df.drop_duplicates(subset=["imgid"]).reset_index(drop=True)
    dataset = ImageFeatureDataset(df_unique, IMAGES_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    dataloader = accelerator.prepare(dataloader)
    
    if accelerator.is_main_process:
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        h5f = h5py.File(output_h5_path, 'w')

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Extracting {df_path.name}", leave=False)

    for batch in pbar:
        imgids = batch["imgid"]
        pixel_values = batch["pixel_values"]
        with torch.no_grad():
            features = encoder(pixel_values).to(torch.float16)
        
        # Gom kết quả từ cả 2 GPU về lại Main Process an toàn (loại bỏ padding dummy data)
        gathered_ids, gathered_features = accelerator.gather_for_metrics((imgids, features))
        
        if accelerator.is_main_process:
            gathered_ids = gathered_ids.cpu().numpy()
            gathered_features = gathered_features.cpu().numpy()
            
            for imgid, feat in zip(gathered_ids, gathered_features):
                # Ép kiểu int về string vì HDF5 chỉ nhận key là chuỗi
                imgid = str(imgid)
                if imgid not in h5f:
                    h5f.create_dataset(imgid, data=feat, dtype='float16')

    if accelerator.is_main_process:
        h5f.close()
    accelerator.print(f"Đã lưu đặc trưng tại {output_h5_path}")

def main():
    accelerator = Accelerator(mixed_precision="fp16")
    
    accelerator.print(f"Khởi động môi trường Multi-GPU ({accelerator.num_processes} processes)")
    accelerator.print("Đang tải CLIPViTB16Encoder...")
        
    encoder = CLIPViTB16Encoder(d_model=DMODEL)
    encoder.eval()
    encoder = accelerator.prepare(encoder)
    
    transform = create_clip_transform()
    
    datasets = [
        (TRAIN_DF_PATH, TRAIN_VISUAL_FEATURES_PATH),
        (VAL_DF_PATH, VAL_VISUAL_FEATURES_PATH),
        (TEST_DF_PATH, TEST_VISUAL_FEATURES_PATH)
    ]
    
    for df_path, h5_path in datasets:
        if df_path.exists():
            process_and_save(df_path, h5_path, encoder, transform, accelerator, batch_size=256)
        else:
            accelerator.print(f"Cảnh báo: Không tìm thấy {df_path}")

if __name__ == "__main__":
    main()
