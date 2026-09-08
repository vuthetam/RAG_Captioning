# Chạy bằng nn.DataParallel

import sys
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import h5py
from PIL import Image

# Đảm bảo import được src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
    TRAIN_VISUAL_FEATURES_PATH, VAL_VISUAL_FEATURES_PATH, TEST_VISUAL_FEATURES_PATH,
    IMAGES_DIR,
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
            "pixel_values": pixel_values
        }

def process_and_save(df_path, output_h5_path, encoder, transform, device, batch_size=256):
    print(f"\nĐang xử lý {df_path.name}...")
        
    df = pd.read_parquet(df_path)
    df_unique = df.drop_duplicates(subset=["imgid"]).reset_index(drop=True)
    dataset = ImageFeatureDataset(df_unique, IMAGES_DIR, transform=transform)
    # shuffle=False là bắt buộc để giữ thứ tự tuần tự với index
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    
    output_h5_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_h5_path, 'w') as h5f:
        h5f.create_dataset("imgids", data=df_unique["imgid"].to_numpy(dtype="int64"))
        h5f.attrs["feature_layout"] = "features[i] belongs to imgids[i]"
        h5f.attrs["chunk_rows"] = 1

        feature_store = None
        current_idx = 0

        pbar = tqdm(dataloader, desc=f"Extracting {df_path.name}", leave=False)

        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        features = encoder(pixel_values)
                else:
                    features = encoder(pixel_values)
                    
                # Ép kiểu chuẩn về float16 để ghi H5
                features = features.to(torch.float16).cpu().numpy()

            if feature_store is None:
                feature_shape = tuple(features.shape[1:])
                feature_store = h5f.create_dataset(
                    "features",
                    shape=(len(df_unique), *feature_shape),
                    dtype="float16",
                    chunks=(1, *feature_shape),
                )

            batch_len = features.shape[0]
            feature_store[current_idx:current_idx + batch_len] = features
            current_idx += batch_len

    print(f"Đã lưu đặc trưng tại {output_h5_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    
    print(f"Khởi động môi trường (Sử dụng {num_gpus} GPUs trên {device})")
    print("Đang tải CLIPViTB16Encoder...")
        
    encoder = CLIPViTB16Encoder()
    encoder.eval()
    encoder.to(device)
    
    if num_gpus > 1:
        encoder = nn.DataParallel(encoder)
    
    transform = create_clip_transform()
    
    datasets = [
        (TRAIN_DF_PATH, TRAIN_VISUAL_FEATURES_PATH),
        (VAL_DF_PATH, VAL_VISUAL_FEATURES_PATH),
        (TEST_DF_PATH, TEST_VISUAL_FEATURES_PATH)
    ]
    
    for df_path, h5_path in datasets:
        if df_path.exists():
            process_and_save(
                df_path,
                h5_path,
                encoder,
                transform,
                device,
                batch_size=256,
            )
        else:
            print(f"Cảnh báo: Không tìm thấy {df_path}")

if __name__ == "__main__":
    main()
