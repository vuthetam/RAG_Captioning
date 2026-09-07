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
            "feature_index": idx,
            "pixel_values": pixel_values
        }

def process_and_save(df_path, output_h5_path, encoder, transform, accelerator, batch_size=256):
    accelerator.print(f"\nĐang xử lý {df_path.name}...")
        
    df = pd.read_parquet(df_path)
    df_unique = df.drop_duplicates(subset=["imgid"]).reset_index(drop=True)
    dataset = ImageFeatureDataset(df_unique, IMAGES_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    
    dataloader = accelerator.prepare(dataloader)

    feature_store = None
    if accelerator.is_main_process:
        output_h5_path.parent.mkdir(parents=True, exist_ok=True)
        h5f = h5py.File(output_h5_path, 'w')
        h5f.create_dataset("imgids", data=df_unique["imgid"].to_numpy(dtype="int64"))
        h5f.attrs["feature_layout"] = "features[i] belongs to imgids[i]"
        h5f.attrs["chunk_rows"] = 1

    pbar = tqdm(dataloader, disable=not accelerator.is_local_main_process, desc=f"Extracting {df_path.name}", leave=False)

    for batch in pbar:
        feature_indices = batch["feature_index"]
        pixel_values = batch["pixel_values"]
        with torch.no_grad():
            features = encoder(pixel_values).to(torch.float16)

        # Retain the source row index because gathered batches are not guaranteed to be ordered.
        gathered_indices, gathered_features = accelerator.gather_for_metrics(
            (feature_indices, features)
        )

        if accelerator.is_main_process:
            if feature_store is None:
                feature_shape = tuple(gathered_features.shape[1:])
                feature_store = h5f.create_dataset(
                    "features",
                    shape=(len(df_unique), *feature_shape),
                    dtype="float16",
                    # One image per chunk minimizes read amplification for random access.
                    chunks=(1, *feature_shape),
                )

            sorted_indices, sort_order = torch.sort(gathered_indices)
            feature_store[sorted_indices.cpu().numpy()] = (
                gathered_features[sort_order].cpu().numpy()
            )

    if accelerator.is_main_process:
        h5f.close()
    accelerator.print(f"Đã lưu đặc trưng tại {output_h5_path}")

def main():
    accelerator = Accelerator(mixed_precision="fp16")
    
    accelerator.print(f"Khởi động môi trường Multi-GPU ({accelerator.num_processes} processes)")
    accelerator.print("Đang tải CLIPViTB16Encoder...")
        
    encoder = CLIPViTB16Encoder()
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
            process_and_save(
                df_path,
                h5_path,
                encoder,
                transform,
                accelerator,
                batch_size=256,
            )
        else:
            accelerator.print(f"Cảnh báo: Không tìm thấy {df_path}")

if __name__ == "__main__":
    main()
