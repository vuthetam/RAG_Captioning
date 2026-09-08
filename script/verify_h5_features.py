import sys
from pathlib import Path
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm

# Cấu hình path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
    TRAIN_VISUAL_FEATURES_PATH, VAL_VISUAL_FEATURES_PATH, TEST_VISUAL_FEATURES_PATH,
    IMAGES_DIR
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
            "pixel_values": pixel_values,
            "imgid": self.imgids[idx]
        }

def verify_h5_batched(df_path, h5_path, batch_size=256, max_images=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    print(f"Đang dùng thiết bị: {device} với {num_gpus} GPUs")

    # 1. Khởi tạo model và transform
    encoder = CLIPViTB16Encoder().eval().to(device)
    if num_gpus > 1:
        encoder = nn.DataParallel(encoder)
    transform = create_clip_transform()

    if not h5_path.exists():
        print(f"Lỗi: Không tìm thấy file {h5_path}")
        return

    # 2. Chuẩn bị DataLoader
    df = pd.read_parquet(df_path)
    df_unique = df.drop_duplicates(subset=["imgid"]).reset_index(drop=True)
    if max_images is not None:
        df_unique = df_unique.head(max_images)

    dataset = ImageFeatureDataset(df_unique, IMAGES_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    with h5py.File(h5_path, 'r') as h5f:
        h5_imgids = h5f['imgids'][:]
        h5_features = h5f['features']
        
        # Tạo Hash Map để tra cứu O(1)
        imgid_to_idx = {int(imgid): idx for idx, imgid in enumerate(h5_imgids)}
        
        print(f"Bắt đầu kiểm tra {h5_path.name} (Chế độ Batch + Multi-GPU)...")
        
        min_cos_sim = 1.0
        max_diff = 0.0
        bad_count = 0

        pbar = tqdm(dataloader, desc=f"Verifying {h5_path.name}")
        for batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            batch_imgids = batch["imgid"].numpy()
            
            # Encode ảnh theo batch
            with torch.no_grad():
                if device.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        features_direct = encoder(pixel_values)
                else:
                    features_direct = encoder(pixel_values)
                features_direct = features_direct.to(torch.float16).cpu()

            batch_len = features_direct.shape[0]
            
            # Lấy vector tương ứng từ H5
            features_h5 = []
            for imgid in batch_imgids:
                if imgid not in imgid_to_idx:
                    print(f"Lỗi: Không tìm thấy imgid {imgid} trong file H5.")
                    continue
                features_h5.append(torch.tensor(h5_features[imgid_to_idx[imgid]]))
                
            if len(features_h5) != batch_len:
                bad_count += (batch_len - len(features_h5))
                continue
                
            features_h5 = torch.stack(features_h5)

            # So khớp
            diff = torch.abs(features_direct - features_h5).view(batch_len, -1).max(dim=1)[0]
            cos_sim = F.cosine_similarity(features_direct.view(batch_len, -1).float(), features_h5.view(batch_len, -1).float(), dim=1)

            min_cos_sim = min(min_cos_sim, cos_sim.min().item())
            max_diff = max(max_diff, diff.max().item())

            # In cảnh báo
            bad_indices = (cos_sim < 0.95).nonzero(as_tuple=True)[0]
            for bad_idx in bad_indices:
                if bad_count < 10:
                    imgid = batch_imgids[bad_idx].item()
                    tqdm.write(f"\n[Mismatched!] imgid: {imgid} | Cosine: {cos_sim[bad_idx].item():.4f}")
                bad_count += 1

    print("\n" + "="*50)
    print(f"KẾT QUẢ KIỂM TRA FILE: {h5_path.name}")
    print(f"Số ảnh đã kiểm tra: {len(df_unique)}")
    print(f"Max Absolute Difference: {max_diff:.6f}")
    print(f"Min Cosine Similarity: {min_cos_sim:.6f}")
    print(f"Tổng số ảnh bị lệch đặc trưng (Cosine < 0.95): {bad_count}/{len(df_unique)}")
    if bad_count == 0:
        print("=> PERFECT! File H5 hoàn toàn chính xác.")
    else:
        print("=> THẤT BẠI: File H5 có chứa đặc trưng bị lỗi/lệch index.")

if __name__ == "__main__":
    datasets_to_check = [
        # (TEST_DF_PATH, TEST_VISUAL_FEATURES_PATH),
        (VAL_DF_PATH, VAL_VISUAL_FEATURES_PATH),
        (TRAIN_DF_PATH, TRAIN_VISUAL_FEATURES_PATH)
    ]
    
    for df_path, h5_path in datasets_to_check:
        print(f"\n{'='*70}")
        print(f"BẮT ĐẦU KIỂM TRA: {df_path.name} <-> {h5_path.name}")
        print(f"{'='*70}")
        
        if not df_path.exists() or not h5_path.exists():
            print(f"-> Bỏ qua vì không tìm thấy file dataframe hoặc file H5 tương ứng.\n")
            continue
            
        verify_h5_batched(df_path, h5_path, batch_size=256)
