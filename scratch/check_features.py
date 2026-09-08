import sys
from pathlib import Path
import h5py
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd

PROJECT_ROOT = Path("/home/tam/Link to workspace/ML/rag_captioning")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH,
    TRAIN_VISUAL_FEATURES_PATH, VAL_VISUAL_FEATURES_PATH, TEST_VISUAL_FEATURES_PATH,
    IMAGES_DIR,
)
from src.encoder import CLIPViTB16Encoder
from src.dataset import create_clip_transform

imgids_to_check = [39361, 39368, 39403, 39415]

# Find which split these images belong to
df_paths = [TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH]
h5_paths = [TRAIN_VISUAL_FEATURES_PATH, VAL_VISUAL_FEATURES_PATH, TEST_VISUAL_FEATURES_PATH]

encoder = CLIPViTB16Encoder()
encoder.eval()
transform = create_clip_transform()

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.to(device)

for df_path, h5_path in zip(df_paths, h5_paths):
    if not df_path.exists() or not h5_path.exists():
        continue
    df = pd.read_parquet(df_path)
    # Check if any imgids_to_check is in this df
    found_ids = [i for i in imgids_to_check if i in df['imgid'].values]
    if not found_ids:
        continue
        
    print(f"Found {found_ids} in {df_path.name}")
    
    with h5py.File(h5_path, 'r') as h5f:
        h5_imgids = h5f['imgids'][:]
        h5_features = h5f['features']
        
        for imgid in found_ids:
            row = df[df['imgid'] == imgid].iloc[0]
            img_path = IMAGES_DIR / row['filepath'] / row['filename']
            print(f"\nProcessing imgid: {imgid}")
            print(f"Image path: {img_path}")
            
            with Image.open(img_path) as img:
                pixel_values = transform(img.convert("RGB")).unsqueeze(0).to(device)
                
            with torch.no_grad():
                # Tương đương với quá trình extract: encode -> to(float16)
                feature_direct = encoder(pixel_values).to(torch.float16).cpu()
                
            # Find in h5
            idx = (h5_imgids == imgid).nonzero()[0][0]
            feature_h5 = torch.tensor(h5_features[idx])
            
            # Compare
            diff = torch.abs(feature_direct - feature_h5).max().item()
            print(f"Max absolute difference: {diff}")
            
            cos_sim = F.cosine_similarity(feature_direct.flatten().float(), feature_h5.flatten().float(), dim=0).item()
            print(f"Cosine similarity: {cos_sim}")

