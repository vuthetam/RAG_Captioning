"""Copy all test set images to a separate folder for easy viewing or sharing."""

import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import IMAGES_DIR, TEST_DF_PATH


def main():
    df = pd.read_parquet(TEST_DF_PATH)
    
    output_dir = Path(f"/home/tam/Downloads/images/{df.iloc[0]["filepath"]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Bắt đầu copy {len(df):,} ảnh sang thư mục {output_dir}...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        src_path = IMAGES_DIR / row["filepath"] / row["filename"]
        dst_path = output_dir / row["filename"]
        
        if src_path.exists():
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Lỗi: Không tìm thấy ảnh {src_path}")
            
    print("Xong!")


if __name__ == "__main__":
    main()

