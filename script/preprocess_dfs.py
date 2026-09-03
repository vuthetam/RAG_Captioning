import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_COCO_PATH, TEST_DF_PATH, TRAIN_DF_PATH, VAL_DF_PATH


def resolve_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Khong tim thay file dataset_coco.json. Hay cap nhat DATASET_COCO_PATH hoac dat file dung vi tri."
    )


def get_dataset_path() -> Path:
    return resolve_path(
        [
            DATASET_COCO_PATH,
            Path("dataset/mscoco/dataset_coco.json"),
            Path("../dataset/mscoco/dataset_coco.json"),
            Path("../../dataset/mscoco/dataset_coco.json"),
        ]
    )


def build_split_dataframes(dataset_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images_df = pd.json_normalize(
        data["images"],
        record_path="sentences",
        meta=["filepath", "filename", "split"],
    )

    train_df = images_df[images_df["split"].isin(["train", "restval"])].reset_index(drop=True)
    val_df = images_df[images_df["split"] == "val"].reset_index(drop=True)
    # test set: one row per unique image, all 5 reference captions collected for scoring
    test_raw = images_df[images_df["split"] == "test"]
    test_df = (
        test_raw.groupby(["imgid", "filepath", "filename", "split"], sort=False)
        .agg(all_tokens=("tokens", list), all_raws=("raw", list))
        .reset_index()
    )
    return train_df, val_df, test_df


def save_dataframes(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[Path, Path, Path]:
    for path in [TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH]:
        path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(TRAIN_DF_PATH, index=False)
    val_df.to_parquet(VAL_DF_PATH, index=False)
    test_df.to_parquet(TEST_DF_PATH, index=False)

    return TRAIN_DF_PATH, VAL_DF_PATH, TEST_DF_PATH


def main() -> None:
    dataset_path = get_dataset_path()
    train_df, val_df, test_df = build_split_dataframes(dataset_path)

    print(f"train_df: {len(train_df):,}")
    print(f"val_df:   {len(val_df):,}")
    print(f"test_df:  {len(test_df):,}")

    train_path, val_path, test_path = save_dataframes(train_df, val_df, test_df)
    print(train_path)
    print(val_path)
    print(test_path)


if __name__ == "__main__":
    main()
