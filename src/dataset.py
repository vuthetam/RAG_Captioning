from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import IMAGES_DIR
from src.vocabulary import Vocabulary


def create_clip_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


class MSCOCODataset(Dataset):
    def __init__(
        self,
        df,
        vocab: Vocabulary,
        images_dir: str | Path | None = None,
        transform=None,
        max_length: int | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.images_dir = Path(images_dir) if images_dir is not None else Path(IMAGES_DIR)
        self.transform = transform or create_clip_transform()
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def _get_image_path(self, row) -> Path:
        return self.images_dir / row["filepath"] / row["filename"]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self._get_image_path(row)

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        input_ids, attention_mask = self.vocab.encode_from_tokens(row["tokens"], self.max_length)

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return image, input_ids, attention_mask

class ImageOnlyDataset(Dataset):
    def __init__(
        self,
        df,
        images_dir: str | Path | None = None,
        transform=None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir) if images_dir is not None else Path(IMAGES_DIR)
        self.transform = transform or create_clip_transform()

    def __len__(self) -> int:
        return len(self.df)

    def _get_image_path(self, row) -> Path:
        return self.images_dir / row["filepath"] / row["filename"]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path = self._get_image_path(row)

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        return image, row["imgid"]


class _H5FeatureStore:
    """Lazily reads CLIP features from the project's row-aligned HDF5 format."""

    def __init__(self, features_path: str | Path) -> None:
        self.features_path = Path(features_path)
        self._h5_file: h5py.File | None = None
        with h5py.File(self.features_path, "r") as h5_file:
            stored_imgids = np.asarray(h5_file["imgids"], dtype=np.int64)
            self.feature_shape = tuple(h5_file["features"].shape[1:])

        self._imgid_to_index = {
            int(imgid): index for index, imgid in enumerate(stored_imgids.tolist())
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def _feature_for_imgid(self, imgid: int) -> torch.Tensor:
        imgid = int(imgid)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.features_path, "r")
        feature_index = self._imgid_to_index[imgid]
        feature = np.asarray(self._h5_file["features"][feature_index])
        return torch.from_numpy(feature)

class PrecomputedFeatureDataset(_H5FeatureStore, Dataset):
    """Caption dataset that uses pre-extracted CLIP visual tokens, not image files."""

    def __init__(
        self,
        df,
        vocab: Vocabulary,
        features_path: str | Path,
        max_length: int | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_length = max_length
        _H5FeatureStore.__init__(self, features_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        input_ids, attention_mask = self.vocab.encode_from_tokens(
            row["tokens"], self.max_length
        )
        image_feature = self._feature_for_imgid(row["imgid"])
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        return (
            image_feature,
            input_ids,
            attention_mask,
        )


class PrecomputedFeatureOnlyDataset(_H5FeatureStore, Dataset):
    """One visual-token tensor per image for caption generation."""

    def __init__(self, df, features_path: str | Path) -> None:
        self.df = df.reset_index(drop=True)
        _H5FeatureStore.__init__(self, features_path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_feature = self._feature_for_imgid(row["imgid"])
        image_id = int(row["imgid"])

        return image_feature, image_id
