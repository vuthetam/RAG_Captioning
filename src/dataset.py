from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from src.config import IMAGES_PATH
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
        self.images_dir = Path(images_dir) if images_dir is not None else Path(IMAGES_PATH)
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
