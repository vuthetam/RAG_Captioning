"""Generate test captions from pre-extracted CLIP visual-token HDF5 files."""

import json
import sys
from pathlib import Path

import pandas as pd
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint
from src.config import (
    BATCH_SIZE,
    BEAM_SIZE,
    BEST_CHECKPOINT_PATH,
    DMODEL,
    DROPOUT,
    MAX_LENGTH,
    NHEADS,
    NLAYERS,
    NUM_WORKERS,
    PREDICTIONS_PATH,
    TEST_DF_PATH,
    TEST_VISUAL_FEATURES_PATH,
    VOCAB_PATH,
)
from src.dataset import PrecomputedFeatureOnlyDataset
from src.inference import generate_captions
from src.models.baseline import BaselineCaptioner
from src.vocabulary import Vocabulary


def main() -> None:
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)
    test_df = pd.read_parquet(TEST_DF_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)
    test_dataset = PrecomputedFeatureOnlyDataset(test_df, TEST_VISUAL_FEATURES_PATH)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = BaselineCaptioner(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        use_precomputed_features=True,
        visual_feature_dim=test_dataset.feature_shape[-1],
    )
    if not BEST_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {BEST_CHECKPOINT_PATH}")
    load_checkpoint(BEST_CHECKPOINT_PATH, model, device=accelerator.device)

    model, test_loader = accelerator.prepare(model, test_loader)
    caption_tokens_list = generate_captions(
        model, test_loader, vocab, BEAM_SIZE, MAX_LENGTH, accelerator, show_progress=True
    )

    if accelerator.is_main_process:
        predictions = [
            {"imgid": int(imgid), "caption": " ".join(caption_tokens)}
            for imgid, caption_tokens in zip(test_df["imgid"], caption_tokens_list)
        ]
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, ensure_ascii=False, indent=2)
        accelerator.print(f"Saved {len(predictions):,} captions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
