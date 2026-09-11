"""Train the caption decoder from pre-extracted CLIP visual-token HDF5 files."""

import sys
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import load_checkpoint, save_checkpoint
from src.config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    DMODEL,
    DROPOUT,
    LAST_CHECKPOINT_PATH,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MAX_LENGTH,
    NHEADS,
    NLAYERS,
    NUM_EPOCHS,
    NUM_WORKERS,
    TRAIN_DF_PATH,
    IMAGES_DIR,
    VAL_DF_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
)
from src.dataset import ImageCaptionDataset
from src.engine import evaluate_one_epoch, train_one_epoch
from src.models.baseline import BaselineCaptioner
from src.utils import trainable_parameters
from src.vocabulary import Vocabulary


def main() -> None:
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)

    train_dataset = ImageCaptionDataset(
        train_df, vocab, images_dir=IMAGES_DIR, max_length=MAX_LENGTH
    )
    val_dataset = ImageCaptionDataset(
        val_df, vocab, images_dir=IMAGES_DIR, max_length=MAX_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
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
        use_precomputed_features=False,
    )
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    start_epoch = 0
    best_val_loss = float("inf")
    if LAST_CHECKPOINT_PATH.exists():
        start_epoch, _, best_val_loss = load_checkpoint(
            LAST_CHECKPOINT_PATH, model, optimizer, device=accelerator.device
        )
        accelerator.print(f"Resuming after epoch {start_epoch}.")

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    accelerator.print(
        f"train rows={len(train_dataset):,}; val rows={len(val_dataset):,}; "
        f"trainable params="
        f"{sum(parameter.numel() for parameter in trainable_parameters(model)):,}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, vocab.pad_idx(), accelerator, MAX_GRAD_NORM, True
        )
        val_loss = evaluate_one_epoch(model, val_loader, vocab.pad_idx(), accelerator, True)
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] "
            f"train loss = {train_loss:.4f} val loss = {val_loss:.4f}"
        )

        if accelerator.is_main_process:
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    BEST_CHECKPOINT_PATH, model, optimizer, epoch + 1, train_loss, best_val_loss, accelerator
                )


if __name__ == "__main__":
    main()
