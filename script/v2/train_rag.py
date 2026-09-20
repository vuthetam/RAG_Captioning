"""Train RagModelV2 từ pre-extracted CLIP visual-token HDF5 files."""

import sys
from pathlib import Path

import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.checkpoint import load_checkpoint, save_checkpoint
from src.shared.config import (
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
    TRAIN_VISUAL_FEATURES_PATH,
    VAL_DF_PATH,
    VAL_VISUAL_FEATURES_PATH,
    VOCAB_PATH,
    WEIGHT_DECAY,
    TRAIN_RAG_CONTEXTS_PATH,
    VAL_RAG_CONTEXTS_PATH,
    TOP_K_CAPTIONS,
    MAX_CTX_LEN,
    MAX_OBJ_LEN,
    MAX_REL_LEN,
)
from src.shared.utils import trainable_parameters
from src.shared.vocabulary import Vocabulary
from src.v2.dataset import RagV2CaptionDataset
from src.v2.engine import evaluate_one_epoch, train_one_epoch
from src.v2.models.rag import RagModelV2


def main() -> None:
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    # ------------------------------------------------------------------
    # 1. Load Dữ liệu
    # ------------------------------------------------------------------
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df   = pd.read_parquet(VAL_DF_PATH)
    vocab    = Vocabulary.load(VOCAB_PATH)

    train_dataset = RagV2CaptionDataset(
        df=train_df,
        vocab=vocab,
        features_path=TRAIN_VISUAL_FEATURES_PATH,
        rag_contexts_path=TRAIN_RAG_CONTEXTS_PATH,
        max_length=MAX_LENGTH,
        top_k=TOP_K_CAPTIONS,
        max_ctx_len=MAX_CTX_LEN,
        max_obj_len=MAX_OBJ_LEN,
        max_rel_len=MAX_REL_LEN,
    )
    val_dataset = RagV2CaptionDataset(
        df=val_df,
        vocab=vocab,
        features_path=VAL_VISUAL_FEATURES_PATH,
        rag_contexts_path=VAL_RAG_CONTEXTS_PATH,
        max_length=MAX_LENGTH,
        top_k=TOP_K_CAPTIONS,
        max_ctx_len=MAX_CTX_LEN,
        max_obj_len=MAX_OBJ_LEN,
        max_rel_len=MAX_REL_LEN,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 2. Khởi tạo Model
    # ------------------------------------------------------------------
    model = RagModelV2(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        visual_feature_dim=train_dataset.feature_shape[-1],
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters(model),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # ------------------------------------------------------------------
    # 3. Resume Checkpoint (nếu có)
    # ------------------------------------------------------------------
    start_epoch   = 0
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
        f"train rows={len(train_dataset):,}  val rows={len(val_dataset):,}\n"
        f"feature shape={train_dataset.feature_shape}  "
        f"trainable params={sum(p.numel() for p in trainable_parameters(model)):,}"
    )

    # ------------------------------------------------------------------
    # 4. Vòng lặp Training
    # ------------------------------------------------------------------
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            vocab.pad_idx(), accelerator, MAX_GRAD_NORM,
        )
        val_loss = evaluate_one_epoch(
            model, val_loader,
            vocab.pad_idx(), accelerator,
        )
        accelerator.print(
            f"[Epoch {epoch + 1:02d}/{NUM_EPOCHS}] "
            f"train loss = {train_loss:.4f}  val loss = {val_loss:.4f}"
        )

        if accelerator.is_main_process:
            save_checkpoint(
                LAST_CHECKPOINT_PATH, model, optimizer,
                epoch + 1, train_loss, best_val_loss, accelerator,
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    BEST_CHECKPOINT_PATH, model, optimizer,
                    epoch + 1, train_loss, best_val_loss, accelerator,
                )


if __name__ == "__main__":
    main()

