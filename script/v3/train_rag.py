import os
from pathlib import Path
import torch
import pandas as pd
from accelerate import Accelerator
from torch.utils.data import DataLoader
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import (
    VOCAB_PATH,
    TRAIN_DF_PATH,
    VAL_DF_PATH,
    TRAIN_VISUAL_FEATURES_PATH,
    VAL_VISUAL_FEATURES_PATH,
    TRAIN_RAG_CONTEXTS_PATH,
    VAL_RAG_CONTEXTS_PATH,
    RUN_CHECKPOINT_DIR,
    LAST_CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    MAX_LENGTH,
    MAX_RAG_LEN,
    TOP_K_CAPTIONS,
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    DMODEL,
    NHEADS,
    NLAYERS,
    CTX_NLAYERS,
    DROPOUT
)
from src.shared.vocabulary import Vocabulary
from src.shared.utils import trainable_parameters
from src.shared.checkpoint import save_checkpoint
from src.v3.dataset import RAGFeatureCaptionDatasetV3
from src.v3.models.rag import RagModelV3
from src.v3.engine import train_one_epoch_v3, evaluate_one_epoch_v3


def main():
    # 1. Khởi tạo bộ tăng tốc phần cứng (Hỗ trợ Multi-GPU & Mixed Precision)
    accelerator = Accelerator()
    
    accelerator.print("========== RAG CAPTIONING V3 ==========")
    accelerator.print("[1/5] Loading Vocabulary...")
    vocab = Vocabulary.load(VOCAB_PATH)
    pad_idx = vocab.pad_idx()
    
    accelerator.print("[2/5] Loading DataFrames...")
    train_df = pd.read_parquet(TRAIN_DF_PATH)
    val_df = pd.read_parquet(VAL_DF_PATH)
    
    accelerator.print("[3/5] Building Datasets & DataLoaders...")
    train_dataset = RAGFeatureCaptionDatasetV3(
        df=train_df,
        vocab=vocab,
        features_path=TRAIN_VISUAL_FEATURES_PATH,
        rag_contexts_path=TRAIN_RAG_CONTEXTS_PATH,
        max_length=MAX_LENGTH,
        max_rag_len=MAX_RAG_LEN,
        top_k=TOP_K_CAPTIONS
    )
    
    val_dataset = RAGFeatureCaptionDatasetV3(
        df=val_df,
        vocab=vocab,
        features_path=VAL_VISUAL_FEATURES_PATH,
        rag_contexts_path=VAL_RAG_CONTEXTS_PATH,
        max_length=MAX_LENGTH,
        max_rag_len=MAX_RAG_LEN,
        top_k=TOP_K_CAPTIONS
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    
    accelerator.print("[4/5] Initializing FiD RagModelV3 & Optimizer...")
    model = RagModelV3(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nhead=NHEADS,
        num_encoder_layers=CTX_NLAYERS,
        num_decoder_layers=NLAYERS,
        dim_feedforward=DMODEL * 4,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        max_rag_len=MAX_RAG_LEN,
        top_k=TOP_K_CAPTIONS
    )
    
    optimizer = torch.optim.AdamW(
        trainable_parameters(model), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    accelerator.print("[5/5] Preparing environment with Accelerator...")
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    best_val_loss = float('inf')
    os.makedirs(RUN_CHECKPOINT_DIR, exist_ok=True)
    
    accelerator.print("\n🚀 STARTING TRAINING LOOP 🚀")
    for epoch in range(1, NUM_EPOCHS + 1):
        accelerator.print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")
        
        train_loss = train_one_epoch_v3(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            pad_idx=pad_idx,
            accelerator=accelerator,
            max_grad_norm=MAX_GRAD_NORM,
        )
        
        val_loss = evaluate_one_epoch_v3(
            model=model,
            dataloader=val_loader,
            pad_idx=pad_idx,
            accelerator=accelerator,
        )
        
        accelerator.print(f"✅ Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Chỉ tiến trình chính (Main Process) mới được quyền lưu file
        if accelerator.is_main_process:
            # Lưu Checkpoint mỗi Epoch (Bao gồm cả Trọng số, Trạng thái Optimizer và Loss)
            save_checkpoint(
                path=LAST_CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                best_val_loss=best_val_loss,
                accelerator=accelerator
            )
            
            # Cập nhật Best Checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    path=BEST_CHECKPOINT_PATH,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    best_val_loss=best_val_loss,
                    accelerator=accelerator
                )
                accelerator.print(f" 🏆 New Best Checkpoint Saved: {BEST_CHECKPOINT_PATH}")

    accelerator.print("\n🎉 TRAINING COMPLETE! 🎉")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
