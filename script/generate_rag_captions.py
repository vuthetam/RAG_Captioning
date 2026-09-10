"""Generate test captions using the RAG captioner and pre-extracted visual features."""

import json
import sys
import os
from pathlib import Path

import pandas as pd
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force RUN_MODE to rag for checkpoints and predictions path
os.environ["RUN_MODE"] = "rag"

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
    TEST_RAG_CONTEXTS_PATH,
    CTX_NLAYERS,
    MAX_CTX_LENGTH,
    TOP_K_CAPTIONS,
)
from src.dataset import RAGFeatureDataset
from src.inference import generate_captions
from src.models.rag import RAGCaptioner
from src.vocabulary import Vocabulary


def main() -> None:
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)
    test_df = pd.read_parquet(TEST_DF_PATH)
    vocab = Vocabulary.load(VOCAB_PATH)
    
    test_dataset = RAGFeatureDataset(
        test_df, TEST_VISUAL_FEATURES_PATH, TEST_RAG_CONTEXTS_PATH,
        vocab, max_ctx_length=MAX_CTX_LENGTH, top_k=TOP_K_CAPTIONS
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = RAGCaptioner(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        use_precomputed_features=True,
        visual_feature_dim=test_dataset.feature_shape[-1],
        ctx_nlayers=CTX_NLAYERS,
        max_ctx_length=MAX_CTX_LENGTH,
    )
    
    if not BEST_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {BEST_CHECKPOINT_PATH}")
    load_checkpoint(BEST_CHECKPOINT_PATH, model, device=accelerator.device)

    model, test_loader = accelerator.prepare(model, test_loader)
    caption_dict = generate_captions(
        model, test_loader, vocab, BEAM_SIZE, MAX_LENGTH, accelerator, show_progress=True
    )

    if accelerator.is_main_process:
        predictions = [
            {"imgid": int(imgid), "caption": " ".join(caption_dict[imgid])}
            for imgid in test_df["imgid"]
        ]
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", encoding="utf-8") as output_file:
            json.dump(predictions, output_file, ensure_ascii=False, indent=2)
        accelerator.print(f"Saved {len(predictions):,} captions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

