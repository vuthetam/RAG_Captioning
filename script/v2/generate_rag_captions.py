"""Generate captions cho tập Test bằng RagModelV2 + pre-extracted CLIP features."""

import json
import sys
from pathlib import Path

import pandas as pd
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.checkpoint import load_checkpoint
from src.shared.config import (
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
    TEST_RAG_CONTEXTS_PATH,
    VOCAB_PATH,
    TOP_K_CAPTIONS,
    MAX_CTX_LEN,
    MAX_OBJ_LEN,
    MAX_REL_LEN,
)
from src.shared.vocabulary import Vocabulary
from src.v2.dataset import RagV2InferenceDataset
from src.v2.inference import generate_captions
from src.v2.models.rag import RagModelV2


def main() -> None:
    accelerator = Accelerator(mixed_precision="fp16")
    set_seed(42)

    test_df = pd.read_parquet(TEST_DF_PATH)
    vocab   = Vocabulary.load(VOCAB_PATH)

    test_dataset = RagV2InferenceDataset(
        df=test_df,
        vocab=vocab,
        features_path=TEST_VISUAL_FEATURES_PATH,
        rag_contexts_path=TEST_RAG_CONTEXTS_PATH,
        top_k=TOP_K_CAPTIONS,
        max_ctx_len=MAX_CTX_LEN,
        max_obj_len=MAX_OBJ_LEN,
        max_rel_len=MAX_REL_LEN,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = RagModelV2(
        vocab_size=len(vocab),
        d_model=DMODEL,
        nheads=NHEADS,
        nlayers=NLAYERS,
        dropout=DROPOUT,
        max_length=MAX_LENGTH,
        pad_idx=vocab.pad_idx(),
        visual_feature_dim=test_dataset.feature_shape[-1],
    )

    if not BEST_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {BEST_CHECKPOINT_PATH}")
    load_checkpoint(BEST_CHECKPOINT_PATH, model, device=accelerator.device)

    model, test_loader = accelerator.prepare(model, test_loader)

    caption_dict = generate_captions(
        model, test_loader, vocab,
        beam_size=BEAM_SIZE,
        max_length=MAX_LENGTH,
        accelerator=accelerator,
        show_progress=True,
    )

    if accelerator.is_main_process:
        predictions = [
            {"imgid": int(imgid), "caption": " ".join(caption_dict[imgid])}
            for imgid in test_df["imgid"]
            if imgid in caption_dict
        ]
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PREDICTIONS_PATH.open("w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        accelerator.print(f"Saved {len(predictions):,} captions -> {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()

