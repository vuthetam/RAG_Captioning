import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_PATH = Path(__file__).resolve().parent.parent
ROOT_DIR = ROOT_PATH
ENV_PATH = ROOT_PATH / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

# Environment-driven settings
DATASET_COCO_PATH = Path(
    os.getenv("DATASET_COCO_PATH", str(ROOT_PATH / "dataset" / "mscoco" / "dataset_coco.json"))
)
IMAGES_PATH = Path(
    os.getenv("IMAGES_PATH", str(ROOT_PATH / "dataset" / "mscoco" / "images"))
)
FREQ_THRESHOLD = int(os.getenv("FREQ_THRESHOLD", "5"))
DMODEL = int(os.getenv("DMODEL", "512"))
NHEADS = int(os.getenv("NHEADS", "8"))
NLAYERS = int(os.getenv("NLAYERS", "4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "40"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))

# Thêm cờ để phân biệt đang chạy Baseline hay RAG
RUN_MODE = os.getenv("RUN_MODE", "baseline")  # 'baseline' hoặc 'rag'

# Tự động chia nhánh Checkpoint theo RUN_MODE
_save_last = os.getenv("SAVE_LAST_CHECKPOINT_DIR")
_save_best = os.getenv("SAVE_BEST_CHECKPOINT_DIR")
_load_last = os.getenv("LOAD_LAST_CHECKPOINT_DIR")
_load_best = os.getenv("LOAD_BEST_CHECKPOINT_DIR")

SAVE_LAST_CHECKPOINT_DIR = str(Path(_save_last) / RUN_MODE) if _save_last else None
SAVE_BEST_CHECKPOINT_DIR = str(Path(_save_best) / RUN_MODE) if _save_best else None
LOAD_LAST_CHECKPOINT_DIR = str(Path(_load_last) / RUN_MODE) if _load_last else None
LOAD_BEST_CHECKPOINT_DIR = str(Path(_load_best) / RUN_MODE) if _load_best else None

if SAVE_LAST_CHECKPOINT_DIR:
    os.makedirs(SAVE_LAST_CHECKPOINT_DIR, exist_ok=True)
if SAVE_BEST_CHECKPOINT_DIR:
    os.makedirs(SAVE_BEST_CHECKPOINT_DIR, exist_ok=True)

# Static project paths
ARTIFACTS_DIR = ROOT_PATH / "artifacts"
VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
TRAIN_DF_PATH = ARTIFACTS_DIR / "train_df.parquet"
VAL_DF_PATH = ARTIFACTS_DIR / "val_df.parquet"
TEST_DF_PATH = ARTIFACTS_DIR / "test_df.parquet"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SAVE_LAST_CHECKPOINT_PATH = (
    Path(SAVE_LAST_CHECKPOINT_DIR) / "last_checkpoint.pth" if SAVE_LAST_CHECKPOINT_DIR else None
)
SAVE_BEST_CHECKPOINT_PATH = (
    Path(SAVE_BEST_CHECKPOINT_DIR) / "best_checkpoint.pth" if SAVE_BEST_CHECKPOINT_DIR else None
)
LOAD_LAST_CHECKPOINT_PATH = (
    Path(LOAD_LAST_CHECKPOINT_DIR) / "last_checkpoint.pth" if LOAD_LAST_CHECKPOINT_DIR else None
)
LOAD_BEST_CHECKPOINT_PATH = (
    Path(LOAD_BEST_CHECKPOINT_DIR) / "best_checkpoint.pth" if LOAD_BEST_CHECKPOINT_DIR else None
)


# Cấu hình cho Knowledge Base / Retrieval
KB_MODEL_ID = "openai/clip-vit-large-patch14-336"


KB_DIR = ARTIFACTS_DIR / "kb"
KB_DIR.mkdir(parents=True, exist_ok=True)

KB_FAISS_INDEX_PATH = KB_DIR / "kb_text_index.faiss"
KB_METADATA_PATH = KB_DIR / "kb_metadata.parquet"

