import os
from pathlib import Path
from dotenv import load_dotenv

# ==========================================
# 1. ROOT & ENVIRONMENT
# ==========================================
ROOT_PATH = Path(__file__).resolve().parent.parent
ROOT_DIR = ROOT_PATH
ENV_PATH = ROOT_PATH / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)

RUN_MODE = os.getenv("RUN_MODE", "baseline")  # 'baseline' hoặc 'rag'

# ==========================================
# 2. DATASET PATHS (Inputs)
# ==========================================
DATASET_COCO_PATH = Path(
    os.getenv("DATASET_COCO_PATH", str(ROOT_PATH / "dataset" / "mscoco" / "dataset_coco.json"))
)
IMAGES_PATH = Path(
    os.getenv("IMAGES_PATH", str(ROOT_PATH / "dataset" / "mscoco" / "images"))
)

# ==========================================
# 3. ARTIFACTS PATHS (Intermediates & Outputs)
# ==========================================
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", str(ROOT_PATH / "artifacts")))
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / f"{RUN_MODE}_predictions.json"

# Splits
SPLITS_DIR = ARTIFACTS_DIR / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_DF_PATH = SPLITS_DIR / "train_df.parquet"
VAL_DF_PATH = SPLITS_DIR / "val_df.parquet"
TEST_DF_PATH = SPLITS_DIR / "test_df.parquet"

# Knowledge Base (FAISS)
KB_MODEL_ID = "openai/clip-vit-large-patch14-336"
KB_DIR = ARTIFACTS_DIR / "kb"
KB_DIR.mkdir(parents=True, exist_ok=True)
KB_FAISS_INDEX_PATH = KB_DIR / "kb_text_index.faiss"
KB_METADATA_PATH = KB_DIR / "kb_metadata.parquet"

# RAG
RAG_DIR = ARTIFACTS_DIR / "rag"
RAG_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_VISUAL_FEATURES_PATH = RAG_DIR / "train_visual_features.h5"
VAL_VISUAL_FEATURES_PATH = RAG_DIR / "val_visual_features.h5"
TEST_VISUAL_FEATURES_PATH = RAG_DIR / "test_visual_features.h5"
TRAIN_RAG_CONTEXTS_PATH = RAG_DIR / "train_rag_contexts.parquet"
VAL_RAG_CONTEXTS_PATH = RAG_DIR / "val_rag_contexts.parquet"
TEST_RAG_CONTEXTS_PATH = RAG_DIR / "test_rag_contexts.parquet"


# ==========================================
# 4. CHECKPOINT PATHS (Weights)
# ==========================================
# Nhờ thủ thuật copy sang Working Dir, ta chỉ cần 1 thư mục duy nhất để vừa Load vừa Save
CHECKPOINTS_DIR = Path(os.getenv("CHECKPOINTS_DIR", str(ROOT_PATH / "checkpoints")))

# Tự động chia nhánh Checkpoint theo RUN_MODE ('baseline' hoặc 'rag')
RUN_CHECKPOINT_DIR = CHECKPOINTS_DIR / RUN_MODE
RUN_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LAST_CHECKPOINT_PATH = RUN_CHECKPOINT_DIR / "last_checkpoint.pth"
BEST_CHECKPOINT_PATH = RUN_CHECKPOINT_DIR / "best_checkpoint.pth"

# ==========================================
# 5. HYPERPARAMETERS
# ==========================================
# Model Architecture
DMODEL = int(os.getenv("DMODEL", "512"))
NHEADS = int(os.getenv("NHEADS", "8"))
NLAYERS = int(os.getenv("NLAYERS", "4"))
DROPOUT = float(os.getenv("DROPOUT", "0.1"))

# Training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "1e-4"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "4"))

# Text & Retrieval Settings
FREQ_THRESHOLD = int(os.getenv("FREQ_THRESHOLD", "5"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "40"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "5"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "4"))
