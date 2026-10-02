"""
script/v3/build_vocab.py
Load vocab V1/V2, thêm 3 dấu câu cần cho prompt template của V3, rồi lưu ra vocab_v3.json.
Không làm thay đổi file vocab.json gốc.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.config import VOCAB_PATH, ARTIFACTS_DIR
from src.shared.vocabulary import Vocabulary

# Các token bổ sung cần cho prompt template:
# "similar image shows: man hold hand, man look row."
EXTRA_TOKENS = [":", ",", "."]

VOCAB_V3_PATH = ARTIFACTS_DIR / "vocab_v3.json"

def main() -> None:
    # 1. Load vocab gốc
    vocab = Vocabulary.load(VOCAB_PATH)
    print(f"Loaded vocab from : {VOCAB_PATH}")
    print(f"Original size     : {len(vocab):,} tokens")

    # 2. Thêm các dấu câu vào cuối
    added = []
    for token in EXTRA_TOKENS:
        if token not in vocab.stoi:
            vocab._add_token(token)
            added.append(token)
        else:
            print(f"  '{token}' already exists at ID {vocab.stoi[token]}, skipping.")

    if added:
        print(f"Added tokens      : {added}")

    # 3. Kiểm tra ID của từng token mới
    print()
    for token in EXTRA_TOKENS:
        print(f"  '{token}' → ID {vocab.token_to_idx(token)}")

    # 4. Lưu ra vocab_v3.json
    vocab.save(VOCAB_V3_PATH)
    print(f"\nNew size          : {len(vocab):,} tokens")
    print(f"Saved to          : {VOCAB_V3_PATH}")


if __name__ == "__main__":
    main()
