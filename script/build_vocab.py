import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FREQ_THRESHOLD, TRAIN_DF_PATH, VOCAB_PATH
from src.vocabulary import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vocabulary from train_df and save it as JSON.")
    parser.add_argument("--train-df-path", type=str, default=str(TRAIN_DF_PATH))
    parser.add_argument("--output-path", type=str, default=str(VOCAB_PATH))
    parser.add_argument("--freq-threshold", type=int, default=FREQ_THRESHOLD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_df = pd.read_parquet(args.train_df_path)

    vocab = Vocabulary(freq_threshold=args.freq_threshold)
    vocab.build_vocabulary(train_df["tokens"].tolist())
    output_path = vocab.save(args.output_path)

    print(f"vocab_size: {len(vocab):,}")
    print(f"saved_to: {output_path}")


if __name__ == "__main__":
    main()
