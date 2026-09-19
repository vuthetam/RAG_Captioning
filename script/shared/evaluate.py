import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.shared.metrics import compute_metrics
from src.shared.config import METRICS_PATH, PREDICTIONS_PATH, TEST_DF_PATH, RUN_MODE


def main():
    print(f"Run mode  : {RUN_MODE}")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Metrics out: {METRICS_PATH}")
    print()

    metrics = compute_metrics(PREDICTIONS_PATH, TEST_DF_PATH)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved metrics to {METRICS_PATH}")

    print("\n" + "=" * 35)
    print(f"  [{RUN_MODE.upper()}] METRICS")
    print("=" * 35)
    for metric, score in metrics.items():
        print(f"  {metric:>10}: {score * 100:.2f}")
    print("=" * 35)


if __name__ == "__main__":
    main()
