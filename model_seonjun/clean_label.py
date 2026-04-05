from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

RAW_TRAIN_JSON = BASE_DIR / "json" / "training2.json"
SP500_CSV = BASE_DIR / "csv" / "sp500_list.csv"
CLEAN_TRAIN_JSON = BASE_DIR / "json" / "training2_clean.json"


def load_sp500_universe() -> set[str]:
    df = pd.read_csv(SP500_CSV)
    tickers = df["ticker"].astype(str).str.upper().tolist()
    return set(tickers)


def load_raw_training() -> List[Dict[str, Any]]:
    with RAW_TRAIN_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    sp500_universe = load_sp500_universe()
    raw_data = load_raw_training()

    cleaned: List[Dict[str, Any]] = []

    for row in raw_data:
        tickers = [t.strip().upper() for t in row.get("tickers", [])]
        sp500_labels = sorted({t for t in tickers if t in sp500_universe})

        # S&P500 교집합이 하나도 없으면 학습/평가 대상에서 제외
        if not sp500_labels:
            continue

        row["tickers"] = tickers
        row["sp500_labels"] = sp500_labels
        cleaned.append(row)

    # reproducible shuffle
    random.seed(42)
    random.shuffle(cleaned)

    n = len(cleaned)
    n_train = int(0.7 * n)
    n_valid = int(0.15 * n)
    # train: 0 ~ n_train-1
    # valid: n_train ~ n_train+n_valid-1
    # test:  나머지
    for i, row in enumerate(cleaned):
        if i < n_train:
            split = "train"
        elif i < n_train + n_valid:
            split = "valid"
        else:
            split = "test"
        row["split"] = split

    CLEAN_TRAIN_JSON.parent.mkdir(parents=True, exist_ok=True)
    with CLEAN_TRAIN_JSON.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Raw samples: {len(raw_data)}")
    print(f"[INFO] After S&P500 label cleaning: {len(cleaned)}")
    print(
        "[INFO] Split counts:",
        f"train={sum(r['split']=='train' for r in cleaned)},",
        f"valid={sum(r['split']=='valid' for r in cleaned)},",
        f"test={sum(r['split']=='test' for r in cleaned)}",
    )


if __name__ == "__main__":
    main()
