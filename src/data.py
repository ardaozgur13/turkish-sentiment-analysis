"""Downloads and cleans the data, then builds the train/val/test subsets.

Usage:
    python -m src.data                      # default (CPU) sizes
    python -m src.data --n-train 40000      # larger subset if a GPU is available
"""
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, DATASET_NAME, LABELS, N_TEST, N_TRAIN, N_VAL, SEED

RAW_TRAIN = DATA_DIR / "raw_train.parquet"
RAW_TEST = DATA_DIR / "raw_test.parquet"


def download_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Downloads the dataset from Hugging Face and caches it under data/ as parquet.
    Later runs read from disk instead of the internet."""
    if RAW_TRAIN.exists() and RAW_TEST.exists():
        return pd.read_parquet(RAW_TRAIN), pd.read_parquet(RAW_TEST)

    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME)
    train, test = ds["train"].to_pandas(), ds["test"].to_pandas()
    train.to_parquet(RAW_TRAIN, index=False)
    test.to_parquet(RAW_TEST, index=False)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drops empty texts, unknown labels and duplicate texts."""
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    df = df[df["label"].isin(LABELS)]
    return df.drop_duplicates(subset="text").reset_index(drop=True)


def make_splits(n_train: int = N_TRAIN, n_val: int = N_VAL, n_test: int = N_TEST):
    raw_train, raw_test = download_raw()
    raw_train, raw_test = clean(raw_train), clean(raw_test)

    # Prevent data leakage: texts that also appear in train must not be in test.
    raw_test = raw_test[~raw_test["text"].isin(set(raw_train["text"]))]

    # stratify=label keeps the original class proportions in every subset.
    train_val, _ = train_test_split(
        raw_train, train_size=n_train + n_val, stratify=raw_train["label"], random_state=SEED
    )
    train, val = train_test_split(
        train_val, test_size=n_val, stratify=train_val["label"], random_state=SEED
    )
    test, _ = train_test_split(
        raw_test, train_size=n_test, stratify=raw_test["label"], random_state=SEED
    )

    for name, df in (("train", train), ("val", val), ("test", test)):
        df.reset_index(drop=True).to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(df)} examples | {df['label'].value_counts().to_dict()}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(pd.read_csv(DATA_DIR / f"{n}.csv") for n in ("train", "val", "test"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-train", type=int, default=N_TRAIN)
    p.add_argument("--n-val", type=int, default=N_VAL)
    p.add_argument("--n-test", type=int, default=N_TEST)
    a = p.parse_args()
    make_splits(a.n_train, a.n_val, a.n_test)
