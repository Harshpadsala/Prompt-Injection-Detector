"""Dataset loading, cleaning and the train/test/external split used by every approach."""
import re

import pandas as pd
from datasets import load_dataset

from src.config import EXTERNAL_DATASET, MAIN_DATASET, PROCESSED_DIR

_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase and collapse repeated whitespace. Punctuation is kept: quotes, brackets and
    code symbols are often part of an attack."""
    return _WHITESPACE.sub(" ", text.lower()).strip()


def _to_frame(split) -> pd.DataFrame:
    df = split.to_pandas()[["text", "label"]]
    return df.rename(columns={"text": "text_raw"})


def load_main() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Main dataset (train, test) straight from Hugging Face."""
    ds = load_dataset(MAIN_DATASET)
    return _to_frame(ds["train"]), _to_frame(ds["test"])


def load_external() -> pd.DataFrame:
    """deepset dataset, train and test merged: it is only used as an external test set."""
    ds = load_dataset(EXTERNAL_DATASET)
    return pd.concat([_to_frame(ds["train"]), _to_frame(ds["test"])], ignore_index=True)


def _drop_empty_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["text_raw"].str.strip() != ""].drop_duplicates(subset="text_raw")
    return df.reset_index(drop=True)


def prepare_datasets() -> dict[str, pd.DataFrame]:
    """Clean all splits, remove leakage between them, and save them as parquet.

    Leakage rules:
      * test rows that also appear in train are dropped
      * main-dataset rows that also appear in the external set are dropped
    """
    train, test = load_main()
    external = load_external()
    splits = {name: _drop_empty_and_duplicates(df)
              for name, df in {"train": train, "test": test, "external": external}.items()}

    for df in splits.values():
        df["text_clean"] = df["text_raw"].map(clean_text)

    external_texts = set(splits["external"]["text_clean"])
    train_texts = set(splits["train"]["text_clean"])
    splits["train"] = splits["train"][~splits["train"]["text_clean"].isin(external_texts)]
    splits["test"] = splits["test"][
        ~splits["test"]["text_clean"].isin(external_texts | train_texts)
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        splits[name] = df.reset_index(drop=True)
        splits[name].to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)
    return splits


def load_split(name: str) -> pd.DataFrame:
    """Load a processed split: 'train', 'test' or 'external'."""
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
