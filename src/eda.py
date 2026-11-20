"""Exploratory data analysis: summary tables and figures saved to reports/figures."""
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from wordcloud import STOPWORDS, WordCloud

from src.config import FIGURES_DIR, LABEL_NAMES, SEED

WORD_CLOUD_STOP_WORDS = set(ENGLISH_STOP_WORDS) | STOPWORDS


def _save(fig, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / filename, dpi=120)


def summarize(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Rows, class balance and prompt length per split."""
    rows = {}
    for name, df in splits.items():
        words = df["text_raw"].str.split().str.len()
        rows[name] = {"rows": len(df), "injection_share": df["label"].mean(),
                      "median_words": words.median(), "max_words": words.max()}
    return pd.DataFrame(rows).T


def plot_class_balance(splits: dict[str, pd.DataFrame]) -> None:
    counts = pd.DataFrame({n: df["label"].map(LABEL_NAMES).value_counts()
                           for n, df in splits.items()})
    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts.T.plot.bar(ax=ax, rot=0)
    ax.set(title="Class balance per split", ylabel="Prompts")
    _save(fig, "eda_class_balance.png")


def plot_prompt_length(df: pd.DataFrame) -> None:
    """Histogram of words per prompt (log x-axis: lengths are heavily skewed)."""
    words = df["text_raw"].str.split().str.len()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for label, name in LABEL_NAMES.items():
        ax.hist(words[df["label"] == label].clip(lower=1), bins=60, alpha=0.6, label=name)
    ax.set(xscale="log", title="Prompt length (train)", xlabel="Words", ylabel="Prompts")
    ax.legend()
    _save(fig, "eda_prompt_length.png")


def top_words(df: pd.DataFrame, label: int, n: int = 15) -> list[tuple[str, int]]:
    counter = Counter(
        # set(): count each word once per prompt so very long prompts do not dominate
        w for text in df.loc[df["label"] == label, "text_clean"] for w in set(text.split())
        if w.isalpha() and w not in ENGLISH_STOP_WORDS
    )
    return counter.most_common(n)


def plot_top_words(df: pd.DataFrame) -> None:
    """Most common words per class, counted once per prompt, stop words removed."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (label, name) in zip(axes, LABEL_NAMES.items()):
        words, counts = zip(*top_words(df, label))
        ax.barh(words[::-1], counts[::-1])
        ax.set_title(f"{name}: most common words")
    _save(fig, "eda_top_words.png")


def plot_word_clouds(df: pd.DataFrame, max_prompts: int = 30_000) -> None:
    """One word cloud per class. Each class is subsampled to max_prompts to keep it fast."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, (label, label_name) in zip(axes, LABEL_NAMES.items()):
        texts = df.loc[df["label"] == label, "text_clean"]
        texts = texts.sample(min(max_prompts, len(texts)), random_state=SEED)
        cloud = WordCloud(width=900, height=500, background_color="white",
                          stopwords=WORD_CLOUD_STOP_WORDS, collocations=False,  # single words only
                          max_words=80, random_state=SEED).generate(" ".join(texts))
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(f"{label_name} prompts", fontsize=20, fontweight="bold", pad=12)
        ax.axis("off")
    _save(fig, "eda_word_clouds.png")
