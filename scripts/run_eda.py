"""Generate EDA tables and figures."""
from src.data import load_split
from src.eda import (plot_class_balance, plot_prompt_length, plot_top_words, plot_word_clouds,
                     summarize)


def main() -> None:
    splits = {s: load_split(s) for s in ("train", "test", "external")}
    print(summarize(splits))
    plot_class_balance(splits)
    plot_prompt_length(splits["train"])
    plot_top_words(splits["train"])
    plot_word_clouds(splits["train"])


if __name__ == "__main__":
    main()
