"""Download, clean and save the train, test and external splits."""
from src.data import prepare_datasets


def main() -> None:
    for name, df in prepare_datasets().items():
        print(f"{name:9s} rows={len(df):>7,}  injection share={df['label'].mean():.3f}")


if __name__ == "__main__":
    main()
