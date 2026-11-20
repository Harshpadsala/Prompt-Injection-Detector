"""Approach 1: TF-IDF features with supervised classifiers."""
import argparse

import joblib

from src.config import MODELS_DIR
from src.data import load_split
from src.evaluate import evaluate_model, record_result
from src.models import select, tfidf_specs, tune_and_fit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="subset to (re)train, e.g. --models xgboost")
    args = parser.parse_args()
    train, test, external = (load_split(s) for s in ("train", "test", "external"))
    MODELS_DIR.mkdir(exist_ok=True)

    for spec in select(tfidf_specs(), args.models):
        model = tune_and_fit(spec, train["text_clean"], train["label"])
        joblib.dump(model, MODELS_DIR / f"tfidf_{spec.name}.joblib")

        run_name = f"tfidf_{spec.name}"
        metrics = {
            "best_params": spec.best_params,
            "test": evaluate_model(model, test["text_clean"], test["label"], "test", run_name),
            "external": evaluate_model(
                model, external["text_clean"], external["label"], "external", run_name),
        }
        record_result(run_name, metrics)
        print(f"[{spec.name}] test={metrics['test']}\n[{spec.name}] external={metrics['external']}")


if __name__ == "__main__":
    main()
