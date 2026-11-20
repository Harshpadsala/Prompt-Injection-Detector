"""Central configuration: paths, seeds and constants shared by every module."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MAIN_DATASET = "jayavibhav/prompt-injection"
EXTERNAL_DATASET = "deepset/prompt-injections"

SEED = 42
LABEL_NAMES = {0: "Safe", 1: "Injection"}

# Grid search runs on a random subsample (classes are balanced);
# the final model is refit on all data.
TUNING_SAMPLE_SIZE = 30_000
CV_FOLDS = 5

TFIDF_MAX_FEATURES = 50_000  # caps memory and training time; rarer n-grams add little
LSA_COMPONENTS = 100  # dense enough for K-means distances, small enough to stay fast
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast general-purpose sentence encoder

# K-means: candidate K values and the subsample used to pick K.
K_RANGE = range(2, 21)
SWEEP_SAMPLE = 30_000
