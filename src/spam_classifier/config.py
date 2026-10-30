"""Project paths and fixed experiment settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "SMSSpamCollection"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
LOG_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "docs"
RANDOM_STATE = 42
MAX_INPUT_CHARS = 1000
TARGET_SPAM_PRECISION = 0.98
SPAM_LABEL = "spam"
HAM_LABEL = "ham"
