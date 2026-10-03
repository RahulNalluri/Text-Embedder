"""Configuration for the Indian financial Word2Vec training pipeline."""

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "annual_reports"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "indian_financial"
VERSION_3_REPORTS_DIR = RAW_DATA_DIR / "version_3"
VERSION_3_APPROVED_DIR = VERSION_3_REPORTS_DIR / "approved"
VERSION_3_PROCESSED_DIR = PROCESSED_DATA_DIR / "version_3"
VERSION_3_ARTIFACTS_DIR = ARTIFACTS_DIR / "version_3"


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters shared by collection, training, and evaluation."""

    domain_name: str = "indian_corporate_financial_reporting"
    architecture: str = "skip_gram"
    context_window: int = 3
    embedding_dimension: int = 50
    negative_samples: int = 5
    minimum_word_frequency: int = 5
    epochs: int = 5
    learning_rate: float = 0.025
    random_seed: int = 42
    workers: int = 1

    def __post_init__(self) -> None:
        positive_integer_fields = {
            "context_window": self.context_window,
            "embedding_dimension": self.embedding_dimension,
            "negative_samples": self.negative_samples,
            "minimum_word_frequency": self.minimum_word_frequency,
            "epochs": self.epochs,
            "workers": self.workers,
        }

        for field_name, value in positive_integer_fields.items():
            if value < 1:
                raise ValueError(
                    f"{field_name} must be a positive integer."
                )

        if not 0 < self.learning_rate <= 1:
            raise ValueError(
                "learning_rate must be greater than 0 and at most 1."
            )

        if self.architecture not in {"skip_gram", "cbow"}:
            raise ValueError("architecture must be either 'skip_gram' or 'cbow'.")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation of the settings."""

        return asdict(self)


DEFAULT_CONFIG = TrainingConfig()
