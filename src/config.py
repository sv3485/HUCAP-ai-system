"""
Configuration module for the protein function prediction pipeline.

Defines data, training, model, and path configurations. Values are loaded
from ``configs/training_config.yaml`` when present, and fall back to
sensible defaults otherwise.
"""

import os
from dataclasses import dataclass
from typing import Any, Optional

import yaml


@dataclass
class DataConfig:
    """Dataset-related configuration."""

    # Data source mode: "cafa" or "swissprot_gaf"
    data_source: str = "swissprot_gaf"

    # CAFA-6 paths
    dataset_root: str = "cafa-6-protein-function-prediction (1)"
    cafa_terms_path: str = "cafa-6-protein-function-prediction (1)/Train/train_terms.tsv"

    # Swiss-Prot + GAF paths
    fasta_path: str = "cafa-6-protein-function-prediction (1)/Train/train_sequences.fasta"
    gaf_path: str = "filtered_goa_uniprot_all_noiea.gaf"

    # General
    aspect: str = "F"
    max_seq_len: int = 1024
    min_term_frequency: int = 20
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Filtering
    min_seq_len: int = 30
    max_seq_len_filter: int = 2000
    max_go_terms_per_protein: int = 100
    top_n_go_terms: int = 1000

    # Training cap
    max_train_samples: int = 100000


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_workers: int = 0
    random_seed: int = 42
    label_smoothing: float = 0.05
    gradient_clipping: bool = True
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    vocab_size: int = 28  # 20 AAs + special tokens
    embedding_dim: int = 64
    conv_channels: int = 128
    conv_kernel_sizes: tuple = (3, 5, 7)
    dropout: float = 0.3


class Paths:
    """Filesystem paths derived from the project root."""

    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @property
    def models_dir(self) -> str:
        path = os.path.join(self.project_root, "models")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def outputs_dir(self) -> str:
        path = os.path.join(self.project_root, "outputs")
        os.makedirs(path, exist_ok=True)
        return path


paths = Paths()

# ---------------------------------------------------------------------------
# Load optional YAML configuration overlay
# ---------------------------------------------------------------------------
_yaml_path = os.path.join(paths.project_root, "configs", "training_config.yaml")
_config_data: dict = {}
if os.path.exists(_yaml_path):
    with open(_yaml_path, "r", encoding="utf-8") as _f:
        _config_data = yaml.safe_load(_f) or {}


def _get_val(section: Optional[str], key: str, default: Any) -> Any:
    if section is None:
        return _config_data.get(key, default)
    return (_config_data.get(section) or {}).get(key, default)


data_config = DataConfig(
    data_source=_get_val(None, "data_source", DataConfig.data_source),
    dataset_root=_get_val(None, "dataset_root", DataConfig.dataset_root),
    fasta_path=_get_val(None, "fasta_path", DataConfig.fasta_path),
    gaf_path=_get_val(None, "gaf_path", DataConfig.gaf_path),
    cafa_terms_path=_get_val(None, "cafa_terms_path", DataConfig.cafa_terms_path),
    aspect=_get_val(None, "aspect", DataConfig.aspect),
    max_seq_len=_get_val(None, "max_seq_len", DataConfig.max_seq_len),
    min_term_frequency=_get_val(None, "min_term_frequency", DataConfig.min_term_frequency),
    val_ratio=_get_val(None, "val_ratio", DataConfig.val_ratio),
    test_ratio=_get_val(None, "test_ratio", DataConfig.test_ratio),
    min_seq_len=_get_val(None, "min_seq_len", DataConfig.min_seq_len),
    max_seq_len_filter=_get_val(None, "max_seq_len_filter", DataConfig.max_seq_len_filter),
    max_go_terms_per_protein=_get_val(None, "max_go_terms_per_protein", DataConfig.max_go_terms_per_protein),
    top_n_go_terms=_get_val(None, "top_n_go_terms", DataConfig.top_n_go_terms),
    max_train_samples=_get_val(None, "max_train_samples", DataConfig.max_train_samples),
)

train_config = TrainConfig(
    batch_size=_get_val("train", "batch_size", TrainConfig.batch_size),
    num_epochs=_get_val("train", "num_epochs", TrainConfig.num_epochs),
    learning_rate=_get_val("train", "learning_rate", TrainConfig.learning_rate),
    weight_decay=_get_val("train", "weight_decay", TrainConfig.weight_decay),
    num_workers=_get_val("train", "num_workers", TrainConfig.num_workers),
    random_seed=_get_val("train", "random_seed", TrainConfig.random_seed),
    label_smoothing=_get_val("train", "label_smoothing", TrainConfig.label_smoothing),
    gradient_clipping=_get_val("train", "gradient_clipping", TrainConfig.gradient_clipping),
    mixed_precision=_get_val("train", "mixed_precision", TrainConfig.mixed_precision),
    gradient_accumulation_steps=_get_val("train", "gradient_accumulation_steps", TrainConfig.gradient_accumulation_steps),
)

model_config = ModelConfig(
    embedding_dim=_get_val("model", "embedding_dim", ModelConfig.embedding_dim),
    dropout=_get_val("model", "dropout", ModelConfig.dropout),
)
