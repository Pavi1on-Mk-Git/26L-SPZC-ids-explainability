import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_FILE_NAME = "config.json"


@dataclass
class DataConfig:
    dataset_name: str
    label_column: str
    classes_to_keep: list[str]
    csv_null_values: list[str]
    pca_components: int
    test_size: float
    random_seed: int
    target_total_samples: int | None = None
    processed_data_name: str = "data.npz"
    processed_metadata_name: str = "metadata.json"
    raw_data_dir: Path = field(init=False)
    config_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)

    def __post_init__(self):
        data_dir = Path("data")
        self.raw_data_dir = data_dir / "raw" / self.dataset_name
        self.config_dir = data_dir / "processed" / self.dataset_name / self.config_hash()
        self.processed_data_dir = self.config_dir / f"{self.random_seed}"

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init and f.name != "random_seed"}

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def write_or_verify_config(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        config_path = self.config_dir / CONFIG_FILE_NAME
        current = self.to_dict()
        if config_path.exists():
            existing = json.loads(config_path.read_text())
            if existing != current:
                raise ValueError(
                    f"Config hash collision detected at {config_path}: the stored "
                    f"config does not match the current config.\nstored:  {existing}\n"
                    f"current: {current}"
                )
        else:
            config_path.write_text(json.dumps(current, indent=4))


@dataclass
class OracleConfig:
    hidden_dim: int
    n_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    max_epochs: int
    val_size: float
    early_stopping_patience: int
    early_stopping_monitor: str
    best_ckpt_name: str = "best_ckpt"
    model_name: str = "oracle.pt"


@dataclass
class ExplainerConfig:
    k_frac: float
    tree_max_depth: int
    n_search: int
    data_dir_name: str = field(init=False)
    centroids_name: str = "centroids.npy"
    trees_name: str = "trees.joblib"
    cluster_indices_name: str = "cluster_indices.joblib"

    def __post_init__(self):
        self.data_dir_name = f"explainer_k{self.k_frac}"


CICIDS2017_CONFIG = DataConfig(
    dataset_name="CICIDS2017",
    label_column="Label",
    classes_to_keep=[
        "BENIGN",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "FTP-Patator",
        "PortScan",
        "SSH-Patator",
    ],
    csv_null_values=["Infinity", "NaN"],
    pca_components=35,
    test_size=0.25,
    random_seed=42,
    target_total_samples=1_971_937,
)

DEFAULT_ORACLE_CONFIG = OracleConfig(
    hidden_dim=512,
    n_layers=5,
    dropout=0.2,
    learning_rate=1e-3,
    batch_size=10_000,
    max_epochs=100,
    val_size=0.25,
    early_stopping_patience=5,
    early_stopping_monitor="val_loss",
)

DEFAULT_EXPLAINER_CONFIG = ExplainerConfig(k_frac=0.005, tree_max_depth=4, n_search=3)
