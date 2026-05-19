from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    dataset_name: str
    label_column: str
    classes_to_keep: list[str]
    csv_null_values: list[str]
    pca_components: int
    test_size: float
    random_seed: int
    processed_data_name: str = "data.npz"
    processed_metadata_name: str = "metadata.json"
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)

    def __post_init__(self):
        data_dir = Path("data")
        self.raw_data_dir = data_dir / "raw" / self.dataset_name
        self.processed_data_dir = data_dir / "processed" / self.dataset_name


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
)
