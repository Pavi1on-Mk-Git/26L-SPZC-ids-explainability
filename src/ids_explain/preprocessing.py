import json
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .config import DataConfig
from .data_loader import DatasetSplit


@dataclass
class ProcessedData:
    X_train_raw: np.ndarray
    X_train_pca: np.ndarray
    X_test_raw: np.ndarray
    X_test_pca: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_map: dict[str, int]
    feature_names: list[str]


def build_preprocessing_pipeline(n_components: int, random_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("minmax", MinMaxScaler()),
            ("standard", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=random_seed)),
        ]
    )


def fit_pipeline(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.fit_transform(X).astype(np.float32)


def apply_pipeline(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    return pipeline.transform(X).astype(np.float32)


def save_processed(data: ProcessedData, config: DataConfig):
    config.write_or_verify_config()
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.processed_data_dir / config.processed_data_name,
        X_train_raw=data.X_train_raw,
        X_train_pca=data.X_train_pca,
        X_test_raw=data.X_test_raw,
        X_test_pca=data.X_test_pca,
        y_train=data.y_train,
        y_test=data.y_test,
    )
    metadata = {
        "label_map": data.label_map,
        "feature_names": data.feature_names,
    }
    (config.processed_data_dir / config.processed_metadata_name).write_text(json.dumps(metadata, indent=2))


def load_processed(config: DataConfig) -> ProcessedData:
    config.write_or_verify_config()
    arrays = np.load(config.processed_data_dir / config.processed_data_name)
    metadata = json.loads((config.processed_data_dir / config.processed_metadata_name).read_text())
    return ProcessedData(
        X_train_raw=arrays["X_train_raw"],
        X_train_pca=arrays["X_train_pca"],
        X_test_raw=arrays["X_test_raw"],
        X_test_pca=arrays["X_test_pca"],
        y_train=arrays["y_train"],
        y_test=arrays["y_test"],
        label_map=metadata["label_map"],
        feature_names=metadata["feature_names"],
    )


def preprocess(split: DatasetSplit, config: DataConfig) -> ProcessedData:
    pipeline = build_preprocessing_pipeline(config.pca_components, config.random_seed)
    X_train_pca = fit_pipeline(pipeline, split.X_train)
    X_test_pca = apply_pipeline(pipeline, split.X_test)

    data = ProcessedData(
        X_train_raw=split.X_train,
        X_train_pca=X_train_pca,
        X_test_raw=split.X_test,
        X_test_pca=X_test_pca,
        y_train=split.y_train,
        y_test=split.y_test,
        label_map=split.label_map,
        feature_names=split.feature_names,
    )
    save_processed(data, config)
    return data
