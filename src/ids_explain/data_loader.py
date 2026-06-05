from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from .config import DataConfig


@dataclass
class DatasetSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    label_map: dict[str, int]
    feature_names: list[str]


def _scan_csvs(data_dir: Path, null_values: list[str]) -> pl.LazyFrame:
    files = sorted(data_dir.glob("*.csv"))
    frames = [pl.scan_csv(f, null_values=null_values, try_parse_dates=False, infer_schema_length=10_000) for f in files]
    return pl.concat(frames, how="diagonal_relaxed")


def _strip_column_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    rename_map = {col: col.strip() for col in schema.names()}
    return lf.rename(rename_map)


def _drop_duplicate_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    duplicate_cols = [name for name in lf.collect_schema().names() if "_duplicated_" in name]
    return lf.drop(duplicate_cols) if duplicate_cols else lf


def _replace_inf_with_null(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    float_cols = [name for name, dtype in schema.items() if dtype in (pl.Float32, pl.Float64)]
    if not float_cols:
        return lf
    return lf.with_columns(
        pl.when(pl.col(c).is_infinite() | pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
        for c in float_cols
    )


def _filter_classes(
    lf: pl.LazyFrame,
    label_column: str,
    classes: list[str],
) -> pl.LazyFrame:
    return lf.filter(pl.col(label_column).is_in(classes))


def _encode_labels(
    df: pl.DataFrame,
    label_column: str,
) -> tuple[pl.DataFrame, dict[str, int]]:
    unique_labels: list[str] = sorted(df[label_column].unique().to_list())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = df.with_columns(
        pl.col(label_column).replace_strict(
            old=pl.Series(list(label_map.keys())),
            new=pl.Series(list(label_map.values()), dtype=pl.Int64),
            return_dtype=pl.Int64,
        )
    )
    return encoded, label_map


def _to_numpy(
    df: pl.DataFrame,
    label_col: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    feature_cols = [c for c in df.columns if c != label_col]
    X = df.select(pl.col(c).cast(pl.Float32) for c in feature_cols).to_numpy(allow_copy=True).astype(np.float32)
    y = df[label_col].cast(pl.Int64).to_numpy().astype(np.int64)
    return X, y, feature_cols


def _stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )


def _undersample_majority(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int,
    majority_target: int,
) -> tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(y_train, return_counts=True)
    majority_idx = int(counts.argmax())
    majority_label = int(unique[majority_idx])
    majority_available = int(counts[majority_idx])

    if not 0 < majority_target <= majority_available:
        raise ValueError(
            f"Majority undersampling target {majority_target:,} is out of range "
            f"(1..{majority_available:,} available in the training split)."
        )

    sampler = RandomUnderSampler(
        sampling_strategy={majority_label: majority_target},
        random_state=random_seed,
    )
    return sampler.fit_resample(X_train, y_train)


def _majority_target_for_total(y_train: np.ndarray, n_test: int, target_total_samples: int) -> int:
    _, counts = np.unique(y_train, return_counts=True)
    non_majority = int(counts.sum() - counts.max())
    return target_total_samples - n_test - non_majority


def load_dataset(data_cfg: DataConfig) -> DatasetSplit:
    lf = _scan_csvs(data_cfg.raw_data_dir, data_cfg.csv_null_values)
    lf = _drop_duplicate_columns(lf)
    lf = _strip_column_names(lf)
    lf = _replace_inf_with_null(lf)
    lf = _filter_classes(lf, data_cfg.label_column, data_cfg.classes_to_keep)
    lf = lf.drop_nulls()

    df = lf.collect()
    df, label_map = _encode_labels(df, data_cfg.label_column)
    X, y, feature_names = _to_numpy(df, data_cfg.label_column)

    X_train, X_test, y_train, y_test = _stratified_split(X, y, data_cfg.test_size, data_cfg.random_seed)
    majority_target = None
    if data_cfg.target_total_samples is not None:
        majority_target = _majority_target_for_total(y_train, len(y_test), data_cfg.target_total_samples)
    X_train, y_train = _undersample_majority(X_train, y_train, data_cfg.random_seed, majority_target)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_map=label_map,
        feature_names=feature_names,
    )
