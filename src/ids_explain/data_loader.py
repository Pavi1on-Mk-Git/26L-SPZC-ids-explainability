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


def _scan_csvs(data_dir: Path, null_values: list[str], filenames: list[str] | None = None) -> pl.LazyFrame:
    if filenames:
        files = [data_dir / name for name in filenames]
        missing = [str(f) for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"Configured CSV files not found: {missing}")
    else:
        files = sorted(data_dir.glob("*.csv"))
    frames = [pl.scan_csv(f, null_values=null_values, try_parse_dates=False, infer_schema_length=None) for f in files]
    return pl.concat(frames, how="diagonal_relaxed")


def _strip_column_names(lf: pl.LazyFrame) -> pl.LazyFrame:
    schema = lf.collect_schema()
    rename_map = {col: col.strip() for col in schema.names()}
    return lf.rename(rename_map)


def _drop_duplicate_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    duplicate_cols = [name for name in lf.collect_schema().names() if "_duplicated_" in name]
    return lf.drop(duplicate_cols) if duplicate_cols else lf


def _drop_columns(lf: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    present = set(lf.collect_schema().names())
    to_drop = [c for c in columns if c in present]
    return lf.drop(to_drop) if to_drop else lf


def _normalize_label(lf: pl.LazyFrame, label_column: str) -> pl.LazyFrame:
    return lf.with_columns(pl.col(label_column).cast(pl.String).str.strip_chars())


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


def _collect_xy(
    lf: pl.LazyFrame,
    label_column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int], list[str]]:
    feature_names = [name for name in lf.collect_schema().names() if name != label_column]
    lf = lf.with_columns(pl.col(c).cast(pl.Float32) for c in feature_names)
    df = lf.collect(engine="streaming")

    unique_labels: list[str] = sorted(df[label_column].unique().to_list())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = (
        df[label_column]
        .replace_strict(
            old=pl.Series(list(label_map.keys())),
            new=pl.Series(list(label_map.values()), dtype=pl.Int64),
            return_dtype=pl.Int64,
        )
        .to_numpy()
    )

    X = df.drop(label_column).to_numpy()
    return X, y, label_map, feature_names


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


def _default_majority_target(y_train: np.ndarray) -> int:
    _, counts = np.unique(y_train, return_counts=True)
    majority_available = int(counts.max())
    non_majority = int(counts.sum() - counts.max())
    return min(2 * non_majority, majority_available)


def load_dataset(data_cfg: DataConfig, raw_data_dir: Path) -> DatasetSplit:
    lf = _scan_csvs(raw_data_dir, data_cfg.csv_null_values, data_cfg.csv_filenames)
    lf = _drop_duplicate_columns(lf)
    lf = _strip_column_names(lf)
    lf = _drop_columns(lf, data_cfg.columns_to_drop)
    lf = _replace_inf_with_null(lf)
    lf = _normalize_label(lf, data_cfg.label_column)
    lf = _filter_classes(lf, data_cfg.label_column, data_cfg.classes_to_keep)
    lf = lf.drop_nulls()

    X, y, label_map, feature_names = _collect_xy(lf, data_cfg.label_column)

    X_train, X_test, y_train, y_test = _stratified_split(X, y, data_cfg.test_size, data_cfg.random_seed)
    del X, y
    if data_cfg.target_total_samples is not None:
        majority_target = _majority_target_for_total(y_train, len(y_test), data_cfg.target_total_samples)
    else:
        majority_target = _default_majority_target(y_train)
    X_train, y_train = _undersample_majority(X_train, y_train, data_cfg.random_seed, majority_target)

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_map=label_map,
        feature_names=feature_names,
    )
