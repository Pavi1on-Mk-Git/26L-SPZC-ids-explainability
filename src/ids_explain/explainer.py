from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .config import DataConfig, ExplainerConfig
from .mdav import mdav
from .preprocessing import ProcessedData


@dataclass
class Explainer:
    centroids: np.ndarray
    trees: list[DecisionTreeClassifier]
    cluster_indices: list[np.ndarray]


def _centroid(X: np.ndarray) -> np.ndarray:
    return X.mean(axis=0)


def train_explainer(data: ProcessedData, data_cfg: DataConfig, explainer_cfg: ExplainerConfig) -> Explainer:
    X, y = data.X_train_raw, data.y_train
    k = max(1, round(explainer_cfg.k_frac * len(X)))

    cluster_indices = mdav(X, k)

    centroids = []
    trees = []

    for indices in cluster_indices:
        X_cluster, y_cluster = X[indices], y[indices]
        centroids.append(_centroid(X_cluster))

        tree = DecisionTreeClassifier(max_depth=explainer_cfg.tree_max_depth, random_state=data_cfg.random_seed)
        tree.fit(X_cluster, y_cluster)
        trees.append(tree)

    return Explainer(
        centroids=np.stack(centroids),
        trees=trees,
        cluster_indices=cluster_indices,
    )


def _explainer_dir(data_cfg: DataConfig, explainer_cfg: ExplainerConfig) -> Path:
    return data_cfg.processed_data_dir / explainer_cfg.data_dir_name


def save_explainer(explainer: Explainer, data_cfg: DataConfig, explainer_cfg: ExplainerConfig):
    out_dir = _explainer_dir(data_cfg, explainer_cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / explainer_cfg.centroids_name, explainer.centroids)
    joblib.dump(explainer.trees, out_dir / explainer_cfg.trees_name)
    joblib.dump(explainer.cluster_indices, out_dir / explainer_cfg.cluster_indices_name)


def load_explainer(data_cfg: DataConfig, explainer_cfg: ExplainerConfig) -> Explainer:
    out_dir = _explainer_dir(data_cfg, explainer_cfg)
    return Explainer(
        centroids=np.load(out_dir / explainer_cfg.centroids_name),
        trees=joblib.load(out_dir / explainer_cfg.trees_name),
        cluster_indices=joblib.load(out_dir / explainer_cfg.cluster_indices_name),
    )


def guided_search(
    sample: np.ndarray,
    oracle_pred: int,
    explainer: Explainer,
    n_search: int,
) -> tuple[int, DecisionTreeClassifier, int]:
    diff = explainer.centroids - sample
    distances = (diff**2).sum(axis=1)
    sorted_indices = np.argsort(distances)

    for cluster_idx in sorted_indices[:n_search]:
        tree: DecisionTreeClassifier = explainer.trees[cluster_idx]
        pred = int(tree.predict(sample.reshape(1, -1))[0])
        if pred == oracle_pred:
            return pred, tree, int(cluster_idx)

    fallback = int(sorted_indices[0])
    tree = explainer.trees[fallback]
    pred = int(tree.predict(sample.reshape(1, -1))[0])
    return pred, tree, fallback


def predict_explainer(
    explainer: Explainer,
    X: np.ndarray,
    oracle_preds: np.ndarray,
    n_search: int,
) -> np.ndarray:
    preds = np.empty(len(X), dtype=np.int64)
    for i, (sample, oracle_pred) in enumerate(zip(X, oracle_preds)):
        pred, _, _ = guided_search(sample, int(oracle_pred), explainer, n_search)
        preds[i] = pred
    return preds
