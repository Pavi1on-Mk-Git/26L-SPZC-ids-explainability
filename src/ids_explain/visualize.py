import copy
from pathlib import Path

import dtreeviz
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from .explainer import Explainer, guided_search
from .preprocessing import ProcessedData


def _class_names_for_tree(
    tree_classes: np.ndarray,
    label_map: dict[str, int],
) -> list[str]:
    inv = {idx: label for label, idx in label_map.items()}
    return [inv[int(c)] for c in tree_classes]


def _make_viz_tree(
    tree: DecisionTreeClassifier,
    y_cluster: np.ndarray,
) -> tuple[DecisionTreeClassifier, np.ndarray]:
    original_classes = tree.classes_
    class_to_idx = {int(c): i for i, c in enumerate(original_classes)}
    y_remapped = np.vectorize(class_to_idx.__getitem__)(y_cluster).astype(np.int64)
    viz_tree = copy.deepcopy(tree)
    viz_tree.classes_ = np.arange(len(original_classes), dtype=original_classes.dtype)
    return viz_tree, y_remapped


def explain_sample(
    sample: np.ndarray,
    oracle_pred: int,
    explainer: Explainer,
    data: ProcessedData,
    n_search: int,
) -> dtreeviz.DTreeVizRender:
    _, tree, cluster_idx = guided_search(sample, oracle_pred, explainer, n_search)
    indices = explainer.cluster_indices[cluster_idx]
    X_cluster, y_cluster = data.X_train_raw[indices], data.y_train[indices]

    class_names = _class_names_for_tree(tree.classes_, data.label_map)
    viz_tree, y_viz = _make_viz_tree(tree, y_cluster)

    viz_model = dtreeviz.model(
        viz_tree,
        X_train=X_cluster,
        y_train=y_viz,
        feature_names=data.feature_names,
        target_name="class",
        class_names=class_names,
    )
    return viz_model.view(x=sample, show_just_path=False, fancy=True, fontname="Liberation Sans")


def save_svg(render: dtreeviz.DTreeVizRender, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    render.save(str(path.with_suffix(".svg")))
