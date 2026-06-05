import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from ids_explain.config import (
    CICIDS2017_CONFIG,
    DataConfig,
    ExplainerConfig,
    OracleConfig,
)
from ids_explain.data_loader import load_dataset
from ids_explain.explainer import (
    Explainer,
    load_explainer,
    predict_explainer,
    save_explainer,
    train_explainer,
)
from ids_explain.oracle import OracleMLP, load_oracle, predict, save_oracle, train_oracle
from ids_explain.preprocessing import ProcessedData, load_processed, preprocess

RESULTS_DIR = Path("results")
REPORT_NAME = "report.json"


def _load_or_preprocess(data_cfg: DataConfig, save_processed_data: bool = True) -> ProcessedData:
    processed_path = data_cfg.processed_data_dir / data_cfg.processed_data_name
    if processed_path.exists():
        return load_processed(data_cfg)
    split = load_dataset(data_cfg)
    return preprocess(split, data_cfg, save=save_processed_data)


def _load_or_train_oracle(
    data: ProcessedData,
    data_cfg: DataConfig,
    oracle_cfg: OracleConfig,
) -> OracleMLP:
    oracle_path = data_cfg.processed_data_dir / oracle_cfg.model_name
    if oracle_path.exists():
        return load_oracle(data_cfg, oracle_cfg)
    mlp = train_oracle(data, data_cfg, oracle_cfg)
    save_oracle(mlp, data_cfg, oracle_cfg)
    return mlp


def _load_or_train_explainer(
    data: ProcessedData,
    data_cfg: DataConfig,
    explainer_cfg: ExplainerConfig,
) -> Explainer:
    explainer_dir = data_cfg.processed_data_dir / explainer_cfg.data_dir_name
    if explainer_dir.exists():
        return load_explainer(data_cfg, explainer_cfg)
    explainer = train_explainer(data, data_cfg, explainer_cfg)
    save_explainer(explainer, data_cfg, explainer_cfg)
    return explainer


def _report(y_true: np.ndarray, y_pred: np.ndarray, label_map: dict[str, int]) -> dict:
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in sorted(inv_label_map)]
    return classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)


def _aggregate(reports: list[dict]) -> dict:
    metrics = ["precision", "recall", "f1-score"]
    aggregate: dict = {}
    for key in reports[0]:
        first = reports[0][key]
        if isinstance(first, dict):
            row: dict = {}
            for m in metrics:
                vals = [r[key][m] for r in reports]
                row[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            row["support"] = float(np.mean([r[key]["support"] for r in reports]))
            aggregate[key] = row
        else:  # accuracy is a scalar
            vals = [r[key] for r in reports]
            aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return aggregate


def main(save_processed_data: bool = True):
    n_seeds = 5
    base_cfg = CICIDS2017_CONFIG
    oracle_cfg = OracleConfig(
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
    explainer_specs = [
        (ExplainerConfig(k_frac=0.2, tree_max_depth=4, n_search=3), "Table III"),
        (ExplainerConfig(k_frac=0.005, tree_max_depth=4, n_search=3), "Table IV"),
    ]

    runs: list[dict] = []
    # Per-table classification reports across seeds, for aggregation.
    collected: dict[str, list[dict]] = {}

    for random_seed in range(n_seeds):
        data_cfg = replace(base_cfg, random_seed=random_seed)
        data = _load_or_preprocess(data_cfg, save_processed_data=save_processed_data)

        mlp = _load_or_train_oracle(data, data_cfg, oracle_cfg)
        oracle_preds = predict(mlp, data.X_test_pca, batch_size=oracle_cfg.batch_size)
        oracle_title = "Oracle (ANN + PCA)  —  Table I"
        oracle_report = _report(data.y_test, oracle_preds, data.label_map)
        collected.setdefault(oracle_title, []).append(oracle_report)

        explainer_results = []
        for explainer_cfg, table in explainer_specs:
            explainer = _load_or_train_explainer(data, data_cfg, explainer_cfg)
            explainer_preds = predict_explainer(
                explainer, data.X_test_raw, oracle_preds, n_search=explainer_cfg.n_search
            )
            title = f"Explainer  k={explainer_cfg.k_frac}  —  {table}"
            report = _report(data.y_test, explainer_preds, data.label_map)
            collected.setdefault(title, []).append(report)
            explainer_results.append(
                {
                    "table": table,
                    "k_frac": explainer_cfg.k_frac,
                    "clusters": len(explainer.trees),
                    "k": max(1, round(explainer_cfg.k_frac * len(data.X_train_raw))),
                    "oracle_agreement": float((explainer_preds == oracle_preds).mean()),
                    "report": report,
                }
            )

        runs.append(
            {
                "seed": random_seed,
                "data": {
                    "train": len(data.y_train),
                    "test": len(data.y_test),
                    "features_raw": data.X_train_raw.shape[1],
                    "features_pca": data.X_train_pca.shape[1],
                    "classes": len(data.label_map),
                },
                "oracle": {"table": "Table I", "report": oracle_report},
                "explainers": explainer_results,
            }
        )

    report = {
        "config_hash": base_cfg.config_hash(),
        "seeds": list(range(n_seeds)),
        "config": {
            "data": base_cfg.to_dict(),
            "oracle": asdict(oracle_cfg),
            "explainers": [asdict(cfg) for cfg, _ in explainer_specs],
        },
        "runs": runs,
        "aggregate": {title: _aggregate(reports) for title, reports in collected.items()},
    }

    results_dir = RESULTS_DIR / base_cfg.config_hash()
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / REPORT_NAME).write_text(json.dumps(report, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the oracle/explainer reproduction experiments.")
    parser.add_argument(
        "--no-save-processed",
        action="store_true",
        help="Do not write the processed dataset (data.npz/metadata.json) to disk; "
        "preprocess in-memory each run instead.",
    )
    args = parser.parse_args()
    main(save_processed_data=not args.no_save_processed)
