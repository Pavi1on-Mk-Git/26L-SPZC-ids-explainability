import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report

from ids_explain.config import CICIDS2017_CONFIG, DataConfig, ExplainerConfig, OracleConfig, config_hash, get_dirs
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
ORACLE_REPORT_NAME = "results_oracle.json"


def _load_or_preprocess(
    data_cfg: DataConfig,
    raw_data_dir: Path,
    processed_data_dir: Path,
    save_processed_data: bool = True,
) -> ProcessedData:
    processed_path = processed_data_dir / data_cfg.processed_data_name
    if processed_path.exists():
        print("[data] loading preprocessed data from disk")
        return load_processed(data_cfg, processed_data_dir)
    print(f"[data] preprocessing raw CSVs ({'caching to disk' if save_processed_data else 'in-memory'})")
    split = load_dataset(data_cfg, raw_data_dir)
    return preprocess(split, data_cfg, processed_data_dir, save=save_processed_data)


def _load_or_train_oracle(
    data: ProcessedData,
    data_cfg: DataConfig,
    oracle_cfg: OracleConfig,
    processed_data_dir: Path,
) -> OracleMLP:
    oracle_path = processed_data_dir / oracle_cfg.model_name
    if oracle_path.exists():
        print("[oracle] loading trained oracle from disk")
        return load_oracle(data_cfg, oracle_cfg, processed_data_dir)
    print("[oracle] training from scratch")
    mlp = train_oracle(data, data_cfg, oracle_cfg, processed_data_dir)
    save_oracle(mlp, oracle_cfg, processed_data_dir)
    return mlp


def _load_or_train_explainer(
    data: ProcessedData,
    data_cfg: DataConfig,
    explainer_cfg: ExplainerConfig,
    processed_data_dir: Path,
) -> Explainer:
    explainer_dir = processed_data_dir / explainer_cfg.data_dir_name
    if explainer_dir.exists():
        print(f"[explainer k_frac={explainer_cfg.k_frac}] loading from disk")
        return load_explainer(explainer_cfg, processed_data_dir)
    print(f"[explainer k_frac={explainer_cfg.k_frac}] training from scratch")
    explainer = train_explainer(data, data_cfg, explainer_cfg)
    save_explainer(explainer, explainer_cfg, processed_data_dir)
    return explainer


def _report(y_true: np.ndarray, y_pred: np.ndarray, label_map: dict[str, int]) -> dict:
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in sorted(inv_label_map)]
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    for value in report.values():
        if isinstance(value, dict):
            value.pop("support", None)
    return report


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
            aggregate[key] = row
        else:  # accuracy is a scalar
            vals = [r[key] for r in reports]
            aggregate[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return aggregate


def main(
    save_processed_data: bool = True,
    include_explainers: bool = True,
    learning_rate: float = 1e-3,
    early_stopping_monitor: str = "val_loss",
):
    n_seeds = 5
    base_cfg = CICIDS2017_CONFIG
    oracle_cfg = OracleConfig(
        hidden_dim=512,
        n_layers=5,
        dropout=0.2,
        learning_rate=learning_rate,
        batch_size=10_000,
        max_epochs=100,
        val_size=0.25,
        early_stopping_patience=5,
        early_stopping_monitor=early_stopping_monitor,
    )
    explainers = [
        ExplainerConfig(k_frac=0.2, tree_max_depth=4, n_search=3),
        ExplainerConfig(k_frac=0.005, tree_max_depth=4, n_search=3),
    ]

    experiment_hash = config_hash(base_cfg, oracle_cfg, explainers[0])

    runs: list[dict] = []
    oracle_reports: list[dict] = []
    explainer_reports: dict[str, list[dict]] = {}

    for random_seed in range(n_seeds):
        print(f"\n=== seed {random_seed + 1}/{n_seeds} ===")
        data_cfg = replace(base_cfg, random_seed=random_seed)
        raw_data_dir, processed_data_dir = get_dirs(data_cfg, oracle_cfg, explainers[0])

        data = _load_or_preprocess(data_cfg, raw_data_dir, processed_data_dir, save_processed_data=save_processed_data)

        mlp = _load_or_train_oracle(data, data_cfg, oracle_cfg, processed_data_dir)
        oracle_preds = predict(mlp, data.X_test_pca, batch_size=oracle_cfg.batch_size)
        oracle_report = _report(data.y_test, oracle_preds, data.label_map)
        oracle_reports.append(oracle_report)

        run = {
            "seed": random_seed,
            "data": {
                "train": len(data.y_train),
                "test": len(data.y_test),
                "features_raw": data.X_train_raw.shape[1],
                "features_pca": data.X_train_pca.shape[1],
                "classes": len(data.label_map),
            },
            "oracle": oracle_report,
        }

        if include_explainers:
            run["explainers"] = {}
            for explainer_cfg in explainers:
                key = f"k={explainer_cfg.k_frac}"
                explainer = _load_or_train_explainer(data, data_cfg, explainer_cfg, processed_data_dir)
                explainer_preds = predict_explainer(
                    explainer, data.X_test_raw, oracle_preds, n_search=explainer_cfg.n_search
                )
                report = _report(data.y_test, explainer_preds, data.label_map)
                explainer_reports.setdefault(key, []).append(report)
                run["explainers"][key] = {
                    "k_frac": explainer_cfg.k_frac,
                    "clusters": len(explainer.trees),
                    "k": max(1, round(explainer_cfg.k_frac * len(data.X_train_raw))),
                    "oracle_agreement": float((explainer_preds == oracle_preds).mean()),
                    "report": report,
                }

        runs.append(run)

    config_block: dict = {"data": base_cfg.to_dict(), "oracle": asdict(oracle_cfg)}
    aggregate: dict = {"oracle": _aggregate(oracle_reports)}
    if include_explainers:
        config_block["explainers"] = {
            f"k={explainer_cfg.k_frac}": asdict(explainer_cfg) for explainer_cfg in explainers
        }
        aggregate["explainers"] = {key: _aggregate(reports) for key, reports in explainer_reports.items()}

    report = {
        "config_hash": experiment_hash,
        "seeds": list(range(n_seeds)),
        "config": config_block,
        "runs": runs,
        "aggregate": aggregate,
    }

    results_dir = RESULTS_DIR / experiment_hash
    results_dir.mkdir(parents=True, exist_ok=True)
    report_name = REPORT_NAME if include_explainers else ORACLE_REPORT_NAME
    out_path = results_dir / report_name
    out_path.write_text(json.dumps(report, indent=4))
    print(f"\n[results] wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the oracle/explainer reproduction experiments.")
    parser.add_argument(
        "--no-save-processed",
        action="store_true",
        help="Do not write the processed dataset to disk; preprocess in-memory each run instead.",
    )
    parser.add_argument(
        "--oracle-only",
        action="store_true",
        help="Only train and evaluate the oracle; skip explainer training and evaluation.",
    )
    parser.add_argument("--learning-rate", type=float, help="Learning rate used for training the oracle.")
    parser.add_argument(
        "--early-stopping-monitor",
        type=str,
        help="Metric to monitor when deciding whether to stop the oracle training early.",
        choices=["val_loss", "val_acc", "train_loss", "train_acc"],
    )
    args = parser.parse_args()
    main(
        save_processed_data=not args.no_save_processed,
        include_explainers=not args.oracle_only,
        learning_rate=args.learning_rate,
        early_stopping_monitor=args.early_stopping_monitor,
    )
