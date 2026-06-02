import numpy as np
from sklearn.metrics import classification_report

from ids_explain.config import DataConfig, ExplainerConfig, OracleConfig
from ids_explain.data_loader import load_dataset
from ids_explain.explainer import (
    Explainer,
    load_explainer,
    predict_explainer,
    save_explainer,
    train_explainer,
)
from ids_explain.oracle import OracleMLP, load_oracle, predict, save_oracle, train_oracle
from ids_explain.preprocessing import ProcessedData, load_processed, preprocess, save_processed


def _load_or_preprocess(data_cfg: DataConfig) -> ProcessedData:
    processed_path = data_cfg.processed_data_dir / data_cfg.processed_data_name
    if processed_path.exists():
        print("[data] Loading preprocessed data from disk.")
        return load_processed(data_cfg)
    print("[data] Preprocessed data not found — loading raw CSVs and preprocessing.")
    split = load_dataset(data_cfg)
    processed = preprocess(split, data_cfg)
    save_processed(processed, data_cfg)
    return processed


def _load_or_train_oracle(
    data: ProcessedData,
    data_cfg: DataConfig,
    oracle_cfg: OracleConfig,
) -> OracleMLP:
    oracle_path = data_cfg.processed_data_dir / oracle_cfg.model_name
    if oracle_path.exists():
        print("[oracle] Loading trained oracle from disk.")
        return load_oracle(data_cfg, oracle_cfg)
    print("[oracle] Trained oracle not found — training from scratch.")
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
        print(f"[explainer k={explainer_cfg.k_frac}] Loading trained explainer from disk.")
        return load_explainer(data_cfg, explainer_cfg)
    print(f"[explainer k={explainer_cfg.k_frac}] Trained explainer not found — training from scratch.")
    explainer = train_explainer(data, data_cfg, explainer_cfg)
    save_explainer(explainer, data_cfg, explainer_cfg)
    return explainer


def _report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_map: dict[str, int],
    title: str,
) -> None:
    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map[i] for i in sorted(inv_label_map)]
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=2))


def main():
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
    explainer_cfg_coarse = ExplainerConfig(k_frac=0.2, tree_max_depth=4, n_search=3)
    explainer_cfg_fine = ExplainerConfig(k_frac=0.005, tree_max_depth=4, n_search=3)

    for random_seed in range(5):
        data_cfg = DataConfig(
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
            random_seed=random_seed,
        )

        data = _load_or_preprocess(data_cfg)
        print(
            f"[data] train={len(data.y_train):,}  test={len(data.y_test):,}  "
            f"features(raw)={data.X_train_raw.shape[1]}  features(pca)={data.X_train_pca.shape[1]}  "
            f"classes={len(data.label_map)} seed={random_seed}"
        )

        mlp = _load_or_train_oracle(data, data_cfg, oracle_cfg)
        oracle_preds = predict(mlp, data.X_test_pca, batch_size=oracle_cfg.batch_size)
        _report(data.y_test, oracle_preds, data.label_map, "Oracle (ANN + PCA)  —  Table I")

        for cfg, table in [
            (explainer_cfg_coarse, "Table III"),
            (explainer_cfg_fine, "Table IV"),
        ]:
            explainer = _load_or_train_explainer(data, data_cfg, cfg)
            explainer_preds = predict_explainer(explainer, data.X_test_raw, oracle_preds, n_search=cfg.n_search)
            _report(
                data.y_test,
                explainer_preds,
                data.label_map,
                f"Explainer  k={cfg.k_frac}  —  {table}",
            )

            n_clusters = len(explainer.trees)
            k_int = max(1, round(cfg.k_frac * len(data.X_train_raw)))
            agreement = (explainer_preds == oracle_preds).mean()
            print(f"  clusters={n_clusters}  k={k_int:,}  oracle-agreement={agreement:.2%}")


if __name__ == "__main__":
    main()
