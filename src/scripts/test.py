from pathlib import Path

import numpy as np

from ids_explain.config import CICIDS2017_CONFIG, DEFAULT_EXPLAINER_CONFIG, DEFAULT_ORACLE_CONFIG, get_dirs
from ids_explain.data_loader import load_dataset
from ids_explain.explainer import Explainer, load_explainer, predict_explainer, save_explainer, train_explainer
from ids_explain.oracle import OracleMLP, load_oracle, predict, save_oracle, train_oracle
from ids_explain.preprocessing import ProcessedData, load_processed, preprocess
from ids_explain.visualize import explain_sample, save_svg


def test(processed: ProcessedData, oracle: OracleMLP, explainer: Explainer):
    oracle_preds = predict(oracle, processed.X_test_pca, 10_000)

    explainer_preds = predict_explainer(
        explainer, processed.X_test_raw, oracle_preds, DEFAULT_EXPLAINER_CONFIG.n_search
    )

    print(f"Oracle accuracy: {np.mean(oracle_preds == processed.y_test)}")
    print(f"Explainer accuracy: {np.mean(explainer_preds == processed.y_test)}")
    print(f"Comparative accuracy: {np.mean(oracle_preds == explainer_preds)}")

    sample_idx = 2137

    tree = explain_sample(
        processed.X_test_raw[sample_idx],
        oracle_preds[sample_idx],
        explainer,
        processed,
        DEFAULT_EXPLAINER_CONFIG.n_search,
    )

    save_svg(tree, Path.cwd() / "test")


def full_main():
    raw_data_dir, processed_data_dir = get_dirs(CICIDS2017_CONFIG, DEFAULT_ORACLE_CONFIG, DEFAULT_EXPLAINER_CONFIG)

    dataset = load_dataset(CICIDS2017_CONFIG, raw_data_dir)
    processed = preprocess(dataset, CICIDS2017_CONFIG, processed_data_dir)

    oracle = train_oracle(processed, CICIDS2017_CONFIG, DEFAULT_ORACLE_CONFIG, processed_data_dir)

    save_oracle(oracle, DEFAULT_ORACLE_CONFIG, processed_data_dir)

    explainer = train_explainer(processed, CICIDS2017_CONFIG, DEFAULT_EXPLAINER_CONFIG)

    save_explainer(explainer, DEFAULT_EXPLAINER_CONFIG, processed_data_dir)

    test(processed, oracle, explainer)


def loaded_main():
    raw_data_dir, processed_data_dir = get_dirs(CICIDS2017_CONFIG, DEFAULT_ORACLE_CONFIG, DEFAULT_EXPLAINER_CONFIG)
    processed = load_processed(CICIDS2017_CONFIG, processed_data_dir)
    oracle = load_oracle(CICIDS2017_CONFIG, DEFAULT_ORACLE_CONFIG, processed_data_dir)
    explainer = load_explainer(DEFAULT_EXPLAINER_CONFIG, processed_data_dir)
    test(processed, oracle, explainer)


if __name__ == "__main__":
    full_main()
