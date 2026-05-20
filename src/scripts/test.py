import numpy as np

from ids_explain.config import CICIDS2017_CONFIG, DEFAULT_ORACLE_CONFIG
from ids_explain.data_loader import load_dataset
from ids_explain.oracle import predict, train_oracle
from ids_explain.preprocessing import preprocess


def main():
    dataset = load_dataset(CICIDS2017_CONFIG)
    processed = preprocess(dataset, CICIDS2017_CONFIG)
    oracle = train_oracle(processed, CICIDS2017_CONFIG, DEFAULT_ORACLE_CONFIG)
    preds = predict(oracle, processed.X_test_pca, 10_000)

    print(f"Accuracy: {np.mean(preds == processed.y_test)}")


if __name__ == "__main__":
    main()
