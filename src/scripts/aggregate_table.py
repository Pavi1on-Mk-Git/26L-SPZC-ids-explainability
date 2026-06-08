"""Print a LaTeX table of overall accuracy and macro-F1 for a chosen model.

Three rows are emitted:

    CIC IDS 2017 (oryginał) -- the paper's reported accuracy, with macro-F1
        computed from the per-class precision/recall of its tables.
    CIC IDS 2017 (nasz)     -- our results on CICIDS2017.
    CSE-CIC-IDS 2018 (nasz) -- our results on CSE-CIC-IDS-2018.

The model can be the oracle (Table I / Table II oracle accuracy) or one of the
explainers (Tables III/IV, with the explainer accuracy from Table II). Our rows
are read from the ``report.json`` files written by ``experiments.py`` as
``avg +- stdev`` over seeds, at the default learning rate with ``val_acc`` early
stopping.

Usage::

    python src/scripts/aggregate_table.py [oracle | k=0.2 | k=0.005]
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_NAME = "report.json"
DEFAULT_LR = 1e-3
MONITOR = "val_acc"

# Per-class (precision, recall) as percentages, straight from the paper tables,
# and the overall accuracy the paper reports for each model (Tables I-IV).
PAPER = {
    "oracle": {  # Table I, accuracy from Table II
        "accuracy": 0.98,
        "classes": {
            "Benign": (99, 98),
            "DDoS": (100, 98),
            "DoS GoldenEye": (96, 99),
            "DoS Hulk": (89, 96),
            "DoS Slowhttptest": (87, 99),
            "DoS Slowloris": (97, 97),
            "FTP-Patator": (91, 98),
            "PortScan": (88, 97),
            "SSH-Patator": (100, 51),
        },
    },
    "k=0.2": {  # Table III, accuracy from Table II
        "accuracy": 0.95,
        "classes": {
            "Benign": (98, 98),
            "DDoS": (82, 76),
            "DoS GoldenEye": (53, 19),
            "DoS Hulk": (81, 91),
            "DoS Slowhttptest": (23, 19),
            "DoS Slowloris": (0, 0),
            "FTP-Patator": (15, 35),
            "PortScan": (99, 99),
            "SSH-Patator": (0, 0),
        },
    },
    "k=0.005": {  # Table IV, accuracy from Table II
        "accuracy": 0.99,
        "classes": {
            "Benign": (99, 99),
            "DDoS": (99, 99),
            "DoS GoldenEye": (93, 87),
            "DoS Hulk": (96, 98),
            "DoS Slowhttptest": (93, 94),
            "DoS Slowloris": (58, 66),
            "FTP-Patator": (91, 93),
            "PortScan": (99, 99),
            "SSH-Patator": (94, 97),
        },
    },
}


def paper_macro_f1(model: str) -> float:
    classes = PAPER[model]["classes"].values()
    f1s = [2 * p * r / (p + r) if p + r else 0.0 for p, r in classes]
    return sum(f1s) / len(f1s) / 100


def find_aggregate(dataset_name: str, model: str) -> dict:
    """Return the ``model`` aggregate for ``dataset_name`` from the final report.

    Only the ``report.json`` runs at the default learning rate with ``val_acc``
    early stopping are considered.
    """
    for report_dir in sorted(RESULTS_DIR.iterdir()):
        path = report_dir / REPORT_NAME
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report["config"]["data"].get("dataset_name") != dataset_name:
            continue
        oracle_cfg = report["config"]["oracle"]
        if oracle_cfg["learning_rate"] != DEFAULT_LR:
            continue
        if oracle_cfg["early_stopping_monitor"] != MONITOR:
            continue
        agg = report["aggregate"]
        sub = agg["oracle"] if model == "oracle" else agg.get("explainers", {}).get(model)
        if sub is not None:
            return sub
    raise SystemExit(f"no {dataset_name} {REPORT_NAME} found with {model!r} at lr={DEFAULT_LR}, monitor={MONITOR}")


def num(value: float) -> str:
    """Format a 0..1 value as ``$0{,}xx$`` (comma decimal separator)."""
    return f"${value:.2f}$".replace(".", "{,}")


def fmt(stat: dict) -> str:
    """Format a ``{mean, std}`` aggregate as ``$mean \\pm std$``."""
    return f"${stat['mean']:.2f} \\pm {stat['std']:.2f}$".replace(".", "{,}")


def main(model: str) -> None:
    ours_2017 = find_aggregate("CICIDS2017", model)
    ours_2018 = find_aggregate("CSE-CIC-IDS-2018", model)

    rows = [
        ("Oryginalny artykuł", "CIC IDS 2017", num(PAPER[model]["accuracy"]), num(paper_macro_f1(model))),
        ("Nasze eksperymenty", "CIC IDS 2017", fmt(ours_2017["accuracy"]), fmt(ours_2017["macro avg"]["f1-score"])),
        ("Nasze eksperymenty", "CSE-CIC-IDS 2018", fmt(ours_2018["accuracy"]), fmt(ours_2018["macro avg"]["f1-score"])),
    ]

    lines = [
        r"\begin{tabular}{|c|c|c|c|}",
        r"\hline",
        r"\textbf{Źródło} & \textbf{Zbiór danych} & \textbf{Dokładność} & \textbf{Miara F1} \\",
        r"\hline",
    ]
    for source, dataset, acc, f1 in rows:
        lines.append(f"{source} & {dataset} & {acc} & {f1} \\\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print an accuracy/macro-F1 summary LaTeX table.")
    parser.add_argument(
        "model",
        nargs="?",
        default="oracle",
        choices=sorted(PAPER),
        help="Which model to summarise (default: oracle).",
    )
    args = parser.parse_args()
    main(args.model)
