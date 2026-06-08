"""Print a LaTeX table comparing per-class precision/recall: paper vs ours.

The "original" columns hold the per-class precision and recall reported in
"Achieving Explainability of Intrusion Detection System by Hybrid
Oracle-Explainer Approach" (Szczepanski et al., 2020):

    Table I   -> oracle (ANN + PCA)
    Table III -> explainer, k=0.2
    Table IV  -> explainer, k=0.005

The "ours" columns are read from the JSON reports written by ``experiments.py``
(``report.json``) as ``avg +- stdev`` over seeds.
Only runs at the default learning rate and ``val_loss`` early stopping are used,
so the learning-rate / early-stopping sweeps do not leak in.

Usage::

    python src/scripts/comparison_table.py [oracle | k=0.2 | k=0.005]
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_NAME = "report.json"
DEFAULT_LR = 1e-3
MONITOR = "val_acc"

# Per-class (precision, recall) as percentages, straight from the paper tables.
PAPER = {
    "oracle": {  # Table I
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
    "k=0.2": {  # Table III
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
    "k=0.005": {  # Table IV
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
}


def find_aggregate(model: str, classes: set[str]) -> dict:
    """Return the per-class aggregate for ``model`` from the matching report.

    Only the final ``report.json`` runs at the default learning rate with
    ``val_acc`` early stopping whose classes match the paper's (i.e. the
    CICIDS2017 dataset, not e.g. CSE-CIC-IDS-2018) are considered.
    """
    for report_dir in sorted(RESULTS_DIR.iterdir()):
        path = report_dir / REPORT_NAME
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        oracle_cfg = report["config"]["oracle"]
        if oracle_cfg["learning_rate"] != DEFAULT_LR:
            continue
        if oracle_cfg["early_stopping_monitor"] != MONITOR:
            continue
        agg = report["aggregate"]
        sub = agg["oracle"] if model == "oracle" else agg.get("explainers", {}).get(model)
        if sub is not None and classes <= {k.lower() for k in sub}:
            return sub
    raise SystemExit(f"no {REPORT_NAME} found with {model!r} at lr={DEFAULT_LR}, monitor={MONITOR}")


def num(value: float) -> str:
    """Format a 0..1 value as ``$0{,}xx$`` (comma decimal separator)."""
    return f"${value:.2f}$".replace(".", "{,}")


def fmt(stat: dict) -> str:
    """Format a ``{mean, std}`` aggregate as ``$mean \\pm std$``."""
    return f"${stat['mean']:.2f} \\pm {stat['std']:.2f}$".replace(".", "{,}")


def main(model: str) -> None:
    paper = PAPER[model]
    ours = find_aggregate(model, {cls.lower() for cls in paper})
    # Match our (possibly differently-cased) class names to the paper ones.
    ours_by_lower = {k.lower(): v for k, v in ours.items()}

    lines = [
        r"\begin{tabular}{|c|c|c|c|c|}",
        r"\hline",
        r"\multirow{2}{*}{\textbf{Klasa}} & \multicolumn{2}{c|}{\textbf{oryginał}} "
        r"& \multicolumn{2}{c|}{\textbf{nasz}} \\",
        r"\cline{2-5}",
        r" & Precyzja & Czułość & Precyzja & Czułość \\",
        r"\hline",
    ]
    for cls, (p, r) in paper.items():
        stat = ours_by_lower[cls.lower()]
        lines.append(
            f"{cls.replace('_', r'\\_')} & {num(p / 100)} & {num(r / 100)} "
            f"& {fmt(stat['precision'])} & {fmt(stat['recall'])} \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a paper-vs-ours per-class LaTeX table.")
    parser.add_argument(
        "model",
        nargs="?",
        default="oracle",
        choices=sorted(PAPER),
        help="Which model to compare (default: oracle).",
    )
    args = parser.parse_args()
    main(args.model)
