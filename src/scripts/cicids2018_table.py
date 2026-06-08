"""Print a LaTeX table of per-class precision/recall on CSE-CIC-IDS-2018.

Analogous to ``comparison_table.py`` but for the CSE-CIC-IDS-2018 dataset, which
the paper does not report, so there are no "original" columns -- only our own
results read from the JSON reports written by ``experiments.py``
(``report.json``) as ``avg +- stdev`` over seeds.

Usage::

    python src/scripts/cicids2018_table.py [oracle | k=0.2 | k=0.005]
"""

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_NAME = "report.json"
DATASET_NAME = "CSE-CIC-IDS-2018"
DEFAULT_LR = 1e-3
MONITOR = "val_acc"
MODELS = ("oracle", "k=0.2", "k=0.005")

METRIC_KEYS = ("accuracy", "macro avg", "weighted avg")


def find_aggregate(model: str) -> dict:
    """Return the per-class aggregate for ``model`` on the CSE-CIC-IDS-2018 set."""
    for report_dir in sorted(RESULTS_DIR.iterdir()):
        path = report_dir / REPORT_NAME
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        if report["config"]["data"].get("dataset_name") != DATASET_NAME:
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
    raise SystemExit(f"no {DATASET_NAME} {REPORT_NAME} found with {model!r} at lr={DEFAULT_LR}, monitor={MONITOR}")


def fmt(stat: dict) -> str:
    """Format a ``{mean, std}`` aggregate as ``$mean \\pm std$``."""
    return f"${stat['mean']:.3f} \\pm {stat['std']:.3f}$".replace(".", "{,}")


def main(model: str) -> None:
    ours = find_aggregate(model)

    lines = [
        r"\begin{tabular}{|c|c|c|}",
        r"\hline",
        r"\textbf{Klasa} & Precyzja & Czułość \\",
        r"\hline",
    ]
    for cls, stat in ours.items():
        if cls in METRIC_KEYS:
            continue
        lines.append(
            f"{cls.replace('_', r'\\_')} & {fmt(stat['precision'])} & {fmt(stat['recall'])} \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a CSE-CIC-IDS-2018 per-class LaTeX table.")
    parser.add_argument(
        "model",
        nargs="?",
        default="oracle",
        choices=MODELS,
        help="Which model to report (default: oracle).",
    )
    args = parser.parse_args()
    main(args.model)
