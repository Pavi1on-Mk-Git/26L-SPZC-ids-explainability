"""Collect oracle results across experiments and print a LaTeX table to stdout.

Scans the ``results/`` directory for the JSON reports written by
``experiments.py`` (``report.json`` or ``results_oracle.json``) and emits one
table row per learning rate with accuracy and macro-F1 as ``avg +- stdev``.
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_NAMES = ("report.json", "results_oracle.json")
MONITOR = "val_loss"


def collect_rows() -> list[tuple[float, dict, dict]]:
    rows: list[tuple[float, dict, dict]] = []
    seen: set[float] = set()
    for report_dir in sorted(RESULTS_DIR.iterdir()):
        for name in REPORT_NAMES:
            path = report_dir / name
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            oracle_cfg = report["config"]["oracle"]
            if oracle_cfg["early_stopping_monitor"] != MONITOR:
                continue
            lr = oracle_cfg["learning_rate"]
            if lr in seen:
                continue
            seen.add(lr)
            oracle = report["aggregate"]["oracle"]
            rows.append((lr, oracle["accuracy"], oracle["macro avg"]["f1-score"]))
            break
    rows.sort(key=lambda r: r[0])
    return rows


def fmt(stat: dict) -> str:
    return f"{stat['mean']:.3f} $\\pm$ {stat['std']:.3f}"


def main() -> None:
    rows = collect_rows()

    lines = [
        r"\begin{tabular}{ccc}",
        r"\hline",
        r"learning rate & accuracy & macro-F1 \\",
        r"\hline",
    ]
    for lr, acc, f1 in rows:
        lines.append(f"{lr:g} & {fmt(acc)} & {fmt(f1)} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
