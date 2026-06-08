import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_NAME = "report.json"
DEFAULT_LR = 1e-3
MONITOR = "val_acc"

PAPER = {
    "oracle": {
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
    "k=0.2": {
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
    "k=0.005": {
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
    return f"${value:.2f}$".replace(".", "{,}")


def fmt(stat: dict) -> str:
    return f"${stat['mean']:.2f} \\pm {stat['std']:.2f}$".replace(".", "{,}")


def main(model: str) -> None:
    paper = PAPER[model]
    ours = find_aggregate(model, {cls.lower() for cls in paper})
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
