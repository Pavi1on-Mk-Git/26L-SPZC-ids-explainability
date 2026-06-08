import json
from pathlib import Path

RESULTS_DIR = Path("results")
REPORT_NAMES = ("report.json", "results_oracle.json")
DEFAULT_LR = 1e-3


def collect_rows() -> list[tuple[str, dict, dict]]:
    rows: list[tuple[str, dict, dict]] = []
    seen: set[str] = set()
    for report_dir in sorted(RESULTS_DIR.iterdir()):
        for name in REPORT_NAMES:
            path = report_dir / name
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            oracle_cfg = report["config"]["oracle"]
            if oracle_cfg["learning_rate"] != DEFAULT_LR:
                continue
            monitor = oracle_cfg["early_stopping_monitor"]
            if monitor in seen:
                continue
            seen.add(monitor)
            oracle = report["aggregate"]["oracle"]
            rows.append((monitor, oracle["accuracy"], oracle["macro avg"]["f1-score"]))
            break
    order = {"val_loss": 0, "train_loss": 1, "val_acc": 2, "train_acc": 3}
    rows.sort(key=lambda r: order.get(r[0], len(order)))
    return rows


def fmt(stat: dict) -> str:
    return f"${stat['mean']:.3f} \\pm {stat['std']:.3f}$".replace(".", "{,}")


def main() -> None:
    rows = collect_rows()

    lines = [
        r"\begin{tabular}{|c|c|c|}",
        r"\hline",
        r"early stopping & accuracy & macro-F1 \\",
        r"\hline",
    ]
    for monitor, acc, f1 in rows:
        lines.append(f"{monitor.replace('_', r'\_')} & {fmt(acc)} & {fmt(f1)} \\\\")
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
