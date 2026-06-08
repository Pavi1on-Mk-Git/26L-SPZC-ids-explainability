TABLES = {
    "Oracle (Table I)": {
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
    "Explainer k=0.2 (Table III)": {
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
    "Explainer k=0.005 (Table IV)": {
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


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main() -> None:
    for table_name, classes in TABLES.items():
        print(table_name)
        f1s = []
        for cls, (p, r) in classes.items():
            score = f1(p, r)
            f1s.append(score)
            print(f"  {cls:<18} P={p:3d}%  R={r:3d}%  F1={score:6.2f}%")
        macro = sum(f1s) / len(f1s)
        print(f"  {'macro-F1':<18} = {macro:.2f}%\n")


if __name__ == "__main__":
    main()
