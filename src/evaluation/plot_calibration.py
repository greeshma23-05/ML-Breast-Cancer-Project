from __future__ import annotations

from pathlib import Path
import json
import matplotlib.pyplot as plt


repo_root = Path(__file__).resolve().parents[2]
reports_dir = repo_root / "reports"
figures_dir = reports_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def plot_reliability(name: str, calib: dict) -> None:
    acc = calib["bin_acc"]
    conf = calib["bin_conf"]

    plt.plot(conf, acc, marker="o", label=name)


def main() -> None:
    path = reports_dir / "calibration_cv.json"
    with path.open() as f:
        data = json.load(f)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    for name, obj in data.items():
        plot_reliability(name, obj["calibration"])

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability Diagram (5-fold CV, OOF)")
    plt.legend()
    plt.tight_layout()

    out_path = figures_dir / "calibration_curve.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved calibration plot to: {out_path}")


if __name__ == "__main__":
    main()
