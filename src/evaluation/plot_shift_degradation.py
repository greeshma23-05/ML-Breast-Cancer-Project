from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib.pyplot as plt


repo_root = Path(__file__).resolve().parents[2]
reports_dir = repo_root / "reports"
figures_dir = reports_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    path = reports_dir / "shift_test_results.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.models.run_shift_tests"
        )

    with path.open() as f:
        rows = json.load(f)

    # Extract baseline
    baseline = next(r for r in rows if r["name"] == "baseline_test")
    noise_rows = [r for r in rows if r["name"].startswith("noise_sigma_")]

    def sigma_from_name(name: str) -> float:
        m = re.search(r"noise_sigma_(\d+\.\d+)", name)
        return float(m.group(1)) if m else float("nan")

    noise_rows = sorted(noise_rows, key=lambda r: sigma_from_name(r["name"]))

    sigmas = [sigma_from_name(r["name"]) for r in noise_rows]
    roc = [r["roc_auc"] for r in noise_rows]
    pr = [r["pr_auc"] for r in noise_rows]
    sens = [r["sensitivity"] for r in noise_rows]
    spec = [r["specificity"] for r in noise_rows]

    plt.figure(figsize=(7, 5))
    plt.plot(sigmas, roc, marker="o", label="ROC-AUC")
    plt.plot(sigmas, pr, marker="o", label="PR-AUC")
    plt.plot(sigmas, sens, marker="o", label="Sensitivity")
    plt.plot(sigmas, spec, marker="o", label="Specificity")

    # Baseline reference lines
    plt.axhline(baseline["roc_auc"], linestyle="--", linewidth=1, label="Baseline ROC-AUC")
    plt.axhline(baseline["pr_auc"], linestyle="--", linewidth=1, label="Baseline PR-AUC")

    plt.xlabel("Gaussian noise sigma (added to all features)")
    plt.ylabel("Metric value")
    plt.title("Performance Degradation Under Measurement Noise")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()

    out_path = figures_dir / "shift_noise_degradation.png"
    plt.savefig(out_path, dpi=160)
    print(f"Saved plot to: {out_path}")


if __name__ == "__main__":
    main()
