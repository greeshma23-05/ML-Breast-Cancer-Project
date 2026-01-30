from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.evaluation.reject_option import evaluate_reject_option


repo_root = Path(__file__).resolve().parents[2]
processed_path = repo_root / "data" / "processed" / "dataset.parquet"
reports_dir = repo_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(processed_path)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy().astype(int).ravel()

    base_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    iso = CalibratedClassifierCV(
        estimator=base_lr, cv=cv, method="isotonic"
    )

    # OOF calibrated probabilities
    p_iso = cross_val_predict(
        iso, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    results = []
    thresholds = np.linspace(0.5, 0.95, 10)

    for t in thresholds:
        m = evaluate_reject_option(y, p_iso, confidence_threshold=t)
        results.append(m.__dict__)

    out_path = reports_dir / "reject_option.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("Reject option analysis done.")
    for r in results:
        print(
            f"conf>={r['threshold']:.2f} | "
            f"coverage={r['coverage']:.2f} | "
            f"error={r['error_rate']:.3f} | "
            f"FN rate={r['false_negative_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
