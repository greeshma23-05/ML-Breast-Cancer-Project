from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from src.evaluation.metrics import evaluate_binary
from src.evaluation.shift_tests import add_gaussian_noise, drift_feature, subpopulation_slice


repo_root = Path(__file__).resolve().parents[2]
processed_path = repo_root / "data" / "processed" / "dataset.parquet"
reports_dir = repo_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(processed_path)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy().astype(int).ravel()

    # Hold-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    base_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000)),
        ]
    )

    # Fit isotonic calibration using CV
    calibrated = CalibratedClassifierCV(
        estimator=base_lr,
        method="isotonic",
        cv=5
    )
    calibrated.fit(X_train, y_train)

    def eval_on(X_eval, y_eval, name: str):
        p = calibrated.predict_proba(X_eval)[:, 1]
        m = evaluate_binary(y_eval, p, threshold=0.5).__dict__
        return {"name": name, **m}

    results = []

    #Baseline (no shift)
    results.append(eval_on(X_test, y_test, "baseline_test"))

    #Noise injection
    for sigma in [0.05, 0.10, 0.20]:
        Xn = add_gaussian_noise(X_test, sigma=sigma, seed=42)
        results.append(eval_on(Xn, y_test, f"noise_sigma_{sigma:.2f}"))

    #Feature drift
    drift_feat = "radius_mean" if "radius_mean" in X.columns else X.columns[0]
    for shift_std in [0.5, 1.0, 2.0]:
        Xd = drift_feature(X_test, feature=drift_feat, shift_in_std=shift_std)
        results.append(eval_on(Xd, y_test, f"drift_{drift_feat}_plus_{shift_std:.1f}std"))

    #Subpopulation slice
    slice_feat = "concave_points_mean" if "concave_points_mean" in X.columns else X.columns[1]
    X_slice, y_slice, thr = subpopulation_slice(X_test, y_test, feature=slice_feat, quantile=0.8)
    results.append(eval_on(X_slice, y_slice, f"slice_{slice_feat}_top20pct_thr_{thr:.3f}"))

    out_path = reports_dir / "shift_test_results.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("Shift tests done. Saved to:", out_path)
    for r in results:
        print(
            f"{r['name']}: ROC-AUC={r['roc_auc']:.3f} PR-AUC={r['pr_auc']:.3f} "
            f"Sens={r['sensitivity']:.3f} Spec={r['specificity']:.3f}"
        )


if __name__ == "__main__":
    main()
