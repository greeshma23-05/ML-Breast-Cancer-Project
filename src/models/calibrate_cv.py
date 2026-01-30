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

from src.evaluation.metrics import evaluate_binary
from src.evaluation.calibration import expected_calibration_error


repo_root = Path(__file__).resolve().parents[2]
processed_path = repo_root / "data" / "processed" / "dataset.parquet"
reports_dir = repo_root / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)


def load_processed() -> pd.DataFrame:
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing {processed_path}. Run: python -m src.data.make_dataset"
        )
    return pd.read_parquet(processed_path)


def oof_proba(model, X, y, seed: int = 42) -> np.ndarray:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]


def main() -> None:
    df = load_processed()
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy().astype(int).ravel()

    base_lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )

    # Uncalibrated OOF probs
    p_uncal = oof_proba(base_lr, X, y)

    # Calibrated models
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    platt = CalibratedClassifierCV(estimator=base_lr, cv=cv, method="sigmoid")
    p_platt = oof_proba(platt, X, y)

    iso = CalibratedClassifierCV(estimator=base_lr, cv=cv, method="isotonic")
    p_iso = oof_proba(iso, X, y)

    # Metrics at threshold 0.5
    out = {
        "uncalibrated": {
            "eval": evaluate_binary(y, p_uncal, threshold=0.5).__dict__,
            "calibration": expected_calibration_error(y, p_uncal, n_bins=10).__dict__,
        },
        "platt_sigmoid": {
            "eval": evaluate_binary(y, p_platt, threshold=0.5).__dict__,
            "calibration": expected_calibration_error(y, p_platt, n_bins=10).__dict__,
        },
        "isotonic": {
            "eval": evaluate_binary(y, p_iso, threshold=0.5).__dict__,
            "calibration": expected_calibration_error(y, p_iso, n_bins=10).__dict__,
        },
    }

    out_path = reports_dir / "calibration_cv.json"
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)

    print("Calibration done.")
    print(f"Saved calibration report to: {out_path}")
    for k, v in out.items():
        print(
            f"{k}: ECE={v['calibration']['ece']:.4f}  "
            f"ROC-AUC={v['eval']['roc_auc']:.3f} PR-AUC={v['eval']['pr_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
