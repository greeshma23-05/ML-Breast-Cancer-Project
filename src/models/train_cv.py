from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.evaluation.metrics import evaluate_binary

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

from sklearn.model_selection import StratifiedKFold, cross_val_predict

def cv_predict_proba(model, X, y, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    y = np.asarray(y).ravel()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return proba[:, 1]


def main() -> None:
    df = load_processed()
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy().astype(int).ravel()

    results = {}

    # Model 1: Logistic Regression (CV)
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    lr_oof = cv_predict_proba(lr, X, y)
    results["logistic_regression_cv"] = evaluate_binary(y, lr_oof, threshold=0.5).__dict__

    # Model 2: Gradient Boosting (CV)
    gbdt = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )
    gbdt_oof = cv_predict_proba(gbdt, X, y)
    results["hist_gradient_boosting_cv"] = evaluate_binary(y, gbdt_oof, threshold=0.5).__dict__

    out_path = reports_dir / "metrics_cv.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("Cross-validation training done.")
    print(f"Saved CV metrics to: {out_path}")
    for name, m in results.items():
        print(
            f"{name}: ROC-AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  "
            f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}"
        )


if __name__ == "__main__":
    main()