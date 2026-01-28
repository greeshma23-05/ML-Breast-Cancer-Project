from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

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

def evalutate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    roc = float(roc_auc_score(y_true, y_prob))
    pr = float(average_precision_score(y_true, y_prob))

    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred).tolist()

    tn, fp = cm[0]
    fn, tp = cm[1]
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float ("nan")
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float ("nan")

    return{
        "roc-auc": roc,
        "pr_auc": pr,
        "threshold": threshold,
        "confusion_matrix": cm,
        "sensitivity": sensitivity,
        "specificity": specificity
    }

def main() -> None:
    #Load data
    df = load_processed()
    X = df.drop(columns=["target"])
    y = df["target"].astype(int).to_numpy()

    #train/test splot
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results: dict[str, dict] = {}

    #baseline 1: logistic regression
    lr = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, solver="lbfgs")),
        ]
    )
    lr.fit(X_train, y_train)
    lr_prob = lr.predict_proba(X_test)[:, 1]
    results["logistic regression"] = evalutate_binary(y_test, lr_prob, threshold=0.5)

    #baseline 2: random forest
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = evalutate_binary(y_test, rf_prob, threshold=0.5)

    #save metrics
    out_path = reports_dir / "metrics_baseline.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)

    #print summary
    print("Baseline training complete")
    print(f"Saved metrics to: {out_path}")
    for name, m in results.items():
        print(
            f"{name}: ROC-AUC={m['roc-auc']:.3f} PR-AUC={m['pr_auc']:.3f}  "
            f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}"
        )

if __name__ == "__main__":
    main()