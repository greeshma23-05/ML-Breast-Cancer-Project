from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

from src.evaluation.metrics import evaluate_binary


repo_root = Path(__file__).resolve().parents[2]
processed_path = repo_root / "data" / "processed" / "dataset.parquet"
figuers_dir = repo_root / "reports" / "figures"
figuers_dir.mkdir(parents=True, exist_ok=True)


def pick_cases(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Return indices for TP, FP, FN (if they exist) within the provided arrays.
    """
    # y: 0 benign, 1 malignant
    tp = np.where((y_true == 1) & (y_pred == 1))[0]
    fp = np.where((y_true == 0) & (y_pred == 1))[0]
    fn = np.where((y_true == 1) & (y_pred == 0))[0]

    def first_or_none(arr):
        return int(arr[0]) if len(arr) > 0 else None

    return first_or_none(tp), first_or_none(fp), first_or_none(fn)


def main() -> None:
    df = pd.read_parquet(processed_path)
    X = df.drop(columns=["target"])
    y = df["target"].to_numpy().astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    #tree model
    model = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # Evaluate
    p = model.predict_proba(X_test)[:, 1]
    eval_out = evaluate_binary(y_test, p, threshold=0.5)
    print(
        f"RF test: ROC-AUC={eval_out.roc_auc:.3f} PR-AUC={eval_out.pr_auc:.3f} "
        f"Sens={eval_out.sensitivity:.3f} Spec={eval_out.specificity:.3f}"
    )

    y_pred = (p >= 0.5).astype(int)
    tp_i, fp_i, fn_i = pick_cases(y_test, y_pred)
    print("Picked case indices (within test set):", {"TP": tp_i, "FP": fp_i, "FN": fn_i})

    # SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # shap_values[0] for class 0, shap_values[1] for class 1
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    #bar plot of mean
    plt.figure()
    shap.summary_plot(sv, X_test, plot_type="bar", show=False)
    out_global =figuers_dir / "shap_global_bar.png"
    plt.tight_layout()
    plt.savefig(out_global, dpi=160)
    plt.close()
    print("Saved:", out_global)

    #beeswarm
    plt.figure()
    shap.summary_plot(sv, X_test, show=False)
    out_swarm = figuers_dir / "shap_beeswarm.png"
    plt.tight_layout()
    plt.savefig(out_swarm, dpi=160)
    plt.close()
    print("Saved:", out_swarm)

    #waterfall plots for TP / FP / FN
    def save_waterfall(idx: int | None, label: str):
        if idx is None:
            print(f"Skipping {label}: no example found.")
            return

        row_sv = sv[idx]
        if isinstance(row_sv, np.ndarray) and row_sv.ndim == 2:
            row_sv = row_sv[:, 1]

        base = explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1]  # class 1 baseline

        feature_names = X_test.columns.tolist()
        feature_values = X_test.iloc[idx].to_numpy()

        exp = shap.Explanation(
            values=row_sv,
            base_values=base,
            data=feature_values,
            feature_names=feature_names,
        )

        plt.figure()
        shap.waterfall_plot(exp, max_display=12, show=False)
        out_path = figuers_dir / f"shap_waterfall_{label}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        print("Saved:", out_path)

    save_waterfall(tp_i, "TP")
    save_waterfall(fp_i, "FP")
    save_waterfall(fn_i, "FN")


if __name__ == "__main__":
    main()
