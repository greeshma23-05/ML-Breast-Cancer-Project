from __future__ import annotations

from pathlib import Path
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
processed_path = repo_root / "data" / "processed" / "dataset.parquet"

def validate_dataset(df: pd.DataFrame) -> None:
    #Needs target
    if "target" not in df.columns:
        raise KeyError("Processed dataset must include a 'target' column.")

    #Target has to be binary
    unique_targets = set(df["target"].unique().tolist())
    if not unique_targets.issubset({0, 1}):
        raise ValueError(f"Target must be binary 0/1. Found: {unique_targets}")

    #No missing values in target
    if df["target"].isna().any():
        raise ValueError("Target column contains missing values.")

    #No missing values in features
    feature_cols = [c for c in df.columns if c != "target"]
    if df[feature_cols].isna().any().any():
        missing_counts = df[feature_cols].isna().sum()
        missing_counts = missing_counts[missing_counts > 0]
        raise ValueError(f"Missing values found in features:\n{missing_counts}")

    #Features must be numeric
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric}")

    #Drop obvious leakage
    suspicious_cols = [c for c in df.columns if "id" == c or c.endswith("_id")]
    if suspicious_cols:
        raise ValueError(
            f"Suspicious ID-like columns present (should not be features): {suspicious_cols}"
        )

    #Duplicate rows check
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        print(f"WARNING: Found {dup_count} duplicate rows in processed dataset.")

    #Shape sanity
    if df.shape[0] < 100:
        raise ValueError(f"Dataset seems too small: {df.shape[0]} rows.")
    if df.shape[1] < 5:
        raise ValueError(f"Dataset seems to have too few columns: {df.shape[1]} cols.")

    #Class balance sanity
    pos_rate = float(df["target"].mean())
    if pos_rate < 0.05 or pos_rate > 0.95:
        raise ValueError(f"Class balance looks suspicious. Positive rate={pos_rate:.3f}")

    print("Dataset validation passed.")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]} (includes target)")
    print(f"Positive rate (target=1): {pos_rate:.3f}")


def main() -> None:
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {processed_path}. "
            "Run: python -m src.data.make_dataset"
        )

    df = pd.read_parquet(processed_path)
    validate_dataset(df)


if __name__ == "__main__":
    main()
