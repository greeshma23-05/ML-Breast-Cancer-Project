from __future__ import annotations

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def find_single_csv(raw_dir: Path) -> Path:
    csv_files = sorted(raw_dir.glob("*.csv"))
    if len(csv_files) == 0:
        raise FileNotFoundError(
            f"No CSV files found in {raw_dir}. Put the Kaggle CSV in data/raw/."
        )
    if len(csv_files) > 1:
        raise RuntimeError(
            f"Found multiple CSVs in {raw_dir}: {[p.name for p in csv_files]}. "
            "Keep only one CSV for now."
        )
    return csv_files[0]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace(r"[^\w_]+", "", regex=True)
    )
    return df


def build_dataset(input_csv: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    # Standardize column names
    df = clean_column_names(df)

    # Common Kaggle versions have columns like: id, diagnosis, ... , unnamed:_32
    # Drop unnamed columns (usually empty)
    unnamed_cols = [c for c in df.columns if c.startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Basic checks for expected label column
    if "diagnosis" not in df.columns:
        raise KeyError(
            f"Expected a 'diagnosis' column but got columns: {list(df.columns)[:15]}..."
        )

    # Map labels: M -> 1, B -> 0
    label_map = {"m": 1, "b": 0}
    df["target"] = df["diagnosis"].astype(str).str.lower().map(label_map)

    if df["target"].isna().any():
        bad = df.loc[df["target"].isna(), "diagnosis"].unique().tolist()
        raise ValueError(f"Unexpected diagnosis labels found: {bad}")

    # Drop original label column (keep only target)
    df = df.drop(columns=["diagnosis"])

    # Remove ID column if present (not a feature)
    for id_col in ["id"]:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])

    # Ensure all remaining columns are numeric
    feature_cols = [c for c in df.columns if c != "target"]
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise TypeError(f"Non-numeric feature columns found: {non_numeric}")

    # Save processed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    return df


def main() -> None:
    PROCESSED_PATH = PROCESSED_DIR / "dataset.parquet"

    input_csv = find_single_csv(RAW_DIR)
    df = build_dataset(input_csv=input_csv, output_path=PROCESSED_PATH)

    # Quick summary printout
    n_rows, n_cols = df.shape
    pos_rate = df["target"].mean()
    print(f"Built processed dataset: {PROCESSED_PATH}")
    print(f"Shape: {n_rows} rows x {n_cols} columns (includes target)")
    print(f"Malignant rate (target=1): {pos_rate:.3f}")
    print("First 5 columns:", list(df.columns[:5]))


if __name__ == "__main__":
    main()
