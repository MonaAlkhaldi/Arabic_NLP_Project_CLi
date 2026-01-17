from __future__ import annotations
from pathlib import Path
import pandas as pd


class ShowColumnsError(ValueError):
    """Raised when required columns are missing."""


def load_csv(csv_path: str) -> pd.DataFrame:
  
    path = Path(csv_path)

    #cheacking if the file exisit 
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    
    df = pd.read_csv(path)

    
    if df.empty:
        raise ValueError("CSV loaded but it is empty.")

    return df


def validate_columns(df: pd.DataFrame, text_col: str, label_col: str) -> None:
  
    cols = list(df.columns)

    missing = []
    if text_col not in df.columns:
        missing.append(f"text_col='{text_col}'")
    if label_col not in df.columns:
        missing.append(f"label_col='{label_col}'")

    if missing:
        raise ShowColumnsError(
            f"Missing column(s): {', '.join(missing)}.\n"
            f"Available columns: {', '.join(cols)}"
        )


def basic_dataset_summary(df: pd.DataFrame, text_col: str, label_col: str) -> dict:
  
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "text_missing": int(df[text_col].isna().sum()),
        "num_classes": int(df[label_col].nunique(dropna=True)),
    }
