from pathlib import Path

import pandas as pd


def load_reports(path: str | Path, text_col: str = "narrative") -> pd.DataFrame:
    """Load raw ASRS reports from CSV and drop rows with no narrative."""
    df = pd.read_csv(path)
    return df.dropna(subset=[text_col]).reset_index(drop=True)
