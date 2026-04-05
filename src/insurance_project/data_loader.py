from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    print("Data loaded successfully.")
    print(f"Shape: {df.shape}")
    return df
        
