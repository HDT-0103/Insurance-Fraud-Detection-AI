from __future__ import annotations

import pandas as pd


def add_age_and_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    age_bins = [0, 25, 35, 45, 65]
    age_labels = ["18-25", "26-35", "36-45", "46-65"]
    df["age_group"] = pd.cut(df["age"], bins=age_bins, labels=age_labels, include_lowest=True)

    tenure_bins = [0, 60, 180, 300, 500]
    tenure_labels = ["New (<5yr)", "Mid (5-15yr)", "Long (15-25yr)", "Legacy (>25yr)"]
    df["tenure_group"] = pd.cut(
        df["months_as_customer"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
    )

    return df

