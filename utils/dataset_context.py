"""Build a compact text summary of a DataFrame for LLM context."""

from __future__ import annotations

import io

import pandas as pd


def dataframe_profile(df: pd.DataFrame, *, preview_rows: int = 25) -> str:
    buf = io.StringIO()
    buf.write(f"shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n")
    buf.write("dtypes:\n")
    buf.write(df.dtypes.astype(str).to_string())
    buf.write("\n\n")
    buf.write("head (CSV):\n")
    buf.write(df.head(preview_rows).to_csv(index=False))
    buf.write("\n")
    num = df.select_dtypes(include=["number"])
    if not num.empty:
        buf.write("\nnumeric describe():\n")
        buf.write(num.describe().T.to_string())
        buf.write("\n")
    cat = df.select_dtypes(exclude=["number", "datetime", "datetimetz"])
    low_card = [c for c in cat.columns if df[c].nunique(dropna=False) <= 30]
    if low_card:
        buf.write("\nvalue counts (low-cardinality columns):\n")
        for c in low_card[:12]:
            buf.write(f"\n[{c}]\n")
            buf.write(df[c].value_counts(dropna=False).head(15).to_string())
            buf.write("\n")
    return buf.getvalue()
