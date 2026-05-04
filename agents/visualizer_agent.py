"""Visualizer agent: turns insights into matplotlib code executed safely in a narrow namespace."""

from __future__ import annotations

import builtins
import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from client.groq_client import chat_completion, extract_python_code
from utils.dataset_context import dataframe_profile

VISUALIZER_SYSTEM = """You are a visualization engineer. You receive:
1) A dataset profile (schema + sample + stats)
2) An analyst's written insights and suggested charts

Write Python code ONLY (no prose) that uses:
- pandas as pd, numpy as np, matplotlib.pyplot as plt
- the variable `df` (a pandas DataFrame) — already provided; do not reload CSV
Rules:
- Create 1–3 subplots that support the analyst's conclusions (distributions, comparisons, trends).
- Use readable labels, titles, and rotate x tick labels if crowded.
- After all plotting, set `fig = plt.gcf()` and call `fig.tight_layout()` if appropriate.
- Do not use plt.show(), do not read/write files, no network, no input().
- Keep the script self-contained and robust to moderate missing values (dropna where needed).
Output a single ```python fenced block with the full script."""


def run_visualizer(df: pd.DataFrame, analyst_report: str) -> tuple[str, bytes | None, str | None]:
    profile = dataframe_profile(df)
    user = (
        f"Dataset profile:\n\n{profile}\n\n"
        f"Analyst report:\n\n{analyst_report}\n\n"
        "Generate matplotlib code as instructed."
    )
    raw = chat_completion(system=VISUALIZER_SYSTEM, user=user, temperature=0.25)
    code = extract_python_code(raw)

    err: str | None = None
    png: bytes | None = None
    buf = io.BytesIO()

    plt.close("all")
    namespace: dict[str, Any] = {
        "__builtins__": builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df,
    }
    try:
        exec(code, namespace, None)  # noqa: S102 — narrow namespace; portfolio/demo use
        fig = namespace.get("fig")
        if fig is None:
            fig = plt.gcf()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        png = buf.read()
    except Exception as e:  # noqa: BLE001 — surface any plotting failure to the UI
        err = f"{type(e).__name__}: {e}"
        plt.close("all")

    return code, png, err
