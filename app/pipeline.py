"""Orchestrate Analyst → Visualizer."""

from __future__ import annotations

import pandas as pd

from agents.analyst_agent import run_analyst
from agents.viz_interpreter_agent import run_viz_interpreter
from agents.visualizer_agent import run_visualizer


def run_pipeline(csv_path: str) -> tuple[str, str, bytes | None, str | None, str]:
    df = pd.read_csv(csv_path)
    insights = run_analyst(df)
    code, png, plot_error = run_visualizer(df, insights)
    viz_writeup = run_viz_interpreter(insights, code, png, plot_error)
    return insights, code, png, plot_error, viz_writeup

