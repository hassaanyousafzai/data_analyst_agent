"""Analyst agent: interprets the dataset and produces insights."""

from __future__ import annotations

import pandas as pd

from utils.dataset_context import dataframe_profile
from client.groq_client import chat_completion

ANALYST_SYSTEM = """You are a senior data analyst. You receive a structured profile of a CSV \
dataset (schema, sample rows, basic statistics). Your job is to:
- Summarize what the dataset appears to represent.
- Highlight notable patterns, outliers, missing data risks, and relationships worth plotting.
- Suggest 2–4 concrete visualization ideas (what to plot and why).
Write clearly for a technical reader. Use headings and bullet points. Do not write code."""


def run_analyst(df: pd.DataFrame) -> str:
    profile = dataframe_profile(df)
    user = f"Here is the dataset profile:\n\n{profile}\n\nProvide your analysis."
    return chat_completion(system=ANALYST_SYSTEM, user=user, temperature=0.35)
