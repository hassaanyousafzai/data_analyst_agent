"""
Gradio UI: CSV upload → Analyst → Visualizer → Visualization writer.

Run from the **repository root** (where `.env` and `fifa_eda.csv` live)::

    python -m app.app

Or::

    python app/app.py

Configure with environment variables or ``.env`` (no argparse). Optional: python-dotenv.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image

from app.pipeline import run_pipeline

load_dotenv(_PROJECT_ROOT / ".env")


def _upload_path(file_obj: Any) -> str | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, str) and os.path.isfile(file_obj):
        return file_obj
    if isinstance(file_obj, dict):
        for key in ("path", "name", "temp_file_path"):
            p = file_obj.get(key)
            if isinstance(p, str) and os.path.isfile(p):
                return p
    path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
    if isinstance(path, str) and os.path.isfile(path):
        return path
    return None


def _png_to_numpy(png: bytes) -> np.ndarray:
    """Gradio's Image (numpy) expects uint8 RGB; float RGBA from imread often renders blank."""
    im = Image.open(io.BytesIO(png)).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def process_file(file_obj: Any) -> tuple[str, np.ndarray | None, str, str, str]:
    path = _upload_path(file_obj)
    if not path:
        return "", None, "", "", "Upload a CSV file, then click **Run pipeline**."

    try:
        insights, code, png, plot_error, viz_writeup = run_pipeline(path)
    except RuntimeError as e:
        return "", None, "", "", str(e)
    except Exception as e:  # noqa: BLE001
        return "", None, "", "", f"Pipeline error: {type(e).__name__}: {e}"

    status = "Pipeline complete (Analyst → Visualizer → Visualization writer)."
    if plot_error:
        status = (
            f"Plot step failed: {plot_error}. "
            "Agent 3 used code + analyst text only (no image). See generated code below."
        )
    if not png:
        status = "No PNG produced; Agent 3 inferred from code and analyst report only."

    img: np.ndarray | None = _png_to_numpy(png) if png else None
    return insights, img, viz_writeup, code, status


def _on_file_change(_file_obj: Any) -> tuple[str, np.ndarray | None, str, str, str]:
    return "", None, "", "", "Click **Run pipeline** after selecting a CSV."


with gr.Blocks(title="Data Analyst + Visualizer + Writer") as demo:
    gr.Markdown(
        "## Data Analyst + Visualizer + Visualization writer\n"
        "Three Groq agents: **Analyst** (CSV insights) → **Visualizer** (matplotlib) → "
        "**Visualization writer** (narrative from the chart, using vision when the plot renders).\n\n"
        "Set `GROQ_API_KEY`. Optional: `GROQ_MODEL` (text agents), `GROQ_VISION_MODEL` (Agent 3 image). "
        "See `env.example`."
    )
    file_in = gr.File(label="Upload CSV", file_types=[".csv"])
    run_btn = gr.Button("Run pipeline", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    insights_out = gr.Markdown(label="Agent 1: Analyst insights")
    plot_out = gr.Image(label="Agent 2: Visualizer output", type="numpy")
    viz_writeup_out = gr.Markdown(label="Agent 3: Analysis of the visualization")
    code_out = gr.Code(label="Agent 2: Generated matplotlib code", language="python")

    run_btn.click(
        fn=process_file,
        inputs=[file_in],
        outputs=[insights_out, plot_out, viz_writeup_out, code_out, status],
    )
    file_in.change(
        fn=_on_file_change,
        inputs=[file_in],
        outputs=[insights_out, plot_out, viz_writeup_out, code_out, status],
    )


if __name__ == "__main__":
    demo.launch()
