"""Agent 3: interprets the rendered figure in light of the analyst report and plotting code."""

from __future__ import annotations

from client.groq_client import chat_completion_with_optional_image

INTERPRETER_SYSTEM = """You are a data visualization interpreter. You combine:
- the analyst's written findings about the dataset,
- the matplotlib code that produced the figure,
- and (when provided) the actual rendered chart image.

Your job is to write a concise narrative analysis of what the visualization communicates: chart type, axes, \
scales, legends, clusters or outliers visible in the image, and how the graphic supports, extends, or \
qualifies the analyst's earlier conclusions. Call out caveats (e.g. log scales, missing data, crowded labels).

If no image is available, infer what the figure was intended to show from the code and prior analysis, and \
state clearly that you could not verify it against a rendered plot."""


def run_viz_interpreter(
    analyst_insights: str,
    viz_code: str,
    png: bytes | None,
    plot_error: str | None,
) -> str:
    parts = [
        "### Analyst insights (Agent 1)\n",
        analyst_insights.strip(),
        "\n\n### Visualization code (Agent 2)\n```python\n",
        viz_code.strip(),
        "\n```",
    ]
    if plot_error:
        parts.append(f"\n\n### Plot execution note\n{plot_error}")
    parts.append(
        "\n\nWrite the visualization-focused analysis as requested. "
        "Use short titled sections (e.g. Overview, Reading the chart, Tie-in to prior analysis, Caveats)."
    )
    user_text = "".join(parts)
    return chat_completion_with_optional_image(
        system=INTERPRETER_SYSTEM,
        user_text=user_text,
        image_png=png,
        temperature=0.35,
    )
