# AI Agents: Analyst → Visualizer → Writer

Upload a CSV in the Gradio UI. Three Groq agents run in order: **analyst** (insights), **visualizer** (matplotlib code + plot), **writer** (reads the chart when possible and summarizes it).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env   # add GROQ_API_KEY
```

## Run

From the repo root:

```bash
python -m app.app
```

Or: `python app/app.py`

## Env

| Variable | Purpose |
|----------|---------|
| `GROQ_API_KEY` | Required |
| `GROQ_MODEL` | Text agents (default: `llama-3.3-70b-versatile`) |
| `GROQ_VISION_MODEL` | Agent 3 image (default: `meta-llama/llama-4-scout-17b-16e-instruct`) |
