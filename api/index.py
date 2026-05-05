from __future__ import annotations

from fastapi import FastAPI
import gradio as gr

from app.app import demo

# Vercel Python runtime looks for an ASGI `app` object.
fastapi_app = FastAPI()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
