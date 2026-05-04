"""Shared Groq chat completion helper."""

import base64
import io
import os

from groq import Groq
from PIL import Image


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY in your environment or in a .env file (see env.example)."
        )
    return Groq(api_key=api_key)


def default_model() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def default_vision_model() -> str:
    return os.environ.get(
        "GROQ_VISION_MODEL",
        "meta-llama/llama-4-scout-17b-16e-instruct",
    )


def _jpeg_data_url_for_vision(png_or_image_bytes: bytes) -> str:
    """Groq base64 image payloads are capped (~4MB); JPEG + downscale keeps requests small."""
    im = Image.open(io.BytesIO(png_or_image_bytes)).convert("RGB")
    quality = 88
    side_limit = 1600
    for _ in range(12):
        w, h = im.size
        if max(w, h) > side_limit:
            im = im.resize(
                (max(1, int(w * side_limit / max(w, h))), max(1, int(h * side_limit / max(w, h)))),
                Image.Resampling.LANCZOS,
            )
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        raw = buf.getvalue()
        b64 = base64.standard_b64encode(raw).decode("ascii")
        if len(b64) <= 3_200_000:
            return f"data:image/jpeg;base64,{b64}"
        quality = max(55, quality - 8)
        side_limit = int(side_limit * 0.85)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=50, optimize=True)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def chat_completion(
    *,
    system: str,
    user: str,
    temperature: float = 0.4,
    max_tokens: int = 4096,
) -> str:
    client = get_client()
    model = default_model()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    if not choice.message.content:
        return ""
    return choice.message.content.strip()


def chat_completion_with_optional_image(
    *,
    system: str,
    user_text: str,
    image_png: bytes | None,
    temperature: float = 0.35,
    max_tokens: int = 4096,
) -> str:
    """Text chat, or vision chat when ``image_png`` is set (uses ``GROQ_VISION_MODEL``)."""
    client = get_client()
    if image_png:
        model = default_vision_model()
        url = _jpeg_data_url_for_vision(image_png)
        user_content: list | str = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": url}},
        ]
    else:
        model = default_model()
        user_content = user_text
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    choice = response.choices[0]
    if not choice.message.content:
        return ""
    return choice.message.content.strip()


def extract_python_code(text: str) -> str:
    """Pull the first fenced python block, or the whole string if none."""
    marker = "```python"
    if marker in text:
        start = text.index(marker) + len(marker)
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        if text[start : start + 6].lower() == "python":
            start += 6
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
    return text.strip()
