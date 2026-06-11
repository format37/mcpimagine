"""Gemini "Nano Banana" image generation + editing tools for the MCP server.

Models (Google Gemini image generation, codename "Nano Banana"):
  - gemini-3-pro-image-preview    -> Nano Banana Pro  (highest quality, text rendering, optional grounding)
  - gemini-3.1-flash-image-preview-> Nano Banana 2    (fast, high volume)

Each tool returns a downscaled inline preview (so the caller can see the result)
plus an HTTPS URL to the persisted full-resolution image.
"""

import base64
import io
import logging
import mimetypes
import os
import threading
import uuid
from pathlib import Path
from typing import Any, List, Optional

import anyio
from google import genai
from google.genai import types
from PIL import Image as PILImage

from mcp_image_utils import (
    MAX_DOWNLOAD_BYTES,
    retrieve_image_from_url,
    sniff_mime,
    to_mcp_image,
)
from request_logger import log_request

logger = logging.getLogger(__name__)

# Friendly aliases -> concrete model ids.
MODEL_ALIASES = {
    "pro": "gemini-3-pro-image-preview",
    "nano-banana-pro": "gemini-3-pro-image-preview",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "flash": "gemini-3.1-flash-image-preview",
    "nano-banana": "gemini-3.1-flash-image-preview",
    "nano-banana-2": "gemini-3.1-flash-image-preview",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
}
DEFAULT_MODEL = os.getenv("GEMINI_IMAGE_MODEL", "gemini-3-pro-image-preview")

VALID_ASPECT = {"1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"}
VALID_SIZE = {"1K", "2K", "4K"}

PREVIEW_MAX = int(os.getenv("MCP_PREVIEW_MAX", "1536"))
PREVIEW_QUALITY = int(os.getenv("MCP_PREVIEW_JPEG_QUALITY", "85"))
# Gemini SDK timeout in milliseconds (generation typically takes 15-25s).
GENAI_TIMEOUT_MS = int(os.getenv("GEMINI_TIMEOUT_MS", "300000"))
# Bound concurrent generations to protect the host and the API quota.
MAX_CONCURRENCY = int(os.getenv("IMAGE_MAX_CONCURRENCY", "2"))
_gen_semaphore = threading.Semaphore(MAX_CONCURRENCY)

MAX_INPUT_IMAGES = int(os.getenv("MAX_INPUT_IMAGES", "8"))

# A single long-lived client is reused across requests. It MUST be held in a
# stable reference: if a genai.Client is created inline and garbage-collected
# mid-call, its finalizer closes the underlying httpx transport and the in-flight
# request fails with "Cannot send a request, as the client has been closed".
_client_lock = threading.Lock()
_cached_client: Optional[genai.Client] = None


def _resolve_model(model: Optional[str]) -> str:
    if not model:
        return DEFAULT_MODEL
    return MODEL_ALIASES.get(model.strip().lower(), model.strip())


def _client() -> genai.Client:
    global _cached_client
    if _cached_client is None:
        with _client_lock:
            if _cached_client is None:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise RuntimeError("GEMINI_API_KEY is not configured on the server")
                _cached_client = genai.Client(
                    api_key=api_key,
                    http_options=types.HttpOptions(timeout=GENAI_TIMEOUT_MS),
                )
    return _cached_client


def _validate(aspect_ratio: str, image_size: str) -> None:
    if aspect_ratio not in VALID_ASPECT:
        raise ValueError(
            f"Invalid aspect_ratio '{aspect_ratio}'. Valid: {sorted(VALID_ASPECT)}"
        )
    if image_size not in VALID_SIZE:
        raise ValueError(
            f"Invalid image_size '{image_size}'. Valid: {sorted(VALID_SIZE)}"
        )


def _load_input_parts(
    image_urls: Optional[List[str]],
    image_base64: Optional[List[str]],
) -> List[types.Part]:
    """Build Gemini input image Parts from URLs and/or base64 strings."""
    parts: List[types.Part] = []
    items: List[tuple[str, str]] = []
    for url in image_urls or []:
        if url and str(url).strip():
            items.append(("url", str(url).strip()))
    for b64 in image_base64 or []:
        if b64 and str(b64).strip():
            items.append(("b64", str(b64).strip()))

    if len(items) > MAX_INPUT_IMAGES:
        raise ValueError(
            f"Too many input images ({len(items)}); max is {MAX_INPUT_IMAGES}."
        )

    for kind, value in items:
        if kind == "url":
            try:
                data = retrieve_image_from_url(value)
            except Exception as e:
                raise ValueError(f"Failed to download input image {value!r}: {e}") from e
        else:  # base64 (tolerate data: URIs)
            payload = value.split(",", 1)[1] if value.startswith("data:") else value
            # base64 encodes ~4 chars per 3 bytes; cap before decoding.
            if len(payload) > int(MAX_DOWNLOAD_BYTES / 0.74) + 1024:
                raise ValueError(
                    f"base64 input image too large (decoded would exceed {MAX_DOWNLOAD_BYTES} bytes)"
                )
            try:
                data = base64.b64decode(payload)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 input image: {e}") from e
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise ValueError(f"base64 input image exceeds max size of {MAX_DOWNLOAD_BYTES} bytes")
        mime = sniff_mime(data)
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
    return parts


def _generate(
    prompt: str,
    input_parts: List[types.Part],
    aspect_ratio: str,
    image_size: str,
    model: Optional[str],
    grounding: bool,
):
    """Call Gemini and return (image_bytes, mime, text, model_id, finish_reason)."""
    model_id = _resolve_model(model)
    _validate(aspect_ratio, image_size)
    if grounding and model_id != "gemini-3-pro-image-preview":
        raise ValueError(
            "grounding=True is only supported with the 'pro' model (gemini-3-pro-image-preview)."
        )

    parts = list(input_parts) + [types.Part.from_text(text=prompt)]
    contents = [types.Content(role="user", parts=parts)]

    cfg_kwargs: dict[str, Any] = {
        "response_modalities": ["IMAGE", "TEXT"],
        "image_config": types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
    }
    if grounding:
        cfg_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    config = types.GenerateContentConfig(**cfg_kwargs)

    logger.info(
        "Generating image: model=%s aspect=%s size=%s inputs=%d grounding=%s prompt=%r",
        model_id, aspect_ratio, image_size, len(input_parts), grounding, prompt[:120],
    )

    client = _client()  # keep a strong reference for the whole call (see _client docstring)
    response = client.models.generate_content(
        model=model_id, contents=contents, config=config
    )

    image_bytes: Optional[bytes] = None
    mime: Optional[str] = None
    text: Optional[str] = None
    finish_reason = None

    candidates = response.candidates or []
    if candidates:
        cand = candidates[0]
        finish_reason = getattr(cand, "finish_reason", None)
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            if part.inline_data and part.inline_data.data:
                image_bytes = part.inline_data.data
                mime = part.inline_data.mime_type
            elif part.text:
                text = (text or "") + part.text

    return image_bytes, mime, text, model_id, finish_reason


def _ext_for_mime(mime: Optional[str]) -> str:
    ext = mimetypes.guess_extension(mime or "image/jpeg") or ".jpg"
    return ".jpg" if ext in (".jpe", ".jpeg") else ext


def _persist_and_respond(
    image_bytes: bytes,
    mime: Optional[str],
    model_text: Optional[str],
    model_id: str,
    aspect_ratio: str,
    image_size: str,
    images_dir: Path,
    public_asset_base: Optional[str],
    host_images_dir: Optional[str] = None,
) -> List[Any]:
    """Save full-res image, build inline preview, return [info_text, preview].

    The info text comes FIRST so the saved-file path and download URL survive
    even if the client truncates the response after the large inline image."""
    # Validate the bytes decode to a real image BEFORE persisting, so a corrupt
    # response from the model never leaves a junk file on disk.
    try:
        im = PILImage.open(io.BytesIO(image_bytes))
        im.load()
    except Exception as e:
        raise ValueError(f"Model returned undecodable image data: {e}") from e

    images_dir.mkdir(parents=True, exist_ok=True)
    uid = uuid.uuid4().hex
    filename = f"{uid}{_ext_for_mime(mime)}"
    (images_dir / filename).write_bytes(image_bytes)

    # Downscaled JPEG preview keeps the inline MCP response small.
    try:
        resample = PILImage.Resampling.LANCZOS
    except AttributeError:  # Pillow < 10
        resample = PILImage.LANCZOS
    preview_img = im.copy()
    preview_img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), resample=resample)
    preview_content = to_mcp_image(
        preview_img, format="jpeg", quality=PREVIEW_QUALITY, optimize=True
    ).to_image_content()

    public_url = f"{public_asset_base}/{filename}" if public_asset_base else None
    saved_dir = host_images_dir.rstrip("/") if host_images_dir else str(images_dir)
    saved_path = f"{saved_dir}/{filename}"

    info: List[str] = []
    info.append(f"Saved file: {saved_path}")
    if public_url:
        info.append(f"Full-resolution image: {public_url}")
    else:
        info.append(
            "Set MCP_PUBLIC_BASE_URL (or MCP_PUBLIC_ASSET_BASE_URL) to expose an HTTPS "
            "download link for the full-resolution image."
        )
    info.append(
        f"(model={model_id}, aspect_ratio={aspect_ratio}, image_size={image_size}, "
        f"bytes={len(image_bytes)})"
    )
    if model_text:
        info.append(f"Model note: {model_text.strip()}")

    # Text first: if the client truncates after the inline preview, the path
    # and URL are still delivered.
    return ["\n".join(info), preview_content]


def register_image_tools(
    mcp,
    images_dir: Path,
    requests_dir: Path,
    public_asset_base: Optional[str],
    host_images_dir: Optional[str] = None,
):
    """Register generate_image and edit_image tools on the FastMCP instance."""

    def _blocking_run(
        tool_name: str,
        prompt: str,
        image_urls: Optional[List[str]],
        image_base64: Optional[List[str]],
        aspect_ratio: str,
        image_size: str,
        model: Optional[str],
        grounding: bool,
        requester: str,
    ) -> List[Any]:
        """All blocking work (input download, Gemini call, PIL preview). Runs in a
        worker thread so the event loop stays free to serve health/asset/other
        requests while a ~20s generation is in flight."""
        log_input = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size,
            "model": model or DEFAULT_MODEL,
            "grounding": grounding,
        }
        try:
            input_parts = _load_input_parts(image_urls, image_base64)
            if tool_name == "edit_image" and not input_parts:
                raise RuntimeError(
                    "edit_image requires at least one source image via image_urls or "
                    "image_base64. Use generate_image for text-only image creation."
                )
            log_input["n_input_images"] = len(input_parts)

            with _gen_semaphore:
                image_bytes, mime, text, model_id, finish_reason = _generate(
                    prompt, input_parts, aspect_ratio, image_size, model, grounding
                )

            if not image_bytes:
                msg = (
                    "No image was returned by the model "
                    f"(finish_reason={finish_reason}). "
                    "This usually means the request was blocked by safety filters or the "
                    "prompt asked for text only. "
                )
                if text:
                    msg += f"Model said: {text.strip()}"
                logger.warning("%s: %s", tool_name, msg)
                log_request(requests_dir, requester, tool_name, log_input, msg)
                return [msg]

            result = _persist_and_respond(
                image_bytes, mime, text, model_id, aspect_ratio, image_size,
                images_dir, public_asset_base, host_images_dir,
            )
            # Log only the text part (never the image bytes).
            log_request(
                requests_dir, requester, tool_name, log_input,
                next((x for x in result if isinstance(x, str)), "ok"),
            )
            return result
        except Exception as e:
            logger.exception("%s failed", tool_name)
            log_request(requests_dir, requester, tool_name, log_input, f"ERROR: {e}")
            raise RuntimeError(f"{tool_name} failed: {e}") from e

    @mcp.tool()
    async def generate_image(
        prompt: str,
        aspect_ratio: str = "1:1",
        image_size: str = "2K",
        model: str = "pro",
        grounding: bool = False,
        requester: str = "unknown",
    ) -> List[Any]:
        """Generate an image from a text prompt using Gemini "Nano Banana".

        Returns an info text (saved file path on the server + HTTPS URL to the
        full-resolution image) followed by a downscaled inline preview image.
        Always share the file path and full-resolution URL with the user.

        Parameters:
            prompt (str): Detailed description of the image to generate. Nano Banana
                renders text inside images very well — quote any text you want shown.
            aspect_ratio (str): One of "1:1" (default), "2:3", "3:2", "3:4", "4:3",
                "4:5", "5:4", "9:16", "16:9", "21:9".
            image_size (str): "1K", "2K" (default) or "4K". Higher = slower/larger.
            model (str): "pro" (default, gemini-3-pro-image-preview / Nano Banana Pro —
                best quality and text rendering) or "flash"
                (gemini-3.1-flash-image-preview / Nano Banana 2 — faster).
            grounding (bool): If true, let the model use Google Search to ground the
                image in real-world facts (pro model only; adds latency). Default false.
            requester (str): Identifier of who is calling, used for request logging.

        Returns:
            list: [info text with saved file path and full-resolution HTTPS URL,
            inline preview image]

        Notes:
            - Generation typically takes 15-25 seconds.
            - All generated images carry an invisible SynthID watermark.

        Example:
            generate_image(prompt="A neon-lit ramen shop in the rain, cinematic",
                           aspect_ratio="16:9", image_size="2K")
        """
        return await anyio.to_thread.run_sync(
            _blocking_run, "generate_image", prompt, None, None,
            aspect_ratio, image_size, model, grounding, requester,
        )

    @mcp.tool()
    async def edit_image(
        prompt: str,
        image_urls: Optional[List[str]] = None,
        image_base64: Optional[List[str]] = None,
        aspect_ratio: str = "1:1",
        image_size: str = "2K",
        model: str = "pro",
        requester: str = "unknown",
    ) -> List[Any]:
        """Edit or compose images with Gemini "Nano Banana" using input image(s) + a prompt.

        Provide one or more source images (as HTTPS URLs and/or base64 strings) plus an
        instruction describing the edit or composition (e.g. "place this product on a
        marble table", "combine these two people into one photo", "make it night time").
        Returns an info text (saved file path + HTTPS URL to the full-resolution
        result) followed by a downscaled inline preview.

        Parameters:
            prompt (str): The edit / composition instruction.
            image_urls (list[str]): Optional http(s) URLs of source images.
            image_base64 (list[str]): Optional base64-encoded source images
                (raw base64 or a data: URI). Up to 8 input images total.
            aspect_ratio (str): Output aspect ratio (see generate_image). Default "1:1".
            image_size (str): "1K", "2K" (default), or "4K".
            model (str): "pro" (default) or "flash". See generate_image.
            requester (str): Identifier of who is calling, used for request logging.

        Returns:
            list: [info text with saved file path and full-resolution HTTPS URL,
            inline preview image]

        Example:
            edit_image(prompt="Put this character on a snowy mountain at sunrise",
                       image_urls=["https://example.com/character.png"],
                       aspect_ratio="3:4")
        """
        return await anyio.to_thread.run_sync(
            _blocking_run, "edit_image", prompt, image_urls, image_base64,
            aspect_ratio, image_size, model, False, requester,
        )
