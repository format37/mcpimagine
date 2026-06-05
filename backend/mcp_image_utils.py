"""Helpers for loading input images and converting images to MCP content."""

import io
import ipaddress
import os
import socket
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image as PILImage
from mcp.server.fastmcp import Image as MCPImage

# Cap decoded pixel count to defuse decompression bombs from input images.
PILImage.MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(64_000_000)))

# Max bytes to pull for a single input image download.
MAX_DOWNLOAD_BYTES = int(os.getenv("MAX_DOWNLOAD_BYTES", str(25 * 1024 * 1024)))
# Escape hatch for local testing against localhost/private asset URLs.
ALLOW_PRIVATE_IMAGE_URLS = os.getenv("ALLOW_PRIVATE_IMAGE_URLS", "").lower() in ("1", "true", "yes")

# Map PIL format -> mime type used when sniffing input images.
_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


def _assert_fetchable_url(image_url: str) -> None:
    """SSRF guard: only allow http(s) URLs that resolve to public IP addresses.

    Prevents a token holder from making the server fetch internal services on
    the shared Docker network, cloud metadata (169.254.169.254), or localhost.
    """
    parsed = urlparse(image_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https image URLs are allowed (got scheme {parsed.scheme!r})")
    host = parsed.hostname
    if not host:
        raise ValueError("Image URL has no host")
    if ALLOW_PRIVATE_IMAGE_URLS:
        return
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve image URL host {host!r}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not _is_public_ip(ip):
            raise ValueError(
                f"Refusing to fetch image from non-public address {ip} (host {host!r})"
            )


def retrieve_image_from_url(image_url: str, timeout: int = 30, max_bytes: int = None) -> bytes:
    """Download raw image bytes from a public http(s) URL, size-capped."""
    _assert_fetchable_url(image_url)
    limit = max_bytes or MAX_DOWNLOAD_BYTES
    with requests.get(image_url, timeout=timeout, stream=True) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > limit:
            raise ValueError(f"Input image too large ({declared} bytes > {limit} limit)")
        chunks, total = [], 0
        for chunk in response.iter_content(chunk_size=65536):
            total += len(chunk)
            if total > limit:
                raise ValueError(f"Input image exceeds max size of {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks)


def sniff_mime(data: bytes, fallback: str = "image/jpeg") -> str:
    """Detect the mime type of raw image bytes via PIL, with a fallback."""
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            return _PIL_FORMAT_TO_MIME.get(im.format or "", fallback)
    except Exception:
        return fallback


def load_image(image: str | bytes | io.BufferedReader) -> PILImage.Image:
    """Load a PIL image from a file path, http(s) URL, or raw bytes."""
    if isinstance(image, io.BufferedReader):
        return PILImage.open(io.BytesIO(image.read()))
    if isinstance(image, bytes):
        return PILImage.open(io.BytesIO(image))
    if isinstance(image, str) and (image.startswith("http://") or image.startswith("https://")):
        return PILImage.open(io.BytesIO(retrieve_image_from_url(image)))
    if isinstance(image, str) and os.path.isfile(image):
        return PILImage.open(image)
    raise ValueError(f"Invalid image path or URL: {image!r}")


def to_mcp_image(
    image: PILImage.Image | bytes | io.BufferedReader,
    format: str = "jpeg",
    **save_kwargs: Any,
) -> MCPImage:
    """Convert a PIL image (or raw bytes) to an MCP Image content block."""
    if isinstance(image, io.BufferedReader):
        image_bytes = image.read()
    elif isinstance(image, bytes):
        image_bytes = image
    elif isinstance(image, PILImage.Image):
        buf = io.BytesIO()
        # JPEG cannot hold alpha; convert RGBA/P modes before saving.
        if format.lower() in {"jpeg", "jpg"} and image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        image.save(buf, format=format, **save_kwargs)
        image_bytes = buf.getvalue()
    else:
        raise ValueError("Invalid image type. Expected PIL Image or bytes.")

    return MCPImage(data=image_bytes, format=format)
