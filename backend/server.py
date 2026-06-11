"""mcpimagine — MCP image generation service powered by Gemini "Nano Banana".

Exposes generate_image / edit_image tools over MCP Streamable HTTP. Generated
images are persisted and served back as HTTPS download links; an inline preview
is returned with every result. TLS/routing is handled by Caddy in front.
"""

import asyncio
import contextlib
import contextvars
import datetime
import logging
import os
import re
import time
from pathlib import Path

import sentry_sdk
import uvicorn
from dotenv import load_dotenv
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from image_tools import DEFAULT_MODEL, register_image_tools

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional Sentry
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(dsn=sentry_dsn, enable_logs=True)
    logger.info("Sentry initialized")

MCP_TOKEN_CTX = contextvars.ContextVar("mcp_token", default=None)

MCP_NAME = os.getenv("MCP_NAME", "imagine")
_safe_name = re.sub(r"[^a-z0-9_-]", "-", MCP_NAME.lower()).strip("-") or "service"
BASE_PATH = f"/{_safe_name}"
STREAM_PATH = f"{BASE_PATH}/"
logger.info("Service name: %s  stream path: %s", _safe_name, STREAM_PATH)

# --- Storage layout -------------------------------------------------------
DATA_DIR = Path(os.getenv("MCP_DATA_DIR", "./data")).resolve()
IMAGES_DIR = DATA_DIR / "images"
REQUESTS_DIR = DATA_DIR / "requests"
for d in (IMAGES_DIR, REQUESTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

ASSETS_ROUTE = f"{BASE_PATH}/assets"

# --- Public URL config (for download links) -------------------------------
PUBLIC_BASE_URL = (os.getenv("MCP_PUBLIC_BASE_URL") or "").rstrip("/") or None
PUBLIC_ASSET_BASE_URL = (os.getenv("MCP_PUBLIC_ASSET_BASE_URL") or "").rstrip("/") or None
if not PUBLIC_ASSET_BASE_URL and PUBLIC_BASE_URL:
    PUBLIC_ASSET_BASE_URL = f"{PUBLIC_BASE_URL}{ASSETS_ROUTE}"
if not PUBLIC_ASSET_BASE_URL:
    logger.warning(
        "Neither MCP_PUBLIC_BASE_URL nor MCP_PUBLIC_ASSET_BASE_URL is set; tools will "
        "return generated images inline only, with no downloadable HTTPS URL."
    )
# Host-side path of the images dir (for display in tool responses). Inside the
# container IMAGES_DIR is the mount target; this tells callers where the same
# files live on the host.
HOST_IMAGES_DIR = (os.getenv("MCP_HOST_IMAGES_DIR") or "").rstrip("/") or None

# Tokens (also used to redact access logs). The middleware re-reads these itself.
_TOKENS = {t.strip() for t in os.getenv("MCP_TOKENS", "").split(",") if t.strip()}

# Disk-cleanup policy for persisted images + request logs.
IMAGE_MAX_AGE_DAYS = int(os.getenv("IMAGE_MAX_AGE_DAYS", "30"))
REQUESTS_MAX_FILES = int(os.getenv("REQUESTS_MAX_FILES", "10000"))
CLEANUP_INTERVAL_S = int(os.getenv("CLEANUP_INTERVAL_S", "21600"))  # 6h

# --- Transport security (DNS-rebinding protection behind Caddy) ------------
PORT = int(os.getenv("PORT", "8012"))
_allowed_hosts = ["localhost", "127.0.0.1", "0.0.0.0", _safe_name, f"mcp-{_safe_name}"]
_allowed_hosts += [f"{h}:{PORT}" for h in list(_allowed_hosts)]
_allowed_origins = ["http://localhost", "http://127.0.0.1"]
_extra_hosts = os.getenv("MCP_ALLOWED_HOSTS", "scriptlab.duckdns.org")
for h in [x.strip() for x in _extra_hosts.split(",") if x.strip()]:
    _allowed_hosts.append(h)
    _allowed_origins.append(f"https://{h}")
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=_allowed_hosts,
    allowed_origins=_allowed_origins,
)

mcp = FastMCP(
    _safe_name,
    streamable_http_path=STREAM_PATH,
    json_response=True,
    transport_security=transport_security,
)

# Suppress noisy expected disconnect errors.
class _StreamErrorFilter(logging.Filter):
    def filter(self, record):
        return "ClosedResourceError" not in str(record.getMessage())

logging.getLogger("mcp.server.streamable_http").addFilter(_StreamErrorFilter())


# Redact URL/query tokens from access logs (MCP_ALLOW_URL_TOKENS=true means tokens
# can appear in the request line). Without this, `docker logs` would leak them.
class _TokenRedactingFilter(logging.Filter):
    def __init__(self, tokens):
        super().__init__()
        self._tokens = sorted((t for t in tokens if t), key=len, reverse=True)

    def filter(self, record):
        if self._tokens:
            try:
                msg = record.getMessage()
                redacted = msg
                for t in self._tokens:
                    redacted = redacted.replace(t, "***")
                if redacted != msg:
                    record.msg = redacted
                    record.args = ()
            except Exception:
                pass
        return True


# Apply redaction at the ROOT handler so it covers every logger that propagates
# here — our middleware's own logs AND uvicorn's access logs (uvicorn is started
# with log_config=None in main(), so its records propagate to this handler).
_redaction_filter = _TokenRedactingFilter(_TOKENS)
for _handler in logging.getLogger().handlers:
    _handler.addFilter(_redaction_filter)
logging.getLogger().addFilter(_redaction_filter)

# --- Register tools + resources -------------------------------------------
register_image_tools(mcp, IMAGES_DIR, REQUESTS_DIR, PUBLIC_ASSET_BASE_URL, HOST_IMAGES_DIR)


@mcp.resource(
    f"{_safe_name}://documentation",
    name="Imagine MCP Documentation",
    description="How to use the Nano Banana image generation MCP server",
    mime_type="text/markdown",
)
def get_documentation_resource() -> str:
    return f"""# Imagine MCP — Gemini Nano Banana image generation

## Tools
- `generate_image(prompt, aspect_ratio="1:1", image_size="2K", model="pro", grounding=False)`
  Text -> image.
- `edit_image(prompt, image_urls=[], image_base64=[], aspect_ratio="1:1", image_size="2K", model="pro")`
  Input image(s) + instruction -> edited / composed image.

## Models
- `pro`  = gemini-3-pro-image-preview (Nano Banana Pro) — best quality + text rendering. Default.
- `flash`= gemini-3.1-flash-image-preview (Nano Banana 2) — faster.

## Output
Each call returns an inline downscaled preview plus an HTTPS URL to the persisted
full-resolution image. Generation usually takes 15-25s. Images carry a SynthID watermark.

Aspect ratios: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9.
Image sizes: 1K, 2K, 4K.
"""

# --- ASGI app -------------------------------------------------------------
mcp_asgi = mcp.streamable_http_app()


def _prune_once():
    """Delete images older than the retention window and cap request-log count."""
    removed = 0
    if IMAGE_MAX_AGE_DAYS > 0:
        cutoff = time.time() - IMAGE_MAX_AGE_DAYS * 86400
        for p in IMAGES_DIR.glob("*"):
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    removed += 1
            except OSError:
                pass
    try:
        logs = sorted(
            (f for f in REQUESTS_DIR.glob("*.json") if f.is_file()),
            key=lambda f: f.stat().st_mtime,
        )
        for f in logs[: max(0, len(logs) - REQUESTS_MAX_FILES)]:
            try:
                f.unlink()
            except OSError:
                pass
    except OSError:
        pass
    if removed:
        logger.info("cleanup: removed %d images older than %d days", removed, IMAGE_MAX_AGE_DAYS)


async def _cleanup_loop():
    while True:
        try:
            await asyncio.to_thread(_prune_once)  # filesystem scan off the event loop
        except Exception as e:  # never let cleanup kill the loop
            logger.warning("cleanup loop error: %s", e)
        await asyncio.sleep(CLEANUP_INTERVAL_S)


@contextlib.asynccontextmanager
async def lifespan(_: Starlette):
    cleanup_task = asyncio.create_task(_cleanup_loop())
    try:
        async with mcp.session_manager.run():
            yield
    finally:
        cleanup_task.cancel()


async def health_check(request):
    healthy = bool(os.getenv("GEMINI_API_KEY")) and IMAGES_DIR.is_dir()
    return JSONResponse(
        {
            "status": "healthy" if healthy else "unhealthy",
            "service": _safe_name,
            "default_model": DEFAULT_MODEL,
            "gemini_api_key_configured": bool(os.getenv("GEMINI_API_KEY")),
            "public_asset_base_url": PUBLIC_ASSET_BASE_URL,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        },
        status_code=200 if healthy else 503,
    )


app = Starlette(
    routes=[
        Route("/health", health_check, methods=["GET"]),
        Route(f"{BASE_PATH}/health", health_check, methods=["GET"]),
        Mount(ASSETS_ROUTE, app=StaticFiles(directory=IMAGES_DIR, check_dir=False), name="assets"),
        Mount("/", app=mcp_asgi),
    ],
    lifespan=lifespan,
)


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Bearer/URL token gate for everything under BASE_PATH.

    Health and asset (/imagine/assets/*) routes stay public — assets use
    unguessable UUID filenames. Tokens come from MCP_TOKENS (comma-separated).
    MCP_REQUIRE_AUTH=true rejects requests without a valid token; MCP_ALLOW_URL_TOKENS=true
    also accepts ?token= and /imagine/<token>/ forms.
    """

    def __init__(self, app):
        super().__init__(app)
        raw = os.getenv("MCP_TOKENS", "")
        self.allowed_tokens = {t.strip() for t in raw.split(",") if t.strip()}
        self.allow_url_tokens = os.getenv("MCP_ALLOW_URL_TOKENS", "").lower() in ("1", "true", "yes")
        self.require_auth = os.getenv("MCP_REQUIRE_AUTH", "").lower() in ("1", "true", "yes")
        if self.require_auth and not self.allowed_tokens:
            logger.warning("MCP_REQUIRE_AUTH=true but MCP_TOKENS empty -> all requests rejected")
        elif not self.allowed_tokens:
            logger.warning("MCP_TOKENS empty -> token auth DISABLED for %s", BASE_PATH)

    async def dispatch(self, request, call_next):
        path = request.url.path or "/"
        if not path.startswith(BASE_PATH):
            return await call_next(request)
        # Public: health + assets (UUID filenames).
        if path in ("/health", f"{BASE_PATH}/health") or path.startswith(ASSETS_ROUTE):
            return await call_next(request)

        async def proceed(token_value, source):
            scope = MCP_TOKEN_CTX.set(token_value)
            request.state.mcp_token = token_value
            logger.info("Authenticated %s %s via %s", request.method, path, source)
            try:
                return await call_next(request)
            finally:
                MCP_TOKEN_CTX.reset(scope)

        if not self.allowed_tokens:
            if self.require_auth:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            return await call_next(request)

        # Bearer header
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        token = auth.split(" ", 1)[1].strip() if auth and auth.lower().startswith("bearer ") else None
        if token and token in self.allowed_tokens:
            return await proceed(token, "header")

        if self.allow_url_tokens:
            url_token = request.query_params.get("token")
            if url_token and url_token in self.allowed_tokens:
                return await proceed(url_token, "query")
            segs = [s for s in path.split("/") if s]
            if len(segs) >= 2 and segs[0] == _safe_name and segs[1] in self.allowed_tokens:
                candidate = segs[1]
                remainder = "/".join([_safe_name] + segs[2:])
                new_path = "/" + (remainder + "/" if path.endswith("/") and not remainder.endswith("/") else remainder)
                if new_path == BASE_PATH:
                    new_path = STREAM_PATH
                request.scope["path"] = new_path
                if "raw_path" in request.scope:
                    request.scope["raw_path"] = new_path.encode("utf-8")
                return await proceed(candidate, "path")

        detail = "Unauthorized" if self.allow_url_tokens else \
            "Use Authorization: Bearer <token>; URL/query tokens are not allowed"
        return JSONResponse({"detail": detail}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})


app.add_middleware(TokenAuthMiddleware)


def main():
    logger.info("Starting %s MCP server on port %s at %s", MCP_NAME, PORT, STREAM_PATH)
    uvicorn.run(
        app=app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=PORT,
        log_level=os.getenv("LOG_LEVEL", "info"),
        access_log=True,
        # log_config=None lets uvicorn's access logs propagate to the root handler,
        # which carries the token-redaction filter (see _redaction_filter above).
        log_config=None,
        proxy_headers=True,
        forwarded_allow_ips="*",
        timeout_keep_alive=660,
    )


if __name__ == "__main__":
    main()
