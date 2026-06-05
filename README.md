# mcpimagine

MCP (Streamable HTTP) image-generation service for Google Gemini **"Nano Banana"**.
Each result returns an inline preview plus an HTTPS link to the persisted full-resolution image.

**Live:** `https://scriptlab.duckdns.org/imagine/` (Bearer-token auth)

## Tools

| tool | purpose |
|------|---------|
| `generate_image(prompt, aspect_ratio="1:1", image_size="2K", model="pro", grounding=False)` | text → image |
| `edit_image(prompt, image_urls=[], image_base64=[], aspect_ratio="1:1", image_size="2K", model="pro")` | source image(s) + instruction → image |
| `save_tool_notes` / `read_tool_notes` | file-backed notes |

`model`: `pro` = `gemini-3-pro-image-preview` (best quality, optional Google-Search grounding) ·
`flash` = `gemini-3.1-flash-image-preview` (faster).
Aspect ratios `1:1 2:3 3:2 3:4 4:3 4:5 5:4 9:16 16:9 21:9` · sizes `1K 2K 4K`.
Generation takes ~15–25 s; images carry a SynthID watermark and are auto-deleted after
`IMAGE_MAX_AGE_DAYS` (default 30).

## Run

```bash
cp .env.example .env     # set GEMINI_API_KEY and MCP_TOKENS
./compose.sh             # build + (re)start container on the mcp-shared network
./logs.sh                # tail logs
```

No host ports are published — Caddy reaches the container over `mcp-shared` by name.
Add the block from `Caddyfile.snippet` to the Caddy site, then
`sudo docker exec reverse-proxy caddy reload --config /etc/caddy/Caddyfile`.

## Connect a client

**Claude Desktop / Claude Code / `mcp-remote`** — Bearer header (preferred):

```json
{ "mcpServers": { "imagine": { "command": "npx",
  "args": ["mcp-remote", "https://scriptlab.duckdns.org/imagine/",
           "--header", "Authorization: Bearer <MCP_TOKEN>"] } } }
```

Claude Code one-liner: `claude mcp add --transport http imagine https://scriptlab.duckdns.org/imagine/ --header "Authorization: Bearer <MCP_TOKEN>" --scope user`

**Claude web (claude.ai)** — custom connectors support only OAuth/no-auth (no static
headers), so put the token in the URL path instead. Settings → Connectors →
**Add custom connector**, URL (keep the trailing slash, leave OAuth fields blank):

```
https://scriptlab.duckdns.org/imagine/<MCP_TOKEN>/
```

Requires a paid plan. Tokens in the URL are redacted from logs.

Generated asset URLs (`/imagine/assets/<uuid>.jpg`) are public via unguessable UUIDs.
Health: `curl https://scriptlab.duckdns.org/imagine/health`.

## Configuration (`.env`)

```bash
GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
MCP_TOKENS=...       # access token(s); python3 -c "import secrets;print(secrets.token_urlsafe(24))"
```

Operational defaults live in `docker-compose.yml`; optional overrides are listed in `.env.example`.

## Layout

```
backend/  server.py · image_tools.py · notes_tools.py · mcp_image_utils.py · request_logger.py · Dockerfile
docker-compose.yml · compose.sh · logs.sh · Caddyfile.snippet
```
