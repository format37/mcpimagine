#!/bin/bash
# Build and (re)deploy the mcp-imagine service.
set -e
cd "$(dirname "$0")"

# Ensure the shared reverse-proxy network exists.
if ! sudo docker network inspect mcp-shared >/dev/null 2>&1; then
    echo "📡 Creating mcp-shared network..."
    sudo docker network create mcp-shared
fi

mkdir -p ./data/images

echo "🛑 Stopping existing container..."
sudo docker compose down --remove-orphans 2>/dev/null || true

echo "🏗️  Building and starting..."
sudo docker compose up --build -d --force-recreate

echo "✅ mcp-imagine is up."
echo "   Internal endpoint : http://mcp-imagine:8012/imagine/"
echo "   Public endpoint   : https://scriptlab.duckdns.org/imagine/  (Bearer token required)"
echo "   Health            : sudo docker exec mcp-imagine python -c \"import urllib.request;print(urllib.request.urlopen('http://localhost:8012/health').read().decode())\""
