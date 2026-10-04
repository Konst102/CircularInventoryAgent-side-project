#!/usr/bin/env python3
"""
Circular Inventory Agent - Webhook Server
FastAPI-based webhook receiver for e-commerce return notifications.

Features:
- Async background processing
- API key authentication
- Rate limiting (100 req/hour per client)
- Structured logging (file-based, China-compatible)
- Minimal attack surface (no version info exposure)

Environment Variables:
    DASHSCOPE_API_KEY: DashScope API key
    WEBHOOK_API_KEY:   Client API key for webhook auth (default: sk-test-12345)
    HOST:              Bind host (default: 0.0.0.0)
    PORT:              Bind port (default: 8000)
"""

import os
import sys
import json
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from agent_v2 import Config, init_db, process_return, _mask_sensitive

try:
    from fastapi import FastAPI, HTTPException, Header, BackgroundTasks, Request
    from fastapi.responses import JSONResponse, HTMLResponse
    import uvicorn
except ImportError as exc:
    print("Missing dependencies. Install: pip install fastapi uvicorn")
    raise exc

# ---------------------------------------------------------------------------
# Security Configuration
# ---------------------------------------------------------------------------

# Client API keys for webhook authentication (NOT the AI provider key)
_DEFAULT_CLIENT_KEY = os.getenv("WEBHOOK_API_KEY", "sk-test-12345")
VALID_KEYS = {
    _DEFAULT_CLIENT_KEY: "demo",
    "sk-demo-china": "china_pilot"
}

# In-memory rate limiter: {client_id: [timestamp, ...]}
_request_counts: dict = {}


def _check_auth(api_key: str = Header(None, alias="X-API-Key")) -> str:
    """Validate client API key."""
    if not api_key or api_key not in VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return VALID_KEYS[api_key]


def _check_rate_limit(client_id: str) -> None:
    """Enforce 100 requests/hour per client."""
    import time
    now = time.time()
    window = now - 3600

    _request_counts.setdefault(client_id, [])
    _request_counts[client_id] = [t for t in _request_counts[client_id] if t > window]

    if len(_request_counts[client_id]) >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded (100/hour)")

    _request_counts[client_id].append(now)


# ---------------------------------------------------------------------------
# Logging (File-based, no external services)
# ---------------------------------------------------------------------------

def _log_event(message: str) -> None:
    """Append event to local log file."""
    try:
        log_path = Path(__file__).parent / "notifications.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass


def _log_error(context: str, detail: str = "") -> None:
    """Secure error logging — masks potential secrets."""
    try:
        log_path = Path(__file__).parent / "errors.log"
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe = _mask_sensitive(detail)[:120]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {context} | {safe}\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Circular Inventory Agent",
    description="AI-powered returns processing webhook API",
    version="2.2.0",
    docs_url=None,      # Disable /docs (reduces attack surface)
    redoc_url=None,     # Disable /redoc
)


@app.on_event("startup")
async def _startup() -> None:
    """Initialize database and directories on server start."""
    init_db()
    Path(Config.upload_dir).mkdir(parents=True, exist_ok=True)
    print("🚀 Server ready")


@app.get("/", response_class=HTMLResponse)
async def _root() -> str:
    """Minimal landing page."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Circular Inventory Agent</title>
        <style>
            body { font-family: system-ui, -apple-system, sans-serif;
                   max-width: 720px; margin: 60px auto; padding: 20px;
                   background: #f5f7fa; color: #1a1a2e; }
            .box { background: #fff; padding: 32px; border-radius: 12px;
                   box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
            h1 { margin-bottom: 8px; }
            .status { display: inline-flex; align-items: center; gap: 8px;
                      background: #10b981; color: #fff; padding: 6px 14px;
                      border-radius: 20px; font-size: 13px; font-weight: 600; }
            .dot { width: 8px; height: 8px; background: #fff;
                   border-radius: 50%; animation: pulse 2s infinite; }
            @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
            .endpoint { background: #1e293b; color: #e2e8f0; padding: 14px;
                        border-radius: 8px; margin: 10px 0;
                        font-family: ui-monospace, SFMono-Regular, monospace;
                        font-size: 13px; overflow-x: auto; }
            .note { background: #fef3c7; border-left: 4px solid #f59e0b;
                    padding: 12px 16px; margin: 20px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="box">
            <h1>🔄 Circular Inventory Agent</h1>
            <span class="status"><span class="dot"></span>Operational</span>
            <div class="note">
                ⚠️ Beta testing phase. Seeking e-commerce partners for pilot program.
            </div>
            <h3>API Endpoints</h3>
            <div class="endpoint">POST /webhook/return</div>
            <div class="endpoint">POST /webhook/shopify</div>
            <div class="endpoint">GET /stats</div>
            <div class="endpoint">GET /health</div>
            <p style="color:#64748b; font-size:14px; margin-top:20px;">
               AI vision condition assessment for returns processing.
            </p>
        </div>
    </body>
    </html>
    """


@app.post("/webhook/return")
async def _generic_webhook(
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Header(..., alias="X-API-Key")
) -> JSONResponse:
    """Generic webhook endpoint accepting JSON payload."""
    client = _check_auth(api_key)
    _check_rate_limit(client)

    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    return_id = data.get("return_id")
    sku = data.get("sku")
    image_url = data.get("image_url")

    if not all([return_id, sku, image_url]):
        return JSONResponse(
            {"status": "error", "message": "Missing required fields: return_id, sku, image_url"},
            status_code=400
        )

    _log_event(f"📥 New return: {return_id} | SKU: {sku} | Client: {client}")

    async def _process() -> None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as resp:
                    if resp.status != 200:
                        _log_error(f"Download failed: {return_id}", f"HTTP {resp.status}")
                        return

                    img_path = Path(Config.upload_dir) / f"{return_id}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(await resp.read())

                    result = await process_return(str(img_path), return_id, sku)
                    disp = result.get("disposition", "unknown")
                    grade = result.get("condition_grade", "unknown")

                    _log_event(f"✅ Processed: {return_id} | {grade} → {disp}")
        except Exception as exc:
            _log_error(f"Processing error: {return_id}", str(exc))

    background_tasks.add_task(_process)

    return JSONResponse({
        "status": "accepted",
        "return_id": return_id,
        "message": "Processing in background",
        "estimated_time": "3–5 seconds"
    })


@app.post("/webhook/shopify")
async def _shopify_webhook(
    background_tasks: BackgroundTasks,
    request: Request,
    api_key: str = Header(..., alias="X-API-Key")
) -> JSONResponse:
    """Shopify-compatible webhook endpoint."""
    client = _check_auth(api_key)
    _check_rate_limit(client)

    try:
        payload = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    return_id = payload.get("return_id", f"SHOPIFY-{datetime.now().strftime('%Y%m%d%H%M%S')}")
    order_id = payload.get("order_id", "unknown")
    sku = payload.get("sku", "unknown")
    image_url = payload.get("image_url", "")

    if not image_url:
        return JSONResponse({"status": "rejected", "error": "No image_url provided"}, status_code=400)

    _log_event(f"📥 Shopify: {return_id} (Order {order_id})")

    async def _process() -> None:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url, timeout=30) as resp:
                    if resp.status != 200:
                        _log_error(f"Shopify download failed: {return_id}", f"HTTP {resp.status}")
                        return

                    img_path = Path(Config.upload_dir) / f"{return_id}.jpg"
                    with open(img_path, "wb") as f:
                        f.write(await resp.read())

                    result = await process_return(str(img_path), return_id, sku)
                    disp = result.get("disposition", "unknown")
                    grade = result.get("condition_grade", "unknown")

                    _log_event(f"✅ Shopify done: {return_id} | {grade} → {disp}")
        except Exception as exc:
            _log_error(f"Shopify error: {return_id}", str(exc))

    background_tasks.add_task(_process)

    return JSONResponse({
        "status": "accepted",
        "source": "shopify",
        "return_id": return_id,
        "message": "Queued for AI processing"
    })


@app.get("/stats")
async def _stats(api_key: str = Header(..., alias="X-API-Key")) -> JSONResponse:
    """Return processing statistics."""
    _check_auth(api_key)

    try:
        conn = sqlite3.connect(Config.db_path)
        c = conn.cursor()
        c.execute("SELECT COUNT(*), SUM(cost_rmb) FROM return_assessments")
        total, cost = c.fetchone()
        c.execute("SELECT disposition, COUNT(*) FROM return_assessments GROUP BY disposition")
        by_disp = dict(c.fetchall())
        conn.close()
    except Exception as exc:
        _log_error("Stats query failed", str(exc))
        return JSONResponse({"error": "Database unavailable"}, status_code=500)

    return {
        "summary": {
            "total_processed": total or 0,
            "total_cost_rmb": round(cost, 4) if cost else 0.0,
            "avg_cost_per_item": round(cost / total, 4) if total else 0.0
        },
        "disposition_breakdown": by_disp,
        "system": "healthy"
    }


@app.get("/health")
async def _health() -> dict:
    """Minimal health check — no version or provider exposure."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not Config.validate():
        print("\nSet required environment variables and restart.")
        print("Example: export DASHSCOPE_API_KEY=sk-xxxxx")
        sys.exit(1)

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print(f"🌐 Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
