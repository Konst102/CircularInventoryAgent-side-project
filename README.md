# 🔄 Circular Inventory Agent

Autonomous AI system for e-commerce returns processing. Uses multimodal vision AI (Qwen-VL-Max) to assess item condition from customer-submitted photos and intelligently route inventory — resell, refurbish, recycle, or liquidate.

> **Status:** Technical MVP validated. Project paused after market analysis identified EU compliance costs (€12K–22K) incompatible with student-founder budget. Infrastructure preserved for future pivot.

---

## ✨ What It Does

| Step | Action |
|------|--------|
| 1. **Receive** | Webhook fires when customer uploads return photo |
| 2. **Analyze** | Qwen-VL-Max vision model grades condition (New → Damaged) |
| 3. **Decide** | AI routes to optimal channel: resell primary, secondary, refurbish, parts, recycle |
| 4. **Log** | SQLite audit trail for compliance & cost tracking |
| 5. **Notify** | File-based logging (async background tasks) |

**Unit economics:** ¥0.001 per assessment (~$0.00014)

---

## 🛠 Tech Stack

- **Python 3.12** — Asyncio, type hints
- **FastAPI** — Webhook receiver, auto-validation
- **Qwen-VL-Max** (DashScope/BaiLian) — Multimodal vision + reasoning
- **SQLite** — Audit-compliant logging
- **Alibaba Cloud** — Hong Kong deployment (1 vCPU / 1 GB)

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
export DASHSCOPE_API_KEY="sk-your-key-from-bailian-console"
export WEBHOOK_API_KEY="sk-your-client-key"
```

### 3. Run CLI test

```bash
python agent_v2.py --image ./photos/returned_pi.jpg --sku "raspberry-pi-4"
```

### 4. Start webhook server

```bash
python webhook_server.py
# Server runs at http://0.0.0.0:8000
```

### 5. Send test webhook

```bash
curl -X POST http://localhost:8000/webhook/return \
  -H "X-API-Key: sk-your-client-key" \
  -H "Content-Type: application/json" \
  -d '{
    "return_id": "RET-001",
    "sku": "pi-4-8gb",
    "image_url": "https://example.com/photo.jpg"
  }'
```

---

## 📁 Project Structure

```
.
├── agent_v2.py           # Core AI pipeline (vision → assessment → DB)
├── webhook_server.py     # FastAPI webhook receiver
├── requirements.txt      # Python dependencies
├── .env.example          # Environment variable template
├── data/                 # SQLite database (created at runtime)
└── uploads/              # Downloaded return photos (created at runtime)
```

---

## 🔒 Security Features

- **No hardcoded secrets** — all keys via environment variables
- **API key authentication** — per-client rate limiting (100 req/hour)
- **Input sanitization** — masked error logs (no key leakage)
- **Minimal attack surface** — `/docs` and `/redoc` disabled
- **Safe error responses** — generic messages to clients, detailed logs server-side

---

## 📊 Validation Results

Tested with 4 real product images (Raspberry Pi 5 variants):

| Photo | Condition | Disposition | Confidence |
|-------|-----------|-------------|------------|
| New | New | `resell_primary` | 1.00 |
| Good | Good | `refurbish` | 0.90 |
| Damaged | Damaged | `parts_harvest` | 0.95 |

---

## 🌍 Deployment

- **Cloud:** Alibaba Cloud Lightweight Application Server (Hong Kong)
- **OS:** Ubuntu 22.04
- **Process manager:** GNU Screen (24/7 session)
- **Firewall:** Port 8000 open, port 22 (SSH) only

---

## 📜 License

MIT — feel free to fork and adapt.

---

*Built with ¥35/month infrastructure and a lot of coffee.*
