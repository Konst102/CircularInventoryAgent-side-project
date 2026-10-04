#!/usr/bin/env python3
"""
Circular Inventory Agent - Core AI Processing Module
Autonomous returns processing using multimodal vision AI.

Features:
- Vision-based condition assessment (Qwen-VL-Max)
- Smart disposition routing (resell/refurbish/recycle)
- SQLite audit trail for compliance
- Cost tracking per inference

Environment Variables Required:
    DASHSCOPE_API_KEY: DashScope/BaiLian API key (sk-...)
"""

import os
import sys
import json
import base64
import asyncio
import aiohttp
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration loaded from environment variables."""
    provider: str = os.getenv("AI_PROVIDER", "dashscope")
    dashscope_key: str = os.getenv("DASHSCOPE_API_KEY", "")
    db_path: str = os.getenv("DB_PATH", "./data/returns.db")
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    cost_per_1k_tokens: float = 0.003

    @classmethod
    def validate(cls) -> bool:
        """Check that required secrets are configured."""
        if cls.provider == "dashscope" and not cls.dashscope_key:
            print("[Config Error] DASHSCOPE_API_KEY environment variable not set.")
            return False
        return True


# Electronics defect taxonomy for structured assessment
DEFECT_TAXONOMY = {
    "screen": ["scratch_minor", "scratch_deep", "crack_minor", "crack_major", "discoloration"],
    "casing": ["scratch_superficial", "dent_minor", "dent_major", "crack", "chipping"],
    "ports": ["charging_port_loose", "charging_port_broken", "audio_jack_issues", "button_malfunction"],
    "battery": ["swelling", "reduced_capacity", "not_charging", "fast_drain"],
    "accessories": ["missing_cable", "missing_adapter", "missing_case", "damaged_cable"],
    "functional": ["power_on_failure", "audio_issues", "connectivity_issues", "overheating"]
}

DISPOSITION_OPTIONS = [
    "resell_primary",
    "resell_secondary",
    "refurbish",
    "parts_harvest",
    "recycle",
    "liquidate"
]


def init_db() -> None:
    """Initialize SQLite database with audit-compliant schema."""
    Path(Config.db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS return_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            return_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            sku TEXT,
            condition_grade TEXT CHECK(condition_grade IN (
                'New', 'Like New', 'Good', 'Fair', 'Poor', 'Damaged', 'Unsellable'
            )),
            disposition TEXT CHECK(disposition IN (
                'resell_primary', 'resell_secondary', 'refurbish',
                'parts_harvest', 'recycle', 'liquidate'
            )),
            reasoning TEXT,
            recovery_value_eur TEXT,
            defects_detected TEXT,
            packaging_state TEXT,
            confidence_score REAL,
            tokens_used INTEGER,
            cost_rmb REAL,
            api_provider TEXT,
            raw_response TEXT
        )
    ''')
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON return_assessments(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_disposition ON return_assessments(disposition)")
    conn.commit()
    conn.close()


def _build_assessment_prompt() -> str:
    """Construct structured prompt for consistent JSON output."""
    return """Analyze this returned electronics item for circular inventory processing.

DEFECT CATEGORIES TO CHECK:
- screen: scratch_minor, scratch_deep, crack_minor, crack_major, discoloration
- casing: scratch_superficial, dent_minor, dent_major, crack, chipping
- ports: charging_port_loose, charging_port_broken, audio_jack_issues, button_malfunction
- battery: swelling, reduced_capacity, not_charging, fast_drain
- accessories: missing_cable, missing_adapter, missing_case, damaged_cable
- functional: power_on_failure, audio_issues, connectivity_issues, overheating

OUTPUT STRICT JSON (no markdown, no extra text):
{
  "condition_grade": "New|Like New|Good|Fair|Poor|Damaged|Unsellable",
  "confidence": 0.0-1.0,
  "defects_detected": ["category:specific_defect"],
  "packaging_state": "intact|damaged|missing|none",
  "functional_status": "fully_functional|partial_issues|non_functional",
  "disposition": "resell_primary|resell_secondary|refurbish|parts_harvest|recycle|liquidate",
  "reasoning": "1-2 sentence justification",
  "recovery_value_eur": "XX-XX"
}

Be objective. If uncertain, flag for manual review."""


def _mask_sensitive(text: str) -> str:
    """Redact potential API keys from error strings."""
    if not text:
        return text
    import re
    # Mask DashScope-style keys
    return re.sub(r"sk-[a-zA-Z0-9]{16,}", "[REDACTED]", text)


async def call_vision_api(image_base64: str) -> Dict[str, Any]:
    """Call Qwen-VL-Max for vision-based condition assessment.

    Returns:
        dict with keys: content (parsed assessment), tokens, cost_rmb, request_id
    Raises:
        Exception on API failure (message sanitized)
    """
    if not Config.dashscope_key:
        raise RuntimeError("DASHSCOPE_API_KEY not configured")

    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
    headers = {
        "Authorization": f"Bearer {Config.dashscope_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": Config.vision_model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": image_base64},
                        {"text": _build_assessment_prompt()}
                    ]
                }
            ]
        }
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
            if resp.status != 200:
                raw = await resp.text()
                safe_error = _mask_sensitive(raw)
                raise RuntimeError(f"API returned HTTP {resp.status}")

            data = await resp.json()

    usage = data.get("usage", {})
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    total_tokens = tokens_in + tokens_out
    cost = (total_tokens / 1000) * Config.cost_per_1k_tokens

    content = data.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", "")

    return {
        "content": content,
        "tokens": total_tokens,
        "cost_rmb": round(cost, 4),
        "request_id": data.get("request_id", "unknown")
    }


def parse_assessment(response_text: Any) -> Dict[str, Any]:
    """Parse and validate AI response into structured assessment.

    Handles markdown code blocks and list-wrapped responses.
    """
    try:
        text = response_text
        if isinstance(text, list) and text:
            text = text[0].get("text", "") if isinstance(text[0], dict) else str(text[0])

        if not isinstance(text, str):
            text = str(text)

        # Strip markdown
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].strip()

        parsed = json.loads(text)

        # Validate required fields with safe defaults
        required = {
            "condition_grade": "unknown",
            "disposition": "manual_review",
            "reasoning": "No reasoning provided",
            "recovery_value_eur": "0-0",
            "confidence": 0.0
        }
        for key, default in required.items():
            if key not in parsed:
                parsed[key] = default

        return parsed

    except Exception:
        return {
            "condition_grade": "unknown",
            "disposition": "manual_review",
            "reasoning": "AI response parsing failed",
            "recovery_value_eur": "0-0",
            "confidence": 0.0,
            "defects_detected": []
        }


def save_assessment(
    return_id: str,
    sku: str,
    assessment: Dict[str, Any],
    api_result: Dict[str, Any],
    customer_email: str = ""
) -> None:
    """Persist assessment to SQLite audit trail."""
    conn = sqlite3.connect(Config.db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR REPLACE INTO return_assessments (
                return_id, sku, condition_grade, disposition, reasoning,
                recovery_value_eur, defects_detected, packaging_state,
                confidence_score, tokens_used, cost_rmb, api_provider,
                customer_email, raw_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            return_id,
            sku,
            assessment.get("condition_grade"),
            assessment.get("disposition"),
            assessment.get("reasoning"),
            assessment.get("recovery_value_eur"),
            json.dumps(assessment.get("defects_detected", [])),
            assessment.get("packaging_state", "unknown"),
            assessment.get("confidence", 0.0),
            api_result.get("tokens", 0),
            api_result.get("cost_rmb", 0.0),
            Config.provider,
            customer_email,
            json.dumps(assessment)
        ))
        conn.commit()
    except Exception as e:
        # Log DB error without exposing data
        print(f"[DB Error] {str(e)[:60]}")
    finally:
        conn.close()


async def process_return(
    image_path: str,
    return_id: str,
    sku: str,
    customer_email: str = ""
) -> Dict[str, Any]:
    """End-to-end return processing pipeline.

    Args:
        image_path: Local path to return photo
        return_id: Unique return identifier
        sku: Product SKU
        customer_email: Optional customer contact (stored as-is, hash upstream if needed)

    Returns:
        Structured assessment dict
    """
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        api_result = await call_vision_api(image_b64)
        assessment = parse_assessment(api_result["content"])

        save_assessment(return_id, sku, assessment, api_result, customer_email)

        return assessment

    except Exception as e:
        safe_msg = _mask_sensitive(str(e))
        return {
            "condition_grade": "error",
            "disposition": "manual_review",
            "reasoning": f"Processing error: {safe_msg[:80]}",
            "recovery_value_eur": "0-0",
            "confidence": 0.0
        }


async def cli_mode() -> None:
    """Interactive CLI for local testing and development."""
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Circular Inventory Agent - CLI Mode"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to return photo")
    parser.add_argument("--sku", "-s", default="unknown", help="Product SKU")
    parser.add_argument("--return-id", "-r", default=None, help="Return ID (auto-generated if empty)")
    args = parser.parse_args()

    if not Config.validate():
        sys.exit(1)

    if not args.return_id:
        args.return_id = f"CLI-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    Path(Config.upload_dir).mkdir(parents=True, exist_ok=True)
    dest = Path(Config.upload_dir) / f"{args.return_id}.jpg"
    shutil.copy(args.image, dest)

    print("\n" + "=" * 50)
    print("🔄 Circular Inventory Agent v2.0")
    print("=" * 50)

    result = await process_return(str(dest), args.return_id, args.sku)

    print("\n📋 Assessment Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("disposition") != "manual_review":
        print(f"\n✅ {result.get('condition_grade', 'Unknown')} → {result.get('disposition', 'Unknown')}")
    else:
        print("\n⚠️  Flagged for manual review")


if __name__ == "__main__":
    init_db()
    if not Config.validate():
        sys.exit(1)
    asyncio.run(cli_mode())
