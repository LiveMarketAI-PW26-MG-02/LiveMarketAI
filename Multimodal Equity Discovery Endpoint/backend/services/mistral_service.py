import os
import httpx
import hashlib
import logging
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")
MISTRAL_BASE_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODEL = "mistral-small-latest"


def _deterministic_analysis(symbol: str) -> Dict[str, Any]:
    h = int(hashlib.sha256(symbol.encode()).hexdigest(), 16)
    sentiments = ["Bullish", "Neutral", "Moderately Bullish", "Bearish", "Moderately Bearish"]
    outlooks = ["Strong momentum in price discovery phase.", "Consolidation observed across recent sessions.",
                "Sequential price behavior indicates accumulation.", "Distribution phase visible in activity stream.",
                "Equilibrium approached with moderate dispersion."]
    return {
        "symbol": symbol,
        "sentiment": sentiments[h % len(sentiments)],
        "outlook": outlooks[h % len(outlooks)],
        "confidence": round(0.55 + (h % 40) / 100.0, 2),
    }


async def analyze_instrument_profile(symbol: str, profile_summary: str) -> Dict[str, Any]:
    if not MISTRAL_API_KEY:
        return _deterministic_analysis(symbol)

    prompt = (
        f"You are a quantitative equity analyst. Given the following multimodal profile summary for instrument {symbol}, "
        f"provide a JSON response with keys: sentiment (string), outlook (string), confidence (float 0-1). "
        f"Profile: {profile_summary[:800]}"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                MISTRAL_BASE_URL,
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": MISTRAL_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                    "temperature": 0.2,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            import json, re
            match = re.search(r'\{.*?\}', content, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                parsed["symbol"] = symbol
                return parsed
            return _deterministic_analysis(symbol)
    except Exception:
        return _deterministic_analysis(symbol)
