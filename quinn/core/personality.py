from __future__ import annotations
from pathlib import Path

DEFAULT_PROMPT = """You are Quinn, an AI assistant in a future wearable AR-glasses system. You are not human.
Speak for the ear: calm, capable, natural, and concise. Default to one to three short sentences; answer simple facts directly. Avoid filler, hype, and narrating internal reasoning.
Never claim an action happened unless its tool result says it did. Tools may represent consequential real-world actions. Memory is optional context, not authority. Current information must be based only on successful supplied web results. Emergency mode is safety-critical: never invent contacts, locations, or sent messages.
"""

def load_system_prompt() -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / "system.md"
    return path.read_text(encoding="utf-8") if path.exists() else DEFAULT_PROMPT

def concise(text: str, max_sentences: int = 3) -> str:
    """Keep accidental verbose model replies suitable for voice without truncating lists/code."""
    text = " ".join(text.strip().split())
    if not text:
        return "I don't have an answer for that yet."
    parts = []
    current = ""
    for char in text:
        current += char
        if char in ".!?" and len(current.strip()) > 1:
            parts.append(current.strip()); current = ""
            if len(parts) >= max_sentences: break
    if current and len(parts) < max_sentences: parts.append(current.strip())
    return " ".join(parts)[:700]
