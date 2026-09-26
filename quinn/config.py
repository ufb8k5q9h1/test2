"""Runtime configuration loaded from environment variables (optionally .env)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

def _load_dotenv() -> None:
    path = ROOT.parent / ".env"
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).lower() in {"1", "true", "yes", "on"}

@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")
    ollama_think: bool = _bool("OLLAMA_THINK", False)
    log_level: str = os.getenv("QUINN_LOG_LEVEL", "INFO")
    memory_enabled: bool = _bool("MEMORY_ENABLED", True)
    memory_top_k: int = int(os.getenv("MEMORY_TOP_K", "5"))
    web_search_enabled: bool = _bool("WEB_SEARCH_ENABLED", True)
    conversation_timeout_seconds: int = int(os.getenv("CONVERSATION_TIMEOUT_SECONDS", "180"))
    emergency_location_interval_seconds: int = int(os.getenv("EMERGENCY_LOCATION_INTERVAL_SECONDS", "300"))
    emergency_enabled: bool = _bool("EMERGENCY_ENABLED", True)
    emergency_contacts: tuple[str, ...] = tuple(x.strip() for x in os.getenv("EMERGENCY_CONTACTS", "").split(",") if x.strip())
    data_dir: Path = Path(os.getenv("QUINN_DATA_DIR", str(ROOT / "data")))

def load_settings() -> Settings:
    _load_dotenv()
    # Reconstruct after reading .env; dataclass defaults are evaluated at import.
    return Settings(
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:4b"),
        ollama_think=_bool("OLLAMA_THINK", False), log_level=os.getenv("QUINN_LOG_LEVEL", "INFO"),
        memory_enabled=_bool("MEMORY_ENABLED", True), memory_top_k=int(os.getenv("MEMORY_TOP_K", "5")),
        web_search_enabled=_bool("WEB_SEARCH_ENABLED", True),
        conversation_timeout_seconds=int(os.getenv("CONVERSATION_TIMEOUT_SECONDS", "180")),
        emergency_location_interval_seconds=int(os.getenv("EMERGENCY_LOCATION_INTERVAL_SECONDS", "300")),
        emergency_enabled=_bool("EMERGENCY_ENABLED", True),
        emergency_contacts=tuple(x.strip() for x in os.getenv("EMERGENCY_CONTACTS", "").split(",") if x.strip()),
        data_dir=Path(os.getenv("QUINN_DATA_DIR", str(ROOT / "data"))),
    )
