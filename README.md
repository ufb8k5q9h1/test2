# Quinn

Quinn is a local-first, voice-first AI assistant foundation for AR glasses and a wearable backpack computer. Today it runs in a terminal; its core is intentionally independent from Ollama, search, speech, audio transport, and hardware.

## Run it

Requirements: Python 3.11+ and [Ollama](https://ollama.com). The terminal demo uses only the Python standard library. From this repository:

```bash
cp .env.example .env
ollama pull gemma3:4b
ollama serve
python -m quinn.main
```

In another terminal, use `python -m quinn.main`. It offers `/status`, `/memory`, `/clear`, and `/exit`; normal language is the main interface. If Ollama is offline, Quinn tells you that directly rather than fabricating an answer.

## Architecture

`core/` owns behavior: routing, state, context limits, memory policy, and permissions. `llm/ollama.py` is only a language provider. `rag/` provides a private SQLite-backed local vector store with dependency-free hashed local embeddings; it can be replaced by Chroma or a model embedder without changing Core. `search/` has a swappable web-search interface. `tools/` validates tool calls through a registry and permission layer.

The system prompt is maintained in `quinn/prompts/system.md`, while dynamic time, state, trimmed history, relevant memories, successful search results, and available tools are assembled by Core.

## Memory and RAG

Say `remember that ...` to write a persistent memory. Say `what do you remember about ...` to retrieve relevant entries. SQLite data is stored at `quinn/data/memory/quinn.sqlite3` by default. The retrieval pipeline is query → local embedding → cosine-ranked top-k memories → context assembly. Only explicit memory language writes by default; Quinn does not save every turn. `MemoryManager` also supports deletion and updates for future UIs/tools.

## Web search

Quinn routes explicit search and likely-current questions (latest/current/news/weather/prices/scores/schedules/versions) to `WebSearchProvider`. It does not search simple math, memory, system commands, or ordinary conversation. The default DuckDuckGo HTML provider needs network access; a failed lookup is reported briefly and is never presented as current information.

## Tools and safety

Tools expose names, descriptions, parameter schemas, handlers and permissions. Search and location are read-only; memory is low-risk; messaging/recording are external actions; emergency controls are critical. Core supplies explicit intent for deterministic user commands—an LLM never runs arbitrary shell, Python, or filesystem code.

Location and messaging are development mocks. They never invent coordinates or send a real message. Recording is explicitly started only and never captures passive audio. The mock records no microphone audio.

`I'm in an emergency` activates the safe mock emergency flow, gathers mock location, logs only mock notifications to configured contacts, and starts configurable periodic mock location checks. `stop emergency mode` cancels it. Ambiguous phrases ask for confirmation. Emergency events are logged locally through standard structured log fields; no contacts or coordinates are fabricated.

## Future hardware and voice

`voice/` defines STT, TTS, wake-word and audio transport seams. Passive audio is never sent to the LLM or persisted by the Core. `hardware/esp32.py` defines the ESP32-S3 adapter boundary (`connect`, `disconnect`, `send_audio`, `receive_audio`, `send_command`, `get_status`) and includes a local mock. Add the serial/Wi-Fi framing and I2S integration there when the board arrives—Quinn Core does not need to change.

The backpack environmental recorder belongs behind `tools/recording.py`; it stays explicitly activated and separate from conversational glasses audio. Production GPS, phone/network location and SMS/messaging providers replace the mock classes, preserving their interfaces.

## Configuration

Copy `.env.example` to `.env`. `OLLAMA_MODEL` can be changed to `qwen3:4b` or another installed Ollama model. `OLLAMA_THINK=false` requests lower-latency no-thinking behavior for models/servers that support it. `MEMORY_TOP_K`, conversation timeout, emergency contacts, and location interval are configurable there.

## Tests

```bash
python -m pip install -U pytest
python -m pytest -q
```

Tests are fully local: they mock model/search behavior and use temporary local data, with no internet, credentials, hardware, message sending, or real emergency actions required.
