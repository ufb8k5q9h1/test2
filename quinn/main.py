from __future__ import annotations

import asyncio
import logging

from quinn.config import load_settings
from quinn.core.quinn import Quinn


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


async def run() -> None:
    settings = load_settings()
    quinn = Quinn(settings)

    print()
    print("╭──────────────────────────────╮")
    print("│          QUINN ONLINE        │")
    print("╰──────────────────────────────╯")
    print()
    print(f"LLM: {settings.ollama_model}")
    print("Type 'exit' or press Ctrl-C to shut down.")
    print()

    try:
        while True:
            try:
                text = input("You: ")

            except EOFError:
                print()
                break

            except KeyboardInterrupt:
                print()
                break

            text = text.strip()

            if not text:
                continue

            if text.lower() in {
                "exit",
                "quit",
                "shutdown",
            }:
                break

            try:
                answer = await quinn.handle(
                    text
                )

                if answer:
                    print(
                        f"Quinn: {answer}"
                    )
                    print()

            except KeyboardInterrupt:
                print()
                break

            except Exception:
                logging.getLogger(
                    "quinn"
                ).exception(
                    "request_failed"
                )

                print(
                    "Quinn: I hit an internal error "
                    "handling that."
                )
                print()

    finally:
        print("Shutting Quinn down...")

        try:
            await quinn.shutdown()

        except Exception:
            logging.getLogger(
                "quinn"
            ).exception(
                "shutdown_failed"
            )

        print("Quinn offline.")


def main() -> None:
    try:
        asyncio.run(run())

    except KeyboardInterrupt:
        # Python 3.14 can propagate KeyboardInterrupt from
        # asyncio.run() even after the coroutine has cleaned up.
        # Keep the terminal clean.
        pass


if __name__ == "__main__":
    main()