#!/usr/bin/env python3
"""Run all three Apple Watch external-information 10-variant experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_applewatch_external_info import main as run_gpt41  # noqa: E402
from scripts.run_applewatch_gpt5nano_medium_external_info import (  # noqa: E402
    main as run_gpt5nano_medium,
)
from scripts.run_applewatch_gpt5nano_minimal_external_info import (  # noqa: E402
    main as run_gpt5nano_minimal,
)


RUNS: list[tuple[str, Callable[[], None]]] = [
    ("gpt-4.1, temperature=1", run_gpt41),
    ("gpt-5-nano, temperature=1, reasoning=minimal", run_gpt5nano_minimal),
    ("gpt-5-nano, temperature=1, reasoning=medium", run_gpt5nano_medium),
]


def main() -> None:
    for label, run in RUNS:
        print(f"\n=== Starting {label} ===")
        run()
        print(f"=== Completed {label} ===")


if __name__ == "__main__":
    main()
