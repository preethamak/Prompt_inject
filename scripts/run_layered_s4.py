#!/usr/bin/env python
from __future__ import annotations

from autoresearch.data import load_split
from autoresearch.runner import generate_candidates, run_single


def main() -> None:
    split = load_split()
    cfg = generate_candidates("small")[-1]
    cfg.enable_defense = True
    result = run_single(cfg, split)

    print("\n=== LAYERED S4 RESULT ===")
    for key, value in sorted(result.items()):
        if (
            key in {
                "run_id",
                "status",
                "protocol",
                "selected_threshold",
                "defense_chunk_threshold",
                "elapsed_s",
                "beats_baseline",
            }
            or key.startswith("val_")
            or key.startswith("test_")
        ):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
