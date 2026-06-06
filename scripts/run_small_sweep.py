#!/usr/bin/env python
from __future__ import annotations

from autoresearch.data import load_split
from autoresearch.runner import generate_candidates, run_sweep
from autoresearch.leaderboard import (
    clear_results,
    display_leaderboard,
    get_best_run,
    save_result,
)


def main() -> None:
    split = load_split()
    clear_results()

    run_sweep(
        split=split,
        candidates=generate_candidates("small"),
        on_result=save_result,
        verbose=True,
    )

    print("\n=== LEADERBOARD ===")
    display_leaderboard(top_n=10)

    print("\n=== BEST RUN ===")
    print(get_best_run())


if __name__ == "__main__":
    main()
