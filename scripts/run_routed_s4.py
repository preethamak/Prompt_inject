#!/usr/bin/env python
from __future__ import annotations

from autoresearch.data import load_split
from autoresearch.runner import generate_candidates, run_single


def main() -> None:
    split = load_split()
    cfg = generate_candidates("small")[-1]
    result = run_single(cfg, split)

    print("\n=== ROUTED S4 RESULT ===")
    for key, value in sorted(result.items()):
        if (
            key in {
                "run_id",
                "status",
                "protocol",
                "selected_threshold",
                "routing_margin",
                "routing_tfidf_threshold",
                "routing_final_threshold",
                "routing_val_used_fusion_fraction",
                "routing_test_used_fusion_fraction",
                "elapsed_s",
                "beats_baseline",
            }
            or key.startswith("val_routed")
            or key.startswith("test_routed")
            or key.startswith("val_tfidf")
            or key.startswith("val_")
            or key.startswith("test_")
        ):
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
