#!/usr/bin/env python3
"""Side-by-side: legacy patterns.py heuristics vs the catalog [S] detectors.

Run BOTH detection paths on the same trajectory so you can verify the catalog
detectors cover (and extend) what the dashboard's legacy heuristics report,
before deleting the duplicate path.

    python scripts/compare_detectors.py path/to/trajectory.json [more.json ...]
"""

from __future__ import annotations

import sys

from trajectory_visualizer.insight.loaders import load_trajectory
from trajectory_visualizer.insight.parser import parse_steps
from trajectory_visualizer.insight.patterns import (
    detect_fruitless_streaks,
    detect_tool_selection_antipatterns,
)
from trajectory_visualizer.insight.catalog_detectors import run_catalog_detectors, summarize

# Legacy heuristic -> the catalog [S] detector(s) covering the same concept.
OVERLAP = {
    "fruitless_streak (empty search output)": ["empty-result-churn", "search-loop"],
    "tool_selection_antipattern (shell read)": ["shell-over-tool"],
}


def _fired_map(results):
    return {r["id"]: len(r["detections"]) for r in results if r["status"] == "fired"}


def compare_one(path: str) -> None:
    raw = load_trajectory(path)
    if "_error" in raw:
        print(f"  ERROR: {raw['_error']}")
        return
    steps = parse_steps(raw)[:2000]
    traj = raw.get("trajectory", [])

    legacy_fruitless = detect_fruitless_streaks(steps, traj)
    legacy_toolsel = detect_tool_selection_antipatterns(steps)
    legacy = {
        "fruitless_streak (empty search output)": len(legacy_fruitless),
        "tool_selection_antipattern (shell read)": len(legacy_toolsel),
    }

    results = run_catalog_detectors(steps)
    fired = _fired_map(results)
    s = summarize(results)

    print(f"\n### {path}  ({len(steps)} steps)")
    print(f"  catalog [S]: {s['fired']} fired, {s['total_detections']} detections, "
          f"{s['clear']} clear, {s['gated']} gated\n")

    print("  OVERLAP (legacy concept -> catalog detector counts):")
    print(f"  {'legacy heuristic':40s} {'legacy':>7s}   catalog [S]")
    for concept, catalog_ids in OVERLAP.items():
        cat_str = ", ".join(f"{cid}={fired.get(cid, 0)}" for cid in catalog_ids)
        print(f"  {concept:40s} {legacy[concept]:>7d}   {cat_str}")

    overlap_ids = {cid for ids in OVERLAP.values() for cid in ids}
    net_new = {k: v for k, v in fired.items() if k not in overlap_ids}
    print("\n  CATALOG-ONLY detectors that fired (net-new coverage the dashboard lacked):")
    if net_new:
        for k, v in sorted(net_new.items(), key=lambda kv: -kv[1]):
            print(f"    {k:32s} x{v}")
    else:
        print("    (none)")


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        sys.exit(1)
    print("=" * 70)
    print("Legacy patterns.py  vs  catalog [S] detectors")
    print("=" * 70)
    for p in paths:
        compare_one(p)


if __name__ == "__main__":
    main()
