"""
Week 2 — Stage 2, part 1: Structured filter pass.

Pure rule-based logic, no AI involved. This runs BEFORE any embedding
search, to cheaply eliminate items that can't possibly match — e.g. a
winter coat should never even be considered for a "beach vacation" query.

This file is standalone (works on the raw wardrobe list) so you can see
the filtering logic clearly, separate from the vector store integration
that comes next in retrieve_chroma.py.
"""

import json
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wardrobe.json")


def load_wardrobe():
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def filter_items(items, category=None, season=None, weather=None):
    """
    Keep only items matching ALL provided constraints.
    Any constraint left as None is ignored (no filtering on that field).

    category: a single string, e.g. "top"
    season:   a single string, e.g. "summer" — item passes if this
              value is anywhere in the item's season list
    weather:  a single string, e.g. "hot" — same logic as season
    """
    results = []
    for item in items:
        if category is not None and item["category"] != category:
            continue
        if season is not None and season not in item["season"]:
            continue
        if weather is not None and weather not in item["weather"]:
            continue
        results.append(item)
    return results


if __name__ == "__main__":
    wardrobe = load_wardrobe()

    print(f"Full wardrobe: {len(wardrobe)} items\n")

    # Example: everything wearable for hot weather
    hot_items = filter_items(wardrobe, weather="hot")
    print(f"Filtered to weather='hot': {len(hot_items)} items")
    for item in hot_items:
        print(f"  {item['id']} — {item['description']}")
    print()

    # Example: only tops, for hot weather
    hot_tops = filter_items(wardrobe, category="top", weather="hot")
    print(f"Filtered to category='top' AND weather='hot': {len(hot_tops)} items")
    for item in hot_tops:
        print(f"  {item['id']} — {item['description']}")
