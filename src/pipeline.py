"""
Week 6: single entry point for the full Loom pipeline.

    style(query, needed_categories, season=None, weather=None, top_k=5)

Validates the inputs, runs retrieval (wardrobe -> gap detection -> shop),
then asks Gemini for grounded styling advice. Returns plain, JSON-ready
data and never prints, so an API layer (or a test) can call it directly.

Importing this module loads the embedding model and builds the Chroma
collections once (via retrieve_gap_fill). Behaviour is controlled by the
same environment switches used in the Week 5 experiments:
    LOOM_EMBED_MODE   description | tags
    LOOM_GAP_MODE     global | per_category
    LOOM_FILTER_MODE  off | on
The active settings are echoed back in every result under "config".
"""

from retrieve_gap_fill import (
    retrieve_for_occasion,
    KNOWN_CATEGORIES,
    SEASONS,
    WEATHERS,
    EMBED_MODE,
    GAP_MODE,
    FILTER_MODE,
    MIN_COVERAGE_SCORE,
)
from generate import generate_styling_advice


def style(query, needed_categories, season=None, weather=None, top_k=5, generate=True):
    """
    query:             natural-language occasion, e.g. "cozy outfit for a cold night out"
    needed_categories: iterable of categories the outfit needs, e.g. ["top", "bottom"]
    season / weather:  optional explicit constraints (hard filters when LOOM_FILTER_MODE=on)
    top_k:             how many wardrobe items to retrieve
    generate:          set False to skip the Gemini call (retrieval only, no API key needed)

    Raises ValueError for invalid inputs.
    """
    # ---- validation: fail loudly instead of silently returning nonsense ----
    query = (query or "").strip()
    if not query:
        raise ValueError("query must not be empty")

    needed = set(needed_categories or [])
    if not needed:
        raise ValueError("needed_categories must contain at least one category")
    unknown = needed - KNOWN_CATEGORIES
    if unknown:
        raise ValueError(f"unknown categories {sorted(unknown)}; valid: {sorted(KNOWN_CATEGORIES)}")

    if season is not None and season not in SEASONS:
        raise ValueError(f"unknown season {season!r}; valid: {list(SEASONS)}")
    if weather is not None and weather not in WEATHERS:
        raise ValueError(f"unknown weather {weather!r}; valid: {list(WEATHERS)}")
    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be an integer between 1 and 20")

    # ---- warnings: things that would otherwise fail silently ----
    warnings = []
    if (season or weather) and FILTER_MODE != "on":
        warnings.append(
            "season/weather were provided but LOOM_FILTER_MODE is off, so they were ignored."
        )

    # ---- retrieval ----
    retrieved = retrieve_for_occasion(query, needed, top_k=top_k, season=season, weather=weather)

    result = {
        "query": query,
        "needed_categories": sorted(needed),
        "season": season,
        "weather": weather,
        "config": {
            "embed_mode": EMBED_MODE,
            "gap_mode": GAP_MODE,
            "filter_mode": FILTER_MODE,
            "min_coverage_score": MIN_COVERAGE_SCORE,
        },
        "owned": retrieved["owned"],
        "shop": retrieved["shop"],
        "gaps": retrieved["gaps"],
        "generation": None,
        "warnings": warnings,
    }

    # ---- generation ----
    if generate:
        generation = generate_styling_advice(query, retrieved["owned"], retrieved["shop"])
        result["generation"] = generation
        if not generation["grounding"]["mentioned_ids"]:
            warnings.append(
                "the generated advice cited no item IDs, so the grounding check had nothing to verify."
            )

    return result


if __name__ == "__main__":
    import json

    demo = style(
        "elegant outfit for a formal winter dinner",
        ["dress", "outerwear", "footwear", "accessory"],
        season="winter",
    )
    print(json.dumps(demo, indent=2, ensure_ascii=False))
