"""
Week 3 — Stage 3: Gap-filling against the shop catalog.

Pipeline:
1. Search the wardrobe (filter + embed, same as Week 2).
2. Detect which categories are missing or weakly represented in the
   wardrobe results — this is the "gap detection" step.
3. Search the shop catalog, restricted to only those gap categories.
4. Present results clearly split: owned (wardrobe) vs. needs-buying (shop).

Gap detection uses the simplest defensible pattern: category coverage.
An occasion implies a set of needed categories (e.g. an outfit needs a
top + bottom, or a dress). If a needed category has zero wardrobe
matches above a minimum similarity, it's a gap.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
WARDROBE_PATH = os.path.join(DATA_DIR, "wardrobe.json")
SHOP_PATH = os.path.join(DATA_DIR, "shop_catalog.json")

# Minimum similarity for a wardrobe match to "count" as covering a
# category. Below this, we treat the category as effectively missing
# even if something technically got retrieved. Tune this by hand —
# it's a judgment call worth being able to justify in an interview.
# KNOWN LIMITATION: this only measures topical similarity, not
# suitability (e.g. formality) — see README.
MIN_COVERAGE_SCORE = 0.30


# ---------------------------------------------------------
# 1. Load both corpora
# ---------------------------------------------------------
with open(WARDROBE_PATH, "r") as f:
    wardrobe = json.load(f)

with open(SHOP_PATH, "r") as f:
    shop_catalog = json.load(f)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.Client()

for name in ("wardrobe", "shop"):
    try:
        client.delete_collection(name)
    except Exception:
        pass

wardrobe_collection = client.create_collection("wardrobe", embedding_function=embedding_fn)
shop_collection = client.create_collection("shop", embedding_function=embedding_fn)

wardrobe_collection.add(
    ids=[item["id"] for item in wardrobe],
    documents=[item["description"] for item in wardrobe],
    metadatas=[{"category": item["category"]} for item in wardrobe],
)

shop_collection.add(
    ids=[item["id"] for item in shop_catalog],
    documents=[item["description"] for item in shop_catalog],
    metadatas=[{"category": item["category"]} for item in shop_catalog],
)


# ---------------------------------------------------------
# 2. Search functions
# ---------------------------------------------------------
# KNOWN LIMITATION: only category is filtered here. Season/weather
# metadata exists on every item but is NOT applied — see README for
# the confirmed failure case (a summer-tagged dress ranking #1 for a
# winter query) and the root cause (multi-value metadata filtering).
def search_wardrobe(query, category=None, top_k=5):
    where_clause = {"category": category} if category else None
    results = wardrobe_collection.query(
        query_texts=[query], n_results=top_k, where=where_clause
    )
    return _format_results(results)


def search_shop(query, category=None, top_k=5):
    where_clause = {"category": category} if category else None
    results = shop_collection.query(
        query_texts=[query], n_results=top_k, where=where_clause
    )
    return _format_results(results)


def _format_results(results):
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    out = []
    for item_id, doc, meta, dist in zip(ids, docs, metas, distances):
        out.append({
            "id": item_id,
            "description": doc,
            "category": meta["category"],
            "similarity": 1 - dist,
        })
    return out


# ---------------------------------------------------------
# 3. Gap detection — coverage check
# ---------------------------------------------------------
def detect_gaps(wardrobe_results, needed_categories):
    """
    A category counts as covered if at least one wardrobe result in
    that category scores above MIN_COVERAGE_SCORE. Anything not
    covered is a gap.
    """
    covered = set()
    for item in wardrobe_results:
        if item["similarity"] >= MIN_COVERAGE_SCORE:
            covered.add(item["category"])

    gaps = needed_categories - covered
    return gaps


# ---------------------------------------------------------
# 4. Core pipeline — returns data (used by generate.py and the
#    printing wrapper below)
# ---------------------------------------------------------
def style_outfit_data(query, needed_categories, top_k=5):
    """
    Runs the full wardrobe -> gap detection -> shop pipeline and
    returns (owned_items, shop_items) as plain lists of dicts.
    No printing — callers decide how to present the result.
    """
    wardrobe_results = search_wardrobe(query, top_k=top_k)
    gaps = detect_gaps(wardrobe_results, needed_categories)

    shop_results = []
    for category in sorted(gaps):
        shop_results.extend(search_shop(query, category=category, top_k=2))

    return wardrobe_results, shop_results


# ---------------------------------------------------------
# 5. Printing wrapper — for running this file directly as a CLI check
# ---------------------------------------------------------
def style_outfit(query, needed_categories, top_k=5):
    print(f'Occasion: "{query}"')
    print(f"Needed categories: {sorted(needed_categories)}\n")

    owned_items, shop_items = style_outfit_data(query, needed_categories, top_k=top_k)

    print("OWNED (from your wardrobe):")
    for item in owned_items:
        print(f"  [{item['similarity']:.3f}] ({item['category']}) {item['id']} — {item['description']}")

    covered = {item["category"] for item in owned_items if item["similarity"] >= MIN_COVERAGE_SCORE}
    gaps = needed_categories - covered

    if not gaps:
        print("\nNo gaps detected — wardrobe covers all needed categories.\n")
        return

    print(f"\nGAPS DETECTED: {sorted(gaps)}")
    print("NEEDS BUYING (from shop catalog):")
    for item in shop_items:
        print(f"  [{item['similarity']:.3f}] ({item['category']}) {item['id']} — {item['description']}")
    print()


if __name__ == "__main__":
    print(f"Wardrobe: {wardrobe_collection.count()} items | Shop: {shop_collection.count()} items\n")

    # An occasion that should mostly be covered by the wardrobe
    style_outfit(
        "effortless outfit for a beach vacation",
        needed_categories={"top", "bottom", "dress", "footwear"},
    )

    # An occasion designed to expose gaps — wardrobe is thin on
    # elegant cold-weather/formal dinner pieces. In practice, this also
    # surfaces two known limitations (see README): the casual hoodie
    # passes gap-detection for "formal" due to no formality field, and
    # the summer-tagged dress ranks #1 despite being season-inappropriate,
    # since season/weather filtering isn't wired into this script yet.
    style_outfit(
        "elegant outfit for a formal winter dinner",
        needed_categories={"dress", "outerwear", "footwear", "accessory"},
    )
