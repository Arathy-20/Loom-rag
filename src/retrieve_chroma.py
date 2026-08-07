"""
Week 2 — Stage 2, part 2: Filter + embed, combined via Chroma.

Chroma stores both the embedding vector AND the metadata (category,
season, weather) for each item together. This lets us do the filter
pass and the embedding search in a single query — Chroma applies the
metadata filter FIRST (cheap), then runs cosine similarity only over
the items that survived filtering (expensive, but now on a much
smaller set).

This replaces the plain-numpy version from Week 1 for anything that
needs filtering. The numpy version (retrieve.py) still stands as your
reference for what's happening "under the hood" with raw cosine
similarity.
"""

import json
import os
import chromadb
from chromadb.utils import embedding_functions

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "wardrobe.json")

# ---------------------------------------------------------
# 1. Load the wardrobe corpus
# ---------------------------------------------------------
with open(DATA_PATH, "r") as f:
    wardrobe = json.load(f)

# ---------------------------------------------------------
# 2. Set up Chroma with the same embedding model as Week 1
# ---------------------------------------------------------
# Chroma needs an "embedding function" wrapper so it knows how to
# turn text into vectors internally, using the same model as before.
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# in-memory client — nothing persists to disk yet, that's a later concern
client = chromadb.Client()

# if this script has been run before in the same session, the
# collection may already exist — delete and recreate for a clean run
try:
    client.delete_collection("wardrobe")
except Exception:
    pass

collection = client.create_collection(
    name="wardrobe",
    embedding_function=embedding_fn,
)

# ---------------------------------------------------------
# 3. Add items to Chroma — description gets embedded automatically,
#    metadata (category, season, weather) is stored alongside it
# ---------------------------------------------------------
# Chroma metadata values must be strings/numbers/bools, not lists —
# so we flatten season/weather lists into comma-separated strings
# and rely on Chroma's "$contains" style matching being unavailable
# for exact list membership. We handle multi-value fields with a
# simple "any season matches" approach at query time instead.

collection.add(
    ids=[item["id"] for item in wardrobe],
    documents=[item["description"] for item in wardrobe],
    metadatas=[
        {
            "category": item["category"],
            "season": ",".join(item["season"]),
            "weather": ",".join(item["weather"]),
        }
        for item in wardrobe
    ],
)

print(f"Added {collection.count()} items to Chroma.\n")


# ---------------------------------------------------------
# 4. Search: filter first, then embed-rank within the filtered set
# ---------------------------------------------------------
def search(query, category=None, weather=None, top_k=5):
    """
    category / weather: exact-match filters, applied before embedding
    search runs. Season is left out of Chroma filtering for now since
    Chroma's `where` clause does exact match, not "is this value inside
    a comma-separated field" — that needs a different approach we'll
    revisit if it matters later.
    """
    where_clause = {}
    if category is not None:
        where_clause["category"] = category

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where_clause if where_clause else None,
    )

    print(f'Query: "{query}"  (filter: category={category})\n')
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results["distances"][0]

    if not ids:
        print("  No items matched the filter.\n")
        return

    for rank, (item_id, doc, dist) in enumerate(zip(ids, docs, distances), start=1):
        # Chroma returns distance (lower = more similar) by default,
        # not similarity. Convert for consistency with Week 1's output.
        similarity = 1 - dist
        print(f"{rank}. [{similarity:.3f}] {item_id} — {doc}")
    print()


if __name__ == "__main__":
    # same query as Week 1, no filter — should roughly match old results
    search("effortless outfit for a beach vacation")

    # same query, but now filtered to only tops — a real use of Stage 2
    search("effortless outfit for a beach vacation", category="top")
