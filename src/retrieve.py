"""
Week 1 — Stage 1: Plain embedding retrieval on the wardrobe corpus.

No filtering, no vector database — just numpy, so every step stays visible.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# 1. Load the wardrobe corpus
# ---------------------------------------------------------
with open("wardrobe.json", "r") as f:
    wardrobe = json.load(f)

print(f"Loaded {len(wardrobe)} wardrobe items.\n")

# ---------------------------------------------------------
# 2. Load the embedding model
# ---------------------------------------------------------
# all-MiniLM-L6-v2 outputs 384-dimensional vectors.
# It's small and fast — good for learning, good enough for a
# few dozen items. Larger models trade speed for slightly
# richer semantic understanding.
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# 3. Embed every item's description
# ---------------------------------------------------------
descriptions = [item["description"] for item in wardrobe]
item_embeddings = model.encode(descriptions)  # shape: (20, 384)

print(f"Embedded {item_embeddings.shape[0]} items, "
      f"each as a {item_embeddings.shape[1]}-dim vector.\n")

# ---------------------------------------------------------
# 4. Cosine similarity function
# ---------------------------------------------------------
def cosine_similarity(a, b):
    """
    a: single vector (query embedding)
    b: matrix of vectors (all item embeddings)
    Returns a similarity score for each row in b.
    """
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(b_norm, a_norm)  # shape: (num_items,)

# ---------------------------------------------------------
# 5. Embed a query and rank items
# ---------------------------------------------------------
def search(query, top_k=5):
    query_embedding = model.encode(query)
    scores = cosine_similarity(query_embedding, item_embeddings)

    # Get indices of the top_k highest scores, sorted descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    print(f'Query: "{query}"\n')
    for rank, idx in enumerate(top_indices, start=1):
        item = wardrobe[idx]
        print(f"{rank}. [{scores[idx]:.3f}] {item['id']} — {item['description']}")
    print()

# ---------------------------------------------------------
# 6. Try it out
# ---------------------------------------------------------
if __name__ == "__main__":
    search("effortless outfit for a beach vacation")
    search("something cozy for a cold night out")
    search("elegant but not too dressy, good for a date")
