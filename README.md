# Loom

A hybrid retrieval-augmented generation (RAG) system for outfit styling — built as a deliberate learning project to understand the full RAG pipeline at interview depth, not to ship a production app.

## What it does

Given a natural-language occasion (e.g. *"beach vacation, mix of daytime and dinner"*), Loom retrieves matching outfit pieces from a personal wardrobe corpus, identifies gaps, and fills them with suggestions from a separate shop catalog — always explicit about what's owned vs. what needs buying.

## Why RAG (not just tag-filtering)

Real closet apps mostly rely on tag-filtering + vision-based auto-tagging. RAG is used here deliberately, not because it's the only way to build this, but because two things justify it:

1. **Fuzzy query handling** — a query like *"effortless but put-together"* has no matching keywords in structured tags. Embedding similarity captures meaning, not just attributes.
2. **Two-corpus cross-referencing** — matching gaps between a wardrobe and a shop catalog benefits from semantic similarity, not just rule-based category matching.

## Architecture — hybrid retrieval, four stages

1. **Structured filter pass** (rule-based) — narrows candidates using hard attributes (category, season, weather) before any semantic matching runs. Cheap and eliminates obviously wrong items early.
2. **Embedding similarity pass** — embeds item descriptions + query with `sentence-transformers` (`all-MiniLM-L6-v2`), ranks by cosine similarity, returns top-k. Catches fuzzy intent fixed tags can't.
3. **Gap-filling against shop corpus** — repeats filter → embed, restricted to categories missing from wardrobe results.
4. **Generation** — an LLM (Gemini) reasons only over retrieved items, never inventing anything outside the two corpora.

Filtering first, embeddings second: hard constraints get enforced cheaply before semantic search runs over a much smaller, already-safe candidate set (~15–20 items instead of 200) — mirroring how production RAG systems are typically built.

## Project structure

```
loom-rag/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── wardrobe.json
│   └── shop_catalog.json
└── src/
    ├── retrieve.py           # Week 1 — plain numpy cosine similarity (reference implementation)
    ├── filter.py              # Week 2 — standalone rule-based filter logic
    ├── retrieve_chroma.py     # Week 2 — filter + embed, combined via Chroma
    └── retrieve_gap_fill.py   # Week 3 — wardrobe search + gap detection + shop search
```

## Tools

- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** Chroma (metadata support alongside vectors)
- **Generation:** Gemini
- **Similarity metric:** cosine similarity

## Known limitations (to revisit in Week 5)

- **Multi-value metadata filtering is unreliable.** Weather and season are stored as comma-joined strings (e.g. `"hot,humid"`) so they can be saved as Chroma metadata. Chroma's `where` clause does exact string matching, not substring/contains matching, so filtering on a single value (e.g. `weather="hot"`) silently fails to match items with multiple values. The `weather` parameter was deliberately left out of `retrieve_chroma.py`'s `search()` signature rather than ship it broken. Proper fix: one-hot boolean metadata fields per value (`weather_hot`, `weather_cold`, etc.).
- **Gap detection has no formality/appropriateness signal.** Gap detection (`retrieve_gap_fill.py`) only measures topical similarity via embedding score, not suitability. Example: a casual hoodie cleared the coverage threshold for a "formal winter dinner" query, because "cold weather" and "night" overlapped semantically even though "casual" and "formal" are in tension. No `formality` field currently exists in the schema. Whether this is worth fixing with a new field vs. handled at the generation stage is an open question, to be decided based on how often it shows up in Week 5's evaluation set.

## Build log

### Week 1 — Plain embedding retrieval (Stage 1)
- Hand-built a 20-item wardrobe corpus (`wardrobe.json`)
- Embedded descriptions with `all-MiniLM-L6-v2`
- Implemented cosine similarity from scratch (numpy, no vector DB yet)
- Verified retrieval ranks by semantic meaning, not keyword overlap

### Week 2 — Vector store & structured filtering (Stage 2)
- Cleaned inconsistent category values in the wardrobe corpus
- Built standalone rule-based filter (`filter.py`)
- Migrated storage into Chroma, combining filter + embed in one query
- Found and documented the multi-value metadata filtering limitation

### Week 3 — Second corpus & gap-filling (Stage 3)
- Hand-built an 18-item shop catalog, deliberately covering wardrobe gaps
- Implemented category-coverage gap detection with a similarity threshold
- Chained wardrobe search → gap detection → shop search
- Found and documented the formality/appropriateness limitation

### Week 4 — Grounded generation (Stage 4)
*In progress*

### Week 5 — Evaluation
*Not started*
