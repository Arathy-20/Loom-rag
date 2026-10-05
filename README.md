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
    ├── retrieve_gap_fill.py   # Week 3 — wardrobe search + gap detection + shop search
    └── generate.py            # Week 4 — grounded generation + hallucination check
```

## Tools

- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** Chroma (metadata support alongside vectors)
- **Generation:** Gemini
- **Similarity metric:** cosine similarity

## Known limitations (to revisit in Week 5)

- **Season/weather metadata exists but isn't applied during retrieval.** Both `search_wardrobe()` and `search_shop()` in `retrieve_gap_fill.py` only filter on `category` — season and weather are never checked, even though every item has that metadata. Confirmed failure case: for the query "elegant outfit for a formal winter dinner," the top-ranked dress (`w008`) is tagged `season: summer, weather: hot` — explicitly not winter-appropriate — yet it ranks #1 purely on embedding similarity to "dinner." This traces back to the multi-value metadata filtering issue below (weather/season are stored as comma-joined strings, which Chroma's exact-match `where` clause can't filter correctly), so the filter was left out rather than shipped broken. Unlike the formality gap below, this is fixable with data that already exists — it just isn't being used yet.
- **Multi-value metadata filtering is unreliable.** Weather and season are stored as comma-joined strings (e.g. `"hot,humid"`) so they can be saved as Chroma metadata. Chroma's `where` clause does exact string matching, not substring/contains matching, so filtering on a single value (e.g. `weather="hot"`) silently fails to match items with multiple values. This is the root cause of the season/weather issue above. Proper fix: one-hot boolean metadata fields per value (`weather_hot`, `weather_cold`, etc.), then wire the filter into both `retrieve_chroma.py` and `retrieve_gap_fill.py`.
- **Gap detection has no formality/appropriateness signal.** Gap detection only measures topical similarity via embedding score, not suitability. Example: a casual hoodie cleared the coverage threshold for the same "formal winter dinner" query, because "cold weather" and "night" overlapped semantically even though "casual" and "formal" are in tension. No `formality` field currently exists in the schema — unlike the issue above, there's no unused data to wire in; this would require adding a new field. Whether it's worth adding vs. handled at the generation stage is an open question, to be decided based on how often it shows up in Week 5's evaluation set.
- **The grounding check only verifies item IDs, not claims made about those items.** `generate.py`'s grounding check confirms every mentioned item ID exists in the retrieved set, but does not check whether descriptive claims attached to an item are actually supported by its retrieved description. Confirmed case: for the beach-vacation query, the model described `w005` ("white skorts... not too casual, good for a cafe date") as "comfortable" — a word that appears nowhere in the original description. The item itself wasn't hallucinated (ID-level grounding held), but an attribute of it was. A stricter check would need to verify generated claims against the retrieved description text itself (e.g. word-overlap checking, or a second LLM call auditing the first), which is meaningfully harder than ID matching. Worth measuring in Week 5: how often this happens, and whether it's worth the extra complexity to catch.

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
- Found and documented the season/weather and formality limitations

### Week 4 — Grounded generation (Stage 4)
- Built `generate.py`: Gemini call constrained to retrieved items only, via a system-instruction/user-message split (rules vs. data)
- Migrated from the deprecated `google-generativeai` SDK to `google-genai`
- Implemented an ID-level grounding check after generation
- Verified end-to-end: no hallucinated item IDs in testing so far
- Found and documented a subtler gap: unverified descriptive claims about real retrieved items (grounding check doesn't catch this yet)

### Week 5 — Evaluation
*Not started*
