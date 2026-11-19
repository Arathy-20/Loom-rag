# Loom

A hybrid retrieval-augmented generation (RAG) system for outfit styling — built as a deliberate learning project to understand the full RAG pipeline at interview depth, not to ship a production app.

## What it does

Given a natural-language occasion (e.g. *"beach vacation, mix of daytime and dinner"*), Loom retrieves matching outfit pieces from a personal wardrobe corpus, identifies gaps, and fills them with suggestions from a separate shop catalog — always explicit about what's owned vs. what needs buying.

## Why RAG (not just tag-filtering)

Real closet apps mostly rely on tag-filtering + vision-based auto-tagging. RAG is used here deliberately, not because it's the only way to build this, but because two things justify it:

1. **Fuzzy query handling** — a query like *"effortless but put-together"* has no matching keywords in structured tags. Embedding similarity captures meaning, not just attributes.
2. **Two-corpus cross-referencing** — matching gaps between a wardrobe and a shop catalog benefits from semantic similarity, not just rule-based category matching.

## Architecture — hybrid retrieval, four stages

1. **Structured filter pass** (rule-based) — narrows candidates using hard attributes (category, season, weather) before any semantic matching runs. Cheap and eliminates obviously wrong items early. Category filtering is always on; season/weather filtering is opt-in via `LOOM_FILTER_MODE=on` (see Evaluation).
2. **Embedding similarity pass** — embeds item descriptions + query with `sentence-transformers` (`all-MiniLM-L6-v2`), ranks by cosine similarity, returns top-k. Catches fuzzy intent fixed tags can't.
3. **Gap-filling against shop corpus** — repeats filter → embed, restricted to categories missing from wardrobe results.
4. **Generation** — an LLM (Gemini) reasons only over retrieved items, never inventing anything outside the two corpora.

Filtering first, embeddings second: hard constraints get enforced before semantic search runs over a smaller, already-safe candidate set. In the 20-item test wardrobe, the season/weather filter cut the candidate pool to 8–17 items for the 7 constrained eval queries — mirroring how production RAG systems are typically built.

## Project structure

```
loom-rag/
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   ├── wardrobe.json
│   ├── shop_catalog.json
│   └── eval_set.json          # Week 5 — 12 labelled queries (relevant items, expected gaps, constraints)
└── src/
    ├── retrieve.py           # Week 1 — plain numpy cosine similarity (reference implementation)
    ├── filter.py              # Week 2 — standalone rule-based filter logic
    ├── retrieve_chroma.py     # Week 2 — filter + embed, combined via Chroma
    ├── retrieve_gap_fill.py   # Week 3 — wardrobe search + gap detection + shop search (+ Week 5 experiment switches)
    ├── generate.py            # Week 4 — grounded generation + hallucination check
    ├── pipeline.py            # Week 6 — single style() entry point: validation, retrieval, generation
    └── eval.py                # Week 5 — precision/recall/hit@k, gap-detection scoring, threshold sweep
```

## Tools

- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** Chroma (metadata support alongside vectors)
- **Generation:** Gemini
- **Similarity metric:** cosine similarity

## Evaluation (Week 5)

**Setup.** 12 hand-written occasion queries in `data/eval_set.json`, each with labelled relevant wardrobe items, relevant shop items, and the categories that are truly missing from the wardrobe. Labels were drafted with AI assistance and reviewed by me, written from item descriptions and season/weather tags. Seven queries carry an explicit season or weather constraint, set only when the query text contains that word (no inference: "beach vacation" gets none). Retrieval and gap detection are scored separately because they fail independently. Generation quality is not scored here.

**Metrics.** Retrieval: precision@k, recall@k, hit@k (k = 3, 5, 8, 10), plus a precision ceiling per query. Gap detection: exact-match queries, missed gaps (false negatives), false-alarm gaps (false positives), and shop precision.

**Experiments.** Each is an environment-variable switch, default off, so the baseline stays reproducible. Run from `src/` with `python eval.py`.

| Switch | What it changes |
|---|---|
| `LOOM_EMBED_MODE=tags` | Embed description + season + weather text (stored description unchanged) |
| `LOOM_GAP_MODE=per_category` | Detect gaps by searching each needed category separately, instead of checking the global top-k |
| `LOOM_FILTER_MODE=on` | Apply a query's season/weather as a hard filter before ranking (one-hot metadata; all-season / all-weather items always pass) |

**Results.**

| Configuration | P@5 | R@5 | R@10 | Exact gaps | Missed | False alarms | Shop precision |
|---|---|---|---|---|---|---|---|
| Baseline | 0.30 | 0.24 | 0.51 | 2/12 | 4 | 10 | 0.40 |
| A: tags in embeddings | 0.32 | 0.27 | 0.63 | 2/12 | 3 | 11 | 0.53 |
| B: per-category gaps | 0.30 | 0.24 | 0.51 | 7/12 | 6 | 3 | 0.50 |
| C: hard season/weather filter | 0.47 | 0.40 | 0.73 | 3/12 | 5 | 8 | 0.54 |
| **C + B (recommended)** | 0.47 | 0.40 | 0.73 | 6/12 | 5 | 5 | 0.56 |
| A + B + C | 0.47 | 0.43 | 0.84 | 6/12 | 7 | 3 | 0.50 |

Recommended configuration (PowerShell): `$env:LOOM_FILTER_MODE="on"; $env:LOOM_GAP_MODE="per_category"`.

**Findings.**

1. **The hard season/weather filter was the largest retrieval gain** (P@5 0.30 → 0.47, R@5 0.24 → 0.40). Gains concentrate on queries with an explicit constraint (for example R@5 on "cozy for a cold night out" 0.17 → 0.67, on "rainy monsoon day" 0.20 → 0.80). The four unconstrained queries are identical to baseline, so the filter has no side effects when no constraint is given.
2. **A soft signal cannot forbid an item.** With season/weather text embedded (A), summer items still ranked highest for winter queries, and both winter queries stayed at 0.00. Only the hard filter removed them.
3. **Per-category gap detection cut false alarms from 10 to 3** (exact matches 2/12 → 7/12) because it no longer depends on an item reaching the global top 5. It also raised misses from 4 to 6.
4. **Embedding the tags was mixed and is not adopted.** It helped at deeper k and in shop search, but lowered recall on two unconstrained queries, and it shifted scores across the fixed gap threshold, turning two caught gaps into misses.
5. **Every remaining missed gap in the recommended configuration is a formality mismatch**: a casual hoodie or pair of sneakers passes as coverage for an elegant or formal need. Scores of true-gap categories (0.26–0.60) and covered categories (0.15–0.77) overlap, so no threshold separates them. On the two formality-heavy winter queries, ranking inside the filtered pool was no better than chance.
6. **"No item survives the filter" is a precise gap signal**: 0 false alarms across these 12 queries, though it catches only 2 of the 9 true gaps.

**Caveats.** 12 queries with 1–6 relevant items each: one item can move an average. Labels were written using the same tags the filter reads, so the filter can never exclude a relevant item here; the filter's gain is therefore an upper bound under ideal, known constraints. Constrained queries have pools of only 8–11 items, which inflates recall@8 and recall@10, so prefer P@5 and R@5. `MIN_COVERAGE_SCORE = 0.30` was chosen by eye and the threshold sweep ran on the same 12 queries (no held-out set). A baseline rerun after the first refactor reproduced identical numbers.

## Known limitations and future work

- **Formality is not in the schema.** Findings above point to it as the top remaining gap: it accounts for all remaining missed gaps in the recommended configuration. Not yet implemented.
- **Season/weather must be supplied as explicit inputs.** The filter works from structured constraints (an explicit `season`/`weather` argument to `pipeline.style()`). Extracting them from free text is not implemented, and the filter is still opt-in via `LOOM_FILTER_MODE=on`.
- **The generation prompt always asks for an outfit from owned items, even when nothing owned fits.** Observed with the filter on: for a formal winter dinner it suggested a hoodie, a sleeveless top and sneakers, while its own gap section said formal items were missing. A prompt that allows "nothing in your wardrobe suits this" is untested.
- **Gap threshold is fragile.** `MIN_COVERAGE_SCORE` changes meaning whenever scores shift (tags, a different embedding model). Needs a held-out query set and re-tuning with any such change. An embedding-model swap was discussed but not tested.
- **The grounding check verifies item IDs, not claims about items.** Confirmed case: the model described `w005` as "comfortable", a word absent from its description. Not measured by `eval.py`; catching it would need claim-level verification (for example a second LLM pass).
- **Untested suspicion:** very short, generic descriptions (for example `w001`) may rank highly across many unrelated queries ("hub" items). A top-k appearance count per item would confirm or rule it out.
- **Resolved during Week 5:** season/weather were previously ignored in retrieval; multi-value tags could not be filtered in Chroma. Both are addressed by the one-hot metadata and `LOOM_FILTER_MODE`, opt-in until the inputs above exist.

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
- Built a 12-query labelled eval set and `eval.py` (precision/recall/hit@k, gap-detection scoring, threshold sweep)
- Ran four controlled experiments, one change at a time, each behind an environment-variable switch
- Implemented one-hot season/weather metadata and a hard filter; implemented per-category gap detection
- Found formality, not season, is the main remaining failure; documented caveats of a 12-query eval

### Week 6 — Packaging (in progress)
- Refactored into a single `pipeline.style()` entry point that validates inputs, returns JSON-ready data, and reports ignored constraints instead of failing silently
- Verified the refactor changed no behaviour: the full eval output was identical before and after
- RAG pipeline work is complete; next: a FastAPI service (`POST /style`, `GET /health`) and a Dockerfile
