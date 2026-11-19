# Loom: interview notes (personal prep, keep out of the public repo)

## 30-second pitch
Loom is a hybrid RAG system for outfit styling. You describe an occasion, it retrieves matching pieces from your own wardrobe, detects which categories you're missing, fills those from a separate shop catalog, and an LLM writes the styling advice, constrained to only the items retrieved. I built it to learn RAG end to end, then measured it with a labelled eval set, changed one thing at a time, and found where it fails and why.

## Pipeline in four lines
1. Structured filter: category always, season/weather opt-in. Rule-based, no AI.
2. Embedding search: `all-MiniLM-L6-v2`, cosine similarity, in Chroma.
3. Gap detection, then the same filter and search against the shop collection, restricted to the gap categories.
4. Gemini generation with rules in the system instruction and only retrieved items in the user message, plus a post-check that every item ID it mentions was retrieved.

## What I measured (12 labelled queries)
- Baseline: P@5 0.30, R@5 0.24, gap detection exact on 2/12 queries.
- **Hard season/weather filter:** P@5 0.47, R@5 0.40. Biggest retrieval gain, concentrated on queries with an explicit constraint; unconstrained queries unchanged.
- **Per-category gap detection:** false alarms 10 → 3, exact 2/12 → 7/12, but misses 4 → 6.
- **Embedding the season/weather text:** mixed. Not adopted.
- **Remaining failures are formality:** a casual hoodie or sneakers passing as coverage for a formal need. Score ranges for true gaps and covered categories overlap, so no threshold fixes it.

## Say this, not that
- Say: "12 queries, labels drafted with AI help and reviewed by me, so I treat results as directional." Not: "my system achieves 0.47 precision."
- Say: "the filter's gain is an upper bound, because my labels used the same tags the filter reads." This shows you understand the eval's weakness.
- Say: "my first threshold of 0.30 was chosen by eye, and I tuned the sweep on the same queries, so it isn't validated." Not: "0.30 is optimal."

## Likely questions
**Why hybrid instead of pure embeddings?** Embeddings can nudge a ranking but can't forbid an item. With season text embedded, summer items still topped winter queries. Only a hard filter removed them. Measured, not assumed.

**Why filter before similarity?** Hard constraints are cheap and exact, and they shrink the candidate pool (8–17 of 20 items here). Caveat: tiny pools inflate recall at large k.

**How did you evaluate?** Labelled queries; precision/recall/hit@k for retrieval, scored separately from gap detection, because they fail independently. I also print a precision ceiling, since a query with 2 relevant items can never exceed 0.4 P@5.

**What was your biggest finding?** That formality, not season, was the main remaining failure. I only knew that because I fixed season first and the winter queries still failed.

**How do you know the LLM isn't hallucinating?** I check that every item ID it mentions was retrieved, and it passes. But that check misses claims about real items: it called a skort "comfortable" when the description never said so. Catching that needs claim-level verification, which I haven't built.

**Why did gap detection give false alarms at first?** It only counted a category as covered if one of its items reached the global top 5. Searching each category separately fixed most of that.

**What went wrong with the threshold?** It's a fixed similarity cutoff, so it changes meaning whenever scores shift. Adding tags moved two footwear scores just over the line and turned two caught gaps into misses.

**What would you do next?** Add a formality field; build a held-out query set before re-tuning the threshold; extract season/weather from free text (they're explicit inputs now); add a claim-level grounding check; then try another embedding model with the threshold re-tuned.

**What would you change to productionize it?** Return values instead of printing, parameterize the query inputs, persist the Chroma store, pin dependencies, wrap it in FastAPI and Docker (planned as a separate phase).
