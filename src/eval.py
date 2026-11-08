"""
Week 5: Evaluation.

Measures two things SEPARATELY, because they can fail independently:

1. Wardrobe retrieval quality: precision@k, recall@k and hit@k, swept
   over k = 3, 5, 8, 10, against the hand-labelled relevant IDs in
   data/eval_set.json.
2. Gap detection + shop retrieval: did the system flag the categories
   that are truly missing (expected_gap_categories), and were the shop
   items it returned in the labelled-relevant set?

Generation quality (e.g. the "comfortable" claim problem) is NOT
measured here. That needs a different kind of check.

Run from the src/ folder:  python eval.py
"""

import json
import os

from retrieve_gap_fill import (
    search_wardrobe,
    style_outfit_data,
    find_gaps,
    category_best_scores,
    MIN_COVERAGE_SCORE,
    EMBED_MODE,
    GAP_MODE,
    FILTER_MODE,
    eligible_wardrobe_count,
)

EVAL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.json")
K_VALUES = [3, 5, 8, 10]


# ---------------------------------------------------------
# Metric functions (pure, no retrieval involved)
# ---------------------------------------------------------
def precision_at_k(retrieved_ids, relevant_ids, k):
    """Of the top k results, what fraction are relevant?"""
    hits = sum(1 for i in retrieved_ids[:k] if i in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    """Of all relevant items, what fraction appear in the top k?"""
    hits = sum(1 for i in retrieved_ids[:k] if i in relevant_ids)
    return hits / len(relevant_ids)


def hit_at_k(retrieved_ids, relevant_ids, k):
    """1 if at least one relevant item is in the top k, else 0."""
    return 1.0 if any(i in relevant_ids for i in retrieved_ids[:k]) else 0.0


def precision_ceiling(relevant_ids, k):
    """
    Best precision@k any system could get for this query. If a query
    only has 2 relevant items, precision@5 can never exceed 0.4. This
    context matters when reading the averages.
    """
    return min(len(relevant_ids), k) / k


# ---------------------------------------------------------
# Part 1: wardrobe retrieval sweep
# ---------------------------------------------------------
def evaluate_retrieval(eval_set):
    rows = []
    for entry in eval_set:
        relevant = set(entry["relevant_wardrobe_ids"])
        if not relevant:
            continue  # nothing labelled, so nothing to measure

        season, weather = entry.get("season"), entry.get("weather")
        results = search_wardrobe(entry["query"], top_k=max(K_VALUES), season=season, weather=weather)
        retrieved_ids = [r["id"] for r in results]

        rows.append({
            "id": entry["id"],
            "query": entry["query"],
            "retrieved": retrieved_ids,
            "relevant": relevant,
            "p": {k: precision_at_k(retrieved_ids, relevant, k) for k in K_VALUES},
            "r": {k: recall_at_k(retrieved_ids, relevant, k) for k in K_VALUES},
            "hit": {k: hit_at_k(retrieved_ids, relevant, k) for k in K_VALUES},
            "ceiling": {k: precision_ceiling(relevant, k) for k in K_VALUES},
            "pool": eligible_wardrobe_count(season, weather),
        })

    print("=" * 78)
    print("PART 1: WARDROBE RETRIEVAL (per query, at k=5)")
    print("=" * 78)
    print(f"{'id':<5}{'P@5':>6}{'R@5':>6}{'hit@5':>7}{'pool':>6}  query")
    for row in rows:
        print(f"{row['id']:<5}{row['p'][5]:>6.2f}{row['r'][5]:>6.2f}"
              f"{row['hit'][5]:>7.0f}{row['pool']:>6}  {row['query']}")

    n = len(rows)
    print("\nAVERAGES ACROSS QUERIES (macro-average)")
    print(f"{'k':>3}{'P@k':>8}{'ceiling':>9}{'R@k':>8}{'hit@k':>8}")
    for k in K_VALUES:
        avg_p = sum(r["p"][k] for r in rows) / n
        avg_c = sum(r["ceiling"][k] for r in rows) / n
        avg_r = sum(r["r"][k] for r in rows) / n
        avg_h = sum(r["hit"][k] for r in rows) / n
        print(f"{k:>3}{avg_p:>8.2f}{avg_c:>9.2f}{avg_r:>8.2f}{avg_h:>8.2f}")

    print("\nWORST QUERIES at k=5 (lowest recall first), with what went wrong:")
    for row in sorted(rows, key=lambda r: r["r"][5])[:3]:
        top5 = row["retrieved"][:5]
        missed = sorted(row["relevant"] - set(top5))
        wrong = [i for i in top5 if i not in row["relevant"]]
        print(f"  {row['id']} \"{row['query']}\"")
        print(f"      relevant items missed from top 5: {missed}")
        print(f"      irrelevant items that took their place: {wrong}")
    print()


# ---------------------------------------------------------
# Part 2: gap detection + shop retrieval
# ---------------------------------------------------------
def evaluate_gaps(eval_set):
    print("=" * 78)
    print(f"PART 2: GAP DETECTION + SHOP RETRIEVAL (gap mode = {GAP_MODE}, filter = {FILTER_MODE}, threshold = {MIN_COVERAGE_SCORE})")
    print("=" * 78)
    print(f"{'id':<5}{'expected gaps':<36}{'detected gaps':<42}")

    exact = 0
    total_missed = 0
    total_false = 0
    shop_returned = 0
    shop_relevant_hits = 0

    for entry in eval_set:
        needed = set(entry["needed_categories"])
        expected = set(entry["expected_gap_categories"])
        relevant_shop = set(entry["relevant_shop_ids"])

        season, weather = entry.get("season"), entry.get("weather")
        owned, shop = style_outfit_data(entry["query"], needed, top_k=5, season=season, weather=weather)
        detected = find_gaps(entry["query"], owned, needed, season, weather)

        missed = expected - detected          # real gaps the system did not flag
        false_alarms = detected - expected    # gaps flagged that were not real

        exact += (not missed and not false_alarms)
        total_missed += len(missed)
        total_false += len(false_alarms)
        shop_returned += len(shop)
        shop_relevant_hits += sum(1 for s in shop if s["id"] in relevant_shop)

        flag = ""
        if missed:
            flag += f"  MISSED {sorted(missed)}"
        if false_alarms:
            flag += f"  FALSE ALARM {sorted(false_alarms)}"
        print(f"{entry['id']:<5}{str(sorted(expected)):<36}{str(sorted(detected)):<42}{flag}")

    n = len(eval_set)
    print(f"\nExact gap match:      {exact}/{n} queries")
    print(f"Missed real gaps:     {total_missed}  (false negatives)")
    print(f"False-alarm gaps:     {total_false}  (false positives)")
    if shop_returned:
        print(f"Shop precision:       {shop_relevant_hits}/{shop_returned} "
              f"= {shop_relevant_hits / shop_returned:.2f}  "
              f"(returned shop items that are in the labelled-relevant set)")
    print()


# ---------------------------------------------------------
# Part 3: per-category scores + threshold sweep
#   Independent of LOOM_GAP_MODE. Pure diagnostics: for every needed
#   category, the best wardrobe score inside that category, plus how
#   each possible threshold would score on the labelled gaps.
#   "*" marks a category that is a TRUE gap according to the labels.
# ---------------------------------------------------------
def evaluate_gap_scores(eval_set):
    print("=" * 78)
    print("PART 3: BEST SCORE PER NEEDED CATEGORY  (* = labelled as a true gap)")
    print("=" * 78)

    all_scores = []
    for entry in eval_set:
        needed = set(entry["needed_categories"])
        expected = set(entry["expected_gap_categories"])
        scores = category_best_scores(entry["query"], needed, entry.get("season"), entry.get("weather"))
        all_scores.append((entry, expected, scores))
        cells = []
        for cat in sorted(scores):
            sc = scores[cat]
            mark = "*" if cat in expected else ""
            cells.append(f"{cat}{mark}={sc:.2f}" if sc is not None else f"{cat}{mark}=n/a")
        print(f"{entry['id']:<5}" + "   ".join(cells))

    true_gap_scores = [sc for _, exp, scs in all_scores for c, sc in scs.items() if c in exp and sc is not None]
    covered_scores = [sc for _, exp, scs in all_scores for c, sc in scs.items() if c not in exp and sc is not None]
    if true_gap_scores and covered_scores:
        print(f"\nTrue-gap categories:    best scores range {min(true_gap_scores):.2f} to {max(true_gap_scores):.2f}")
        print(f"Covered categories:     best scores range {min(covered_scores):.2f} to {max(covered_scores):.2f}")
        print("(If these ranges overlap, no single threshold can separate them.)")

    print("\nTHRESHOLD SWEEP (per-category scoring)")
    print(f"{'thresh':>7}{'exact':>8}{'missed':>8}{'false':>8}")
    n = len(all_scores)
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        exact = missed = false = 0
        for entry, expected, scores in all_scores:
            detected = {c for c, sc in scores.items() if sc is None or sc < t}
            m, f = expected - detected, detected - expected
            exact += (not m and not f)
            missed += len(m)
            false += len(f)
        cur = "  <- current" if abs(t - MIN_COVERAGE_SCORE) < 1e-9 else ""
        print(f"{t:>7.2f}{exact:>5}/{n:<2}{missed:>8}{false:>8}{cur}")
    print()


if __name__ == "__main__":
    with open(EVAL_PATH, "r") as f:
        eval_set = json.load(f)

    print(f"Embedding mode: {EMBED_MODE}")
    print(f"Gap mode: {GAP_MODE}")
    print(f"Filter mode: {FILTER_MODE}")
    print(f"Loaded {len(eval_set)} labelled queries.\n")
    evaluate_retrieval(eval_set)
    evaluate_gaps(eval_set)
    evaluate_gap_scores(eval_set)
