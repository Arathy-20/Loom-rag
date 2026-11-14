"""
Week 4 — Stage 4: Grounded generation.

Takes the retrieved wardrobe + shop items (from retrieve_gap_fill.py's
pipeline) and asks Gemini to produce styling advice — but constrained
to ONLY reference items that were actually retrieved. The LLM never
sees the full corpus, only the short retrieved list, so it physically
cannot invent an item that doesn't exist in either corpus.

Prompt is split into a system_instruction (rules/behavior) and a user
message (query + retrieved data), rather than one combined string —
this puts constraints in the channel models are trained to weight
most consistently, separate from the data being reasoned over.

A grounding check runs after generation: every item ID the model
mentions gets verified against the actual retrieved item IDs. If the
model mentions an ID that was never retrieved, that's a hallucination
and gets flagged.

SDK: uses the unified `google-genai` SDK (the older
`google-generativeai` package is deprecated).

generate_styling_advice() RETURNS its result (advice text + grounding
check) instead of printing, so an API can serialize it. The Gemini client
is created lazily on first use, so importing this module needs no API key.

LIMITATION: the grounding check only finds IDs written like "(w011)".
If the model cites no IDs at all, the check passes vacuously.
"""

import os
import re

from google import genai
from google.genai import types


# Check AI Studio's model list if this name stops working — model
# names get retired over time.
MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------
# SYSTEM INSTRUCTION — the rules. Behavior and grounding
# constraints live here, separate from the actual data, so the
# model treats them as standing instructions rather than one-off
# text to weigh against everything else in the message.
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """You are a wardrobe styling assistant.

You will be given an occasion, a list of OWNED items (from the
user's wardrobe), and a list of SHOP items (things the user does
not own, would need to buy).

Rules, all mandatory:
- Suggest a specific outfit pairing using ONLY the items provided.
  Do NOT mention, reference, or imply any clothing item that is not
  explicitly listed. If you are unsure whether an item exists, do
  not mention it.
- Every item you reference MUST include its exact ID in parentheses,
  e.g. "white crop top (w011)". No exceptions.
- Do not invent details (fabric, brand, fit) beyond what's in the
  given item descriptions.

Follow this exact output structure:

**Suggested outfit (from your wardrobe):**
[Pair 2-3 owned items into a specific outfit. Reference each by name and ID.]

**Gap — you don't own suitable items for:** [category/categories]
[Suggest 1-2 shop items to fill the gap, each with name and ID.]

**Why this works:**
[1-2 sentences of reasoning about the pairing, grounded only in the
items above.]
"""

# ---------------------------------------------------------
# USER MESSAGE TEMPLATE — just the data: the query and the
# retrieved items. No rules here — those live in the system
# instruction above.
# ---------------------------------------------------------
USER_MESSAGE_TEMPLATE = """Occasion: {query}

OWNED items:
{owned_items}

SHOP items (gaps):
{shop_items}
"""


def format_items_for_prompt(items):
    if not items:
        return "(none)"
    lines = []
    for item in items:
        lines.append(f"- {item['description']} (id: {item['id']})")
    return "\n".join(lines)


def extract_mentioned_ids(text):
    """Pull every id like (w011) or (s006) mentioned in the generated text."""
    return set(re.findall(r"\(([ws]\d{3})\)", text))


_client = None


def _get_client():
    """Create the Gemini client once, on first use, and reuse it."""
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable not set. "
                "Get a key from https://aistudio.google.com/apikey and set it "
                "before generating advice."
            )
        _client = genai.Client(api_key=api_key)
    return _client


def check_grounding(generated_text, owned_items, shop_items):
    """Pure function: did the text mention any item ID that was not retrieved?"""
    valid_ids = {item["id"] for item in owned_items} | {item["id"] for item in shop_items}
    mentioned_ids = extract_mentioned_ids(generated_text)
    hallucinated_ids = mentioned_ids - valid_ids
    return {
        "retrieved_ids": sorted(valid_ids),
        "mentioned_ids": sorted(mentioned_ids),
        "hallucinated_ids": sorted(hallucinated_ids),
        "grounded": not hallucinated_ids,
    }


def generate_styling_advice(query, owned_items, shop_items):
    """Returns {"advice": str, "grounding": {...}}. Does not print."""
    client = _get_client()

    user_message = USER_MESSAGE_TEMPLATE.format(
        query=query,
        owned_items=format_items_for_prompt(owned_items),
        shop_items=format_items_for_prompt(shop_items),
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_message,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_INSTRUCTION),
    )
    generated_text = response.text
    if not generated_text:
        raise RuntimeError("Gemini returned no text (the response may have been blocked or empty).")

    return {
        "advice": generated_text,
        "grounding": check_grounding(generated_text, owned_items, shop_items),
    }


def print_advice(result):
    """CLI presentation of a generate_styling_advice() result."""
    grounding = result["grounding"]
    print(result["advice"])
    print("\n--- Grounding check ---")
    print(f"Items retrieved: {grounding['retrieved_ids']}")
    print(f"Items mentioned: {grounding['mentioned_ids']}")
    if grounding["hallucinated_ids"]:
        print(f"⚠ HALLUCINATION DETECTED — mentioned IDs not in retrieved set: {grounding['hallucinated_ids']}")
    else:
        print("✓ All mentioned items are grounded in retrieved results.")


if __name__ == "__main__":
    from retrieve_gap_fill import retrieve_for_occasion

    query = "effortless outfit for a beach vacation"
    retrieved = retrieve_for_occasion(query, {"top", "bottom", "dress", "footwear"})
    print_advice(generate_styling_advice(query, retrieved["owned"], retrieved["shop"]))
