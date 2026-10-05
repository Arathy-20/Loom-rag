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

KNOWN GAP for productionizing: generate_styling_advice() currently
PRINTS its result instead of returning it — needs to return
(generated_text, grounding_result) before this can back a FastAPI
endpoint.
"""

import os
import re

from google import genai
from google.genai import types

from retrieve_gap_fill import style_outfit_data


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

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


def generate_styling_advice(query, owned_items, shop_items):
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable not set. "
            "Get a key from https://aistudio.google.com/apikey and set it "
            "before running this script."
        )

    client = genai.Client(api_key=GEMINI_API_KEY)

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

    # ---------------------------------------------------------
    # Grounding check: did the model mention any ID that wasn't
    # actually retrieved?
    # ---------------------------------------------------------
    valid_ids = {item["id"] for item in owned_items} | {item["id"] for item in shop_items}
    mentioned_ids = extract_mentioned_ids(generated_text)
    hallucinated_ids = mentioned_ids - valid_ids

    print(generated_text)
    print("\n--- Grounding check ---")
    print(f"Items retrieved: {sorted(valid_ids)}")
    print(f"Items mentioned: {sorted(mentioned_ids)}")
    if hallucinated_ids:
        print(f"⚠ HALLUCINATION DETECTED — mentioned IDs not in retrieved set: {sorted(hallucinated_ids)}")
    else:
        print("✓ All mentioned items are grounded in retrieved results.")


if __name__ == "__main__":
    query = "effortless outfit for a beach vacation"
    owned, shop = style_outfit_data(
        query, needed_categories={"top", "bottom", "dress", "footwear"}
    )
    generate_styling_advice(query, owned, shop)
