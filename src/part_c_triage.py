
from pathlib import Path
import sys
import re
import json
import random
import time

import pandas as pd

# Make the project root importable when this file is run from src/
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import mock_llm


# ---------------------------------------------------------
# PATHS AND SETTINGS
# ---------------------------------------------------------

RAW_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "outputs"

TICKETS_FILE = RAW_DIR / "tickets.csv"
SEGMENTS_FILE = OUTPUT_DIR / "customer_segments.csv"

MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
DEMO_TOKEN_BUDGET = 6000
FINAL_TOKEN_BUDGET = 100000

ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}

OUTPUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# PROMPTS
# ---------------------------------------------------------

ZERO_SHOT_PROMPT = """
CLASSIFY the customer's support ticket.

Choose exactly one category from:
billing, delivery, technical, fraud

Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}

Do not include explanations outside the JSON.
"""

FEW_SHOT_PROMPT = """
CLASSIFY the customer's support ticket.

Choose exactly one category from:
billing, delivery, technical, fraud

Return a JSON-only reply in this exact form:
{"category": "...", "reason": "..."}

Do not include explanations outside the JSON.

EXAMPLES:
Ticket: I was charged twice for my order.
Reply: {"category": "billing", "reason": "duplicate charge"}

Ticket: Someone accessed my account without permission.
Reply: {"category": "fraud", "reason": "unauthorized account access"}

Ticket: My delivery has not arrived yet.
Reply: {"category": "delivery", "reason": "late delivery"}
"""


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

tickets = pd.read_csv(TICKETS_FILE, dtype=str).fillna("")
segments = pd.read_csv(SEGMENTS_FILE)

# Ensure the premium flag is a real Boolean.
segments["is_premium"] = (
    segments["is_premium"].astype(str).str.strip().str.lower() == "true"
)

segment_lookup = segments.set_index("customer_id").to_dict("index")

print("=" * 65)
print("PART C: SUPPORT TICKET TRIAGE")
print("=" * 65)
print("Tickets loaded:", len(tickets))


# ---------------------------------------------------------
# TASK 1 AND 2: INPUT GUARDRAILS AND REDACTION
# ---------------------------------------------------------

EMAIL_PATTERN = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE_PATTERN = r"\b[6-9]\d{9}\b"
ORDER_PATTERN = r"ORD-\d{6}"


def redact_text(text):
    """Redact email, phone and order ID in the required order."""
    text = text or ""

    text, email_count = re.subn(EMAIL_PATTERN, "[EMAIL]", text)
    text, phone_count = re.subn(PHONE_PATTERN, "[PHONE]", text)
    text, order_count = re.subn(ORDER_PATTERN, "[ORDER_ID]", text)

    return text, email_count, phone_count, order_count


redacted_texts = []
email_total = 0
phone_total = 0
order_total = 0

for text in tickets["text"]:
    redacted, emails, phones, orders = redact_text(text)

    redacted_texts.append(redacted)
    email_total += emails
    phone_total += phones
    order_total += orders

tickets["redacted_text"] = redacted_texts

print("\nREDACTION COUNTS")
print("Emails redacted:", email_total)
print("Phone numbers redacted:", phone_total)
print("Order IDs redacted:", order_total)
print("Total substitutions:", email_total + phone_total + order_total)

# Verify that no matching personal information remains.
remaining_emails = sum(
    len(re.findall(EMAIL_PATTERN, text)) for text in tickets["redacted_text"]
)
remaining_phones = sum(
    len(re.findall(PHONE_PATTERN, text)) for text in tickets["redacted_text"]
)
remaining_orders = sum(
    len(re.findall(ORDER_PATTERN, text)) for text in tickets["redacted_text"]
)

print("Remaining emails:", remaining_emails)
print("Remaining phones:", remaining_phones)
print("Remaining order IDs:", remaining_orders)

assert remaining_emails == 0
assert remaining_phones == 0
assert remaining_orders == 0

# Verify the expected input guardrail counts.
empty_count = sum(not str(text).strip() for text in tickets["text"])
long_count = sum(
    bool(str(text).strip())
    and mock_llm.est_tokens(str(text)) > MAX_INPUT_TOKENS
    for text in tickets["text"]
)

print("\nINPUT GUARDRAILS")
print("Empty tickets:", empty_count)
print("Long tickets:", long_count)

assert empty_count == 4
assert long_count == 4


# ---------------------------------------------------------
# TASK 5: VALIDATE MODEL OUTPUT
# ---------------------------------------------------------

def validate_response(content):
    """
    Return category and status.
    Invalid JSON is checked using lowercase word extraction.
    Valid JSON with an unsupported category is never guessed.
    """
    try:
        parsed = json.loads(content)

        if (
            isinstance(parsed, dict)
            and parsed.get("category") in ALLOWED_CATEGORIES
        ):
            return parsed["category"], "classified_json"

        return "", "flagged_invalid_output"

    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", str(content).lower())
        found_categories = ALLOWED_CATEGORIES.intersection(words)

        if len(found_categories) == 1:
            return found_categories.pop(), "classified_keyword"

        return "", "flagged_invalid_output"


# ---------------------------------------------------------
# TASK 4 AND 7: RETRY FUNCTION AND TOKEN BUDGET
# ---------------------------------------------------------

def call_with_retry(messages, ticket_id, running_tokens, token_budget):
    """Call the mock with retries and an optional token budget."""

    for retry_number in range(MAX_RETRIES + 1):

        # Check the budget before every actual model call.
        if token_budget is not None and running_tokens >= token_budget:
            return None, 0, "budget_exhausted"

        try:
            response = mock_llm.chat(messages)
            used_tokens = response.usage.total_tokens
            return response, used_tokens, "called"

        except mock_llm.TransientLLMError:
            if retry_number >= MAX_RETRIES:
                raise RuntimeError(
                    f"Retries exhausted for ticket {ticket_id}"
                )

            wait = (
                BASE_DELAY
                * (2 ** retry_number)
                * random.uniform(0.9, 1.15)
            )

            print(
                f"RETRY ticket={ticket_id} "
                f"retry_number={retry_number} "
                f"wait={wait:.5f}s"
            )

            time.sleep(wait)

    raise RuntimeError(f"Unexpected retry failure for {ticket_id}")


# ---------------------------------------------------------
# COMPLETE TRIAGE PIPELINE
# ---------------------------------------------------------

def run_pipeline(prompt, run_name, token_budget=None):
    """Run one complete ticket batch in original CSV order."""

    # Reset immediately before every independent full run.
    mock_llm.reset()

    results = []
    running_tokens = 0
    retry_count = 0

    for _, row in tickets.iterrows():

        ticket_id = row["ticket_id"]
        customer_id = row["customer_id"]
        raw_text = row["text"]
        redacted_text = row["redacted_text"]

        result = {
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": "",
            "category": "",
            "segment_name": "",
            "priority": "",
            "route": "",
            "tokens_used": 0,
            "true_category": row["true_category"],
        }

        # Guardrail 1: empty or whitespace-only text.
        if not str(raw_text).strip():
            result["status"] = "skipped_empty"

        # Guardrail 2: estimate tokens using original ticket text.
        elif mock_llm.est_tokens(str(raw_text)) > MAX_INPUT_TOKENS:
            result["status"] = "flagged_long"

        else:
            # Privacy assertion immediately before prompt creation.
            assert not re.search(EMAIL_PATTERN, redacted_text)
            assert not re.search(PHONE_PATTERN, redacted_text)
            assert not re.search(ORDER_PATTERN, redacted_text)

            # User message content is exactly the redacted ticket text.
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": redacted_text},
            ]

            response, used_tokens, call_status = call_with_retry(
                messages,
                ticket_id,
                running_tokens,
                token_budget,
            )

            if call_status == "budget_exhausted":
                result["status"] = "budget_exhausted"

            else:
                running_tokens += used_tokens
                result["tokens_used"] = used_tokens

                content = response.choices[0].message.content
                category, status = validate_response(content)

                result["category"] = category
                result["status"] = status

        # Customer segment information for routing.
        customer_segment = segment_lookup.get(customer_id, {})
        segment_name = customer_segment.get("segment_name", "")
        is_premium = customer_segment.get("is_premium", False)

        result["segment_name"] = segment_name

        # Apply the required routing rules.
        if result["status"] in {
            "skipped_empty",
            "flagged_long",
            "flagged_invalid_output",
            "budget_exhausted",
        }:
            result["route"] = "human_review"
            result["priority"] = "NORMAL"

        elif result["category"] == "fraud":
            result["route"] = "security_team"
            result["priority"] = "HIGH"

        elif is_premium:
            result["route"] = "premium_desk"
            result["priority"] = "HIGH"

        else:
            result["route"] = "support_queue"
            result["priority"] = "NORMAL"

        results.append(result)

    result_df = pd.DataFrame(results)

    classified = result_df[
        result_df["status"].isin(
            ["classified_json", "classified_keyword"]
        )
    ]

    classified_count = len(classified)

    correct_count = (
        classified["category"] == classified["true_category"]
    ).sum()

    accuracy = (
        correct_count / classified_count
        if classified_count else 0
    )

    print("\n" + "=" * 65)
    print("RUN:", run_name)
    print("=" * 65)

    print("\nStatus counts:")
    print(result_df["status"].value_counts().to_string())

    print("\nTickets classified:", classified_count)
    print("Correct classifications:", correct_count)
    print(f"Accuracy: {accuracy:.4%}")
    print("Total tokens used:", running_tokens)

    if classified_count:
        print(
            "Tokens per classified ticket:",
            round(running_tokens / classified_count, 2),
        )

    print(
        "Budget exhausted:",
        (result_df["status"] == "budget_exhausted").sum(),
    )

    return result_df, running_tokens


# ---------------------------------------------------------
# TASK 6: ZERO-SHOT AND FEW-SHOT COMPARISON
# ---------------------------------------------------------

zero_results, zero_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT FULL RUN",
    token_budget=None,
)

few_results, few_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    "FEW-SHOT FULL RUN",
    token_budget=None,
)

# Confusion table for zero-shot classified tickets.
zero_classified = zero_results[
    zero_results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
]

print("\nZERO-SHOT CONFUSION TABLE")
print(
    pd.crosstab(
        zero_classified["true_category"],
        zero_classified["category"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )
)

few_classified = few_results[
    few_results["status"].isin(
        ["classified_json", "classified_keyword"]
    )
]

zero_accuracy = (
    (zero_classified["category"] == zero_classified["true_category"]).mean()
)
few_accuracy = (
    (few_classified["category"] == few_classified["true_category"]).mean()
)

print("\nPROMPT COMPARISON")
print(f"Zero-shot accuracy: {zero_accuracy:.4%}")
print(f"Few-shot accuracy: {few_accuracy:.4%}")
print("Zero-shot total tokens:", zero_tokens)
print("Few-shot total tokens:", few_tokens)
print(
    "Extra few-shot tokens:",
    few_tokens - zero_tokens,
)
print(
    "Extra tokens per classified ticket:",
    round(
        (few_tokens - zero_tokens) / len(few_classified),
        2,
    ),
)


# ---------------------------------------------------------
# TASK 7A: ZERO-SHOT TOKEN BUDGET DEMO
# ---------------------------------------------------------

budget_results, budget_tokens = run_pipeline(
    ZERO_SHOT_PROMPT,
    "ZERO-SHOT BUDGET DEMO",
    token_budget=DEMO_TOKEN_BUDGET,
)

budget_exhausted_count = (
    budget_results["status"] == "budget_exhausted"
).sum()

# If 6000 is too high to stop the run, demonstrate with 5000.
if budget_exhausted_count == 0:
    print("\nNo ticket exhausted the 6000-token budget.")
    print("Repeating the demo with a 5000-token budget.")

    DEMO_TOKEN_BUDGET = 5000

    budget_results, budget_tokens = run_pipeline(
        ZERO_SHOT_PROMPT,
        "ZERO-SHOT BUDGET DEMO - 5000",
        token_budget=DEMO_TOKEN_BUDGET,
    )

    budget_exhausted_count = (
        budget_results["status"] == "budget_exhausted"
    ).sum()

budget_columns = [
    "ticket_id",
    "customer_id",
    "status",
    "category",
    "segment_name",
    "priority",
    "route",
    "tokens_used",
]

budget_results[budget_columns].to_csv(
    OUTPUT_DIR / "triage_budget_demo.csv",
    index=False,
)

print("\nBUDGET DEMO SAVED")
print("Budget used:", DEMO_TOKEN_BUDGET)
print("Budget-exhausted tickets:", budget_exhausted_count)


# ---------------------------------------------------------
# TASK 7B: FINAL FEW-SHOT RUN
# ---------------------------------------------------------

final_results, final_tokens = run_pipeline(
    FEW_SHOT_PROMPT,
    "FINAL FEW-SHOT RUN",
    token_budget=FINAL_TOKEN_BUDGET,
)

final_results[budget_columns].to_csv(
    OUTPUT_DIR / "triage_results.csv",
    index=False,
)

print("\nFINAL OUTPUT SAVED")
print("File:", OUTPUT_DIR / "triage_results.csv")
print("Rows:", len(final_results))
print("Columns:", budget_columns)

print("\nFinal status counts:")
print(final_results["status"].value_counts().to_string())

print("\nFinal routing counts:")
print(final_results["route"].value_counts().to_string())

print("\nPart C pipeline completed.")
