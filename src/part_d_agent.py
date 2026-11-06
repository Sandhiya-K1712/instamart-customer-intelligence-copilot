
from pathlib import Path
import sys
import re
import json
import csv
import time
import uuid
from datetime import datetime
from collections import Counter

import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import mock_llm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

POLICY_DIR = BASE_DIR / "data" / "policies"
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"
STATE_DIR = BASE_DIR / "state"

OUTPUT_DIR.mkdir(exist_ok=True)
STATE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------
# TASK 1: CHUNK POLICY DOCUMENTS
# ---------------------------------------------------------

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10

chunks = []

for file_path in sorted(POLICY_DIR.glob("*.txt")):
    text = file_path.read_text(encoding="utf-8")
    words = text.split()

    start = 0

    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))

        chunk_text = " ".join(words[start:end])

        chunks.append({
            "source": file_path.name,
            "text": chunk_text
        })

        if end == len(words):
            break

        start += CHUNK_SIZE - CHUNK_OVERLAP


print("=" * 60)
print("TASK 1: POLICY CHUNKING")
print("=" * 60)

print("Total chunks:", len(chunks))

print("\nChunks per document:")
for source, count in sorted(
    Counter(chunk["source"] for chunk in chunks).items()
):
    print(source, ":", count)


# ---------------------------------------------------------
# TASK 2: TF-IDF VECTORISATION AND RETRIEVAL
# ---------------------------------------------------------

chunk_texts = [chunk["text"] for chunk in chunks]

vectorizer = TfidfVectorizer()
chunk_vectors = vectorizer.fit_transform(chunk_texts)


def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        chunk_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1][:k]

    results = []

    for index in ranked_indices:
        results.append({
            "source": chunks[index]["source"],
            "score": float(similarities[index]),
            "text": chunks[index]["text"]
        })

    return results


print("\n" + "=" * 60)
print("TASK 2: RETRIEVAL TEST")
print("=" * 60)

test_results = retrieve("refund policy late delivery", k=3)

for result in test_results:
    print("\nSource:", result["source"])
    print("Score:", round(result["score"], 4))
    print("Text:", result["text"])

# ============================================================
# TASK 3: POLICY ASSISTANT
# ============================================================

def answer(question):
    results = retrieve(question, k=3)

    context_parts = []
    for item in results:
        context_parts.append(
            f"[{item['source']}]\n{item['text']}"
        )

    context = "\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer the question using only "
                "the supplied policy context. Do not use outside knowledge. "
                "If the context does not contain the answer, reply exactly: "
                "I could not find that information in the policy documents."
            )
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n"
                f"QUESTION:\n{question}"
            )
        }
    ]

    response = mock_llm.chat(messages)
    answer_text = response.choices[0].message.content

    sources = list(dict.fromkeys(item["source"] for item in results))

    return {
        "answer": answer_text,
        "sources": sources
    }


# Test the policy assistant
print("\n" + "=" * 60)
print("TASK 3: POLICY ASSISTANT TEST")
print("=" * 60)

test_question = "What delivery time makes a completed order eligible for a full refund?"
test_result = answer(test_question)

print("Question:", test_question)
print("Answer:", test_result["answer"])
print("Sources:", test_result["sources"])

# ============================================================
# TASK 4: POLICY EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("TASK 4: POLICY EVALUATION")
print("=" * 60)

eval_path = RAW_DIR / "policy_eval.csv"
eval_df = pd.read_csv(eval_path)

answerable = eval_df[eval_df["expected_doc"] != "NONE"]
unanswerable = eval_df[eval_df["expected_doc"] == "NONE"]

doc_correct = 0
fact_correct = 0
fallback_correct = 0

for _, row in eval_df.iterrows():
    result = answer(row["question"])
    answer_text = result["answer"]

    if row["expected_doc"] != "NONE":
        if row["expected_doc"] in result["sources"]:
            doc_correct += 1

        if str(row["expected_fact"]) in answer_text:
            fact_correct += 1

    else:
        if answer_text.startswith("I could not find"):
            fallback_correct += 1

print("Total questions:", len(eval_df))
print("Answerable questions:", len(answerable))
print("Unanswerable questions:", len(unanswerable))

print(
    "Expected document retrieved:",
    doc_correct, "/", len(answerable)
)

print(
    "Expected fact found in answer:",
    fact_correct, "/", len(answerable)
)

print(
    "Correct unanswerable fallbacks:",
    fallback_correct, "/", len(unanswerable)
)

# ============================================================
# TASK 4B: ZERO-OVERLAP COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("TASK 4B: ZERO-OVERLAP COMPARISON")
print("=" * 60)

def build_chunks(overlap):
    new_chunks = []

    for file_path in sorted(POLICY_DIR.glob("*.txt")):
        words = file_path.read_text(encoding="utf-8").split()
        start = 0

        while start < len(words):
            end = min(start + CHUNK_SIZE, len(words))

            new_chunks.append({
                "source": file_path.name,
                "text": " ".join(words[start:end])
            })

            if end == len(words):
                break

            start += CHUNK_SIZE - overlap

    return new_chunks


zero_chunks = build_chunks(0)
zero_texts = [item["text"] for item in zero_chunks]

zero_vectorizer = TfidfVectorizer()
zero_vectors = zero_vectorizer.fit_transform(zero_texts)


def retrieve_zero(query, k=3):
    query_vector = zero_vectorizer.transform([query])
    similarities = cosine_similarity(
        query_vector, zero_vectors
    ).flatten()

    indices = similarities.argsort()[::-1][:k]

    return [
        {
            "source": zero_chunks[i]["source"],
            "score": float(similarities[i]),
            "text": zero_chunks[i]["text"]
        }
        for i in indices
    ]



# Compare whether the expected fact appears in the top retrieved chunk
failed_questions = []

for _, row in answerable.iterrows():
    results = retrieve_zero(row["question"], k=3)

    combined_text = " ".join(
        item["text"] for item in results
    )

    if str(row["expected_fact"]) not in combined_text:
        failed_questions.append(row["question"])

print("Zero-overlap total chunks:", len(zero_chunks))
print("Answerable questions tested:", len(answerable))
print("Questions where expected fact is missing from top 3:", len(failed_questions))

for question in failed_questions:
    print("Failed question:", question)

# ============================================================
# TASK 5: REFUND AGENT TOOLS
# ============================================================

DB_PATH = BASE_DIR / "data" / "instamart.db"
LEDGER_PATH = OUTPUT_DIR / "refund_ledger.jsonl"


def get_order(order_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        row = conn.execute(
            """
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,)
        ).fetchone()

        if row is None:
            return {"error": f"Order {order_id} not found"}

        return {
            "order_id": row["order_id"],
            "status": row["status"],
            "order_amount": row["order_amount"],
            "delivery_minutes": row["delivery_minutes"]
        }

    finally:
        conn.close()


def search_policy(query):
    return {
        "results": retrieve(query, k=3)
    }


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes is None:
        refund = 0.0
    elif float(delivery_minutes) > 45:
        refund = float(order_amount)
    else:
        refund = 0.0

    return {"refund_inr": round(refund, 2)}


def write_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2)
    }

    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return {
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
        "recorded": True
    }


def issue_refund(order_id, refund_inr):
    return write_ledger(
        "issue_refund", order_id, refund_inr
    )


def escalate_to_human(order_id, refund_inr):
    return write_ledger(
        "escalate_to_human", order_id, refund_inr
    )


# Tool specifications for the model
TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Get order status, amount and delivery time from the database.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"}
            },
            "required": ["order_id"]
        }
    },
    {
        "name": "search_policy",
        "description": "Search the company policy documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate_refund",
        "description": "Calculate refund using order amount and delivery minutes.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"}
            },
            "required": ["order_amount", "delivery_minutes"]
        }
    },
    {
        "name": "issue_refund",
        "description": "Record an automatically approved refund in the simulated ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    },
    {
        "name": "escalate_to_human",
        "description": "Record a refund requiring human approval in the simulated ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"}
            },
            "required": ["order_id", "refund_inr"]
        }
    }
]


TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}


print("\n" + "=" * 60)
print("TASK 5: TOOL TESTS")
print("=" * 60)

print("Order lookup:", get_order("ORD-100048"))
print("Missing order:", get_order("ORD-999999"))
print("Refund calculation:", calculate_refund(128.25, 60))
print("Zero refund calculation:", calculate_refund(128.25, 45))
print("Number of tools:", len(TOOLS_SPEC))
print("Tool names:", list(TOOL_MAP.keys()))

# ============================================================
# TASK 8: CHECKPOINT HELPER
# ============================================================

class SimulatedCrash(Exception):
    pass


def save_checkpoint(run_id, scenario_id, messages,
                    tokens_used, llm_calls, steps):

    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "llm_calls": llm_calls,
        "steps": steps,
        "step_number": len(steps),
        "tools_executed": [step["tool"] for step in steps]
    }

    checkpoint_path = STATE_DIR / f"{run_id}.json"

    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    return checkpoint_path

# ============================================================
# TASK 6: REACT AGENT LOOP
# ============================================================

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    run_id=None,
    crash_after_tools=None,
    _resume_checkpoint=None
):
    if run_id is None:
        run_id = str(uuid.uuid4())
    
    if _resume_checkpoint is None:
        messages = [
            {
                "role": "system",
                "content": (
                    "AGENT. Handle refund requests using the available tools. "
                    "Only completed orders qualify. Refund the full order amount "
                    "only when delivery exceeds 45 minutes. Refunds above INR "
                    "1000 must be escalated to a human, never issued automatically. "
                    "Use the policy and order tools before deciding."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        steps = []
        llm_calls = 0
        tokens_used = 0

    else:
        messages = _resume_checkpoint["messages"]
        steps = _resume_checkpoint["steps"]
        llm_calls = _resume_checkpoint["llm_calls"]
        tokens_used = _resume_checkpoint["tokens_used"]

    start_time = time.time()


    def fallback(status, reason):
        return {
            "status": status,
            "final_answer": f"{status}: {reason}",
            "steps": steps,
            "llm_calls": llm_calls,
            "tokens_used": tokens_used
        }

    for iteration in range(max_iterations):

        if time.time() - start_time >= time_limit_seconds:
            return fallback(
                "FALLBACK_TIMEOUT",
                "Time limit reached. Please hand this case to a human agent."
            )

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1

        tokens_used += response.usage.total_tokens

        if tokens_used > token_limit:
            return fallback(
                "FALLBACK_TOKEN_LIMIT",
                "Token budget exceeded. Please hand this case to a human agent."
            )

        assistant_message = response.choices[0].message
        tool_calls = assistant_message.tool_calls

        # No more tool calls: model has produced its final answer
        if not tool_calls:
            final_answer = assistant_message.content or ""
            status = final_answer.split(":")[0].strip()

            return {
                "status": status,
                "final_answer": final_answer,
                "steps": steps,
                "llm_calls": llm_calls,
                "tokens_used": tokens_used
            }

        # Record the assistant's tool-call request
        messages.append({
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments
                }
                for tc in tool_calls
            ]
        })

        # Execute each requested tool
        for tc in tool_calls:
            tool_name = tc.name
            arguments = tc.arguments

            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            if tool_name not in TOOL_MAP:
                observation = {"error": "Unknown tool"}
            else:
                try:
                    observation = TOOL_MAP[tool_name](**arguments)
                except Exception as exc:
                    observation = {"error": str(exc)}

            observation_text = json.dumps(observation)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tool_name,
                "content": observation_text
            })

            
            steps.append({
                "tool": tool_name,
                "arguments": arguments,
                "observation": observation
            })

            save_checkpoint(
                run_id,
                scenario_id,
                messages,
                tokens_used,
                llm_calls,
                steps
            )

            if (
                crash_after_tools is not None
                and len(steps) == crash_after_tools
            ):
                raise SimulatedCrash(
                    f"Simulated crash after {len(steps)} tools"
                )

    return fallback(
        "FALLBACK_MAX_ITERATIONS",
        "Maximum agent steps reached. Please hand this case to a human agent."
    )


# ============================================================
# TASK 8: RESUME AGENT
# ============================================================

def resume_agent(run_id):
    checkpoint_path = STATE_DIR / f"{run_id}.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        checkpoint = json.load(f)

    print("Resuming run:", run_id)
    print("Previously completed tools:", checkpoint["tools_executed"])

    mock_llm.reset()

    user_message = checkpoint["messages"][1]["content"]

    return run_agent(
        user_message=user_message,
        scenario_id=checkpoint["scenario_id"],
        max_iterations=6,
        token_limit=6000,
        time_limit_seconds=10,
        run_id=run_id,
        _resume_checkpoint=checkpoint
    )

# Test scenario S1
print("\n" + "=" * 60)
print("TASK 6: REACT AGENT TEST - S1")
print("=" * 60)

s1_result = run_agent(
    "My order ORD-100048 reached me very late. I would like a refund.",
    "S1"
)

print("Status:", s1_result["status"])
print("Final answer:", s1_result["final_answer"])
print("Model calls:", s1_result["llm_calls"])
print("Tokens used:", s1_result["tokens_used"])
print("Tool sequence:")

for step in s1_result["steps"]:
    print(step["tool"], "->", step["observation"])

# ============================================================
# TASK 6B: RUN ALL SEVEN SCENARIOS
# ============================================================

print("\n" + "=" * 60)
print("TASK 6B: ALL AGENT SCENARIOS")
print("=" * 60)

scenarios_df = pd.read_csv(RAW_DIR / "agent_scenarios.csv")

agent_runs_path = OUTPUT_DIR / "agent_runs.jsonl"
scenario_results = []

# Write mode ensures exactly seven scenario records on a fresh run
with open(agent_runs_path, "w", encoding="utf-8") as f:

    for _, scenario in scenarios_df.iterrows():

        mock_llm.reset()

        result = run_agent(
            user_message=scenario["message"],
            scenario_id=scenario["scenario_id"]
        )

        run_record = {
            "scenario_id": scenario["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"]
        }

        f.write(json.dumps(run_record) + "\n")
        scenario_results.append(run_record)

        print(
            scenario["scenario_id"],
            "| Status:", result["status"],
            "| Calls:", result["llm_calls"],
            "| Tokens:", result["tokens_used"],
            "| Answer:", result["final_answer"]
        )

print("\nSaved scenario records:", len(scenario_results))
print("Output file:", agent_runs_path)

# ============================================================
# TASK 7: TOKEN AND TIME LIMIT DEMONSTRATIONS
# ============================================================

print("\n" + "=" * 60)
print("TASK 7: SAFETY LIMIT TESTS")
print("=" * 60)

s1_message = (
    "My order ORD-100048 reached me very late. "
    "I would like a refund."
)

# Test 1: Token limit
mock_llm.reset()

token_result = run_agent(
    user_message=s1_message,
    scenario_id="S1_TOKEN_DEMO",
    token_limit=200
)

print("\nTOKEN LIMIT TEST")
print("Status:", token_result["status"])
print("Model calls:", token_result["llm_calls"])
print("Tokens used:", token_result["tokens_used"])
print("Final answer:", token_result["final_answer"])

# Test 2: Time limit
mock_llm.reset()

time_result = run_agent(
    user_message=s1_message,
    scenario_id="S1_TIMEOUT_DEMO",
    time_limit_seconds=0
)

print("\nTIME LIMIT TEST")
print("Status:", time_result["status"])
print("Model calls:", time_result["llm_calls"])
print("Tokens used:", time_result["tokens_used"])
print("Final answer:", time_result["final_answer"])

# ============================================================
# TASK 8: CRASH AND RESUME DEMONSTRATION
# ============================================================

print("\n" + "=" * 60)
print("TASK 8: CRASH AND RESUME")
print("=" * 60)

mock_llm.reset()

demo_run_id = "S1_CRASH_DEMO"

try:
    run_agent(
        user_message=s1_message,
        scenario_id="S1",
        run_id=demo_run_id,
        crash_after_tools=2
    )

except SimulatedCrash as exc:
    print("Simulated crash:", exc)

# Checkpoint should contain the first two completed tools
checkpoint_path = STATE_DIR / f"{demo_run_id}.json"

with open(checkpoint_path, "r", encoding="utf-8") as f:
    saved = json.load(f)

print("Saved checkpoint step:", saved["step_number"])
print("Tools already executed:", saved["tools_executed"])

# Resume from the saved checkpoint
resumed_result = resume_agent(demo_run_id)

print("\nRESUMED RESULT")
print("Status:", resumed_result["status"])
print("Final answer:", resumed_result["final_answer"])
print("Total model calls:", resumed_result["llm_calls"])
print("Total tokens:", resumed_result["tokens_used"])
print("Complete tool sequence:")

for step in resumed_result["steps"]:
    print(step["tool"])
