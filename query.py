"""
Grounded answer generation for The Unofficial Guide (Milestone 5).

Pipeline: retrieve top-k chunks -> build a context block -> ask Groq's
llama-3.3-70b-versatile to answer USING ONLY that context. Source attribution
is computed programmatically from the retrieved chunks, not trusted to the LLM.

    from query import ask
    result = ask("How much do students pay for rent near campus?")
    print(result["answer"])
    print(result["sources"])
"""

import os

from dotenv import load_dotenv
from groq import Groq

from embed_store import retrieve

load_dotenv()
_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# Grounding is ENFORCED here: the model is told to use only the context and to
# refuse when the context is insufficient. This is the difference between a RAG
# answer and the model free-associating from its training data.
SYSTEM_PROMPT = """You are The Unofficial Guide, a question-answering assistant for \
students about housing near Green River College.

Rules you MUST follow:
1. Answer using ONLY the information in the provided context documents below. \
Do not use any outside or general knowledge.
2. If the context does not contain enough information to answer the question, \
reply with exactly: "I don't have enough information on that." Do not guess or \
fill gaps with general advice.
3. Base every claim on the context. Do not invent specifics (numbers, names, rules) \
that are not present in the context.
4. Write a concise, helpful answer in plain language for a student."""


def _build_context(hits):
    """Format retrieved chunks into a numbered, source-labeled context block."""
    blocks = []
    for i, hit in enumerate(hits, 1):
        blocks.append(
            f"[Document {i} — source: {hit['source']}]\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def ask(question, k=4):
    """Return {answer, sources, hits} for a user question, grounded in retrieval."""
    hits = retrieve(question, k=k)
    context = _build_context(hits)

    user_prompt = (
        f"Context documents:\n\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )

    response = _client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temp = stay close to the context, less invention
    )
    answer = response.choices[0].message.content.strip()

    # Source attribution computed from retrieval, NOT parsed from the LLM output,
    # so a citation is always present and always accurate. De-duplicate, keep order.
    sources = list(dict.fromkeys(hit["source"] for hit in hits))

    # If the model refused (nothing in context answered the question), don't show
    # sources — they were retrieved but not actually used to answer.
    if answer.lower().startswith("i don't have enough information"):
        sources = []

    return {"answer": answer, "sources": sources, "hits": hits}


if __name__ == "__main__":
    # End-to-end test: 2 in-scope questions + 1 out-of-scope (should refuse).
    tests = [
        "How much do students pay for rent near campus?",
        "How do I make sure I get my security deposit back?",
        "What time does the campus library close on weekends?",  # not in our docs
    ]
    for q in tests:
        print("\n" + "=" * 70)
        print(f"Q: {q}")
        print("=" * 70)
        result = ask(q)
        print("ANSWER:\n" + result["answer"])
        print("\nSOURCES: " + ", ".join(result["sources"]))
