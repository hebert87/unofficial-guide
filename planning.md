# Planning — The Unofficial Guide

## Domain

My domain is **off-campus and on-campus student housing for Green River College students** —
the practical, unofficial knowledge students share with each other about renting near campus:
what rent actually costs, which complexes have problems, how to get a security deposit back,
finding roommates, summer/subletting, avoiding scams, and budgeting for the real monthly cost.

This knowledge is valuable and hard to find officially because the college's own housing pages
are marketing ("Welcome to Campus Corner Apartments!") — they tell you what's offered, not what
it's actually like to live there, which complexes have mold, or how a landlord behaves when your
deposit is on the line. That information lives in Reddit threads, Facebook housing groups, and
word-of-mouth between students, scattered across many posts and impossible to search in one place.

## Documents

10 documents in `/data`, chosen to cover different subtopics so the system can answer a range of
questions rather than 10 sources saying the same thing:

| File | Type | Subtopic |
|------|------|----------|
| greenriver_housing.txt | Official prose | On-campus housing (the official line) |
| lease_signing_tips.txt | Checklist | What to ask before signing a lease |
| student_housing_reddit_1.txt | Q&A thread | Rent prices near campus |
| student_housing_reddit_2_complexes.txt | Q&A thread | Which complexes to avoid |
| security_deposit_horror_stories.txt | Q&A thread | Getting your deposit back |
| finding_roommates.txt | Q&A thread | Finding and vetting roommates |
| commute_vs_living_near_campus.txt | Q&A thread | Commute vs. living close |
| subletting_summer_housing.txt | Q&A thread | Summer housing / subletting |
| avoiding_rental_scams.txt | Q&A thread | Spotting rental scams |
| utilities_and_budgeting.txt | Q&A thread | The true monthly cost of housing |

Sources these are drawn from / modeled on: r/college and Green River student discussions, student
housing Facebook groups, and the official Green River College housing page. The corpus deliberately
mixes one official source against nine student-voice sources so the system can contrast the official
story with lived experience.

## Chunking Strategy

**Chunk size: ~500 characters. Overlap: ~75 characters. Method: recursive splitting on
paragraph/response boundaries** (separators in priority order: `"\n\n"`, `"\n"`, `". "`, `" "`).

Why these choices fit my documents:
- My documents are **short and structured**, not long-form guides. Most are Q&A threads where each
  bullet is a single self-contained student opinion (1–4 sentences). The natural semantic unit is
  the individual response, not an arbitrary character count.
- Splitting on `\n\n` first means the splitter **prefers to break between responses**, so a chunk
  tends to be one complete thought ("I got $0 of my $1200 deposit back because I didn't take move-in
  photos") rather than a sentence fragment.
- **Overlap (~75 chars)** protects against a split landing mid-thought and carries a little context
  (like the "Question:" header) from one chunk into the next, so a chunk isn't orphaned from what
  it's responding to.
- **Why not smaller (e.g. 200 chars):** opinion text needs enough surrounding words to carry meaning.
  A 200-char chunk like "Professor Smith's exams are heavily" matches on keywords but answers nothing.
  Too-small chunks → high distance scores and off-topic top results.
- **Why not larger (e.g. 1000+ chars):** my whole documents are only ~800–1,300 chars. A big chunk
  would merge unrelated opinions (rent prices + parking + roommates), diluting the embedding so no
  specific query matches it well. Too-large chunks → relevant detail gets buried.

**Expected chunk count: ~25–40 chunks.** This is below the project's "50+" rule of thumb, but that
guideline assumes a larger corpus — with 10 short documents this count is a function of corpus size,
not over-large chunks. I verified by printing 5 chunks and checking each is a readable, standalone
thought.

## Retrieval Approach

- **Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers`. Runs locally, no API key, no
  rate limits, and it's well-suited to short text. 384-dimensional embeddings, fast on CPU.
- **Vector store:** ChromaDB (local, persistent), storing each chunk with metadata: source filename
  and chunk index within that document (needed for citation later).
- **top-k:** start at **k=4**. Enough to give the LLM a few corroborating opinions without flooding
  the context with loosely-related chunks that pull the answer off-topic. I'll tune after seeing real
  results — if answers miss context I'll raise it; if answers drift off-topic I'll lower it.

Why semantic search works even without shared words: embeddings map text to vectors by *meaning*, so
a query like "how do I get my money back when I move out" lands near a chunk about "security deposit
returns" even though they share almost no words.

**If cost weren't a constraint** I'd weigh a larger API model (e.g. OpenAI `text-embedding-3-large`
or a Cohere/Voyage model) for: higher accuracy on nuanced opinion text, longer context length (so I
could embed bigger chunks without truncation), and multilingual support (Green River has many
international students who may post in other languages). The tradeoffs are cost per call, network
latency, rate limits, and sending student data to a third party. For this project the local model is
the right call; for a real deployment I'd benchmark a paid model against MiniLM on my own eval set
before switching.

## Evaluation Plan

| # | Test question | Expected correct answer (checkable against docs) |
|---|---------------|--------------------------------------------------|
| 1 | How much do students pay for rent near campus? | Roughly $725–$1,450/month; cheapest options are rooms in large shared houses (~$725–$812 with several roommates), most students sharing pay ~$900–$1,200. (student_housing_reddit_1) |
| 2 | What should I do to make sure I get my security deposit back? | Take dated move-in photos/video, document existing damage on the checklist and email a timestamped copy, do a move-out walkthrough with the landlord in writing; in WA the landlord has 21 days to return it or itemize deductions. (security_deposit_horror_stories) |
| 3 | What are my options for housing over the summer if I have a 12-month lease? | Sublet (only if the lease allows it / with written approval), look for summer-only discounted leases, have a roommate cover the room, or use storage; check whether the lease is 9- or 12-month before signing. (subletting_summer_housing) |
| 4 | How can I avoid rental scams when apartment hunting? | Never send money for a place you haven't seen; reverse-image-search photos; refuse gift card / wire / Venmo / crypto deposits; be wary of below-market rent and "send a deposit to hold it" pressure; verify the owner via the county assessor. (avoiding_rental_scams) |
| 5 | Is it cheaper to live in the on-campus Campus Corner apartments or split a house off campus? | Splitting a house off campus is cheaper (students report ~$725–$812/room) and student opinion is that the dedicated complexes price high for a captive market; Campus Corner is more structured (quiet hours, inspections). This requires combining the official doc + reddit_2 + reddit_1 — a harder, multi-source question, likely my failure/partial case. |

(Q5 is intentionally hard — it spans three documents — so my evaluation surfaces a real limitation
rather than 5 easy passes.)

## Anticipated Challenges

1. **Multi-source synthesis (Q5):** the answer is spread across the official page, the rent thread,
   and the complexes thread. Top-k retrieval may pull chunks from only one of them, giving a partial
   answer. This is my expected failure case.
2. **One official doc vs. nine student docs:** the marketing-tone Green River doc uses different
   vocabulary than the casual student posts, so it may rarely be retrieved, or may get retrieved for
   the wrong "positive vibe" queries.
3. **Noisy/inconsistent figures:** rent numbers vary widely across posts ($725–$1,450). The LLM might
   report a single number as "the" rent instead of a range, which would be misleading.
4. **Grounding leakage:** llama-3.3 knows generic housing advice from its training data and could
   answer plausibly even when retrieval returns nothing relevant — I have to enforce grounding in the
   prompt, not hope for it.

## AI Tool Plan

I'll use Claude to help implement specific components, giving it specific sections of this spec as input:

1. **Ingestion + chunking (`pipeline.py`):** I'll give Claude my *Documents* table and *Chunking
   Strategy* section and ask it to write a script that loads every `.txt` in `/data`, strips any
   leftover boilerplate, and produces chunks at ~500 chars / ~75 overlap using a recursive splitter,
   attaching `{source_filename, chunk_index}` metadata. I expect a `load_documents()` and a
   `chunk_text()` function. I'll verify by printing 5 chunks myself.
2. **Embedding + vector store:** I'll give Claude my *Retrieval Approach* section and ask it to embed
   chunks with `all-MiniLM-L6-v2` and store them in a persistent ChromaDB collection with metadata,
   plus a `retrieve(query, k=4)` function returning chunks + sources + distance scores.
3. **Grounded generation + interface:** I'll give Claude my grounding requirement (answer from
   retrieved context only, say "I don't have enough information" otherwise) and ask it to write the
   Groq `llama-3.3-70b-versatile` prompt template with programmatic source attribution, plus the
   Gradio UI skeleton.

For each, I'll read the generated code, ask Claude to explain any ChromaDB/Groq call I don't
recognize, and correct anything that doesn't match my spec. I am NOT asking AI to write this plan or
to make my chunking/eval decisions — those are mine.

## Architecture

```
┌──────────────────┐   ┌──────────────────┐   ┌────────────────────────┐
│ 1. Ingestion     │   │ 2. Chunking      │   │ 3. Embedding + Store   │
│ load 10 .txt     │──▶│ recursive split  │──▶│ all-MiniLM-L6-v2       │
│ files from /data │   │ ~500 chars /     │   │ → ChromaDB (local,     │
│ + clean          │   │ ~75 overlap      │   │   metadata: source,idx)│
└──────────────────┘   └──────────────────┘   └───────────┬────────────┘
                                                           │
        ┌──────────────────────────────────────┐          │
        │ 5. Generation                         │          ▼
        │ Groq llama-3.3-70b-versatile          │   ┌────────────────────┐
        │ grounded prompt (context only) +      │◀──│ 4. Retrieval       │
        │ source attribution → Gradio UI        │   │ semantic search,   │
        └──────────────────────────────────────┘   │ top-k = 4          │
                                                    └────────────────────┘
```

Stages: Document Ingestion (Python loader) → Chunking (recursive splitter) →
Embedding + Vector Store (all-MiniLM-L6-v2 + ChromaDB) → Retrieval (top-k=4 semantic) →
Generation (Groq llama-3.3-70b-versatile + Gradio).
