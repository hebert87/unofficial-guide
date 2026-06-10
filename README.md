# The Unofficial Guide — Student Housing RAG

A Retrieval-Augmented Generation (RAG) system that makes unofficial, student-generated
knowledge about housing near Green River College searchable and answerable. Ask a
plain-language question ("How much do students pay for rent near campus?") and get a
grounded, cited answer drawn only from real student posts and housing documents.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your free Groq key from console.groq.com

python pipeline.py          # inspect chunks
python embed_store.py       # build the vector index + test retrieval
python app.py               # launch the Gradio UI at http://localhost:7860
```

---

## Domain and document sources

**Domain:** off-campus and on-campus student housing for Green River College students —
the practical, unofficial knowledge students share about renting near campus: what rent
actually costs, which complexes have problems, getting a deposit back, finding roommates,
summer/subletting, avoiding scams, and the real monthly cost of housing.

This knowledge is hard to find through official channels because the college's own housing
pages are marketing — they tell you what's offered, not what it's like to live there, which
complexes have mold, or how a landlord behaves when your deposit is on the line. That lives
in Reddit threads, Facebook housing groups, and word-of-mouth, scattered across many posts.

**10 source documents** (in `/data`), chosen to cover different subtopics so the system can
answer a range of questions:

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

Sources modeled on r/college and Green River student discussions, student housing Facebook
groups, and the official Green River College housing page. The corpus deliberately pairs one
official source against nine student-voice sources so the system can contrast the official
story with lived experience.

---

## Chunking strategy and reasoning

- **Chunk size:** ~500 characters
- **Overlap:** ~75 characters (snapped to the nearest word boundary so chunks don't start
  mid-word)
- **Method:** recursive splitting on paragraph/response boundaries, separators in priority
  order: `"\n\n"`, `"\n"`, `". "`, `" "`
- **Result:** 39 chunks across 10 documents (min 192 / max 550 / avg 411 chars)

**Why this fits the documents.** The documents are short and structured — mostly Q&A threads
where each bullet is a single self-contained student opinion (1–4 sentences). The natural
semantic unit is the individual response, not an arbitrary character count. Splitting on
`\n\n` first makes the splitter prefer to break between responses, so a chunk tends to be one
complete thought rather than a fragment. The ~75-char overlap protects a thought that lands on
a boundary and carries a little context (like the "Question:" header) into the next chunk.

- **Too small (e.g. 200 chars):** opinion text needs surrounding words to carry meaning; tiny
  chunks match on keywords but answer nothing, producing high distance scores and off-topic
  results.
- **Too large (e.g. 1000+ chars):** the whole documents are only ~800–1,300 chars, so a big
  chunk would merge unrelated opinions (rent + parking + roommates), diluting the embedding so
  no specific query matches well.

The 39-chunk total is below the "50+" rule of thumb, but that guideline assumes a larger
corpus — with 10 short documents this count is a function of corpus size, not over-large
chunks. (Implemented in `pipeline.py`.)

---

## Sample chunks

Five representative chunks, each with its source document. (Chunks may start mid-sentence
because the leading text is the ~75-char overlap carried from the previous chunk — this is
intentional and keeps boundary-spanning thoughts retrievable.)

**1. `student_housing_reddit_1.txt` (chunk 0)**
> Question: How much do students pay for rent near campus? Student responses: - One student
> pays $899/month but says the apartment is horrible. - One student paid: - $1450/month for a
> 1-bedroom apartment (5 minute walk)...

**2. `avoiding_rental_scams.txt` (chunk 0)**
> Source: Reddit thread - "Almost got scammed looking for student housing - PSA" Question: PSA
> for anyone apartment hunting... - The classic scam: a listing with great photos and
> below-market rent. The "landlord" says they're out of town... NEVER send money for a place
> you haven't seen in person.

**3. `security_deposit_horror_stories.txt` (chunk 1)**
> Now I photograph and date everything before unpacking a single box. - Got my full deposit
> back, but only because I did a move-out walkthrough WITH the landlord present and got them to
> sign off in writing... - In Washington State, landlords have 21 days to return your deposit
> or give an itemized list of deductions.

**4. `student_housing_reddit_2_complexes.txt` (chunk 2)**
> hear everything. I could hear my neighbor's alarm every morning at 6am. - Maintenance
> response time is the real thing to ask about. One place I lived took 48 hours to fix a broken
> heater in December... - Anything advertised as "newly renovated" usually means they painted
> over problems.

**5. `greenriver_housing.txt` (chunk 0)**
> Source: Green River College Student Housing. Welcome to Campus Corner Apartments! With all
> the comforts of home, you will surely find your fit at Campus Corner Apartments (CCA). Our
> apartment units are move-in ready for a shared living experience at GRC.

---

## Embedding model and production tradeoffs

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` (384-dimensional embeddings,
runs locally, no API key, no rate limits, fast on CPU). **Vector store:** ChromaDB, local and
persistent, with cosine distance. Each chunk is stored with metadata `{source, chunk_index}`
for citation.

**If I were deploying this for real users and cost weren't a constraint,** I'd weigh a larger
API embedding model (e.g. OpenAI `text-embedding-3-large` or a Voyage/Cohere model) against:
- **Accuracy on opinion text** — bigger models capture nuance in subjective, casual writing
  better, which matters for a corpus of student opinions.
- **Context length** — a longer max input would let me embed bigger chunks without truncation.
- **Multilingual support** — Green River has many international students who may post in other
  languages; MiniLM is English-centric.
- **Latency, cost, rate limits, and privacy** — API models add per-call cost, network latency,
  and send student data to a third party.

For this project the local model is the right call. For a real deployment I'd benchmark a paid
model against MiniLM on my own evaluation set before switching.

---

## Retrieval test results

Distances are **cosine** (0 = identical, lower = more relevant); top results are well below
the 0.5 quality threshold.

**Query 1: "How much do students pay for rent near campus?"**
| Rank | Source (chunk) | Distance |
|------|----------------|----------|
| 1 | student_housing_reddit_1.txt (0) | 0.145 |
| 2 | student_housing_reddit_1.txt (1) | 0.245 |
| 3 | utilities_and_budgeting.txt (0) | 0.314 |
| 4 | commute_vs_living_near_campus.txt (1) | 0.320 |

*Why these are relevant:* The top two chunks are the exact rent-prices thread containing the
specific dollar figures the question asks for, which is why they score lowest (0.145, 0.245).
Ranks 3–4 are about monthly cost and the cost of commuting — related to "what housing costs"
but more peripheral, correctly ranked below the direct matches.

**Query 2: "How can I avoid rental scams when apartment hunting?"**
| Rank | Source (chunk) | Distance |
|------|----------------|----------|
| 1 | avoiding_rental_scams.txt (1) | 0.320 |
| 2 | avoiding_rental_scams.txt (2) | 0.367 |
| 3 | avoiding_rental_scams.txt (0) | 0.376 |
| 4 | avoiding_rental_scams.txt (3) | 0.467 |

*Why these are relevant:* All four top results come from the single document dedicated to
rental scams, and together they cover the full answer — never paying for an unseen place,
reverse image search, avoiding untraceable payments, and verifying ownership. The retrieval
correctly concentrated on the one on-topic source.

**Query 3: "What are my options for housing over the summer if I have a 12-month lease?"**
| Rank | Source (chunk) | Distance |
|------|----------------|----------|
| 1 | subletting_summer_housing.txt (0) | 0.224 |
| 2 | subletting_summer_housing.txt (3) | 0.263 |
| 3 | subletting_summer_housing.txt (2) | 0.387 |
| 4 | subletting_summer_housing.txt (1) | 0.390 |

Semantic search also works without shared vocabulary: "get my money back when I move out"
retrieves the security-deposit document even though the chunks say "deposit returns," not the
query's words — embeddings match on meaning, not keywords.

---

## How grounded generation is enforced

Grounding is enforced in code and prompt design (`query.py`), not left to the model:

1. **Retrieval-only context.** The user question retrieves the top-k=4 chunks; only those
   chunks are placed in the prompt as numbered, source-labeled context blocks.
2. **Strict system prompt.** The system prompt instructs the model to answer using *only* the
   provided context, to never use outside/general knowledge, to not invent specifics, and to
   reply with exactly *"I don't have enough information on that."* when the context is
   insufficient.
3. **Low temperature (0.1)** keeps the model close to the supplied text rather than
   free-associating.
4. **Programmatic source attribution.** Sources are computed from the retrieved chunks'
   metadata — not parsed from the model's output — so a citation is always present and always
   accurate. On a refusal, sources are suppressed (they were retrieved but not used).

---

## Example responses

**In-scope (grounded, cited) — "How much do students pay for rent near campus?"**
> Students pay varying amounts... $899/month for an apartment; $1450/month for a 1-bedroom
> (5-minute walk); $800/month for a bedroom in a 12-person house; $725–$770/month sharing with
> several roommates; $1375/month for a private room/bath in a newer building; several students
> report $900–$1200/month when sharing.
>
> **Sources:** student_housing_reddit_1.txt, utilities_and_budgeting.txt,
> commute_vs_living_near_campus.txt

**In-scope (grounded, cited) — "What should I do to make sure I get my security deposit back?"**
> Take photos and date everything before unpacking when you move in; do a move-out walkthrough
> with the landlord present and get sign-off in writing; take a dated video walkthrough on
> move-out day. In Washington State, landlords have 21 days to return your deposit or give an
> itemized list of deductions.
>
> **Sources:** security_deposit_horror_stories.txt, avoiding_rental_scams.txt

**Out-of-scope (refusal) — "What time does the campus library close on weekends?"**
> I don't have enough information on that.
>
> **Sources:** (none)

The system declines rather than inventing a plausible answer from general knowledge — the
documents contain nothing about library hours.

---

## Query interface

A Gradio web UI (`app.py`, runs at http://localhost:7860).

- **Input field:** "Your question" — a text box (also submits on Enter), plus clickable example
  questions.
- **Output fields:** "Answer" — the grounded response; "Retrieved from" — the bulleted list of
  source documents the answer drew from (or a note that there was not enough information).

**Sample interaction transcript:**
```
Your question:  How can I avoid rental scams when apartment hunting?

Answer:         To avoid rental scams: 1) Never send money for a place you haven't seen in
                person. 2) Be suspicious of below-market rent. 3) Don't pay deposits in gift
                cards, wire, Venmo, Zelle, or crypto. 4) Verify ownership via the county
                assessor. 5) Watch for "send a deposit to hold it" pressure. 6) Reverse image
                search the photos. 7) Be wary of anyone who refuses a phone or video call.

Retrieved from: • avoiding_rental_scams.txt
```

---

## Evaluation report

Run with the 5 test questions from `planning.md`. Distances are cosine.

| # | Question | Expected answer | System response (summary) | Judgment |
|---|----------|-----------------|---------------------------|----------|
| 1 | How much do students pay for rent near campus? | ~$725–$1,450/mo; cheapest are rooms in large shared houses (~$725–$812); most sharing pay ~$900–$1,200 | Listed all the specific figures from the rent thread and the $900–$1,200 shared range | **Accurate** |
| 2 | What should I do to make sure I get my security deposit back? | Dated move-in photos/video; document existing damage; move-out walkthrough with landlord in writing; WA 21-day return law | Gave photos, dated move-out video, written landlord walkthrough, and the WA 21-day law | **Accurate** |
| 3 | Summer housing options on a 12-month lease? | Sublet (if lease allows / with approval), summer-only discounted leases, roommate covers room, storage | Gave sublet (check lease), roommate covers room, storage unit, summer-only discounted leases | **Accurate** |
| 4 | How can I avoid rental scams? | Don't pay for unseen places; reverse-image-search; refuse untraceable payments; verify owner; beware below-market rent & pressure | Gave all of these plus the phone/video-call red flag | **Accurate** |
| 5 | Cheaper: on-campus Campus Corner or splitting a house off campus? | Splitting a house is cheaper (~$725–$812/room); complexes price high for a "captive market"; Campus Corner is more structured | Said it lacks Campus Corner prices and could only note a $770 house; did not deliver the comparison | **Partially accurate (failure case)** |

### Failure case (Q5) — explanation

The expected answer required synthesizing three sources: the official Campus Corner page, the
rent thread, and the complexes thread (whose key line is *"the best deals are houses split
between 5-6 students, not the dedicated student complexes — the complexes know they have a
captive market and price accordingly"*).

The system gave a **partial** answer for two specific, pipeline-tied reasons:
1. **A document limitation.** The official `greenriver_housing.txt` is marketing copy and
   contains **no rent figures** for Campus Corner. So no Campus Corner price exists anywhere in
   the corpus to retrieve — the system correctly refused to invent one (good grounding) but
   therefore couldn't complete a numeric comparison.
2. **A retrieval miss.** The decisive "captive market" opinion lives in
   `student_housing_reddit_2_complexes.txt` chunk 2/3, but top-k=4 retrieved chunk 1 of that
   document instead, so the one piece of text that *does* answer the spirit of the question
   never reached the LLM.

This is exactly the multi-source-synthesis risk anticipated in `planning.md`. Possible fixes:
raise top-k, add the missing price data to the corpus, or add a reranking step so the most
on-point chunk in a document surfaces.

---

## Spec reflection

> _TODO (write this in your own words — required, and it's about YOUR experience):_
>
> - **One way the spec helped:** e.g. deciding chunk size/overlap and the paragraph-boundary
>   strategy in `planning.md` first meant the chunking code came out matching the document
>   structure on the first try, instead of me guessing character counts blind.
> - **One way the implementation diverged from the spec, and why:** e.g. the spec said a flat
>   ~75-char overlap, but on inspecting real chunks they started mid-word ("o go. Living..."),
>   so I changed the overlap to snap to a word boundary. Update this with what actually
>   happened for you.

---

## AI usage

> _TODO (required: at least 2 specific instances — confirm/edit these to match what you
> actually directed and changed):_
>
> 1. **Ingestion + chunking.** I gave the AI my `planning.md` Chunking Strategy section and
>    asked it to implement a loader + recursive splitter at ~500 chars / ~75 overlap with
>    `{source, chunk_index}` metadata. I reviewed the output and **changed the overlap to snap
>    to a word boundary** after inspecting real chunks.
> 2. **Grounded generation.** I gave the AI my grounding requirement and asked it to write the
>    Groq prompt + interface. I directed that **source attribution be computed in code, not
>    parsed from the model**, and added **suppressing sources on a refusal** after seeing the
>    system list sources next to an "I don't have enough information" answer.
>
> Replace/expand with your own specifics for full credit.
