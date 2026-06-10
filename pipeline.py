"""
Document ingestion + chunking pipeline for The Unofficial Guide.

Two jobs (per planning.md, Milestone 3):
  1. Load every .txt document in /data and clean it.
  2. Split each document into chunks (~500 chars, ~75 overlap), splitting on
     paragraph/response boundaries so each chunk is a self-contained thought.

Run directly to inspect output:
    python pipeline.py
"""

import html
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# From planning.md -> Chunking Strategy
CHUNK_SIZE = 500       # target max characters per chunk
CHUNK_OVERLAP = 75     # characters carried from the end of one chunk into the next
# Separators in priority order: prefer breaking between responses/paragraphs,
# then lines, then sentences, then words.
SEPARATORS = ["\n\n", "\n", ". ", " "]


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_documents(data_dir=DATA_DIR):
    """Load every .txt file in data_dir. Returns list of {source, raw_text}."""
    docs = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(data_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        docs.append({"source": filename, "raw_text": raw_text})
    return docs


# ----------------------------------------------------------------------------
# 2. CLEAN
# ----------------------------------------------------------------------------
def clean_text(text):
    """Light, defensive cleaning.

    Our documents are already plain text, but this guards against leftover HTML
    tags, HTML entities (&amp;, &#39;), and inconsistent whitespace in case we
    later add scraped sources.
    """
    text = html.unescape(text)              # &amp; -> &, &#39; -> '
    text = re.sub(r"<[^>]+>", "", text)      # strip any <html> tags
    text = re.sub(r"[ \t]+", " ", text)      # collapse runs of spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse 3+ blank lines to one
    # strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# ----------------------------------------------------------------------------
# 3. CHUNK  (recursive character splitter)
# ----------------------------------------------------------------------------
def _split_recursive(text, separators):
    """Split text into pieces no larger than CHUNK_SIZE, trying separators in
    order so we break on the most natural boundary that fits."""
    if len(text) <= CHUNK_SIZE:
        return [text]

    if not separators:
        # No separators left: hard-cut by character.
        return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    sep = separators[0]
    parts = text.split(sep)
    pieces, current = [], ""

    for part in parts:
        candidate = part if current == "" else current + sep + part
        if len(candidate) <= CHUNK_SIZE:
            current = candidate
        else:
            if current:
                pieces.append(current)
            # If a single part is still too big, recurse with the next separator.
            if len(part) > CHUNK_SIZE:
                pieces.extend(_split_recursive(part, separators[1:]))
                current = ""
            else:
                current = part
    if current:
        pieces.append(current)
    return pieces


def _add_overlap(pieces):
    """Prepend the tail of each previous piece to the next, so a thought split
    across a boundary stays retrievable (planning.md: overlap ~75 chars)."""
    if CHUNK_OVERLAP <= 0 or len(pieces) <= 1:
        return pieces
    out = [pieces[0]]
    for i in range(1, len(pieces)):
        tail = pieces[i - 1][-CHUNK_OVERLAP:]
        # Snap to a word boundary so the overlap starts on a whole word
        # (avoids fragments like "o go." at the start of a chunk).
        space = tail.find(" ")
        if space != -1:
            tail = tail[space + 1:]
        out.append((tail + " " + pieces[i]).strip())
    return out


def chunk_text(text):
    """Return a list of non-empty chunks for one cleaned document."""
    pieces = _split_recursive(text, SEPARATORS)
    pieces = [p.strip() for p in pieces if p.strip()]   # drop empty chunks
    return _add_overlap(pieces)


# ----------------------------------------------------------------------------
# PIPELINE: load -> clean -> chunk, with metadata attached
# ----------------------------------------------------------------------------
def build_chunks(data_dir=DATA_DIR):
    """Return list of chunk dicts: {text, source, chunk_index}."""
    chunks = []
    for doc in load_documents(data_dir):
        cleaned = clean_text(doc["raw_text"])
        for i, piece in enumerate(chunk_text(cleaned)):
            chunks.append({
                "text": piece,
                "source": doc["source"],
                "chunk_index": i,
            })
    return chunks


# ----------------------------------------------------------------------------
# Inspection (Milestone 3 checkpoint)
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DATA_DIR}\n")

    chunks = build_chunks()
    print(f"Total chunks produced: {len(chunks)}\n")

    lengths = [len(c["text"]) for c in chunks]
    print(f"Chunk length: min={min(lengths)}, max={max(lengths)}, "
          f"avg={sum(lengths) // len(lengths)}\n")

    # Print 5 evenly-spaced representative chunks (not just the first 5).
    print("=" * 70)
    print("5 REPRESENTATIVE CHUNKS — read each: is it a standalone thought?")
    print("=" * 70)
    step = max(1, len(chunks) // 5)
    for c in chunks[::step][:5]:
        print(f"\n[{c['source']} #chunk {c['chunk_index']}] "
              f"({len(c['text'])} chars)")
        print("-" * 60)
        print(c["text"])
